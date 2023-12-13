import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import os

from colorama import Fore
from importlib import import_module
from collections import deque

from agents.models.memory.memory import ReplayBufferStorage, make_replay_loader
from utilities.controls import carla_control, PID
from utilities.conversions import convert_11
from utilities.networks import update_target_network, RandomShiftsAug


class Agent():
    def __init__(self, training_config, augmentation_config, vehicle_measurements_config, waypoints_config, image_config, critic_config, actor_config, memory_config, control_config, maximum_speed, experiment_path, init_memory):

        self.maximum_speed = maximum_speed
        self.experiment_path = experiment_path
        self.alpha = training_config['alpha']
        self.automatic_entropy_tuning = training_config['automatic_alpha']
        self.device = torch.device(training_config['device'])
        self.batch_size = training_config['batch_size']
        self.discount_factor = training_config['discount_factor']
        self.state_size = image_config['out_dims'] + waypoints_config['out_dims'] + vehicle_measurements_config['out_dims']
        self.target_update_interval = training_config['target_update_interval']
        self.obs_info = self.parse_obs_info(
            memory_config['obs_info'])
        self.repeat_action = training_config['repeat_action']
        self.n_step = training_config['n_step']
        self.use_aug = augmentation_config['use_aug']
        self.critic_tau = critic_config['tau']
        self.num_waypoints = waypoints_config['num_waypoints']
        self.deque_size = training_config['deque_size']
        self.grad_clip = training_config['grad_clip']
        self.unc_threshold = training_config['unc_threshold']
        self.dem_used = 0.0
        self.image_size = image_config['image_size']

        if self.use_aug:
            self.aug = RandomShiftsAug(pad=augmentation_config['pad'])
            self.aug2 = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.05), ratio=(0.1, 2.1))
                ])

        if init_memory:
            experiment_name = experiment_path.split('/')[-1]
            replay_dir = f"{os.getenv('HOME')}/memory/{experiment_name}"
            self.replay_storage = ReplayBufferStorage(
                obs_info=self.obs_info, replay_dir=replay_dir, n_actions=2)
            
            self.replay_loader = make_replay_loader(replay_dir=replay_dir, obs_info=self.obs_info, max_size=memory_config['capacity'], batch_size=self.batch_size, num_workers=memory_config['num_workers'], nstep=self.n_step, discount=self.discount_factor, deque_size=self.deque_size)
        
        # image encoder
        module_str, class_str = image_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.image_encoder = _Class(lr=training_config['lr'], weight_decay=training_config['weight_decay'], out_dims=image_config['out_dims'], checkpoint_dir=self.experiment_path, device=self.device, input_channels=self.deque_size*3)

        # waypoint encoder
        module_str, class_str = waypoints_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.waypoints_encoder = _Class(lr=training_config['lr'], num_waypoints=waypoints_config['num_waypoints'], fc_dims=waypoints_config['fc_dims'],
                                       out_dims=waypoints_config['out_dims'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path)

        # vehicle measurement encoder
        module_str, class_str = vehicle_measurements_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.vm_encoder = _Class(lr=training_config['lr'], num_inputs=vehicle_measurements_config['num_inputs'], fc_dims=vehicle_measurements_config['fc_dims'],
                                       out_dims=vehicle_measurements_config['out_dims'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path)

        # critic
        module_str, class_str = critic_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.critic = _Class(state_size=self.state_size, fc_dims=critic_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, target=False)

        self.critic_target = _Class(state_size=self.state_size, fc_dims=critic_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, target=True)

        # hard update using tau=1.
        update_target_network(self.critic_target, self.critic, tau=1)

        self.pid = PID(kp=control_config['pid']['kp'], ki=control_config['pid']['ki'],
                       kd=control_config['pid']['kd'], dt=control_config['pid']['dt'], maximum_speed=maximum_speed)

        # actor
        module_str, class_str = actor_config['entry_point_steer'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.policy_steer = _Class(state_size=self.state_size, fc_dims=actor_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, log_sig_min=actor_config['log_sig_min'], log_sig_max=actor_config['log_sig_max'], epsilon=actor_config['epsilon'])

        # actor
        module_str, class_str = actor_config['entry_point_speed'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.policy_speed = _Class(state_size=self.state_size, fc_dims=actor_config['fc_dims'], lr=training_config['lr'], weight_decay=training_config['weight_decay'], device=self.device, checkpoint_dir=self.experiment_path, log_sig_min=actor_config['log_sig_min'], log_sig_max=actor_config['log_sig_max'], epsilon=actor_config['epsilon'])


        if self.automatic_entropy_tuning:
            self.target_entropy = - \
                torch.prod(torch.Tensor([2]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(training_config['alpha']), requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha], lr=training_config['lr_alpha'])
 
 
        # init vars.
        self.action_ctn = 0
        self.prev_action = None
        self.train_ctn = 0
        self._replay_iter = None

    
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @staticmethod
    def parse_obs_info(obs_info):
        for state_key, state_value in obs_info.items():
            for key, value in state_value.items():
                obs_info[state_key][key] = eval(value)
        return obs_info

    def encode(self, obs, detach=False):
        image = obs['image'] 
        waypoints = obs['waypoints']  
        vm = obs['vehicle_measurements']

        state_image = self.image_encoder(image)
        state_waypoints = self.waypoints_encoder(waypoints)
        state_vm = self.vm_encoder(vm)
            
        if detach:
            state_image = state_image.detach()
            state_waypoints = state_waypoints.detach()
            state_vm = state_vm.detach()

        state = torch.cat([state_image, state_waypoints, state_vm], dim=1) 

        return state

    @torch.no_grad()
    def choose_action(self, obs, step, training=True):

        current_velocity = self.get_current_speed(obs=obs)

        if self.action_ctn % self.repeat_action == 0:
            obs = self.filter_obs(obs=obs)
            self.update_deque(obs=obs)
            obs = self.convert_obs_into_torch(
                obs=obs, unsqueeze=True)

            state = self.encode(obs=obs, detach=True)
            
            if training:
                steer, _, _ = self.policy_steer.sample(state)
                
                speed, _, _, unc = self.policy_speed.sample(state)
                
                if unc > self.unc_threshold:
                    speed = convert_11(obs['desired_speed'])
                    self.dem_used = 1.0
                else:
                    self.dem_used = 0.0
                
            else:
                _, _, steer = self.policy_steer.sample(state)
                _, _, speed, _ = self.policy_speed.sample(state)
            
            action = torch.cat([speed, steer], dim=1)
            action = action.detach().cpu().numpy()[0]

        else:
            action = self.prev_action

        controls = carla_control(self.pid.get(
            action=action, velocity=current_velocity))

        self.action_ctn += 1
        self.prev_action = action

        return action, controls

    def random_action(self, obs):

        current_velocity = self.get_current_speed(obs=obs)

        if self.action_ctn % self.repeat_action == 0:

            action = np.asarray([random.uniform(-1, 1), random.uniform(-1, 1)])
        else:
            action = self.prev_action

        controls = carla_control(self.pid.get(
            action=action, velocity=current_velocity))

        self.action_ctn += 1
        self.prev_action = action

        return action, controls

    def filter_obs(self, obs):
        obs_ = {}

        resized_image = cv2.resize(
            obs['image']['data'], self.obs_info['image']['shape'][1:3], interpolation=cv2.INTER_AREA)

        obs_['image'] = np.einsum(
            'kij->jki', resized_image)

        obs_['waypoints'] = np.array(
            obs['waypoints']['location'])[0:self.num_waypoints, 0:2].reshape(self.num_waypoints, 2)

        obs_speed = np.array(
            obs['speed']['speed'][0] / self.maximum_speed, dtype=np.float32).reshape(1)

        obs_steer = np.array(
            obs['control']['steer'][0]).reshape(1)

        obs_['vehicle_measurements'] = np.concatenate([obs_speed, obs_steer]).reshape(2)
        
        obs_['desired_speed'] = np.array(
            obs['desired_speed'] / self.maximum_speed, dtype=np.float32).reshape(1)


        return obs_

    def update_deque(self, obs):
        self.img_deque.append(obs['image'])
        obs['image'] = np.concatenate(list(self.img_deque), axis=0)

    def convert_obs_into_torch(self, obs, unsqueeze=False):
        for key, value in obs.items():
            if unsqueeze:
                obs[key] = torch.from_numpy(
                    value).to(self.device).unsqueeze(0)
            else:
                obs[key] = torch.from_numpy(value).to(self.device)
        return obs

    def convert_obs_into_device(self, obs, unsqueeze=False):
        for key, value in obs.items():
            if unsqueeze:
                obs[key] = value.to(self.device).unsqueeze(0)
            else:
                obs[key] = value.to(self.device)
        return obs

    def remember(self, obs, action, reward, next_obs, done):
        obs = None
        next_obs = self.filter_obs(next_obs)
        self.replay_storage.add(
            action=action, reward=reward, next_obs=next_obs, done=done)


    def augment_obs(self, obs):
        img = self.aug(obs['image'].float()) / 255.
        img = img.view(self.batch_size * self.deque_size, 3, self.image_size, self.image_size)
        img = self.aug2(img)
        img = img.view(self.batch_size, 3*self.deque_size, self.image_size, self.image_size) * 255.
        obs['image'] = img
        
        return obs

    def clone_obs(self, obs):
        obs_ = {}
        for key, value in obs.items():
            obs_[key] = value.clone()

        return obs_

    def train(self, step):
        self.train_ctn += 1

        metrics = dict()
        
        if self.train_ctn < 1024:
            return metrics

        # sample batch from memory.
        obs_batch, action_batch, reward_batch, discount_batch, next_obs_batch, done_batch = tuple(
            next(self.replay_iter))

        obs_batch = self.convert_obs_into_device(
            obs_batch)
        next_obs_batch = self.convert_obs_into_device(
            next_obs_batch)

        if self.use_aug:
            obs_batch = self.augment_obs(
                obs=obs_batch)
            next_obs_batch = self.augment_obs(
                obs=next_obs_batch)
        
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        discount_batch = discount_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # critic networks.
        metrics.update(self.update_critics(
                obs_batch=obs_batch, action_batch=action_batch, reward_batch=reward_batch, discount_batch=discount_batch,
                next_obs_batch=next_obs_batch, done_batch=done_batch))
        
        # actor networks.
        metrics.update(self.update_policy(obs_batch=obs_batch))
        metrics.update(self.update_policy_speed(obs_batch=obs_batch))
        
        metrics['dem_used'] = self.dem_used

        if self.train_ctn % self.target_update_interval == 0:
            update_target_network(target=self.critic_target,
                                  source=self.critic, tau=self.critic_tau)
            
        return metrics

    def update_critics(self, obs_batch, action_batch, reward_batch, discount_batch, next_obs_batch, done_batch):
        metrics = dict()
        
        with torch.no_grad():
            next_state_batch = self.encode(next_obs_batch)
            next_state_action_speed, next_state_log_prob_speed, _, _ = self.policy_speed.sample(next_state_batch)
            next_state_action_steer, next_state_log_prob_steer, _ = self.policy_steer.sample(next_state_batch)
            next_state_action = torch.cat([next_state_action_speed, next_state_action_steer], dim=1)
            next_state_log_prob =  next_state_log_prob_speed + next_state_log_prob_steer
            
            q1_next_target, q2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_q_next_target = torch.min(
                q1_next_target, q2_next_target) - self.alpha * next_state_log_prob
            q_value_target = reward_batch + (discount_batch * min_q_next_target)

        state_batch = self.encode(obs_batch)
        # two q-functions to mitigate positive bias in the policy update step.
        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, q_value_target)
        q2_loss = F.mse_loss(q2, q_value_target)
        q_loss = q1_loss + q2_loss

        self.critic.optimizer.zero_grad(set_to_none=True)
        self.image_encoder.optimizer.zero_grad(set_to_none=True)
        self.waypoints_encoder.optimizer.zero_grad(set_to_none=True)
        self.vm_encoder.optimizer.zero_grad(set_to_none=True)

        q_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.image_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.waypoints_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.vm_encoder.parameters(), self.grad_clip)
        

        self.critic.optimizer.step()
        self.image_encoder.optimizer.step()
        self.waypoints_encoder.optimizer.step()
        self.vm_encoder.optimizer.step()
        
        metrics['critic_loss'] = round(q_loss.item(), 4)


        return metrics

    def update_policy(self, obs_batch):
        metrics = dict()
        
        state_batch = self.encode(obs_batch, detach=True)
        
        # policy steer
        actions_speed, log_prob_speed, _, _ = self.policy_speed.sample(state_batch)
        actions_steer, log_prob_steer, _ = self.policy_steer.sample(state_batch)
        
        actions = torch.cat([actions_speed, actions_steer], dim=1)
        log_prob =  log_prob_speed + log_prob_steer
        
        
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)

        policy_loss = ((self.alpha * log_prob) - min_q).mean()

        self.policy_speed.optimizer.zero_grad(set_to_none=True)
        self.policy_steer.optimizer.zero_grad(set_to_none=True)
        
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_speed.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.policy_steer.parameters(), self.grad_clip)
 
        self.policy_speed.optimizer.step()
        self.policy_steer.optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob +
                           self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

            alpha_logs = self.alpha.clone()
            

        metrics['policy_loss'] = round(policy_loss.item(), 4)
        metrics['alpha_loss'] = round(alpha_loss.item(), 4)
        metrics['alpha_logs'] = round(alpha_logs.item(), 4)
        

        return metrics 

    def update_policy_speed(self, obs_batch):
        metrics = dict()
        state_batch = self.encode(obs_batch)

        dist = self.policy_speed.get_dist(state_batch)

        ground_truth = convert_11(obs_batch['desired_speed'])
        
        loss = -torch.mean(dist.log_prob(ground_truth))

        self.policy_speed.optimizer.zero_grad(set_to_none=True)
        self.image_encoder.optimizer.zero_grad(set_to_none=True)
        self.waypoints_encoder.optimizer.zero_grad(set_to_none=True)
        self.vm_encoder.optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_speed.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.image_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.waypoints_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.vm_encoder.parameters(), self.grad_clip)
        

        self.policy_speed.optimizer.step()
        self.image_encoder.optimizer.step()
        self.waypoints_encoder.optimizer.step()
        self.vm_encoder.optimizer.step()
        
        
        metrics['policy_speed_loss'] = round(loss.item(), 4)
        
        return metrics
        

    
    @staticmethod
    def get_current_speed(obs):
        return obs['speed']['speed'][0]

    def reset(self, obs):
        self.pid.reset()
        self.img_deque = deque([], maxlen=self.deque_size)
        
        obs = self.filter_obs(obs)
        for i in range(self.deque_size):
            self.img_deque.append(obs['image'])

    def set_train_mode(self):
        self.critic.train()
        self.critic_target.train()
        self.policy_speed.train()
        self.policy_steer.train()
        self.image_encoder.train()
        self.waypoints_encoder.train()
        self.vm_encoder.train()

    def set_eval_mode(self):
        self.critic.eval()
        self.critic_target.eval()
        self.policy_speed.eval()
        self.policy_steer.eval()
        self.image_encoder.eval()
        self.waypoints_encoder.eval()
        self.vm_encoder.eval()



    def save_models(self, save_memory=False):
        print(f'{Fore.GREEN} saving models... {Fore.RESET}')

        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.policy_speed.save_checkpoint()
        self.policy_steer.save_checkpoint()
        self.image_encoder.save_checkpoint()
        self.waypoints_encoder.save_checkpoint()
        self.vm_encoder.save_checkpoint()

        
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha,
                       f'{self.experiment_path}/weights/log_alpha.pt')
            torch.save(self.alpha_optim.state_dict(),
                       f'{self.experiment_path}/weights/optimizers/log_alpha.pt')


    
    def load_models(self, save_memory=False):
        print(f'{Fore.GREEN} loading models... {Fore.RESET}')

        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        self.policy_speed.load_checkpoint()
        self.policy_steer.load_checkpoint()
        self.image_encoder.load_checkpoint()
        self.waypoints_encoder.load_checkpoint()
        self.vm_encoder.load_checkpoint()

        
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(
                f'{self.experiment_path}/weights/log_alpha.pt', map_location=self.device)
            self.alpha_optim.load_state_dict(
                torch.load(f'{self.experiment_path}/weights/optimizers/log_alpha.pt', map_location=self.device))
            self.alpha = self.log_alpha.exp()

