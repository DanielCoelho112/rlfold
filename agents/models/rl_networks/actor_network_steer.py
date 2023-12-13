import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init

class ActorNetwork(nn.Module):
    def __init__(self, state_size, fc_dims, lr, weight_decay, device, checkpoint_dir, log_sig_min=-20, log_sig_max=2, epsilon=1e-6):
        super(ActorNetwork, self).__init__()
        
        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/actor_network_steer.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/actor_network_steer.pt"
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        
        self.base = nn.Sequential(
            nn.Linear(state_size, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU())
        
        self.mean_linear = nn.Linear(fc_dims, 1)
        self.log_std_linear = nn.Linear(fc_dims, 1)
        
        self.apply(weights_init)
        
        self.optimizer = optimzer.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)

        
    def forward(self, state):
        x = self.base(state)
        mean = torch.tanh(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick (mean + std * N(0,1)).
        action = torch.tanh(x_t) # convert action into [-1,1].
        
        log_prob = normal.log_prob(x_t)
        
        # enforcing action bound. 
        log_prob -= torch.log((1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
       
        
        return action, log_prob, mean
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
        
        
        
    
        