import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init

class CriticNetwork(nn.Module):
    def __init__(self, state_size, fc_dims, lr, weight_decay, device, checkpoint_dir, target=False):
        super(CriticNetwork, self).__init__()
        
        self.device = device
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/critic_network_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/critic_network_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/critic_network.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/critic_network.pt"

        n_actions = 2
        
        self.Q1 = nn.Sequential(
            nn.Linear(state_size + n_actions, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(state_size + n_actions, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1))

        self.apply(weights_init)
        
        self.optimizer = optimzer.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)
        
        return q1, q2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
        
        
    
