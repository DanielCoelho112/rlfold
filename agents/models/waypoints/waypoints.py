import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init

class WayConv1D(nn.Module):
    def __init__(self, lr, num_waypoints, fc_dims, out_dims, weight_decay, device, checkpoint_dir):
        super(WayConv1D, self).__init__()

        self.num_waypoints = num_waypoints
        self.device = device

        self.checkpoint_file = f"{checkpoint_dir}/weights/waypoint_encoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/waypoint_encoder.pt"

        
        self.conv = nn.Conv1d(2, fc_dims, 2)
        
        n_element = self.get_output_conv()
        
        self.linear_layers = nn.Sequential(
            nn.Linear(n_element, fc_dims),
            nn.LayerNorm(fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, out_dims),
            nn.LayerNorm(out_dims),
            nn.Tanh()   
        )
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)

        
    def get_output_conv(self):
        waypoints = torch.zeros(size=(1,self.num_waypoints, 2))
        waypoints = waypoints.permute(0, 2, 1)
        return self.conv(waypoints).nelement()
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.view(x.shape[0], -1)
        x = self.linear_layers(x)
        
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
