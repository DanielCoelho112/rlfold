import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
import math

from utilities.networks import weights_init

# inspired in https://github.com/Aladoro/Stabilizing-Off-Policy-RL

def custom_parameterized_aug_optimizer_builder(encoder_lr, weight_decay, **kwargs):
    """Apply different optimizer parameters for S"""
    def make_optimizer(encoder,):
        assert isinstance(encoder.aug, ParameterizedReg)
        encoder_params = list(encoder.parameters())
        encoder_aug_parameters = list(encoder.aug.parameters())
        encoder_non_aug_parameters = [p for p in encoder_params if
                                      all([p is not aug_p for aug_p in
                                           encoder_aug_parameters])]
        return torch.optim.Adam([{'params': encoder_non_aug_parameters},
                              {'params': encoder_aug_parameters, **kwargs}],
                             lr=encoder_lr, weight_decay=weight_decay)
    return make_optimizer

def get_local_patches_kernel(kernel_size, device):
    patch_dim = kernel_size ** 2
    k = torch.eye(patch_dim, device=device).view(patch_dim, 1,
                                              kernel_size, kernel_size)
    return k


def extract_local_patches(input, kernel, N=None, padding=0, stride=1):
    b, c, _, _ = input.size()
    if kernel is None:
        kernel = get_local_patches_kernel(kernel_size=N, device=input.device)
    flinput = input.flatten(0, 1).unsqueeze(1)
    patches = F.conv2d(flinput, kernel, padding=padding, stride=stride)
    _, _, h, w = patches.size()
    return patches.view(b, c, -1, h, w)

class LearnS(torch.autograd.Function):
    '''Uses neighborhood around each feature gradient position to calculate the
        spatial divergence of the gradients, and uses it to update the param S,'''

    @staticmethod
    def forward(ctx, input, param, N, target_capped_ratio, eps):
        """
        input : Tensor
            representation to be processed (used for the gradient analysis).
        param : Tensor
            ALIX parameter S to be optimized.
        N : int
            filter size used to approximate the spatial divergence as a
            convolution (to calculate the ND scores), should be odd, >= 3
        target_capped_ratio : float
            target ND scores used to adaptively tune S
        eps : float
            small stabilization constant for the ND scores
        """
        ctx.save_for_backward(param)
        ctx.N = N
        ctx.target_capped_ratio = target_capped_ratio
        ctx.eps = eps
        return input

    @staticmethod
    def backward(ctx, dy):
        N = ctx.N
        target_capped_ratio=ctx.target_capped_ratio
        eps = ctx.eps
        dy_mean_B = dy.mean(0, keepdim=True)
        ave_dy_abs = torch.abs(dy_mean_B)
        pad_Hl = (N - 1) // 2
        pad_Hr = (N - 1) - pad_Hl
        pad_Wl = (N - 1) // 2
        pad_Wr = (N - 1) - pad_Wl
        pad = (pad_Wl, pad_Wr, pad_Hl, pad_Hr)
        padded_ave_dy = F.pad(dy_mean_B, pad, mode='replicate')
        loc_patches_k = get_local_patches_kernel(kernel_size=N, device=dy.device)

        local_patches_dy = extract_local_patches(
            input=padded_ave_dy, kernel=loc_patches_k, stride=1, padding=0)
        ave_dy_sq = ave_dy_abs.pow(2)
        patch_normalizer = (N * N) - 1

        unbiased_sq_signal = (local_patches_dy.pow(2).sum(
            dim=2) - ave_dy_sq) / patch_normalizer  # expected squared signal
        unbiased_sq_noise_signal = (local_patches_dy - dy_mean_B.unsqueeze(2)).pow(2).sum(
            2) / patch_normalizer  # 1 x C x x H x W expected squared noise

        unbiased_sqn2sig = (unbiased_sq_noise_signal) / (unbiased_sq_signal + eps)

        unbiased_sqn2sig_lp1 = torch.log(1 + unbiased_sqn2sig).mean()
        param_grad = target_capped_ratio - unbiased_sqn2sig_lp1

        return dy, param_grad, None, None, None
    
class ParameterizedReg(nn.Module):
    """Augmentation/Regularization wrapper where the strength parameterized
       and is tuned with a custom autograd function
        aug : nn.Module
            augmentation/Regularization layer
        parameter_init : float
            initial strength value
        param_grad_fn : str
            custom autograd function to tune the parameter
        param_grad_fn_args : list
            arguments for the custom autograd function
        """
    def __init__(self, aug, parameter_init, param_grad_fn, param_grad_fn_args):
        super().__init__()
        self.aug = aug
        self.P = nn.Parameter(data=torch.tensor(parameter_init))
        self.param_grad_fn_name = param_grad_fn
        if param_grad_fn == 'alix_param_grad':
            self.param_grad_fn = LearnS.apply
        else:
            raise NotImplementedError
        self.param_grad_fn_args = param_grad_fn_args

    def forward(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out

    def forward_no_learn(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        return out

    def forward_no_aug(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = x
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out


class LocalSignalMixing(nn.Module):
    def __init__(self, pad, fixed_batch=False, ):
        """LIX regularization layer
        pad : float
            maximum regularization shift (maximum S)
        fixed batch : bool
            compute independent regularization for each sample (slower)
        """
        super().__init__()
        # +1 to avoid that the sampled values at the borders get smoothed with 0
        self.pad = int(math.ceil(pad)) + 1
        self.base_normalization_ratio = (2 * pad + 1) / (2 * self.pad + 1)
        self.fixed_batch = fixed_batch

    def get_random_shift(self, n, c, h, w, x):
        if self.fixed_batch:
            return torch.rand(size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        else:
            return torch.rand(size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)

    def forward(self, x, max_normalized_shift=1.0):
        """
        x : Tensor
            input features
        max_normalized_shift : float
            current regularization shift in relative terms (current S)
        """
        if self.training:
            max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
            n, c, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, 'replicate')
            arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
            shift = self.get_random_shift(n, c, h, w, x)
            shift_offset = (1 - max_normalized_shift) / 2
            shift = (shift * max_normalized_shift) + shift_offset
            shift *= (2 * self.pad + 1)  # can start up to idx 2*pad + 1 - ignoring the left pad
            grid = base_grid + shift
            # normalize in [-1, 1]
            grid = grid * 2.0 / (h + 2 * self.pad) - 1
            return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        else:
            return x

    def get_grid(self, x, max_normalized_shift=1.0):
        max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
        shift = self.get_random_shift(n, c, h, w, x)
        shift_offset = (1 - max_normalized_shift) / 2
        shift = (shift * max_normalized_shift) + shift_offset
        shift *= (2 * self.pad + 1)
        grid = base_grid + shift
        # normalize in [-1, 1]
        grid = grid * 2.0 / (h + 2 * self.pad) - 1
        return grid



class NonLearnableParameterizedRegWrapper(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self.aug = aug
        assert isinstance(aug, ParameterizedReg)

    def forward(self, x):
        return self.aug.forward_no_learn(x)


class ImageEncoder(nn.Module):
    def __init__(self, lr, weight_decay, device, out_dims, checkpoint_dir, input_channels=3, pretrained=False):
        super(ImageEncoder, self).__init__()

        self.device = device
        self.out_dim = 64 * 5 * 5

        self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        aug_ = LocalSignalMixing(pad=2, fixed_batch=True)
        self.aug = ParameterizedReg(
            aug=aug_, parameter_init=0.5, param_grad_fn='alix_param_grad', param_grad_fn_args=[3, 0.535, 1e-20])
        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(), 
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     self.aug)

        
        self.linear = nn.Sequential(nn.Linear(self.out_dim, out_dims),
                                    nn.LayerNorm(out_dims), nn.Tanh())
                                   
        self.apply(weights_init)
        
        image_encoder_optim_builder = custom_parameterized_aug_optimizer_builder(
            encoder_lr=lr, weight_decay=weight_decay, lr=2e-3, betas=[0.5, 0.999])
        self.optimizer = image_encoder_optim_builder(self)
        
        self.to(self.device)
    


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)


    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))

