import torch
import torch.nn.functional as F

class Bernoulli:

  def __init__(self, dist=None):
    super().__init__()
    self._dist = dist
    self.mean = dist.mean

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def entropy(self):
    return self._dist.entropy()

  def mode(self):
    _mode = torch.round(self._dist.mean)
    return _mode.detach() +self._dist.mean - self._dist.mean.detach()

  def sample(self, sample_shape=()):
    return self._dist.rsample(sample_shape)

  def log_prob(self, x):
    _logits = self._dist.base_dist.logits
    log_probs0 = -F.softplus(_logits)
    log_probs1 = -F.softplus(-_logits)

    return log_probs0 * (1-x) + log_probs1 * x




class ContDist:

  def __init__(self, dist=None):
    super().__init__()
    self._dist = dist
    self.mean = dist.mean

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def entropy(self):
    return self._dist.entropy()

  def mode(self):
    return self._dist.mean

  def sample(self, sample_shape=()):
    return self._dist.rsample(sample_shape)

  def log_prob(self, x):
    return self._dist.log_prob(x)

    
class SafeTruncatedNormal(torch.distributions.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event
