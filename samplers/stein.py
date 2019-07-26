import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from samplers import Sampler
from collections import defaultdict, Counter
import copy
from functools import reduce


class RBFKernel(torch.nn.Module):
  def __init__(self, sigma=None):
    super().__init__()

    self.sigma = sigma

  def forward(self, X, Y):
    '''
    Works when both are matrices with equal sized row vectors.
    '''
    dnorm2 = torch.cdist(X, Y) ** 2

    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()

    return K_XY


class SVGD(Optimizer):
    def __init__(self, params, optimizer, kernel, num_particles=20, particle_init_fn=None, **kwargs):
        defaults = kwargs
        if 'lr' not in defaults:
            defaults['lr'] = 1e-4
        super().__init__(params, defaults)
        self.optimizer = optimizer
        self.kernel = kernel
        self.num_particles = num_particles
        if particle_init_fn is not None:
            particles = []
            
        self.loss = None

    def _flatten_grads(self):
        views = []
        for p in self.param_groups[0]['params']:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _shape_grads(self, update):
        offset = 0
        for p in self.param_groups[0]['params']:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.grad.data = update[offset:offset + numel].view_as(p.grad.data)
            offset += numel
        assert offset == self._numel()

    def eval_stein_grad(self,):
        pass
        
    def phi(self, X):
        X = X.detach().requires_grad_(True)

        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

        return phi

    def lr_scheduler(self):
        pass

    def get_particles(self):
        pass

    def step(self, closure=None, eval_backward=True, lr_scheduler=None):

        loss = None
        if lr_scheduler is None:
            lr_scheduler = self.lr_scheduler()

        if closure is not None:
            loss = closure()
            if eval_backward:
                loss.backward()
            self.eval_stein_grad()
            lr_scheduler()
            for particle in particles:
                self.optimizer.step()