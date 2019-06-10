import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from samplers import Sampler
from collections import defaultdict, Counter
import copy


class MMALA(Sampler):
    """
    For reference, look at https://arxiv.org/pdf/1309.2983.pdf
    """

    def __init__(self, params, **kwargs):
        defaults = kwargs
        super(MALA, self).__init__(params, defaults)

    def step(self, closure):
        """
        Returns the sample, negative log-posterior w.r.t the old parameters if a closure is provided, and whether the sample is accepted or rejected
        """
        loss = None
        params = None
        accepted = False

        def accept(p_old, p_new):
            '''
            Return accept-reject decision (bool)
            '''
            pass


        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) / np.sqrt(group['lr'])
                    )
                    p.data.add_(-group['lr'],
                                d_p + langevin_noise.sample().cuda())
                else:
                    p.data.add_(-group['lr'], d_p)

        return params, closure(param)

    def sample(self, closure, num_samples, burn_in=100):
        chain = self.chain
        for i in range(burn_in):
            
            logp = closure()
            logp.backward()

            self.step()

class SGRLD(Sampler):

    def __init__(self):
        pass



class HAMCMC(Sampler):

    def __init__():
        pass

