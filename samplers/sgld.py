import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from samplers import Sampler
from collections import defaultdict, Counter
import copy


class MALA(Sampler):
    """
    For reference, look at https://arxiv.org/pdf/1309.2983.pdf
    """

    def __init__(self, params, **kwargs):
        defaults = kwargs

        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        super().__init__(params, defaults)
        
        self.logp = None

    def step(self):
        """
        Returns the sample, negative log-posterior w.r.t the old parameters if a closure is provided, and whether the sample is accepted or rejected
        """

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue # assuming only Parameter objects are posterior vars
                self.state[p]['data'] = p.data
                self.state[p]['grad'] = p.grad.data
                d_p = p.grad.data
                # print("Gradient: {}".format(p.grad.data))
                size = d_p.size()
                if np.isnan(p.data).any():
                    exit()
                if group['add_noise']: 
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*group['lr'])
                    )
                    p.data.add_(-group['lr'],
                                d_p + langevin_noise.sample())
                else:
                    p.data.add_(-group['lr'],
                                d_p)


    def accept_or_reject(self, closure):
        '''
        Accept/Reject Step for MALA. Required due to asymmetrical proposal
        '''
        is_accepted = True
        if self.param_groups[0]['add_noise']:
            self.zero_grad()
            new_loss = closure()
            new_loss.backward()

            # log-ratio of posteriors
            log_alpha = self.loss - new_loss
            # print('log-ratio of posteriors:{}'.format(log_alpha))

            # adding log-ratio of proposal
            for group in self.param_groups:
                for p in group['params']:
                    # reverse proposal prob
                    log_alpha += -1./(4*group['lr'])*(self.state[p]['data'] - p.data + group['lr']*p.grad.data).pow(2).sum()
                    # forward proposal prob
                    log_alpha -= -1./(4*group['lr'])*(p.data - self.state[p]['data'] + group['lr']*self.state[p]['grad']).pow(2).sum()

            if torch.isfinite(log_alpha) and torch.log(torch.rand(1)) < log_alpha:
                pass
            else:
                is_accepted = False
                for group in self.param_groups:
                    for p in group['params']:
                        p.data = self.state[p]['data']
                        # p.grad.data = self.state[p]['grad']

        params = [[p.clone().detach().data.numpy() for p in group['params']]
                  for group in self.param_groups]
        return params, is_accepted


    def sample(self, closure, num_samples=1000, burn_in=100):
        chain = self.samples
        logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            # print('Burn-in iter {}'.format(i+1))
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)

            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step()
            params, is_accepted = self.accept_or_reject(closure)
            if not np.isfinite(params[0][1][0]).all():
                exit()
        
        print("Sampling phase started")
        for i in range(num_samples):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step()
            chain += [self.accept_or_reject(closure)]
            # print(self.loss)
            logp_array += [-(self.loss.item())]
        
        return chain, logp_array


class SGLD(Sampler):
    pass
