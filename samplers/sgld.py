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
        Proposal step
        """

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue # assuming only Parameter objects are posterior vars
                if torch.isnan(p.data).sum() or torch.isnan(p.data).sum():
                    raise ValueError("Encountered NaN/Inf in parameter")
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
                    if torch.isnan(p.data).sum() or torch.isnan(p.data).sum():
                        raise ValueError("Encountered NaN/Inf in parameter")
                    # reverse proposal prob
                    # print(self.state[p]['data'])
                    log_alpha += -1./(4*group['lr'])*(self.state[p]['data']\
                                 - p.data + group['lr']*p.grad.data).pow(2).sum()
                    # forward proposal prob
                    log_alpha -= -1./(4*group['lr'])*(p.data - self.state[p]['data']\
                                 + group['lr']*self.state[p]['grad']).pow(2).sum()

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


    def sample(self, closure, num_samples=1000, burn_in=100, print_loss=False):
        chain = self.samples
        logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)

            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step()
            #### Checking if rejection works
            param_prev = self.param_groups[0]['params'][0]
            self.zero_grad()
            params, is_accepted = self.accept_or_reject(closure)
            if not is_accepted:
                if not torch.eq(self.param_groups[0]['params'][0].data, param_prev.data).all():
                    raise RuntimeError("Rejection step copying does not work")
            #####
            logp_array.append(-(self.loss.item()))
            if print_loss:
                with torch.no_grad():
                    print('Burn-in iter {:04d} | loss {:.06f} | accepted={}'.format(i+1, closure(add_prior=False).item(), is_accepted))
            else:
                print('Burn-in iter {:04d} | accepted={}'.format(i+1, is_accepted))

            
        print("Sampling phase started")
        for i in range(num_samples):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step()
            params, is_accepted = self.accept_or_reject(closure)
            chain.append([params, is_accepted])
            logp_array.append(-(self.loss.item()))
            if print_loss:
                with torch.no_grad():
                    print('Sample iter {:04d} | loss {:.06f} | accepted={}'.format(i+1, closure(add_prior=False).item(), is_accepted))
            else:
                print('Sample iter {:04d} | accepted={}'.format(i+1, is_accepted))
        
        return chain, logp_array



class SGLD(Sampler):

    """
    For reference, look at
    https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

    Requires the following parameters during initialization:
    1. parameters for optimization
    2. learning rate params: lr0, gamma, t0, alpha
    learning rate for each step is given by lr0*(t0+alpha*t)^(-gamma)
    """

    def __init__(self, params, **kwargs):
        defaults = kwargs

        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        super().__init__(params, defaults)
        
        self.logp = None


    def step(self, lr=None):
        """
        Proposal step
        """

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue # assuming only Parameter objects are posterior vars
                if torch.isnan(p.data).sum() or torch.isnan(p.data).sum():
                    raise ValueError("Encountered NaN/Inf in parameter")
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


    def get_lr(self, t):
        lr0 = self.param_groups[0]['lr0']
        gamma = self.param_groups[0]['gamma']
        t0 = self.param_groups[0]['t0']
        alpha = self.param_groups[0]['alpha']
        return lr0/np.power(t0+alpha*t, gamma)
            

    def sample(self, closure, num_samples=1000, burn_in=100, print_loss=False):
        chain = self.samples
        logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)
            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step(lr=self.get_lr(i))
            logp_array.append(-self.loss.item())
            if print_loss:
                with torch.no_grad():
                    print('Burn-in iter {:04d} | loss {:.06f}'.format(i+1, closure(add_prior=False).item()))
            else:
                print('Burn-in iter {:04d}'.format(i+1))


        print("Sampling phase started")
        for i in range(num_samples):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step(lr=self.get_lr(i+burn_in))
            params = [[p.clone().detach().data.numpy() for p in group['params']]
                  for group in self.param_groups]
            chain.append((params, True))
            if print_loss:
                with torch.no_grad():
                    print('Sample iter {:04d} | loss {:.06f}'.format(i+1, closure(add_prior=False).item()))
            else:
                print('Sample iter {:04d}'.format(i+1))

            logp_array.append(-self.loss.item())
        
        return chain, logp_array