import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from samplers import Sampler
from collections import defaultdict, Counter
import copy


class aSGHMC(Sampler):
    '''
    Implementation of Adaptive Stochastic Gradient Hamiltonian Monte Carlo.
    Hyperparameters and preconditioners are adaptively set during
    the burn-in phase.
    Implementation inspired from 
    https://github.com/automl/pybnn/blob/master/pybnn/sampler/adaptive_sghmc.py
    For reference, see paper "Bayesian Optimization with Robust Bayesian Neural
    Networks. (2016)" 
    https://ml.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf
    '''
    
    def __init__(self, params, **kwargs):
        defaults = kwargs

        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        if 'lr' not in defaults:
            defaults['lr'] = 1e-5
        if 'mom_decay' not in defaults:
            defaults['mom_decay'] = 5e-2
        if 'lambda_' not in defaults:
            defaults['lambda_'] = 1e-5
        super().__init__(params, defaults)
        
        self.loss = None

    def step(self, lr, burn_in=False, resample_mom_every=50):
        """
        Proposal step
        """

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue # assuming only Parameter objects are posterior vars
                
                if torch.isnan(p.data).sum() or torch.isnan(p.data).sum():
                    raise ValueError("Encountered NaN/Inf in parameter")
                
                state = self.state[p]

                if len(state) == 0:
                    state['iteration'] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["v_hat"] = torch.ones_like(p)
                    state["momentum"] = torch.zeros_like(p)

                state['iteration'] += 1
                mom_decay, lambda_ = group["mom_decay"], group["lambda_"]
                # scale_grad = torch.tensor(group["scale_grad"], dtype=p.dtype)
                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]

                momentum = state["momentum"]
                gradient = p.grad.data # * scale_grad

                tau_inv = 1. / (tau + 1.)

                # update parameters during burn-in
                if burn_in:
                    tau.add_(- tau * (
                            g * g / (v_hat + lambda_)) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    g.add_(-g * tau_inv + tau_inv * gradient)  # average gradient see Eq 9 in [1] right
                    v_hat.add_(-v_hat * tau_inv + tau_inv * (gradient ** 2))  # gradient variance see Eq 8 in [1]

                minv_t = 1. / (torch.sqrt(v_hat) + lambda_)  # preconditioner

                if not burn_in and resample_mom_every is not None:
                    if state['iteration'] % resample_mom_every == 0:
                        momentum.data = torch.normal(mean=torch.zeros_like(momentum), std = torch.clamp(1./minv_t, max=1e1))

                epsilon_var = (2. * (lr ** 2) * mom_decay * minv_t - (lr ** 4))

                # sample random epsilon
                sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))

                # update momentum (Eq 10 right in [1])
                momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mom_decay * momentum)

                if group['add_noise']:
                    sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)
                    momentum.add_(sample_t)

                # update theta (Eq 10 left in [1])
                p.data.add_(momentum)

    def get_lr(self):
        if 'iter' not in self.__dict__: 
            self.iter = 0
        self.iter += 1
        return self.param_groups[0]['lr']

    def sample(self, closure, num_samples=1000, burn_in=100, print_iters=True,
               print_loss=False, lr_scheduler=None, resample_mom_every=None, arr_closure=None):
        '''
        Requires closure which provides the loss that is divide by the number of data points
        (including the prior part), that is , -logp(data, theta)/#(data points).
        This is so that the conditioner variables can be computed according to the paper.

        TODO: add exact delineation between scaling of prior and likelihood 
              gradients for updates

        '''
        chain = self.samples
        # logp_array = []
        if lr_scheduler is None:
            lr_scheduler = self.get_lr

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)
            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step(lr=lr_scheduler(), burn_in=True, resample_mom_every=resample_mom_every)
            # logp_array.append(-self.loss.item())
            sq_err_loss = closure(add_prior=False)
            if arr_closure is not None:
                arr_closure(self.loss, sq_err_loss)
            if print_iters:
                if print_loss:
                    with torch.no_grad():
                        print('Burn-in iter {:04d} | loss {:.06f}'.format(i+1, sq_err_loss.item()))
                else:
                    print('Burn-in iter {:04d}'.format(i+1))


        print("Sampling phase started")
        for i in range(num_samples):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step(lr=lr_scheduler(), burn_in=False, resample_mom_every=resample_mom_every)
            params = [[p.clone().detach().data.numpy() for p in group['params']]
                  for group in self.param_groups]
            chain.append((params, True))
            sq_err_loss = closure(add_prior=False)
            if arr_closure is not None:
                arr_closure(self.loss, sq_err_loss)
            if print_iters:
                if print_loss:
                    with torch.no_grad():
                        print('Sample iter {:04d} | loss {:.06f}'.format(i+1, sq_err_loss.item()))
                else:
                    print('Sample iter {:04d}'.format(i+1))

            # logp_array.append(-self.loss.item())
        
        return chain#, logp_array


class acSGHMC(Sampler):
    '''
    Implementation of Adaptive Stochastic Gradient Hamiltonian Monte Carlo.
    Hyperparameters and preconditioners are adaptively set during
    the burn-in phase.
    Implementation inspired from 
    https://github.com/automl/pybnn/blob/master/pybnn/sampler/adaptive_sghmc.py
    For reference, see paper "Bayesian Optimization with Robust Bayesian Neural
    Networks. (2016)" 
    https://ml.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf
    '''
    
    def __init__(self, params, **kwargs):
        defaults = kwargs

        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        if 'lr0' not in defaults:
            defaults['lr0'] = 0.01
        if 'M' not in defaults:
            defaults['M'] = 5
        if 'beta' not in defaults:
            defaults['beta'] =  0.25
        if 'mom_decay' not in defaults:
            defaults['mom_decay'] = 5e-2
        if 'lambda_' not in defaults:
            defaults['lambda_'] = 1e-5
        super().__init__(params, defaults)
        
        self.loss = None

    def step(self, lr, iter_num, burn_in=False, resample_mom_every=50):
        """
        Proposal step
        """

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue # assuming only Parameter objects are posterior vars
                
                if torch.isnan(p.data).sum() or torch.isnan(p.data).sum():
                    raise ValueError("Encountered NaN/Inf in parameter")
                
                state = self.state[p]

                if len(state) == 0:
                    state['iteration'] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["v_hat"] = torch.ones_like(p)
                    state["momentum"] = torch.zeros_like(p)

                state['iteration'] += 1
                mom_decay, lambda_ = group["mom_decay"], group["lambda_"]
                # scale_grad = torch.tensor(group["scale_grad"], dtype=p.dtype)
                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]

                momentum = state["momentum"]
                gradient = p.grad.data # * scale_grad

                tau_inv = 1. / (tau + 1.)

                # update parameters during burn-in
                if burn_in:
                    tau.add_(- tau * (
                            g * g / (v_hat + lambda_)) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    g.add_(-g * tau_inv + tau_inv * gradient)  # average gradient see Eq 9 in [1] right
                    v_hat.add_(-v_hat * tau_inv + tau_inv * (gradient ** 2))  # gradient variance see Eq 8 in [1]

                minv_t = 1. / (torch.sqrt(v_hat) + lambda_)  # preconditioner

                if not burn_in and resample_mom_every is not None:
                    if state['iteration'] % resample_mom_every == 0:
                        momentum.data = torch.normal(mean=torch.zeros_like(momentum), std = torch.clamp(1./minv_t, max=1e1))

                epsilon_var = (2. * (lr ** 2) * mom_decay * minv_t - (lr ** 4))

                # sample random epsilon
                sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))

                # update momentum (Eq 10 right in [1])
                momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mom_decay * momentum)

                r = self._r(iter_num)
                if r > group['beta']:
                    if group['add_noise']:
                        sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)
                        momentum.add_(sample_t)

                # update theta (Eq 10 left in [1])
                p.data.add_(momentum)

    def get_lr(self, t):
        r = self._r(t)
        lr = self.param_groups[0]['lr0']/2.
        lr *= (np.cos(np.pi*r) + 1)
        return lr

    def _r(self, t):
        M = self.param_groups[0]['M']
        return ((t-1)%((self.num_iters+M)//M))/((self.num_iters+M)//M)

    def sample(self, closure, num_samples=1000, burn_in=100, print_iters=True,
               print_loss=False, resample_mom_every=None, arr_closure=None):
        '''
        Requires closure which provides the loss that is divide by the number of data points
        (including the prior part), that is , -logp(data, theta)/#(data points).
        This is so that the conditioner variables can be computed according to the paper.

        TODO: add exact delineation between scaling of prior and likelihood 
              gradients for updates

        '''
        chain = self.samples
        self.num_iters = num_samples + burn_in

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)
            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step(lr=self.get_lr(i), iter_num=i, burn_in=True, resample_mom_every=resample_mom_every)
            # logp_array.append(-self.loss.item())
            sq_err_loss = closure(add_prior=False)
            if arr_closure is not None:
                arr_closure(self.loss, sq_err_loss)
            if print_iters:
                if print_loss:
                    with torch.no_grad():
                        print('Burn-in iter {:04d} | loss {:.06f}'.format(i+1, sq_err_loss.item()))
                else:
                    print('Burn-in iter {:04d}'.format(i+1))


        print("Sampling phase started")
        for i in range(num_samples):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step(lr=self.get_lr(i+burn_in), iter_num=i+burn_in, burn_in=False, resample_mom_every=resample_mom_every)
            params = [[p.clone().detach().data.numpy() for p in group['params']]
                  for group in self.param_groups]
            chain.append((params, True))
            sq_err_loss = closure(add_prior=False)
            if arr_closure is not None:
                arr_closure(self.loss, sq_err_loss)
            if print_iters:
                if print_loss:
                    with torch.no_grad():
                        print('Sample iter {:04d} | loss {:.06f}'.format(i+1, sq_err_loss.item()))
                else:
                    print('Sample iter {:04d}'.format(i+1))

            # logp_array.append(-self.loss.item())
        
        return chain#, logp_array


class SGRHMC(Sampler):
    '''
    Uses definite programming 

    '''