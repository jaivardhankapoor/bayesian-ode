import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
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


class MMALA(Sampler):
    """
    For reference, look at https://arxiv.org/pdf/1309.2983.pdf
    Code follows the same structure as LBFGS in pytorch
    """

    def __init__(self, params, metric_obj=None, is_full_hessian=True, **kwargs):
        if metric_obj is None:
            raise RuntimeError('Please provide a metric Object')
        self.metric_obj = metric_obj
        self.metric =  None

        defaults = kwargs
        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        if 'use_full_metric' not in defaults:
            defaults['use_full_metric'] = False
        super().__init__(params, defaults)

        if is_full_hessian and len(self.param_groups) != 1:
            raise ValueError("MMALA doesn't support per-parameter options\
                             for full hessian")
        
        self._numel_cache = None
        self.loss = None

        self.params = self.param_groups[0]['params']
        self.param_vector =  None


    def _get_flattened_grad(self):
        cnt = 0
        grads = [p.grad.data for p in self.params]
        for g in grads:
            g_vector = g.contiguous().view(-1) if cnt == 0\
                     else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        return g_vector


    def step(self):        
        
        # if len(self.samples) and not self.samples[-1][1]: # reuse metric from acc/rej if previous sample is accepted

        self.metric = self.metric_obj()
        self.state['metric'] = {k: v.clone().detach() for k,v in self.metric.items()}

        self.param_vector = parameters_to_vector(self.params)
        if torch.isnan(self.param_vector).sum() or torch.isnan(self.param_vector).sum():
            raise ValueError("Encountered NaN/Inf in parameter")
        
        lr = self.param_groups[0]['lr']

        size = self.param_vector.size()
        langevin_noise = Normal(torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*lr))
        self.grad_vector = self._get_flattened_grad()
        
        temp = -lr*self.metric['invMetric']@self.grad_vector

        self.param_vector.add_(-lr*self.metric['invMetric']@self.grad_vector)
        self.param_vector.add_(-lr*self.metric['sqrtinvMetric']@(langevin_noise.sample()))
        
        vector_to_parameters(self.param_vector, self.params)
    

    def accept_or_reject(self, closure):
        '''
        Accept/Reject Step for MMALA. Required due to asymmetrical proposal
        '''
        # print('Accept reject step:', end=' ')
        is_accepted = True
        self.zero_grad()
        new_loss = closure()
        new_loss.backward()

        # log-ratio of posteriors
        log_alpha = self.loss - new_loss
        # print('log-ratio of posteriors:{}'.format(log_alpha))
        self.metric = self.metric_obj()
        
        param_vector_prev = self.param_vector.clone().detach()
        self.param_vector = parameters_to_vector(self.params)
        
        grad_vector_prev = self.grad_vector.clone().detach()
        self.grad_vector = self._get_flattened_grad()

        # print("Gradient Vector:", self.grad_vector)
        
        lr = self.param_groups[0]['lr']

        # adding log-ratio of proposal
        # reverse proposal prob
        # print(param_vector_prev.size(), self.param_vector.size(), self.metric['invMetric'].size(), grad_vector_prev.size())
        temp = (param_vector_prev - self.param_vector + lr*torch.matmul(self.metric['invMetric'], grad_vector_prev))
        log_alpha += -1./(4*lr)*torch.matmul(temp, torch.matmul(self.metric['invMetric'], temp))
        # forward proposal prob
        temp = (self.param_vector - param_vector_prev + lr*torch.matmul(self.state['metric']['invMetric'], self.grad_vector))
        log_alpha -= -1./(4*lr)*torch.matmul(temp, torch.matmul(self.state['metric']['invMetric'], temp))


        if torch.isfinite(log_alpha) and torch.log(torch.rand(1)) < log_alpha:
            vector_to_parameters(self.param_vector, self.params)
        else:
            is_accepted = False
            vector_to_parameters(param_vector_prev, self.params)

        params = [p.clone().detach().data.numpy() for p in self.params]
        # print(is_accepted)
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
            params, is_accepted = self.accept_or_reject(closure)
            param_prev = self.params[0].clone().detach()
            self.zero_grad()
            if not is_accepted:
                if not torch.eq(self.params[0].data, param_prev.data).all():
                    print("current param:", self.params[0])
                    print("previous param:", param_prev)
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
            print('Sample iter: {}'.format(i+1))
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step()
            chain.append(self.accept_or_reject(closure))
            logp_array.append(-(self.loss.item()))
            if print_loss:
                with torch.no_grad():
                    print('Sample iter {:04d} | loss {:.06f} | accepted={}'.format(i+1, closure(add_prior=False).item(), is_accepted))
            else:
                print('Sample iter {:04d} | accepted={}'.format(i+1, is_accepted))

        
        return chain, logp_array


class pSGLD(Sampler):
    '''Preconditioned Stochastic Gradient Langevin Dynamics
    Implemented according to the paper:
    Li, Chunyuan, et al., 2015
    "Preconditioned stochastic gradient Langevin dynamics
    for deep neural networks."
    Arguments:
        initial_lr: float.
                    The initial learning rate
        alpha: float.
               Balances current vs. historic gradient
        mu: float.
            Controls curvature of preconditioning matrix
            (Corresponds to lambda in the paper)
        use_gamma: boolean.
                   Whether to use the Gamma term which is expensive to compute
    '''

    def __init__(self, params, **kwargs):
        defaults = kwargs

        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        if 'lr' not in defaults:
            defaults['lr'] = 1e-5
        if 'alpha' not in defaults:
            defaults['alpha'] = 0.99
        if 'lambda_' not in defaults:
            defaults['lambda_'] = 1e-5
        super().__init__(params, defaults)
        
        self.logp = None

    def step(self, lr=None, clipping=False):
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
                if 'V' not in self.state[p]:
                    self.state[p]['V'] = torch.zeros_like(p.grad.data)
                # else:
                    # print('alpha:', group['alpha'])
                self.state[p]['V'] = group['alpha']*self.state[p]['V'] + (1-group['alpha'])\
                        *d_p**2
                self.state[p]['G'] = torch.ones_like(self.state[p]['V'])\
                                    /(group['lambda_']*torch.ones_like(self.state[p]['V'])
                                    + torch.sqrt(self.state[p]['V']))
                                    
                # print("Gradient: {}".format(p.grad.data))
                size = d_p.size()
                # print('gradient value for parameter {}:'.format(p), d_p)
                # print('Preconditioner for parameter {}:'.format(p), self.state[p]['G'])
                if np.isnan(p.data).any():
                    exit()
                if group['add_noise']: 
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*group['lr'])
                    )
                    p.data.add_(-group['lr'],
                                self.state[p]['G']*d_p\
                                    + torch.sqrt(self.state[p]['G'])*langevin_noise.sample())
                else:
                    p.data.add_(-group['lr'],
                                self.state[p]['G']*d_p)

    def get_lr(self, t):
        lr0 = self.param_groups[0]['lr0']
        gamma = self.param_groups[0]['lr_gamma']
        t0 = self.param_groups[0]['lr_t0']
        alpha = self.param_groups[0]['lr_alpha']
        return lr0/np.power(t0+alpha*t, gamma)
            

    def sample(self, closure, num_samples=1000, burn_in=100, print_loss=False, clipping=False):
        '''
        Requires closure which provides the loss that is divide by the number of data points
        (including the prior part), that is , -logp(data, theta)/#(data points).
        This is so that the conditioner variables can be computed according to the paper.

        TODO: add exact delineation between scaling of prior and likelihood 
              gradients for updates

        '''
        chain = self.samples
        logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)
            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step(lr=self.get_lr(i), clipping=clipping)
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
            self.step(lr=self.get_lr(i+burn_in), clipping=clipping)
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

    # def _get_updates(self):
    #     n = self.params['batch_size']
    #     N = self.params['train_size']
    #     prec_lik = self.params['prec_lik']
    #     prec_prior = self.params['prec_prior']
    #     gc_norm = self.params['gc_norm']
    #     alpha = self.params['alpha']
    #     mu = self.params['mu']
    #     use_gamma = self.params['use_gamma']

    #     # compute log-likelihood
    #     error = self.model_outputs - self.true_outputs
    #     logliks = log_normal(error, prec_lik)
    #     sumloglik = logliks.sum()
    #     meanloglik = sumloglik / n

    #     # compute gradients
    #     grads = tensor.grad(cost = meanloglik, wrt = self.weights)

    #     # update preconditioning matrix
    #     V_t_next = [alpha * v + (1 - alpha) * g * g for g, v in zip(grads, self.V_t)]
    #     G_t = [1. / (mu + tensor.sqrt(v)) for v in V_t_next]

    #     logprior = log_prior_normal(self.weights, prec_prior)
    #     grads_prior = tensor.grad(cost = logprior, wrt = self.weights)

    #     updates = []
    #     [updates.append((v, v_n)) for v, v_n in zip(self.V_t, V_t_next)]

    #     for p, g, gp, gt in zip(self.weights, grads, grads_prior, G_t):
    #         # inject noise
    #         noise = tensor.sqrt(self.lr * gt) * trng.normal(p.shape)
    #         if use_gamma:
    #             # compute gamma
    #             gamma = nlinalg.extract_diag(tensor.jacobian(gt.flatten(), p).flatten(ndim=2))
    #             gamma = gamma.reshape(p.shape)
    #             updates.append((p, p + 0.5 * self.lr * ((gt * (gp + N * g)) + gamma) + noise))
    #         else:
    #             updates.append((p, p + 0.5 * self.lr * (gt * (gp + N * g)) + noise))

    #     return updates, sumloglik



class HAMCMC(Sampler):
    '''
    For reference, see https://arxiv.org/pdf/1602.03442.pdf
    
    '''
    
    def __init__(self, params, **kwargs):
        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        super().__init__(params, defaults)

        self._numel_cache = None
        self.loss = None

        self.params = self.param_groups[0]['params']
        self.param_vector =  None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _get_flattened_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _compute_hessian_vector_prod(grad, M, t):
        for i in range(t-1, t-M+1, step=-1):
            rho

    def step(self):        
        
        # if len(self.samples) and not self.samples[-1][1]: # reuse metric from acc/rej if previous sample is accepted

        self.metric = self.metric_obj()
        self.state['metric'] = {k: v.clone().detach() for k,v in self.metric.items()}

        self.param_vector = parameters_to_vector(self.params)
        if torch.isnan(self.param_vector).sum() or torch.isnan(self.param_vector).sum():
            raise ValueError("Encountered NaN/Inf in parameter")
        
        lr = self.param_groups[0]['lr']

        size = self.param_vector.size()
        langevin_noise = Normal(torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*lr))
        self.grad_vector = self._get_flattened_grad()
        
        temp = -lr*self.metric['invMetric']@self.grad_vector

        self.param_vector.add_(-lr*self.metric['invMetric']@self.grad_vector)
        self.param_vector.add_(-lr*self.metric['sqrtinvMetric']@(langevin_noise.sample()))
        
        vector_to_parameters(self.param_vector, self.params)
    

    def sample(self, closure, num_samples=1000, burn_in=100, print_loss=False):
        chain = self.samples
        logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step()
            logp_array.append(-(self.loss.item()))
            
            if print_loss:
                with torch.no_grad():
                    print('Burn-in iter {:04d} | loss {:.06f}'.format(i+1, closure(add_prior=False).item()))
            else:
                print('Burn-in iter {:04d}'.format(i+1))
            
        print("Sampling phase started")
        for i in range(num_samples):
            print('Sample iter: {}'.format(i+1))
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step()
            params = [[p.clone().detach().data.numpy() for p in group['params']]
                  for group in self.param_groups]
            chain.append((params, True))
            logp_array.append(-(self.loss.item()))
            if print_loss:
                with torch.no_grad():
                    print('Sample iter {:04d} | loss {:.06f}'.format(i+1, closure(add_prior=False).item()))
            else:
                print('Sample iter {:04d}'.format(i+1))

        
        return chain, logp_array

