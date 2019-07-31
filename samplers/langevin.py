import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from samplers import Sampler
from collections import defaultdict, Counter
import copy
from functools import reduce
import warnings


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


    def sample(self, closure, num_samples=1000, burn_in=100, print_loss=False, print_iters=False, arr_closure=None):
        chain = self.samples
        # logp_array = []

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
            # logp_array.append(-(self.loss.item()))
            sq_err_loss = closure(add_prior=False)
            if arr_closure is not None:
                arr_closure(self.loss, sq_err_loss)
            if print_loss:
                with torch.no_grad():
                    print('Burn-in iter {:04d} | loss {:.06f} | accepted={}'.format(i+1, sq_err_loss.item(), is_accepted))
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
            # logp_array.append(-(self.loss.item()))
            sq_err_loss = closure(add_prior=False)
            if arr_closure is not None:
                arr_closure(self.loss, sq_err_loss)
            if print_iters:
                if print_loss:
                    with torch.no_grad():
                        print('Sample iter {:04d} | loss {:.06f} | accepted={}'.format(i+1, sq_err_loss.item(), is_accepted))
                else:
                    print('Sample iter {:04d} | accepted={}'.format(i+1, is_accepted))
        
        return chain #, logp_array

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
        gamma = self.param_groups[0]['lr_gamma']
        t0 = self.param_groups[0]['lr_t0']
        alpha = self.param_groups[0]['lr_alpha']
        return lr0/np.power(t0+alpha*t, gamma)
            

    def sample(self, closure, num_samples=1000, burn_in=100, print_iters=False, print_loss=False, arr_closure=None):
        chain = self.samples
        # logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)
            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step(lr=self.get_lr(i))
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
            self.step(lr=self.get_lr(i+burn_in))
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


    def sample(self, closure, num_samples=1000, burn_in=100, print_iters=True, print_loss=False):
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
            if print_iters:
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
            if print_iters:
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
        if 'N' not in defaults:
            defaults['N'] = 1
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
            

    def sample(self, closure, num_samples=1000, burn_in=100, print_iters=True,
               print_loss=False, arr_closure=None, clipping=False):
        '''
        Requires closure which provides the loss that is divide by the number of data points
        (including the prior part), that is , -logp(data, theta)/#(data points).
        This is so that the conditioner variables can be computed according to the paper.

        TODO: add exact delineation between scaling of prior and likelihood 
              gradients for updates

        '''
        chain = self.samples
        # logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            self.loss /= self.param_groups[0]['N']
            # print(self.loss)
            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step(lr=self.get_lr(i), clipping=clipping)
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
            self.loss /= self.param_groups[0]['N']
            self.loss.backward()
            self.step(lr=self.get_lr(i+burn_in), clipping=clipping)
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
        
        return chain# , logp_array


    ######### Deprecated ###############################
    def _get_updates(self):
        n = self.params['batch_size']
        N = self.params['train_size']
        prec_lik = self.params['prec_lik']
        prec_prior = self.params['prec_prior']
        gc_norm = self.params['gc_norm']
        alpha = self.params['alpha']
        mu = self.params['mu']
        use_gamma = self.params['use_gamma']

        # compute log-likelihood
        error = self.model_outputs - self.true_outputs
        logliks = log_normal(error, prec_lik)
        sumloglik = logliks.sum()
        meanloglik = sumloglik / n

        # compute gradients
        grads = tensor.grad(cost = meanloglik, wrt = self.weights)

        # update preconditioning matrix
        V_t_next = [alpha * v + (1 - alpha) * g * g for g, v in zip(grads, self.V_t)]
        G_t = [1. / (mu + tensor.sqrt(v)) for v in V_t_next]

        logprior = log_prior_normal(self.weights, prec_prior)
        grads_prior = tensor.grad(cost = logprior, wrt = self.weights)

        updates = []
        [updates.append((v, v_n)) for v, v_n in zip(self.V_t, V_t_next)]

        for p, g, gp, gt in zip(self.weights, grads, grads_prior, G_t):
            # inject noise
            noise = tensor.sqrt(self.lr * gt) * trng.normal(p.shape)
            if use_gamma:
                # compute gamma
                gamma = nlinalg.extract_diag(tensor.jacobian(gt.flatten(), p).flatten(ndim=2))
                gamma = gamma.reshape(p.shape)
                updates.append((p, p + 0.5 * self.lr * ((gt * (gp + N * g)) + gamma) + noise))
            else:
                updates.append((p, p + 0.5 * self.lr * (gt * (gp + N * g)) + noise))

        return updates, sumloglik
    ######### Deprecated ###############################




################## DEBUG!!! #############################################

class HAMCMC(Sampler):
    '''
    For reference, see https://arxiv.org/pdf/1602.03442.pdf

    Assumption:  full batch training
    TODO: convert to minibatch/stochastic training
    
    '''
    
    def __init__(self, params, memory=5, **kwargs):
        defaults = kwargs
        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        super().__init__(params, defaults)

        if 'trust_reg' not in self.param_groups[0]:
            self.param_groups[0]['trust_reg'] = 1e0
        if 'H_gamma' not in self.param_groups[0]:
            self.param_groups[0]['H_gamma'] = 1e0

        self._numel_cache = None
        self.loss = None

        self.params = self.param_groups[0]['params']
        self.param_vector =  None

        self.memory = memory + 1

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self.params, 0)
        return self._numel_cache

    def _get_flattened_grad(self):
        ###### TODO ######
        # Transfer loss computation and backprop and optional param change here
        # for minibatch setting gradient diff.
        ###### TODO ######

        views = []
        for p in self.params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _compute_vector_prod_old(self, grad, noise, add_noise=True):
        '''
        Uses normal L-BFGS recursion with cholesky decoomposition
        for square root of inv. Hessian
        Does not use 2-loop recursion 
        '''

        def mm(a,b):
            '''
            matrix multiplication of 2 tensors
            a:1d, b:1d -- returns inner product
            a:1d, b:2d -- returns matrix mult after broadcasting a and removes dim
            a:2d, b:1d -- returns vector prod
            a:2d, b:2d -- returns matrix mult
            '''
            return torch.matmul(a,b)

        def outer(a,b):
            '''
            returns outer product of 2 vectors
            '''
            return torch.ger(a, b)

        grad_vec, noise_vec = None, None
        mem = self.state['memory']
        M = self.memory
        
        H = self.param_groups[0]['H_gamma']*torch.eye(self._numel())
        
        for i in range(len(mem['param_diff'])):
            si = mem['param_diff'][i]
            yi = mem['grad_diff'][i]
            siTyi = mm(si, yi)
            if siTyi.item() < 0:
                warnings.warn("Curvature condition is not satisfied: yiYsi={:.08f}".format(siTyi.item()), RuntimeWarning)
            H = mm((torch.eye(self._numel())- outer(si, yi)/siTyi), H)
            H = mm(H, (torch.eye(self._numel())- outer(yi, si)/siTyi))
            H = H + outer(si, si)/siTyi
        
        grad_vec = mm(H, grad)
        

        if add_noise:
            S = torch.cholesky(H + self.param_groups[0]['cholesky_factor']*torch.eye(self._numel()))
            noise_vec = mm(S, noise)

        return grad_vec, noise_vec

    def _compute_vector_prod(self, grad, noise, add_noise=True):
        '''
        Ported from:
        https://github.com/yichuan-zhang/Quasi-Newton-MCMC/blob/master/hmcbfgs.m
        '''

        def mm(a,b):
            '''
            matrix multiplication of 2 tensors
            a:1d, b:1d -- returns inner product
            a:1d, b:2d -- returns matrix mult after broadcasting a and removes dim
            a:2d, b:1d -- returns vector prod
            a:2d, b:2d -- returns matrix mult
            '''
            return torch.matmul(a,b)

        def outer(a,b):
            '''
            returns outer product of 2 vectors
            '''
            return torch.ger(a, b)

        def reciprocal(a):
            '''
            Returns inverse of a diagonal matrix
            '''
            return torch.diag(torch.reciprocal(torch.diag(a)))

        def rsqrt(a):
            '''
            Returns sqrt of inverse of a diagonal matrix
            '''
            return torch.diag(torch.rsqrt(torch.diag(a)))

        def Cz_prod(z):
            C0 = torch.sqrt(mem['B0'])
            Ciz = mm(C0, z)
            M = len(mem['u'])
            for i in range(M):
                Ciz = Ciz - mm(outer(mem['v'][i], Ciz), mem['u'][i])
            return Ciz

        def CTz_prod(z):
            C0 = torch.sqrt(mem['B0'])
            # print(C0)
            M = len(mem['u'])
            # print('len of u:', M)
            # print("grad vector:", self.grad_vector)
            Ciz = z
            # print("Z before:", Ciz)
            for i in reversed(range(M)):
                # print(i, "u", mem['u'])
                # print(i, "ui", mem['u'][i])
                # print(i, "vi", mem['v'][i])
                # print(i, "v", mem['v'])
                # print(i, "Ciz", Ciz)
                # print(i, "mm", mm(outer(mem['u'][i], Ciz),mem['v'][i]))
                Ciz = Ciz - mm(outer(mem['u'][i], Ciz),mem['v'][i])
                # print(i, "Ciz", Ciz)
            CTz = mm(C0, Ciz)

            return CTz
                        
        def Sz_prod(z):
            S0 = rsqrt(mem['B0'])
            M = len(mem['u'])
            Siz = mm(S0, z)
            for i in range(M):
                Siz = Siz - mm(outer(mem['q'][i], Siz), mem['p'][i])
            return Siz

        def STz_prod(z):
            S0 = rsqrt(mem['B0'])
            # print(S0)
            M = len(mem['u'])
            Siz = z
            for i in reversed(range(M)):
                Siz = Siz - mm(outer(mem['p'][i], Siz),mem['q'][i])
            STz = mm(S0, Siz)
            return STz

        def Bz_prod(z):
            if len(mem['u'])==0:
                return mm(mem['B0'], z)
            CTz = CTz_prod(z)
            # print("CTz:", CTz)
            Cz = Cz_prod(CTz)
            # print("Cz:", Cz)
            return Cz
            # return Cz_prod(CTz_prod(z))

        def Hz_prod(z):
            if len(mem['u'])==0:
                return mm(reciprocal(mem['B0']), z)
            STz = STz_prod(z)
            # print("STz:", STz)
            Sz = Sz_prod(STz)
            # print("Sz:", Sz)
            return Sz


        grad_vec, noise_vec = None, None
        mem = self.state['memory']
        M = len(mem['param_diff'])

        # Clear memory for u, v, p, q
        del mem['u'][:]
        del mem['v'][:]
        del mem['p'][:]
        del mem['q'][:]

        # Compute u, v, p, q
        # print('param diff:', mem['param_diff'])
        for i in range(M):
            sTyi = mm(mem['param_diff'][i], mem['grad_diff'][i])
            # print("sTyi:", sTyi.item())
            if sTyi.item() < 0:
                # raise RuntimeError("Square root term negative, {}".format(sTyi.item()))
                print("sTyi < 0 with value {}".format(sTyi.item()))
                warnings.warn("sTyi < 0 with value {}".format(sTyi.item()))
            else:
                Bsi = Bz_prod(mem['param_diff'][i])
                # print("Bsi:", Bsi)
                sTBsi = mm(mem['param_diff'][i], Bsi)
                # print("Memory index ", i, "sTBsi:", sTBsi.item())
                mem['q'].append(torch.sqrt(sTyi/sTBsi) * Bsi - mem['grad_diff'][i])
                # print("qi:", mem['q'][-1])
                mem['p'].append(mem['param_diff'][i] / sTyi)
                # print("pi:", mem['p'][-1])
                mem['u'].append(torch.sqrt(sTBsi/sTyi) + Bsi)
                # print("ui:", mem['u'])
                mem['v'].append(mem['param_diff'][i] / sTBsi)
                # print("vi:", mem['v'])

        grad_vec = Hz_prod(grad)
        # print("grad:", grad)
        # print("grad_vec:", grad_vec)

        if add_noise:
            noise_vec = Sz_prod(noise)
            # print("noise:", noise)
            # print("noise_vec:", noise_vec)

        return grad_vec, noise_vec

    def _update_metric_vars(self, param_vec, grad_vec):
        # pop and append current vars
        mem = self.state['memory']
        M = self.memory

        mem['params'].append(param_vec.clone().detach().data)
        mem['grads'].append(grad_vec.clone().detach().data)
        mem['loss'].append(self.loss.item())
        # mem['param_diff'].append(-mem['params'][M-1] + mem['params'][2*M-1]) 
        # mem['grad_diff'].append(-mem['grads'][M-1] + mem['grads'][2*M-1]\
        #     + self.param_groups[0]['trust_reg']*mem['param_diff'][-1]) 
        si = -mem['params'][M-1] + mem['params'][2*M-1]
        yi = -mem['grads'][M-1] + mem['grads'][2*M-1] + self.param_groups[0]['trust_reg']*si
        if torch.matmul(si,yi).item() > 1e-8*torch.matmul(si,si).item():      
            mem['param_diff'].append(si) 
            mem['grad_diff'].append(yi) 
            mem['param_diff'].pop(0)
            mem['grad_diff'].pop(0)
        else:
            print("sTy < 0")
            warnings.warn("sTy < 0")
        # if len(mem['param_diff']):
        #     mem['param_diff'].pop(0)
        #     mem['grad_diff'].pop(0)

        mem['loss'].pop(0)
        mem['params'].pop(0)
        mem['grads'].pop(0)
        
        # print(len(mem['grad_diff']), M-1)
        # print(len(mem['param_diff']), M-1)
        # print(len(mem['params']), 2*M-1)
        # print(len(mem['grads']), 2*M-1)
        
        # assert(len(mem['grad_diff'])==M-1)
        # assert(len(mem['param_diff'])==M-1)
        assert(len(mem['loss'])==2*M-1)
        assert(len(mem['grads'])==2*M-1)
        assert(len(mem['params'])==2*M-1)
            
    def _add_to_memory(self, add_params):
        
        if 'memory' not in self.state:
            self.state['memory'] = {}
            self.state['memory']['u'] = []
            self.state['memory']['v'] = []
            self.state['memory']['p'] = []
            self.state['memory']['q'] = []
            self.state['memory']['params'] = [] # (2*m-1) values
            self.state['memory']['grads'] = [] # (2*m-1) values
            self.state['memory']['loss'] = [] # (2*m-1) values
            self.state['memory']['param_diff'] = [] # (m-1) values
            self.state['memory']['grad_diff'] = [] # (m-1) values
            self.state['memory']['B0'] = 1./self.param_groups[0]['H_gamma']*torch.eye(self._numel()) # first B0
            
        mem = self.state['memory']

        # Append param and grad for that step
        if add_params:
            mem['params'].append(self.param_vector.clone().detach().data)
            mem['grads'].append(self.grad_vector.clone().detach().data)
            mem['loss'].append(self.loss.item())
        
        # Precompute param_diff and grad_diff when memory is full
        if len(mem['params']) >= 2*self.memory - 1:
            for i in range(self.memory-1):
                si = -mem['params'][i] + mem['params'][i+self.memory]
                yi = -mem['grads'][i] + mem['grads'][i+self.memory] + self.param_groups[0]['trust_reg']*si
                if torch.matmul(si,yi).item() > 1e-4*torch.matmul(si,si).item():      
                    mem['param_diff'].append(si) 
                    mem['grad_diff'].append(yi) 
                else:
                    print("sTy < 0")
                    warnings.warn("sTy < 0")
        
                # mem['param_diff'].append(-mem['params'][i] + mem['params'][i+self.memory])
                # mem['grad_diff'].append(-mem['grads'][i] + mem['grads'][i+self.memory]\
                #     + self.param_groups[0]['trust_reg']*mem['param_diff'][i])
    
    def step_without_metric(self, lr, update_metric=True, add_noise=True, add_params=False):

        # Vectorize parameters
        self.param_vector = parameters_to_vector(self.params)

        # Check if encontered NaN values
        if torch.isnan(self.param_vector).sum() or torch.isnan(self.param_vector).sum():
            raise ValueError("Encountered NaN/Inf in parameter")
        
        size = self.param_vector.size()
        langevin_noise = Normal(torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*lr)).sample()
        self.grad_vector = self._get_flattened_grad()
        
        self.param_vector.data.add_(-lr, self.grad_vector)
        if add_noise:
            self.param_vector.data.add_(-lr, langevin_noise)
                
        # Add to memory
        if update_metric:
            self._add_to_memory(add_params=add_params)

        # Reshape params from vector
        vector_to_parameters(self.param_vector, self.params)

    def step(self, lr, use_old_lbfgs=False, add_noise=True):
        
        # Vectorize parameters
        # self.param_vector = parameters_to_vector(self.params)
        self.param_vector = self.state['memory']['params'][self.memory-1]
        # print("param vector before update:", self.param_vector)

        size = self.param_vector.size()
        langevin_noise = Normal(torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*lr)).sample()
        self.grad_vector = self._get_flattened_grad()

        # Compute vector products with metric using recursion algo
        if use_old_lbfgs:
            grad_vec, noise_vec = self._compute_vector_prod_old(self.grad_vector, langevin_noise, add_noise=add_noise)
        else:
            grad_vec, noise_vec = self._compute_vector_prod(self.grad_vector, langevin_noise, add_noise=add_noise)
        
        # Update parameters
        # lr = self.param_groups[0]['lr']
        new_param_vector = self.param_vector.add(-lr, grad_vec)
        if add_noise:
            new_param_vector.data.add_(-lr, noise_vec)
        
        # Check if encontered NaN values
        if torch.isnan(new_param_vector).sum() or torch.isnan(new_param_vector).sum():
            raise ValueError("Encountered NaN/Inf in parameter")
        
        # print("param vector after update:", self.param_vector)

        # Update metric variables
        self._update_metric_vars(new_param_vector, self.grad_vector)
        
        # Reshape params from vector
        vector_to_parameters(new_param_vector, self.params)
    
    
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
        log_alpha = mem['loss'] - new_loss
        # print('log-ratio of posteriors:{}'.format(log_alpha))
        ################# TODO implement asymetric proposal ratio ##############
        # self.metric = self.metric_obj()
        
        # param_vector_prev = self.param_vector.clone().detach()
        # self.param_vector = parameters_to_vector(self.params)
        
        # grad_vector_prev = self.grad_vector.clone().detach()
        # self.grad_vector = self._get_flattened_grad()

        # # print("Gradient Vector:", self.grad_vector)
        
        # lr = self.param_groups[0]['lr']

        # # adding log-ratio of proposal
        # # reverse proposal prob
        # # print(param_vector_prev.size(), self.param_vector.size(), self.metric['invMetric'].size(), grad_vector_prev.size())
        # temp = (param_vector_prev - self.param_vector + lr*torch.matmul(self.metric['invMetric'], grad_vector_prev))
        # log_alpha += -1./(4*lr)*torch.matmul(temp, torch.matmul(self.metric['invMetric'], temp))
        # # forward proposal prob
        # temp = (self.param_vector - param_vector_prev + lr*torch.matmul(self.state['metric']['invMetric'], self.grad_vector))
        # log_alpha -= -1./(4*lr)*torch.matmul(temp, torch.matmul(self.state['metric']['invMetric'], temp))
        ############################ TODO ######################################

        if torch.isfinite(log_alpha) and torch.log(torch.rand(1)) < log_alpha:
            vector_to_parameters(self.param_vector, self.params)
        else:
            is_accepted = False
            vector_to_parameters(param_vector_prev, self.params)

        params = [p.clone().detach().data.numpy() for p in self.params]
        # print(is_accepted)
        return params, is_accepted


    def get_lr(self, t):
        lr0 = self.param_groups[0]['lr0']
        gamma = self.param_groups[0]['lr_gamma']
        t0 = self.param_groups[0]['lr_t0']
        alpha = self.param_groups[0]['lr_alpha']
        return lr0/np.power(t0+alpha*t, gamma)
            
    def sample(self, closure, num_samples=1000, burn_in=100, print_iters=True,
               print_loss=False, use_metric=True, use_old_lbfgs=False, add_noise=True):
        chain = self.samples
        logp_array = []


        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            if i < self.memory*2 - 1 + 100:
                self.step_without_metric(lr=self.get_lr(i), add_noise=add_noise, add_params=(i>=100))
            else:
                if use_metric:
                    print('with metric')
                    self.step(lr=self.get_lr(i), use_old_lbfgs=use_old_lbfgs, add_noise=add_noise)
                else:
                    self.step_without_metric(lr=self.get_lr(i), update_metric=use_metric, add_noise=add_noise)
            logp_array.append(-(self.loss.item()))
            
            if print_iters:
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
            if use_metric:
                print('with metric')
                self.step(lr=self.get_lr(burn_in+i), use_old_lbfgs=use_old_lbfgs, add_noise=add_noise)
            else:
                self.step_without_metric(lr=self.get_lr(burn_in+i), update_metric=use_metric, add_noise=add_noise)
            params = [[p.clone().detach().data.numpy() for p in group['params']]
                  for group in self.param_groups]
            chain.append((params, True))
            logp_array.append(-(self.loss.item()))
            if print_iters:
                if print_loss:
                    with torch.no_grad():
                        print('Sample iter {:04d} | loss {:.06f}'.format(i+1, closure(add_prior=False).item()))
                else:
                    print('Sample iter {:04d}'.format(i+1))

        
        return chain, logp_array

class HAMCMC2(HAMCMC):
    '''
    Version of HAMCMC which uses contiguous samples to calculate s and y.
    For memory length M,
        theta_t = theta_{t-M} - lr*H_t(theta_{t-M+1:t-1})*grad
                + sqrt(2*lr*H_t(theta_{t-M+1:t-1}))*noise
    '''

    def __init__(self, params, memory=5, **kwargs):
        defaults = kwargs
        if 'trust_reg' not in defaults:
            defaults['trust_reg'] = 1e0
        if 'H_gamma' not in defaults:
            defaults['H_gamma'] = 1e0
        super().__init__(params, memory=memory, **defaults)

    def _update_metric_vars(self, param_vec, grad_vec):
        # pop and append current vars
        mem = self.state['memory']
        M = self.memory

        mem['params'].append(param_vec.clone().detach().data)
        mem['grads'].append(grad_vec.clone().detach().data)
        mem['param_diff'].append(-mem['params'][-2] + mem['params'][-1]) 
        mem['grad_diff'].append(-mem['grads'][-2] + mem['grads'][-1]\
            + self.param_groups[0]['trust_reg']*mem['param_diff'][-1]) 

        mem['params'].pop(0)
        mem['grads'].pop(0)
        mem['param_diff'].pop(0)
        mem['grad_diff'].pop(0)


        # print(len(mem['grad_diff']), M-2)
        # print(len(mem['param_diff']), M-2)
        # print(len(mem['params']), M)
        # print(len(mem['grads']), M)
        assert(len(mem['grad_diff'])==M-2)
        assert(len(mem['param_diff'])==M-2)
        assert(len(mem['grads'])==M)
        assert(len(mem['params'])==M)
            
    def _add_to_memory(self):
        
        if 'memory' not in self.state:
            self.state['memory'] = {}
            self.state['memory']['u'] = []
            self.state['memory']['v'] = []
            self.state['memory']['p'] = []
            self.state['memory']['q'] = []
            self.state['memory']['params'] = [] # (2*m-1) values
            self.state['memory']['grads'] = [] # (2*m-1) values
            self.state['memory']['param_diff'] = [] # (m-1) values
            self.state['memory']['grad_diff'] = [] # (m-1) values
            self.state['memory']['B0'] = 1./self.param_groups[0]['H_gamma']*torch.eye(self._numel()) # first B0
            
        mem = self.state['memory']

        # Append param and grad for that step
        mem['params'].append(self.param_vector.clone().detach().data)
        mem['grads'].append(self.grad_vector.clone().detach().data)
        
        # Precompute param_diff and grad_diff when memory is full
        M = self.memory
        if len(mem['params']) >= M:
            for j in range(M-2):
                i = j + 1
                mem['param_diff'].append(-mem['params'][i] + mem['params'][i+1])
                mem['grad_diff'].append(-mem['grads'][i] + mem['grads'][i+1]\
                    + self.param_groups[0]['trust_reg']*mem['param_diff'][-1])
    
    def step_without_metric(self, lr, update_metric=True, add_noise=True):

        # Vectorize parameters
        self.param_vector = parameters_to_vector(self.params)

        # Check if encontered NaN values
        if torch.isnan(self.param_vector).sum() or torch.isnan(self.param_vector).sum():
            raise ValueError("Encountered NaN/Inf in parameter")
        
        size = self.param_vector.size()
        langevin_noise = Normal(torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*lr)).sample()
        self.grad_vector = self._get_flattened_grad()
        
        self.param_vector.data.add_(-lr, self.grad_vector)
        if add_noise:
            self.param_vector.data.add_(-lr, langevin_noise)
                
        # Add to memory
        if update_metric:
            self._add_to_memory()

        # Reshape params from vector
        vector_to_parameters(self.param_vector, self.params)

    def step(self, lr, use_old_lbfgs=False, add_noise=True):
        
        # Vectorize parameters
        # self.param_vector = parameters_to_vector(self.params)
        self.param_vector = self.state['memory']['params'][0]
        # print("param vector before update:", self.param_vector)

        size = self.param_vector.size()
        langevin_noise = Normal(torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*lr)).sample()
        self.grad_vector = self._get_flattened_grad()

        # Compute vector products with metric using recursion algo
        if use_old_lbfgs:
            grad_vec, noise_vec = self._compute_vector_prod_old(self.grad_vector, langevin_noise, add_noise=add_noise)
        else:
            grad_vec, noise_vec = self._compute_vector_prod(self.grad_vector, langevin_noise, add_noise=add_noise)
        
        # Update parameters
        new_param_vector = self.param_vector.add(-lr, grad_vec)
        if add_noise:
            new_param_vector.data.add_(-lr, noise_vec)
        
        # Check if encontered NaN values
        if torch.isnan(new_param_vector).sum() or torch.isnan(new_param_vector).sum():
            raise ValueError("Encountered NaN/Inf in parameter")
        
        # print("param vector after update:", self.param_vector)

        # Update metric variables
        self._update_metric_vars(new_param_vector, self.grad_vector)
        
        # Reshape params from vector
        vector_to_parameters(new_param_vector, self.params)
    
    def accept_or_reject(self):
        pass

    def sample(self, closure, num_samples=1000, burn_in=100, print_loss=False, use_metric=True, use_old_lbfgs=False, add_noise=True):

        chain = self.samples
        logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            if i < self.memory:
                self.step_without_metric(lr=self.get_lr(i), add_noise=add_noise)
            else:
                if use_metric:
                    print('with metric')
                    self.step(lr=self.get_lr(i), use_old_lbfgs=use_old_lbfgs, add_noise=add_noise)
                else:
                    self.step_without_metric(lr=self.get_lr(i), update_metric=use_metric, add_noise=add_noise)
            logp_array.append(-(self.loss.item()))
            
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
            if use_metric:
                print('with metric')
                self.step(lr=self.get_lr(burn_in+i), use_old_lbfgs=use_old_lbfgs, add_noise=add_noise)
            else:
                self.step_without_metric(lr=self.get_lr(burn_in+i), update_metric=use_metric, add_noise=add_noise)
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

class HAMCMC3(HAMCMC2):
    '''
    Version of HAMCMC which uses contiguous samples to calculate s and y.
    For memory length M,
        theta_t = theta_{t-1} - lr*H_t(theta_{t-M:t-2})*grad
                + sqrt(2*lr*H_t(theta_{t-M:t-2}))*noise
    '''

    def __init__(self, params, memory=5, **kwargs):
        print('Updating:')
        defaults = kwargs
        if 'trust_reg' not in defaults:
            defaults['trust_reg'] = 1e0
        if 'H_gamma' not in defaults:
            defaults['H_gamma'] = 1e0
        super().__init__(params, memory=memory, **defaults)

    def _update_metric_vars(self, param_vec, grad_vec):
        # pop and append current vars
        mem = self.state['memory']
        M = self.memory

        mem['params'].append(param_vec.clone().detach().data)
        mem['grads'].append(grad_vec.clone().detach().data)
        mem['param_diff'].append(-mem['params'][-3] + mem['params'][-2]) 
        mem['grad_diff'].append(-mem['grads'][-3] + mem['grads'][-2]\
            + self.param_groups[0]['trust_reg']*mem['param_diff'][-1]) 

        mem['params'].pop(0)
        mem['grads'].pop(0)
        mem['param_diff'].pop(0)
        mem['grad_diff'].pop(0)


        print(len(mem['grad_diff']), M-2)
        print(len(mem['param_diff']), M-2)
        print(len(mem['params']), M)
        print(len(mem['grads']), M)
        assert(len(mem['grad_diff'])==M-2)
        assert(len(mem['param_diff'])==M-2)
        assert(len(mem['grads'])==M)
        assert(len(mem['params'])==M)
            
    def _add_to_memory(self):
        
        if 'memory' not in self.state:
            self.state['memory'] = {}
            self.state['memory']['u'] = []
            self.state['memory']['v'] = []
            self.state['memory']['p'] = []
            self.state['memory']['q'] = []
            self.state['memory']['params'] = [] # (2*m-1) values
            self.state['memory']['grads'] = [] # (2*m-1) values
            self.state['memory']['param_diff'] = [] # (m-1) values
            self.state['memory']['grad_diff'] = [] # (m-1) values
            self.state['memory']['B0'] = 1./self.param_groups[0]['H_gamma']*torch.eye(self._numel()) # first B0
            
        mem = self.state['memory']

        # Append param and grad for that step
        mem['params'].append(self.param_vector.clone().detach().data)
        mem['grads'].append(self.grad_vector.clone().detach().data)
        
        # Precompute param_diff and grad_diff when memory is full
        M = self.memory
        if len(mem['params']) >= M:
            for i in range(M-2):
                mem['param_diff'].append(-mem['params'][i] + mem['params'][i+1])
                mem['grad_diff'].append(-mem['grads'][i] + mem['grads'][i+1]\
                    + self.param_groups[0]['trust_reg']*mem['param_diff'][-1])

    
    def step(self, lr, use_old_lbfgs=False, add_noise=True):
        
        # Vectorize parameters
        # self.param_vector = parameters_to_vector(self.params)
        self.param_vector = self.state['memory']['params'][-1]
        # print("param vector before update:", self.param_vector)

        size = self.param_vector.size()
        langevin_noise = Normal(torch.zeros(size),
                        torch.ones(size) / np.sqrt(0.5*lr)).sample()
        self.grad_vector = self._get_flattened_grad()

        # Compute vector products with metric using recursion algo
        if use_old_lbfgs:
            grad_vec, noise_vec = self._compute_vector_prod_old(self.grad_vector, langevin_noise, add_noise=add_noise)
        else:
            grad_vec, noise_vec = self._compute_vector_prod(self.grad_vector, langevin_noise, add_noise=add_noise)
        
        # Update parameters
        new_param_vector = self.param_vector.add(-lr, grad_vec)
        if add_noise:
            new_param_vector.data.add_(-lr, noise_vec)
        
        # Check if encontered NaN values
        if torch.isnan(new_param_vector).sum() or torch.isnan(new_param_vector).sum():
            raise ValueError("Encountered NaN/Inf in parameter")
        
        # print("param vector after update:", self.param_vector)

        # Update metric variables
        self._update_metric_vars(new_param_vector, self.grad_vector)
        
        # Reshape params from vector
        vector_to_parameters(new_param_vector, self.params)
    
    def accept_or_reject(self):
        pass

class HAMCMC4(HAMCMC3):
    '''
    Version of HAMCMC which uses contiguous samples to calculate s and y.
    For memory length M,
        theta_t = theta_{t-1} - lr*H_t(theta_{t-M:t-1})*grad
                + sqrt(2*lr*H_t(theta_{t-M:t-1}))*noise
    '''

    def __init__(self, params, memory=5, **kwargs):
        defaults = kwargs
        if 'trust_reg' not in defaults:
            defaults['trust_reg'] = 1e0
        if 'H_gamma' not in defaults:
            defaults['H_gamma'] = 1e0
        super().__init__(params, memory=memory, **defaults)

    def _update_metric_vars(self, param_vec, grad_vec):
        # pop and append current vars
        mem = self.state['memory']
        M = self.memory

        mem['params'].append(param_vec.clone().detach().data)
        mem['grads'].append(grad_vec.clone().detach().data)
        mem['param_diff'].append(-mem['params'][-2] + mem['params'][-1]) 
        mem['grad_diff'].append(-mem['grads'][-2] + mem['grads'][-1]\
            + self.param_groups[0]['trust_reg']*mem['param_diff'][-1]) 

        mem['params'].pop(0)
        mem['grads'].pop(0)
        mem['param_diff'].pop(0)
        mem['grad_diff'].pop(0)


        # print(len(mem['grad_diff']), M-1)
        # print(len(mem['param_diff']), M-1)
        # print(len(mem['params']), M)
        # print(len(mem['grads']), M)
        assert(len(mem['grad_diff'])==M-1)
        assert(len(mem['param_diff'])==M-1)
        assert(len(mem['grads'])==M)
        assert(len(mem['params'])==M)
            
    def _add_to_memory(self):
        
        if 'memory' not in self.state:
            self.state['memory'] = {}
            self.state['memory']['u'] = []
            self.state['memory']['v'] = []
            self.state['memory']['p'] = []
            self.state['memory']['q'] = []
            self.state['memory']['params'] = [] # (2*m-1) values
            self.state['memory']['grads'] = [] # (2*m-1) values
            self.state['memory']['param_diff'] = [] # (m-1) values
            self.state['memory']['grad_diff'] = [] # (m-1) values
            self.state['memory']['B0'] = 1./self.param_groups[0]['H_gamma']*torch.eye(self._numel()) # first B0
            
        mem = self.state['memory']

        # Append param and grad for that step
        mem['params'].append(self.param_vector.clone().detach().data)
        mem['grads'].append(self.grad_vector.clone().detach().data)
        
        # Precompute param_diff and grad_diff when memory is full
        M = self.memory
        if len(mem['params']) >= M:
            for i in range(M-1):
                mem['param_diff'].append(-mem['params'][i] + mem['params'][i+1])
                mem['grad_diff'].append(-mem['grads'][i] + mem['grads'][i+1]\
                    + self.param_groups[0]['trust_reg']*mem['param_diff'][-1])

class aSGLD(Sampler):
    '''
    Adaptively Preconditioned SGLD
    Implemented according to the paper:
    
    Chandrasekaran Anirudh Bhardwaj, 2015
    "Adaptively Preconditioned stochastic gradient
    Langevin dynamics."
    '''

    def __init__(self, params, **kwargs):
        defaults = kwargs

        if 'add_noise' not in defaults:
            defaults['add_noise'] = True
        if 'lr' not in defaults:
            defaults['lr'] = 1e-5
        if 'mu' not in defaults:
            defaults['mu'] = 0.99
        # if 'lambda_' not in defaults:
        #     defaults['lambda_'] = 1e-5
        super().__init__(params, defaults)
        
        self.loss = None

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

##########################################################################

class cSGLD(Sampler):

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
        if 'lr0' not in defaults:
            defaults['lr0'] = 0.01
        if 'M' not in defaults:
            defaults['M'] = 5
        if 'beta' not in defaults:
            defaults['beta'] =  0.25

        super().__init__(params, defaults)
        
        self.logp = None


    def step(self, iter_num, lr=None):
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
                r = self._r(iter_num)
                if r > group['beta']:
                    # if group['add_noise']: 
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
        r = self._r(t)
        lr = self.param_groups[0]['lr0']/2.
        lr *= (np.cos(np.pi*r) + 1)
        return lr

    def _r(self, t):
        M = self.param_groups[0]['M']
        return ((t-1)%((self.num_iters+M)//M))/((self.num_iters+M)//M)

    def sample(self, closure, num_samples=1000, burn_in=100, print_iters=False, print_loss=False, arr_closure=None):
        chain = self.samples
        self.num_iters = num_samples + burn_in

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            self.loss = closure()
            # print(self.loss)
            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step(lr=self.get_lr(i), iter_num=i)
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
            self.step(lr=self.get_lr(i+burn_in), iter_num=i+burn_in)
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
