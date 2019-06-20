import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from samplers import Sampler
from collections import defaultdict, Counter
import copy


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
        print('Accept reject step:', end=' ')
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

        print("Gradient Vector:", self.grad_vector)
        
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
        print(is_accepted)
        return params, is_accepted


    def sample(self, closure, num_samples=1000, burn_in=100):
        chain = self.samples
        logp_array = []

        print("Burn-in phase started")
        for i in range(burn_in):
            self.zero_grad()
            print('Burn-in iter {}'.format(i+1))
            self.loss = closure()
            # print(self.loss)

            # print('Loss: {}'.format(self.loss))
            self.loss.backward()
            self.step()
            #### Checking if rejection works
            param_prev = self.params[0].clone().detach()
            self.zero_grad()
            params, is_accepted = self.accept_or_reject(closure)
            if not is_accepted:
                if not torch.eq(self.params[0].data, param_prev.data).all():
                    print("current param:", self.params[0])
                    print("previous param:", param_prev)
                    raise RuntimeError("Rejection step copying does not work")
            #####
            
        print("Sampling phase started")
        for i in range(num_samples):
            print('Sample iter: {}'.format(i+1))
            self.zero_grad()
            self.loss = closure()
            self.loss.backward()
            self.step()
            chain.append(self.accept_or_reject(closure))
            logp_array.append(-(self.loss.item()))
        
        return chain, logp_array



class SGRLD(Sampler):

    def __init__(self):
        pass



class HAMCMC(Sampler):

    def __init__():
        pass



class pSGLD(Trainer):
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

    def __init__(self, initial_lr=1.0e-5, alpha=0.99, mu=1.0e-5, use_gamma = False, **kwargs):
        super(pSGLD, self).__init__(**kwargs)
        self.params['lr'] = initial_lr
        self.params['mu'] = mu
        self.params['alpha'] = alpha
        self.params['use_gamma'] = use_gamma

    def _create_auxiliary_variables(self):
        self.lr = tensor.scalar('lr')
        self.V_t = [theano.shared(np.asarray(np.zeros(p.get_value().shape),
                                             dtype = theano.config.floatX))
                    for p in self.weights]


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