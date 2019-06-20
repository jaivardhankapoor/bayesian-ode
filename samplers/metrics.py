# file for metric objects for use in 2nd-order samplers
# Implementation based on 
# https://github.com/matt-graham/hamiltonian-monte-carlo/blob/master/hmc/metrics.py

import torch
from torch import autograd
import numpy as np
from matplotlib import pyplot as plt


def eval_full_hessian(loss_grad, params):
    '''
    https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/3
    '''
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0\
                   else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = autograd.grad(g_vector[idx], params, create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0\
                 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian

class SoftAbsMetric:
    '''
    Metric object for the SoftAbs metric, for details see
    https://arxiv.org/pdf/1212.4693.pdf

    Requires: closure that returns a Hessian (or any other metric)
              softabs coefficient - lesser coeff means more regularization
    '''

    def __init__(self, closure, softabs_coeff=1.0):
        self.closure = closure
        self.softabs_coeff = softabs_coeff
        self.hess = None
        self.Metric = None
        self.sqrtMetric = None
        self.sqrtinvMetric = None
        self.metric_eigval, self.hess_eigval, self.eigvec = None, None, None

    def __call__(self, plot_invMetric=False, plot_Metric=False):
        sqrtMetric, sqrtinvMetric = self.sqrt()
        self.Metric = self.eigvec * torch.mm(torch.diag(self.metric_eigval), self.eigvec.t())
        self.invMetric = self.eigvec * torch.mm(torch.diag(1./self.metric_eigval), self.eigvec.t())
        if plot_invMetric:
            plt.figure()
            plt.imshow(self.invMetric.clone().detach().numpy())
            plt.title('Inverse Metric')
            plt.colorbar()
        if plot_Metric:
            plt.figure()
            plt.imshow(self.Metric.clone().detach().numpy())
            plt.title('Metric')
            plt.colorbar()
        d = dict(hess=self.hess, Metric=self.Metric, sqrtMetric=self.sqrtMetric,
                 log_det_sqrt=self.log_det_sqrt(), sqrtinvMetric=self.sqrtinvMetric,
                 invMetric=self.Metric)
        return d

    def softabs(self, x):
        return x / torch.tanh(x*self.softabs_coeff)

    def eig(self):
        self.hess = self.closure()
        self.hess_eigval, self.eigvec = torch.symeig(self.hess, eigenvectors=True)
        # print("Hessian:", self.hess)
        # print('Minimum eigenvalue of Hessian:', torch.min(self.hess_eigval).data.item())
        self.metric_eigval = self.softabs(self.hess_eigval)
        # print('Maximum eigenvalue of softabs metric:', torch.max(self.metric_eigval).data.item())
        return self.metric_eigval, self.hess_eigval, self.eigvec

    def sqrt(self):
        metric_eigval, hess_eigval, eigvec = self.eig()
        self.sqrtMetric = eigvec * torch.sqrt(metric_eigval)
        self.sqrtinvMetric = eigvec / torch.sqrt(metric_eigval)
        return self.sqrtMetric, self.sqrtinvMetric
    
    def log_det_sqrt(self):
        return 0.5 * torch.log(self.metric_eigval).sum()
    
class IdentityMetric:
    '''
    Returns the Identity Euclidean Metric
    '''
    def __init__(self, size):
        self.size = size
        self.invsqrtMetric = torch.eye(size)
        self.invMetric = torch.eye(size)

    def __call__(self):
        d = dict(invMetric=self.invMetric, sqrtinvMetric=self.invsqrtMetric)
        return d

class HessianMetric:
    '''
    Returns the Hessian as the metric
    Admits a closure which returns
    a full or approximated Hessian 
    Alos, an identity factor which adds some small
    number to make inverse p.d
    '''
    def __init__(self, closure, rcond=1e-6, identity_factor=1e-8):
        self.closure = closure
        self.rcond = rcond
        self.identity_factor = identity_factor
        self.Metric, self.invMetric, self.sqrtinvMetric = None, None, None
    
    def __call__(self, plot_invMetric=False):
        self.Metric = self.closure()
        self.invMetric = torch.pinverse(self.Metric, rcond=self.rcond)
        # self.sqrtinvMetric = torch.eye(self.invMetric.size()[0])
        self.sqrtinvMetric = torch.cholesky(self.invMetric + \
            self.identity_factor*torch.eye(self.invMetric.size()[0]))
        if plot_invMetric:
            # print('Eigenvalues of cholesky of inverse hessian(corrected):\n',
            # torch.symeig(self.sqrtinvMetric)[0])
            # print('Eigenvalues of inverse hessian corrected with small identity:\n',
            # torch.symeig(self.invMetric + self.identity_factor*torch.eye(self.invMetric.size()[0]))[0])
            plt.figure()
            plt.imshow(self.Metric.clone().detach().numpy())
            plt.colorbar()
        
        # print(torch.isnan(self.invMetric).sum())
        # print(torch.min(torch.symeig(self.invMetric)[0]))

        return dict(Metric=self.Metric, invMetric=self.invMetric,
                    sqrtinvMetric=self.sqrtinvMetric)
    