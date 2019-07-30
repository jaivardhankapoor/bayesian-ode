
# import statements

import os
import json
import argparse
import pickle

import torch
import torch.multiprocessing as multiprocessing

import numpy as np
import scipy.stats as ss

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import seaborn as sns


from optims.LBFGS import FullBatchLBFGS
from samplers.langevin import MALA, SGLD, pSGLD
from samplers.hamiltonian import aSGHMC


from torchdiffeq import odeint_adjoint as odeint

class VDP(torch.nn.Module):
    def forward(self, t, x):
        return torch.cat([x[:,1:2], 1*(1-x[:,0:1]**2)*x[:,1:2]-x[:,0:1]],1)

class FHN(torch.nn.Module):
    def forward(self, t, x):
        return torch.cat([3*(x[:,0:1]-x[:,0:1]**3/3.+x[:,1:2]), (0.2-3*x[:,0:1]-0.2*x[:,1:2])/3.], 1)

class LV(torch.nn.Module):
    def forward(self, t, x):
        return torch.cat([1.5*x[:,0:1]-x[:,0:1]*x[:,1:2], -3*x[:,1:2]+x[:,0:1]*x[:,1:2]], 1)
    

def K(X1, X2, sf, ell):
    dist = sq_dist(X1, X2, ell)
    return sf**2 * torch.exp(-dist / 2)

def _flatten(sequence):
        flat = [p.contiguous().view(-1) for p in sequence]
        return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

def sq_dist(X1, X2, ell):
    X1 = X1 / ell
    X1s = torch.sum(X1**2, dim=1).view([-1,1]) 
    X2 = X2 / ell
    X2s = torch.sum(X2**2, dim=1).view([1,-1])
    return -2*torch.mm(X1,torch.t(X2)) + X1s + X2s

class KernelRegression(torch.nn.Module):
        def __init__(self, U0, Zt, sf, ell, noise):
            super(KernelRegression, self).__init__()
            self.U = torch.nn.Parameter(U0, requires_grad=True)
            self.logsn = torch.nn.Parameter(torch.zeros(2)+np.log(noise), requires_grad=True)
            self.sf = sf
            self.ell = ell
            self.Z = Zt
            self.Kzz = K(Zt, Zt, sf, ell)
            self.Kzzinv = self.Kzz.inverse()
            self.L = torch.cholesky(self.Kzz)
            self.KzzinvL = torch.mm(self.Kzzinv,self.L)

        def forward(self, t, X):
            T = torch.mm(K(X, self.Z, self.sf, self.ell),self.KzzinvL)
            return torch.mm(T,self.U)


def run_optim(config, data, output):

    out_dir = os.path.join(output, config['method'])
    if 'dir_name' in config:
        out_dir = os.path.join(out_dir, str(config['id'])+config['dir_name'])
    else:
        out_dir = os.path.join(out_dir, str(config['id']))

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, str(args.id)+".json"), 'w') as f:
        json.dump(config, f)

    # create variables, functions etc
    M   = config['M'] # MxM inducing grid
    D   = 2
    sf  = config['sf']
    ell = config['ell']

    N, R, noise, x0, t, X, Y, ode_type = data.values()
    if 'noise' in config:
        noise = config['noise']

    torch.set_default_tensor_type(torch.DoubleTensor)
    xv = np.linspace(np.min([np.min(Y_[:,0]) for Y_ in Y]), np.max([np.max(Y_[:,0]) for Y_ in Y]), M)
    yv = np.linspace(np.min([np.min(Y_[:,1]) for Y_ in Y]), np.max([np.max(Y_[:,1]) for Y_ in Y]), M)
    xv,yv = np.meshgrid(xv,yv)
    Z  = np.array([xv.T.flatten(),yv.T.flatten()]).T
    Zt = torch.from_numpy(Z)
    Yt = torch.from_numpy(Y)
    U0 = 0.1*torch.randn(M*M, D)

    # gradient matching
    F_ = (Yt[:,1:,:]-Yt[:,:-1,:]) / (t[1]-t[0])
    F_ = F_.contiguous().view(-1,D)
    Z_ = Yt[:,:-1,:].contiguous().view(-1,D)
    Kxz = K(Zt, Z_, sf, ell)
    Kzzinv = (K(Z_, Z_, sf, ell)+0.2*torch.eye(Z_.shape[0])).inverse()
    U0 = torch.mm(torch.mm(Kxz,Kzzinv),F_) # not whitened
    Linv = torch.cholesky(K(Zt, Zt, sf, ell)).inverse()
    U0 = torch.mm(Linv,U0) # whitened

    kreg = KernelRegression(U0, Zt, sf, ell, noise)
    
    params = [kreg.U,kreg.logsn]
    
    optim = None
    if config['method'] == 'Adam':
        optim = torch.optim.Adam(params, lr=config['lr'])
    if 'LBFGS' in config['method']:
        optim = FullBatchLBFGS(params, lr=config['lr'], line_search=config['line_search'],
                      history_size=config['history_size'])
    if 'SGD' in config['method']:
        optim = torch.optim.SGD(params, lr=config['lr'])
        torch.nn.utils.clip_grad_norm_(params, config["clip"])
    if 'nag' in config['method']:
        optim = torch.optim.SGD(params, lr=config['lr'], momentum=0.5, nesterov=True)
        torch.nn.utils.clip_grad_norm_(params, config["clip"])
    if 'RMSprop' in config['method']:
        if 'rmsprop_alpha' not in config:
            config['rmsprop_alpha'] = 0.99
        optim = torch.optim.RMSprop(params, lr=config['lr'], alpha=config['rmsprop_alpha'])
    if 'Adadelta' in config['method']:
        if 'adadelta_rho' not in config:
            config['adadelta_rho'] = 0.9
        optim = torch.optim.Adadelta(params, lr=config['lr'], rho=config['adadelta_rho'])

     
    # n_iters = 1000
    n_iters = config['num_iters']

    ## learning rate scheduling for sgd+momentum and rmsprop
    lr = config['lr']
    lr_decay = 0
    if 'lr_decay' in config:
        lr_decay = config['lr_decay']

    ## Momentum schedulin for sgd+momentum
    if 'mom' in config:
        mom_init = 0.5
        mom_final = config['mom']
        mom_decay = 0.03

    def closure():
        loss = torch.sum((Yt-xode)**2 / torch.exp(kreg.logsn)**2)
        loss += torch.numel(Yt)*torch.sum(kreg.logsn)/D
        # print(kreg.U.shape)
        # print(kreg.Kzzinv.shape)
        temp = torch.mm(kreg.Kzzinv, kreg.U)
        temp = torch.mm(kreg.U.t(), temp)
        loss += torch.sum(torch.diag(temp))/2
        return loss

    total_loss_arr = []
    sq_err_loss_arr = []

    for itr in range(n_iters):
        optim.zero_grad()
        
        if 'lr_decay' in config:
            for g in optim.param_groups:
                g['lr'] = lr/(1+lr_decay*itr)
        if 'mom' in config:
            for g in optim.param_groups:
                g['momentum'] = mom_final - (mom_final-mom_init)/(1+mom_decay*itr)

        xode = odeint(kreg, x0, t, method='rk4').permute([1,0,2])
        
        loss = closure()
        loss.backward()
        if config['method']=='LBFGS':
            optim.step(options={'closure': closure})
        else:
            optim.step()

        if itr % 1 == 0:
            with torch.no_grad():
                xode = odeint(kreg, x0, t, method='rk4').permute([1,0,2])
                if torch.isnan(loss):
                    raise RuntimeError('Encountered NaN values in inference, method {}'.format(config['method']))
                    break
                sq_err_loss = torch.sum((Yt-xode)**2 )
                # print('Iter {:04d} | Total Loss {:.6f} | Sq. err. Loss {:.6f}'.format(itr, loss.item(), sq_err_loss.item()))
                total_loss_arr.append(loss.item())
                sq_err_loss_arr.append(sq_err_loss.item())

    # Save losses
    with open(os.path.join(out_dir, 'total_loss_arr.pickle'), 'wb') as f:
        pickle.dump(total_loss_arr, f)
    with open(os.path.join(out_dir, 'sq_err_loss_arr.pickle'), 'wb') as f:
        pickle.dump(sq_err_loss_arr, f)

    ## Plot results

    # Plot losses
    fig, ax = plt.subplots()
    ax.plot(total_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Negative log posterior')
    fig.savefig(os.path.join(out_dir, '{}_final{}.pdf'.format('post', str(total_loss_arr[-1]))), adjustable='box')
    
    fig, ax = plt.subplots()
    ax.plot(sq_err_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('sum of squares error')
    fig.savefig(os.path.join(out_dir, '{}_final{}.pdf'.format('loss', str(sq_err_loss_arr[-1]))), adjustable='box')
    
    # log losses
    fig, ax = plt.subplots()
    ax.plot(total_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(Negative log posterior)')
    ax.set_yscale('log')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format('post')), adjustable='box')
    
    fig, ax = plt.subplots()
    ax.plot(sq_err_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(sum of squares error)')
    ax.set_yscale('log')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format('loss')), adjustable='box')

    # plot phase plot
    xode = odeint(kreg, x0, t).detach().numpy()
    U_np = kreg.U.detach().numpy()
    Z_np = kreg.Z.detach().numpy()
    xode = np.transpose(xode,[1,0,2])

    cols = ['r','b','g','m','c','k']
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.scatter(Z_np[:,0],Z_np[:,1],100, facecolors='none', edgecolors='k')
    ax.quiver(Z_np[:,0],Z_np[:,1],U_np[:,0],U_np[:,1],units='height',width=0.006,color='k')
    for i in range(min(6,xode.shape[0])):
        ax.plot(Y[i,:,0],Y[i,:,1],'o',color=cols[i])
        ax.plot(xode[i,:,0],xode[i,:,1],'-',color=cols[i])

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format('phaseplot')), adjustable='box')

    # plot real vs modeled samples (3 different plots to choose the best)
    for plot_itr in range(3):
        R_= R
        x0_ = None
        if ode_type == "LV":
            x0_ = torch.from_numpy(R*ss.beta.rvs(size=[N,2], a=1, b=1.5)+ [3., 1.5])
        else:
            x0_ = torch.from_numpy(2*R_*ss.uniform.rvs(size=[10,2])-R_)
            
        t_ = torch.linspace(0., 28., 160)
        xode_gp = odeint(kreg, x0_, t_).detach().numpy()
        xode_gp = np.transpose(xode_gp,[1,0,2])

        ode_func = None
        if ode_type == "VDP":
            ode_func = VDP
        if ode_type == "LV":
            ode_func = LV
        if ode_type == "FHN":
            ode_func = FHN
        xode_real = odeint(ode_func(), x0_, t_).detach().numpy()
        xode_real = np.transpose(xode_real,[1,0,2])

        num_plots = 3
        fig, ax = plt.subplots(ncols=num_plots, figsize=(15,3))
        axes = ax
        for i in range(num_plots):
            axes[i].plot(xode_real[i,0:80,0],'-',color='k', label='pos_real')
            axes[i].plot(xode_gp[i,0:80,0],'--',color='k', label='pos_gp')
            axes[i].plot(xode_real[i,0:80,1],'-',color='gray', label='vel_real')
            axes[i].plot(xode_gp[i,0:80,1],'--',color='gray', label='vel_gp')
            axes[i].legend()

        fig.savefig(os.path.join(out_dir, '{}-{}.pdf'.format('sample_trajectories', plot_itr+1)), adjustable='box')

    
def run_sampler(config, data, output):

    out_dir = os.path.join(output, config['method'])
    if 'dir_name' in config:
        out_dir = os.path.join(out_dir, str(config['id'])+config['dir_name'])
    else:
        out_dir = os.path.join(out_dir, str(config['id']))

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, str(args.id)+".json"), 'w') as f:
        json.dump(config, f)

    
    # create variables, functions etc
    M   = config['M'] # MxM inducing grid
    D   = 2
    sf  = config['sf']
    ell = config['ell']

    N, R, noise, x0, t, X, Y, ode_type = data.values()
    if 'noise' in config:
        noise = config['noise']

    torch.set_default_tensor_type(torch.DoubleTensor)
    xv = np.linspace(np.min([np.min(Y_[:,0]) for Y_ in Y]), np.max([np.max(Y_[:,0]) for Y_ in Y]), M)
    yv = np.linspace(np.min([np.min(Y_[:,1]) for Y_ in Y]), np.max([np.max(Y_[:,1]) for Y_ in Y]), M)
    xv,yv = np.meshgrid(xv,yv)
    Z  = np.array([xv.T.flatten(),yv.T.flatten()]).T
    Zt = torch.from_numpy(Z)
    Yt = torch.from_numpy(Y)
    U0 = 0.1*torch.randn(M*M, D)

    # gradient matching
    F_ = (Yt[:,1:,:]-Yt[:,:-1,:]) / (t[1]-t[0])
    F_ = F_.contiguous().view(-1,D)
    Z_ = Yt[:,:-1,:].contiguous().view(-1,D)
    Kxz = K(Zt, Z_, sf, ell)
    Kzzinv = (K(Z_, Z_, sf, ell)+0.2*torch.eye(Z_.shape[0])).inverse()
    U0 = torch.mm(torch.mm(Kxz, Kzzinv),F_) # not whitened
    Linv = torch.cholesky(K(Zt, Zt, sf, ell)).inverse()
    U0 = torch.mm(Linv,U0) # whitened

    kreg = KernelRegression(U0, Zt, sf, ell, noise)
    Kzzinv = kreg.Kzzinv

    logsn = torch.nn.Parameter(torch.zeros(2), requires_grad=True)
    params = [kreg.U, kreg.logsn]
    
    total_loss_arr = []
    sq_err_loss_arr = []

    def loss_closure(add_prior=True):
        T   = len(t)
        t_  = t[:T]
        Yt_ = Yt[:,:T,:]
        xode = odeint(kreg, x0, t_, method='rk4').permute([1,0,2])
        if add_prior:
            loss = torch.sum((Yt_-xode)**2/(2*torch.exp(kreg.logsn)**2))
            loss += torch.numel(Yt_)*torch.sum(kreg.logsn)/D
            loss += torch.sum(torch.diag(torch.mm(kreg.U.t(),torch.mm(Kzzinv, kreg.U))))/2
        else:
            loss = torch.sum((Yt_-xode)**2)
        return loss

    def arr_closure(total_loss, sq_err_loss):
        total_loss_arr.append(total_loss.item())
        sq_err_loss_arr.append(sq_err_loss.item())


    sampler = None
    chain = None
    lr_scheduler = None
    if config['method'] == 'MALA':
        sampler = MALA(params, lr=config['lr'], add_noise=True)
        chain = sampler.sample(loss_closure, burn_in=config['burn_in'], num_samples=config['num_samples'],
                                           print_iters=False, arr_closure=arr_closure)
    if config['method'] == 'SGLD':
        sampler = SGLD(params, lr0=config['lr0'], lr_gamma=config['lr_gamma'], lr_t0=config['lr_t0'], lr_alpha=config['lr_alpha'])
        chain = sampler.sample(loss_closure, burn_in=config['burn_in'], num_samples=config['num_samples'],
                                           print_iters=False, arr_closure=arr_closure)
    if config['method'] == 'pSGLD':
        sampler = pSGLD(params, lr0=config['lr0'], lr_gamma=config['lr_gamma'], lr_t0=config['lr_t0'], lr_alpha=config['lr_alpha'],
                       lambda_=config['lambda_'], alpha=config['psgld_alpha'], N=N)
        chain = sampler.sample(loss_closure, burn_in=config['burn_in'], num_samples=config['num_samples'],
                                           print_iters=False, arr_closure=arr_closure)
    if config['method'] == 'aSGHMC':
        sampler = aSGHMC(params, lr=config['lr'], add_noise=True)
        chain = sampler.sample(loss_closure, burn_in=config['burn_in'], num_samples=config['num_samples'], print_iters=False, arr_closure=arr_closure)


    chain_ = chain[config['chain_start']::config['thinning']]

    # Save losses
    with open(os.path.join(out_dir, 'total_loss_arr.pickle'), 'wb') as f:
        pickle.dump(total_loss_arr, f)
    with open(os.path.join(out_dir, 'sq_err_loss_arr.pickle'), 'wb') as f:
        pickle.dump(sq_err_loss_arr, f)
    

    # Plot losses
    fig, ax = plt.subplots()
    ax.plot(total_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Negative log posterior')
    fig.savefig(os.path.join(out_dir, '{}_best{}.pdf'.format('post', str(np.min(total_loss_arr)))), adjustable='box')
    
    fig, ax = plt.subplots()
    ax.plot(sq_err_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('sum of squares error')
    fig.savefig(os.path.join(out_dir, '{}_best{}.pdf'.format('loss', str(np.min(sq_err_loss_arr)))), adjustable='box')
    
    # log losses
    fig, ax = plt.subplots()
    ax.plot(total_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(Negative log posterior)')
    ax.set_yscale('log')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format('post')), adjustable='box')
    
    fig, ax = plt.subplots()
    ax.plot(sq_err_loss_arr)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(sum of squares error)')
    ax.set_yscale('log')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format('loss')), adjustable='box')
    
    # plot phase plot
    chain_mode_post_idx = np.where(total_loss_arr[config['burn_in']:] == np.amin(total_loss_arr[config['burn_in']:]))
    chain_mode = chain[chain_mode_post_idx[0][-1]]
    U_mode = chain_mode[0][0][0]

    kreg.U.data = torch.from_numpy(U_mode)
    xode = odeint(kreg, x0, t).detach().numpy()
    Z_np = kreg.Z.detach().numpy()
    xode = np.transpose(xode,[1,0,2])

    cols = ['r','b','g','m','c','k']
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.scatter(Z_np[:,0],Z_np[:,1],100, facecolors='none', edgecolors='k')
    ax.quiver(Z_np[:,0],Z_np[:,1],U_mode[:,0],U_mode[:,1],units='height',width=0.006,color='k')
    for i in range(min(6,xode.shape[0])):
        ax.plot(Y[i,:,0],Y[i,:,1],'o',color=cols[i])
        ax.plot(xode[i,:,0],xode[i,:,1],'-',color=cols[i])

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format('phaseplot')), adjustable='box')

    # plot real vs modeled samples
    for plot_itr in range(3):
        R_= R
        x0_ = None
        if ode_type == "LV":
            x0_ = torch.from_numpy(R*ss.beta.rvs(size=[N,2], a=1, b=1.5)+ [3., 1.5])
        else:
            x0_ = torch.from_numpy(2*R_*ss.uniform.rvs(size=[10,2])-R_)
            
        t_ = torch.linspace(0., 14., 80)
        t_numpy = t_.clone().detach().numpy()

        xode_gp = []
        for i in range(len(chain)):
            kreg.U.data = torch.from_numpy(chain[i][0][0][0])
            xode_gp.append(odeint(kreg, x0_, t_).detach().numpy())
            xode_gp[-1] = np.transpose(xode_gp[-1],[1,0,2])

        xode_gp_mean = np.zeros_like(xode_gp[0])
        xode_gp_std = np.zeros_like(xode_gp[0])

        for i in range(xode_gp_mean.shape[0]):
            xode_gp_mean[i,:,0] = np.mean([xode_gp[j][i,:,0] for j in range(len(xode_gp))], axis=0)
            xode_gp_mean[i,:,1] = np.mean([xode_gp[j][i,:,1] for j in range(len(xode_gp))], axis=0)
            xode_gp_std[i,:,0] = np.std([xode_gp[j][i,:,0] for j in range(len(xode_gp))], axis=0)
            xode_gp_std[i,:,1] = np.std([xode_gp[j][i,:,1] for j in range(len(xode_gp))], axis=0)
            
        ode_func = None
        if ode_type == "VDP":
            ode_func = VDP
        if ode_type == "LV":
            ode_func = LV
        if ode_type == "FHN":
            ode_func = FHN
        xode_real = odeint(ode_func(), x0_, t_).detach().numpy()
        xode_real = np.transpose(xode_real,[1,0,2])


        num_plots = 3
        # fig, ax = plt.subplots(nrows=(num_plots+1)//2, ncols=2, figsize=(15,10))
        fig, axes = plt.subplots(ncols=num_plots, nrows=1, figsize=(15,3))
        for i in range(num_plots):
            axes[i].plot(t_numpy[:80],xode_real[i,0:80,0],'-',color='r', label='Position(real)')
            axes[i].fill_between(x=t_numpy[:80],
                                y1=xode_gp_mean[i,0:80,0]-5*xode_gp_std[i,0:80,0],
                                y2=xode_gp_mean[i,0:80,0]+5*xode_gp_std[i,0:80,0],
                                linestyle='--',color='k', alpha=0.3)
            axes[i].plot(t_numpy[:80],
                        xode_gp_mean[i,0:80,0],
                        linestyle='-',color='k', label='Position(GP)')
            axes[i].plot(t_numpy[:80],xode_real[i,0:80,1],'-',color='g', label='Velocity(real)')
            axes[i].fill_between(x=t_numpy[:80],
                                y1=xode_gp_mean[i,0:80,1]-5*xode_gp_std[i,0:80,1],
                                y2=xode_gp_mean[i,0:80,1]+5*xode_gp_std[i,0:80,1],
                                linestyle='--',color='gray', alpha=0.3)
            axes[i].plot(t_numpy[:80],
                        xode_gp_mean[i,0:80,1],
                        linestyle='-',color='gray', label='Velocity(GP)')

            axes[i].legend()

        fig.savefig(os.path.join(out_dir, '{}-{}.pdf'.format('sample_trajectories', plot_itr+1)), adjustable='box')

    # plot logsn
    fig, ax = plt.subplots()
    ax.hist([np.mean(np.exp(i[0][0][1])) for i in chain_ if i[1]])
    ax.set_ylabel(r'$exp(logsn)$')
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format('logsn')), adjustable='box')


def worker(config, data, output):
    data_ = {}
    for d in data:
        if 'torch' in type(data[d]).__module__:
            data_[d] = data[d].clone().detach()
        else:
            data_[d] = data[d]
    # try:
    output_ = os.path.join(output, data['ODE'])
    if config['inf_type'] == 'optim':
        output_ = os.path.join(output_, 'optim')
        run_optim(config, data_, output_)
    else:
        output_ = os.path.join(output_, 'samplers')
        run_sampler(config, data_, output_)
    # except Exception as e:
    #     print("Encountered error while inferring with {}:\n{}".format(config['method'], e))


if __name__=='__main__':

    ## Parse arguments and load hyperparameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--json-dir', type=str,
                        required=True,
                        help='path to hyperparameters json directory')
    parser.add_argument('--id', type=int,
                        default=None,
                        help='array id from slurm job batch')

    args = parser.parse_args()

    hyp = json.load(open(os.path.join(args.json_dir, str(args.id)+'.json'), 'r'))
    os.makedirs(hyp['output'], exist_ok=True)
    output = hyp['output']

    data = pickle.load(open(hyp['data']['pickle_file'], 'rb'))
    
    sns.set(context="notebook",
            style="whitegrid",
            font_scale=0.7,
            rc={"axes.axisbelow": False,
                "axes.grid": False,
                })
    sns.set_style('ticks')

    params = []
    for config in hyp['configs']:
        if 'id' not in config:
            config['id'] = str(args.id)
        params.append([config, data, output])
    
    pool = multiprocessing.Pool()
    pool.starmap(worker, params)
