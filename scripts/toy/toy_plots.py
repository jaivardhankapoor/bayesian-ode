import torch 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import seaborn as sns
    
import json
import argparse
import os
import torch.multiprocessing as multiprocessing

def get_banana_fns(x,y, a=0.2, b=2.0, c=1.0):


    def closure_banana(add_prior=True):
        return 0.5*(a*x*x + (b*y+c*x*x)**2)

    def prob_banana(x, y):
        return np.exp(-0.5*(a*x*x + (b*y+ c*x*x)**2))

    def plot_contours_banana(ax):
        
        X = np.linspace(-6, 6, 300)
        Y = np.linspace(-12, 2, 1000)

        X, Y = np.meshgrid(X, Y)

        Z = prob_banana(X, Y)
        cmap = cm.get_cmap('binary')
        cmap.set_under(color='w')
        ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5, extend='both')
        
    return closure_banana, prob_banana, plot_contours_banana


def get_multimodal_fns(x,y, mixture=(0.5,0.5), means=([-1,-1], [1,1]),
                       sigmas=([1,0.5], [0.5,1]), corr=(0.5,-0.5)):


    def prob_multimodal(x__, y__):
        prob = 0.
        for i, w in enumerate(mixture):    
            x_ = (x__ - means[i][0])/sigmas[i][0]
            y_ = (y__ - means[i][1])/sigmas[i][1]
            if 'torch' in type(x__).__module__:
                prob += w*torch.exp(-(x_**2 + y_**2 - 2*corr[i]*x_*y_)/(2*(1-corr[i]**2)))/(sigmas[i][0]*sigmas[i][1]*np.sqrt(1-corr[i]**2))    
            elif 'numpy' in type(x__).__module__:
                prob += w*np.exp(-(x_**2 + y_**2 - 2*corr[i]*x_*y_)/(2*(1-corr[i]**2)))/(sigmas[i][0]*sigmas[i][1]*np.sqrt(1-corr[i]**2))        
        return prob

    def closure_multimodal(add_prior=True):
        return -torch.log(prob_multimodal(x, y))


    def plot_contours_multimodal(ax):
        
        X = np.linspace(-2*max([sigmas[i][0] for i in range(len(sigmas))]) + min([means[i][0] for i in range(len(means))]),
                        2*max([sigmas[i][0] for i in range(len(sigmas))]) + max([means[i][0] for i in range(len(means))]), 1000)
        Y = np.linspace(-2*max([sigmas[i][1] for i in range(len(sigmas))]) + min([means[i][1] for i in range(len(means))]),
                        2*max([sigmas[i][1] for i in range(len(sigmas))]) + max([means[i][1] for i in range(len(means))]), 1000)

        X, Y = np.meshgrid(X, Y)

        Z = prob_multimodal(X, Y)
        cmap = cm.get_cmap('binary')
        cmap.set_under(color='w')
        ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5, extend='both')
        
    return closure_multimodal, prob_multimodal, plot_contours_multimodal


def get_gauss_fns(x, y, sigma1, sigma2, corr):

    def closure_gauss(add_prior=True):
        x_ = (x-2)/sigma1
        y_ = (y-4)/sigma2
        return (x_**2 + y_**2 - 2*corr*x_*y_)/(2*(1-corr**2))    
    
    def prob_gauss(x__,y__):
        x_ = (x__-2)/sigma1
        y_ = (y__-4)/sigma2
        return np.exp(-(x_**2 + y_**2 - 2*corr*x_*y_)/(2*(1-corr**2)))    

    def plot_contours_gauss(ax):    
        X = np.linspace(-max(sigma1*2,2)+2, max(2,sigma1*2)+2, 1000)
        Y = np.linspace(-max(2,sigma2*2)+4, max(2,sigma2*2)+4, 1000)
        
        X, Y = np.meshgrid(X, Y)

        Z = prob_gauss(X, Y)
        cmap = cm.get_cmap('binary')
        cmap.set_under(color='w')
        # ax.set_clim(vmin=np.min(Z), vmax=np.max(Z))
        ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5, extend='both')
        
    return closure_gauss, prob_gauss, plot_contours_gauss


def run_MALA(config, dists, output):

    out_path = os.path.join(output, 'MALA/')
    os.makedirs(out_path, exist_ok=True)
    fig, ax = plt.subplots(ncols=len(dists), dpi=300, figsize=(8,2))
    plt.tight_layout()
    for distnum, (dist_name, dist) in enumerate(dists.items()):
        # closure, prob, plot_contours = dist['closure'], dist['prob'], dist['plot_contours']
        x = torch.nn.Parameter(x_init.clone().detach())
        y = torch.nn.Parameter(y_init.clone().detach())
        params = [x, y]

        if 'banana' in dist_name:
            closure, prob, plot_contours = get_banana_fns(x, y, *dist)
        if 'gauss' in dist_name:
            closure, prob, plot_contours = get_gauss_fns(x, y, *dist)    
        if 'modal' in dist_name:
            closure, prob, plot_contours = get_multimodal_fns(x, y, *dist)

        sampler = MALA(params, lr=config['lr'], add_noise=config['add_noise'])
        chain, logp_array = sampler.sample(closure, burn_in=config['burn_in'],
            num_samples=config['num_samples'], print_loss=config['print_loss'])
        
        plot_contours(ax[distnum])
        for i, (sample, acc) in enumerate(chain[::config['thinning']]):
            x_, y_ = sample[0][0][0], sample[0][1][0]
            if acc:
                ax[distnum].scatter(x_, y_, color='g', s=1)
            else:
                ax[distnum].scatter(x_, y_, color='r', s=1)

            x_mean = np.mean([sample[0][0][0] for sample, acc in chain[::config['thinning']] if acc])
            y_mean = np.mean([sample[0][1][0] for sample, acc in chain[::config['thinning']] if acc]) 
            ax[distnum].annotate('x_mean={:.02f}\ny_mean={:.02f}\nacc_rate={:.02f}'.format(x_mean, y_mean,
                len([1 for _, acc in chain[:] if acc])/len(chain)),
                xy=(0.88, 0.03), xycoords='axes fraction', fontsize=8,
                textcoords='offset points', ha='right', va='bottom')
        
            ax[distnum].set_title(dist_name)
            sns.despine(bottom=False,
                    left=False,
                    right=True,
                    top=True,
                    ax=ax[distnum])
            ax[distnum].xaxis.set_major_locator(ticker.MaxNLocator(5))
            
    fig.savefig(os.path.join(out_path, '{}.pdf'.format(config['id'])), adjustable='box')

def run_SGLD(config, dists, output):

    out_path = os.path.join(output, 'SGLD/')
    os.makedirs(out_path, exist_ok=True)
    fig, ax = plt.subplots(ncols=len(dists), dpi=300, figsize=(8,2))
    plt.tight_layout()
    for distnum, (dist_name, dist) in enumerate(dists.items()):
        x = torch.nn.Parameter(x_init.clone().detach())
        y = torch.nn.Parameter(y_init.clone().detach())
        params = [x, y]

        if 'banana' in dist_name:
            closure, prob, plot_contours = get_banana_fns(x, y, *dist)
        if 'gauss' in dist_name:
            closure, prob, plot_contours = get_gauss_fns(x, y, *dist)    
        if 'modal' in dist_name:
            closure, prob, plot_contours = get_multimodal_fns(x, y, *dist)

        sampler = SGLD(params, lr0=config['lr0'], lr_gamma=config['lr_gamma'],
            lr_t0=config['lr_t0'], lr_alpha=config['lr_alpha'], lambda_=config['lambda_'],
            alpha=config['alpha'])
        chain, logp_array = sampler.sample(closure, burn_in=config['burn_in'],
            num_samples=config['num_samples'], print_loss=config['print_loss'])
        
        plot_contours(ax[distnum])
        for i, (sample, acc) in enumerate(chain[::config['thinning']]):
            if acc:
                x_, y_ = sample[0][0][0], sample[0][1][0]
                ax[distnum].scatter(x_, y_, color='g', s=1)
                
        # Compute mean by weighing with the step sizes
        wt = [sampler.get_lr(i*config['thinning']+config['burn_in'])\
                for i in range(len(chain[::config['thinning']]))]
        wt = np.array(wt)/np.sum(wt)
        x_mean = np.sum([wt[i]*sample[0][0][0] for i, (sample, acc) in enumerate(chain[::config['thinning']]) if acc])
        y_mean = np.sum([wt[i]*sample[0][1][0] for i, (sample, acc) in enumerate(chain[::config['thinning']]) if acc]) 
        ax[distnum].annotate('x_mean={:.02f}\ny_mean={:.02f}'.format(x_mean, y_mean),
            xy=(0.88, 0.03), xycoords='axes fraction', fontsize=8,
            textcoords='offset points', ha='right', va='bottom')
        
        ax[distnum].set_title(dist_name)
        sns.despine(bottom=False,
                left=False,
                right=True,
                top=True,
                ax=ax[distnum])
        ax[distnum].xaxis.set_major_locator(ticker.MaxNLocator(5))
        
    fig.savefig(os.path.join(out_path, '{}.pdf'.format(config['id'])), adjustable='box')

def run_pSGLD(config, dists, output):

    out_path = os.path.join(output, 'pSGLD/')
    os.makedirs(out_path, exist_ok=True)
    fig, ax = plt.subplots(ncols=len(dists), dpi=300, figsize=(8,2))
    plt.tight_layout()
    for distnum, (dist_name, dist) in enumerate(dists.items()):
        x = torch.nn.Parameter(x_init.clone().detach())
        y = torch.nn.Parameter(y_init.clone().detach())
        params = [x, y]

        if 'banana' in dist_name:
            closure, prob, plot_contours = get_banana_fns(x, y, *dist)
        if 'gauss' in dist_name:
            closure, prob, plot_contours = get_gauss_fns(x, y, *dist)    
        if 'modal' in dist_name:
            closure, prob, plot_contours = get_multimodal_fns(x, y, *dist)

        sampler = pSGLD(params, lr0=config['lr0'], lr_gamma=config['lr_gamma'],
            lr_t0=config['lr_t0'], lr_alpha=config['lr_alpha'], lambda_=config['lambda_'],
            alpha=config['alpha'])
        chain, logp_array = sampler.sample(closure, burn_in=config['burn_in'],
            num_samples=config['num_samples'], print_loss=config['print_loss'])
        
        plot_contours(ax[distnum])
        for i, (sample, acc) in enumerate(chain[::config['thinning']]):
            if acc:
                x_, y_ = sample[0][0][0], sample[0][1][0]
                ax[distnum].scatter(x_, y_, color='g', s=1)

        # Compute mean by weighing with the step sizes
        wt = [sampler.get_lr(i*config['thinning']+config['burn_in'])\
                for i in range(len(chain[::config['thinning']]))]
        wt = np.array(wt)/np.sum(wt)
        x_mean = np.sum([wt[i]*sample[0][0][0] for sample, acc in chain[::config['thinning']] if acc])
        y_mean = np.sum([wt[i]*sample[0][1][0] for sample, acc in chain[::config['thinning']] if acc])

        ax[distnum].annotate('x_mean={:.02f}\ny_mean={:.02f}'.format(x_mean, y_mean),
            xy=(0.88, 0.03), xycoords='axes fraction', fontsize=8,
            textcoords='offset points', ha='right', va='bottom')
        
        ax[distnum].set_title(dist_name)
        sns.despine(bottom=False,
                left=False,
                right=True,
                top=True,
                ax=ax[distnum])
        ax[distnum].xaxis.set_major_locator(ticker.MaxNLocator(5))
        
    fig.savefig(os.path.join(out_path, '{}.pdf'.format(config['id'])), adjustable='box')

def run_aSGHMC(config, dists, output):

    out_path = os.path.join(output, 'aSGHMC/')
    
    os.makedirs(out_path, exist_ok=True)
    fig, ax = plt.subplots(ncols=len(dists), dpi=300, figsize=(8,2))
    plt.tight_layout()
    for distnum, (dist_name, dist) in enumerate(dists.items()):
        x = torch.nn.Parameter(x_init.clone().detach())
        y = torch.nn.Parameter(y_init.clone().detach())
        params = [x, y]

        if 'banana' in dist_name:
            closure, prob, plot_contours = get_banana_fns(x, y, *dist)
        if 'gauss' in dist_name:
            closure, prob, plot_contours = get_gauss_fns(x, y, *dist)    
        if 'modal' in dist_name:
            closure, prob, plot_contours = get_multimodal_fns(x, y, *dist)

        sampler = aSGHMC(params, mom_decay=config['mom_decay'], lr=config['lr'], lambda_=config['lambda_'])
        chain, logp_array = sampler.sample(closure, burn_in=config['burn_in'],
            num_samples=config['num_samples'], print_loss=config['print_loss'])
        
        plot_contours(ax[distnum])
        for i, (sample, acc) in enumerate(chain[::config['thinning']]):
            x_, y_ = sample[0][0][0], sample[0][1][0]
            if acc:
                ax[distnum].scatter(x_, y_, color='g', s=1)
            else:
                ax[distnum].scatter(x_, y_, color='r', s=1)


        x_mean = np.mean([sample[0][0][0] for sample, acc in chain[::config['thinning']] if acc])
        y_mean = np.mean([sample[0][1][0] for sample, acc in chain[::config['thinning']] if acc]) 
        ax[distnum].annotate('x_mean={:.02f}\ny_mean={:.02f}'.format(x_mean, y_mean),
            xy=(0.88, 0.03), xycoords='axes fraction', fontsize=8,
            textcoords='offset points', ha='right', va='bottom')
        
        ax[distnum].set_title(dist_name)
        sns.despine(bottom=False,
                left=False,
                right=True,
                top=True,
                ax=ax[distnum])
        ax[distnum].xaxis.set_major_locator(ticker.MaxNLocator(5))

    fig.savefig(os.path.join(out_path, '{}.pdf'.format(config['id'])), adjustable='box')

def worker(config, dists, output):
    try:
        if config['sampler'] == 'MALA':
            run_MALA(config=config, dists=dists, output=output)
        if config['sampler'] == 'SGLD':
            run_SGLD(config=config, dists=dists, output=output)
        if config['sampler'] == 'pSGLD':
            run_pSGLD(config=config, dists=dists, output=output)
        if config['sampler'] == 'aSGHMC':
            run_aSGHMC(config=config, dists=dists, output=output)
    except Exception as e:
        print("Encountered error while sampling with {}:\n{}".format(config['sampler'], str(e)))


if __name__=='__main__':

    ## Parse arguments and load hyperparameters
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str,
                        default='./exp/toy/',
                        help='Output path to exp result')
    parser.add_argument('--hyperparams-json', type=str,
                        required=True,
                        help='path to hyperparameters json file')
    parser.add_argument('--id', type=int,
                        default=None,
                        help='array id from slurm job batch')

    args = parser.parse_args()

    hyp = json.load(open(args.hyperparams_json, 'r'))
    os.makedirs(args.output, exist_ok=True)
    output = args.output

    ## Run inference
    from samplers.langevin import *
    from samplers.hamiltonian import *

    x_init = 3*torch.randn([1]).clone().detach()
    y_init = 3*torch.randn([1]).clone().detach()
    
    dists = {
        # 'gaussian1': [1,1,0],
        # 'gaussian2': [3.0,0.1,0],
        'gaussian': [3,0.3,0.7],
        'bimodal': [],
        'multimodal': [(0.3,0.3,0.2, 0.2), ([-1,-0.5], [-1,1.5], [1,-1.5], [2,1]),
                       ([1,0.5],[1.5,0.3],[0.8,0.5],[1.0,0.2]), (0.5,-0.5, -0.7, 0.2)],
        'banana': [],
    }

    sns.set(context="notebook",
            style="whitegrid",
            font_scale=0.7,
            rc={"axes.axisbelow": False,
                "axes.grid": False,
                })
    sns.set_style('ticks')
    
    params = []
    for sampler in hyp:
        for config in hyp[sampler]:
            if args.id is not None:
                config['id'] = str(args.id)
            config['sampler'] = sampler
            params.append([config, dists, output])
    
    pool = multiprocessing.Pool(3)
    pool.starmap(worker, params)
