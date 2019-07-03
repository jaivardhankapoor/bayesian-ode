# bayesian-ode

Bayesian inference in nonparametric ODE models

### TODO's for Jai

1. Add baselines for Van der Pol with:

    1.1. Neural Network :heavy_check_mark:
  
    1.2. Gaussian Process Mean Function :heavy_check_mark:

    with the following changes:

2. Implement L-BFGS type sampler (Langevin, no correction term):
    
    2.1. [HAMCMC](https://arxiv.org/pdf/1602.03442.pdf)
    
    2.2. [SANTA](https://arxiv.org/pdf/1512.07962.pdf) or [P-SGLD](https://arxiv.org/pdf/1512.07666.pdf) :heavy_check_mark:

3. Implement [Stochastic Gradient HMC](https://arxiv.org/pdf/1402.4102.pdf)

4. (*Optional*) Implement [SVGD+SG-MCMC](https://arxiv.org/pdf/1812.00071.pdf)

5. (*Optional*) Implement L-BFGS+SGHMC