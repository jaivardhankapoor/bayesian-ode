# bayesian-ode

Bayesian inference in nonparametric ODE models

### TODO's for Jai

1. Add baselines for Van der Pol with:

    1.1. Neural Network :heavy_check_mark:
  
    1.2. Gaussian Process Mean Function :heavy_check_mark:

    with the following changes:

2. Implement preconditioned Langevin samplers:

    2.1 [P-SGLD](https://arxiv.org/pdf/1512.07666.pdf) :heavy_check_mark:

    2.2 [Adaptively Preconditioned SGLD](https://arxiv.org/pdf/1906.04324.pdf)

3. Implement L-BFGS type sampler (Langevin, no correction term):
    
    3.1. [HAMCMC](https://arxiv.org/pdf/1602.03442.pdf)

    ....3.1.1 Implement non-contiguous difference L-BFGS with new as well as old recursion :heavy_check_mark:

    ....3.1.2 Implement contiguous difference L-BFGS with new as well as old recursion

    ....3.1.3 Ensure positive definiteness by removing pairs for which siyi<0

    ....3.1.4 Implement accept-reject step

    ....3.1.5 Implement Gradient recomputation in case of mini-batch settings for computing yi

    3.2 (*Optional*) L-BFGS+SGHMC

4. Implement [SVGD](https://arxiv.org/pdf/1608.04471.pdf)

3. Implement [Stochastic Gradient HMC](https://arxiv.org/pdf/1402.4102.pdf)

4. (*Optional*) Implement [SVGD+SG-MCMC](https://arxiv.org/pdf/1812.00071.pdf)