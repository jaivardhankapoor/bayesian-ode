# bayesian-ode

Bayesian inference in nonparametric ODE models

### TODO's

1. Add pipeline for mMALA/SGLD sampling and inference for Kernel regression
2. Use FIM formula in Stapor et. al. for Riemannian Metric
3. Calculate Hessian vector product using this rick: https://discuss.pytorch.org/t/calculating-hessian-vector-product/11240/4
4. In case of non-identity metrics in samplers, concatenate all the gradients into a single vector and then do the preconditioning etc. this is so that the matrix can contain some kind of correlations between different parameters and not just inside the individual tensors themselves. See https://github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py#L57-L76