#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# The following code tests the MALA sampler on a toy problem of 1-dimensional Bayesian Linear Regression.

#%%
# get_ipython().run_line_magic('cd', '"../"')
import torch
import numpy as np
from matplotlib import pyplot as plt


#%%

x_data = torch.Tensor([[x] for x in np.linspace(1, 10)])
w_true = 50
b_true = 400
y_data = x_data*w_true + b_true
y_data += torch.randn(y_data.size())*30
plt.scatter(np.array([i[0] for i in x_data.data]), np.array([i[0] for i in y_data.data]))


#%%

# class MODEL(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = torch.nn.Linear(1, 100)
#         self.linear2 = torch.nn.Linear(100, 1)
        
#     def forward(self, x):
#         y = self.linear(x)
#         z = self.linear2(y)
#         return y
    
class MODEL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        y = self.linear(x)
        return y

model = MODEL()


#%%
from samplers.metrics import *

def closure():
    criterion = torch.nn.MSELoss(reduction='sum')
    loss = criterion(model(x_data), y_data)
    lambda_ = 0.3
    for p in model.parameters():
        loss += lambda_ * p.pow(2).sum()
    return loss

def hess_closure():
    return eval_full_hessian(torch.autograd.grad(closure(), model.parameters(), create_graph=True), model.parameters())

cnt = 0
loss_grad = torch.autograd.grad(closure(), model.parameters())
for g in loss_grad:
    g_vector = g.contiguous().view(-1) if cnt == 0\
                else torch.cat([g_vector, g.contiguous().view(-1)])
    cnt = 1
l = g_vector.size(0)

# metric = SoftAbsMetric(closure=hess_closure)
metric = HessianMetric(closure=hess_closure)
# metric = IdentityMetric(size=l)


#%%
from samplers import MALA, SGLD, MMALA
# sampler = MALA(model.parameters(), lr = 0.0001, add_noise=True)
# sampler = SGLD(model.parameters(), lr0=0.003, gamma=0.55, t0=100, alpha=0.1)
sampler = MMALA(model.parameters(), metric_obj=metric, lr=0.001)


#%%
chain, logp_array = sampler.sample(closure, burn_in=3000, num_samples=500)


#%%
# plt.scatter(np.array([i[0] for i in x_data.data]), np.array([i[0] for i in model(x_data).data]))
plt.hist([i[0][0][0][0] for i in chain if i[1]], bins=100)
np.mean([i[0][0][0][0] for i in chain if i[1]])


#%%
plt.hist([i[0][0][0][0] for i in chain if i[1]], bins=100)

np.mean([i[0][0][0][0] for i in chain if i[1]])


#%%
plt.scatter(np.array([i[0] for i in x_data.data]), np.array([i[0] for i in y_data.data]))
plt.plot(np.array([i[0] for i in x_data.data]), np.array([i[0] for i in model(x_data).data]), c='k')


#%%
from samplers.utils import eval_hessian

hess = eval_hessian(torch.autograd.grad(closure(), model.parameters(), create_graph=True), model)


#%%
chain


#%%



#%%



#%%



