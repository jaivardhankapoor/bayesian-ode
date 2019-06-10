import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required

from collections import defaultdict, Counter
import copy

class Sampler(Optimizer):

    def __init__(self, params, defaults):
        self.samples = [] # in the form of [...[param, accept/reject]...]
        print(type(defaults))
        super().__init__(params, defaults)


    def step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError