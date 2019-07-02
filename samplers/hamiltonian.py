import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from samplers import Sampler
from collections import defaultdict, Counter
import copy


def leapfrog():
    pass

def g_leapfrog():
    pass


class HMC(Sampler):

    def __init__(self):
        pass


class SGHMC(Sampler):

    def __init__():
        pass


class RMHMC(Sampler):

    def __init__(self):
        pass


class SGRHMC(Sampler):

    def __init__(self):
        pass


