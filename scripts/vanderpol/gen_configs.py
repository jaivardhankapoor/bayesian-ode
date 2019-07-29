import os
import itertools
from copy import copy
import numpy
import datetime
import json

CLUSTER_SCRIPT_FILENAME = 'quantitative' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


DEFAULT_VALUES = {
    "M": 6,
    "sf": 1.,
    "ell": 0.75,
    "burn_in": 3000,
    "num_samples": 5000,
    "thinning": 2, ## change this
    "chain_start": 0,
    "num_iters": 1000,
    "lr": 1e-3,
    "lr_decay": 0.03,
    "mom": 0.98,
    "rmsprop_alpha": 0.99,
    "adadelta_rho": 0.9,
    "lr0": 5e-3,
    "lr_gamma": 0.51,
    "lr_t0": 100,
    "lr_alpha": 0.1,
    "psgld_alpha": 0.99,
    "lambda_": 1e-8,
    "noise": 0.1,
}

SENSIBLE_PARAMS = {
    "M": "M",
    "ell": "ell",
    # "burn_in": "burn",
    # "num_samples":"nsampl",
    # "num_iters":"nitr",
    "lr": "lr",
    "lr_decay": "lrdec",
    "mom": "mom",
    "rmsprop_alpha": "alpha",
    "adadelta_rho": "rho",
    "lr0": "lr0",
    "noise": "noise",
    "lr_alpha": "lr_alpha",
    "psgld_alpha": "alpha",
}


OUT_PATH = 'exp/vanderpol/gp'
DATA_DIR = 'data/'
DATA_FILE = 'vdp.pickle'

JSON_DIR = 'data/vanderpol/json/'

os.makedirs(JSON_DIR, exist_ok=True)

json_iter = 1

## Write configs for optim

# Adam
id_=1
# LR = [1e-3,5e-3,1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
LR = [1e-3]
M = [6]
ELL = [0.75]
NUM_ITERS = 1000

for lr, m, ell in itertools.product(LR, M, ELL):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "optim",
        "method": "Adam",
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "ell": ell,
        "lr": lr,
        "num_iters": NUM_ITERS,
        "id": id_,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1

# LBFGS
id_=1
# LR = [1e-3,5e-3,1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
LR = [1e-3]
M = [6]
ELL = [0.75]
NUM_ITERS = 500

for lr, m, ell in itertools.product(LR, M, ELL):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "optim",
        "method": "LBFGS",
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "ell": ell,
        "lr": lr,
        "num_iters": NUM_ITERS,
        "id": id_,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1

# SGD
id_=1
# LR = [3e-4, 1e-3, 3e-3, 1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
LR = [1e-4]
M = [6]
ELL = [0.75]
NUM_ITERS = 1500

for lr, m, ell in itertools.product(LR, M, ELL):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "optim",
        "method": "SGD",
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "ell": ell,
        "lr": lr,
        "num_iters": NUM_ITERS,
        "id": id_,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1


# SGD+mom
id_=1
# LR = [3e-4, 1e-3, 3e-3, 1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# MOM = [0.9,0.95,0.98,0.99]
LR = [1e-3]
M = [6]
ELL = [0.75]
MOM = [0.99]
NUM_ITERS = 1000

for lr, m, ell, mom in itertools.product(LR, M, ELL, MOM):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "optim",
        "method": "SGD+mom",
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "ell": ell,
        "lr": lr,
        "num_iters": NUM_ITERS,
        "id": id_,
        "lr_decay": DEFAULT_VALUES["lr_decay"],
        "mom": mom,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1

# RMSprop
id_ = 1
# LR = [3e-4, 1e-3, 3e-3, 1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# LR_DECAY = [0, 0.02]
# ALPHA = [0.95, 0.98, 0.99]
LR = [1e-3]
M = [6]
ELL = [0.75]
LR_DECAY = [0.02]
ALPHA = [0.99]
NUM_ITERS = 1000
for lr, m, ell, lrdec, alpha in itertools.product(LR, M, ELL, LR_DECAY, ALPHA):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "optim",
        "method": "RMSprop",
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "ell": ell,
        "lr": lr,
        "num_iters": NUM_ITERS,
        "id": id_,
        "lr_decay": lrdec,
        "rmsprop_alpha": alpha,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1

# RMSprop+mom
id_ = 1
# LR = [3e-4, 1e-3, 3e-3, 1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# LR_DECAY = [0.01, 0.03]
# ALPHA = [0.96, 0.99]
# MOM = [0.9, 0.95, 0.99]
LR = [1e-3]
M = [6]
ELL = [0.75]
LR_DECAY = [0.02]
ALPHA = [0.99]
MOM = [0.98]
NUM_ITERS = 1000
for lr, m, ell, lrdec, alpha, mom in itertools.product(LR, M, ELL, LR_DECAY, ALPHA, MOM):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "optim",
        "method": "RMSprop+mom",
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "ell": ell,
        "lr": lr,
        "num_iters": NUM_ITERS,
        "id": id_,
        "lr_decay": lrdec,
        "rmsprop_alpha": alpha,
        "mom": mom,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1


# Adadelta
id_ = 1
# LR = [3e-4, 1e-3, 3e-3, 1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# LR_DECAY = [0.0, 0.03, 0.1]
LR = [1e-3]
M = [6]
ELL = [0.75]
LR_DECAY = [0.02]
NUM_ITERS = 1000
for lr, m, ell, lrdec in itertools.product(LR, M, ELL, LR_DECAY):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "optim",
        "method": "Adadelta",
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "ell": ell,
        "lr": lr,
        "num_iters": NUM_ITERS,
        "id": id_,
        "lr_decay": lrdec,
        "adadelta_rho": DEFAULT_VALUES["adadelta_rho"],
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1


## Write configs for samplers

# MALA
id_=1
# LR = [1e-4, 3e-4, 1e-3]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# NOISE = [0.03, 0.1, 0.3, 1.0]
BURN_IN = DEFAULT_VALUES["burn_in"]
NUM_SAMPLES = DEFAULT_VALUES["num_samples"]
LR = [1e-4]
M = [6]
ELL = [0.75]
NOISE = [0.1]

for lr, m, ell, noise in itertools.product(LR, M, ELL, NOISE):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "samplers",
        "method": "MALA",
        "burn_in": BURN_IN,
        "num_samples": NUM_SAMPLES,
        "chain_start": DEFAULT_VALUES['chain_start'],
        "thinning": DEFAULT_VALUES['thinning'],
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "noise": noise,
        "ell": ell,
        "lr": lr,
        "id": id_,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1


# SGLD
id_=1
# LR0 = [1e-3, 3e-3, 1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# NOISE = [0.03, 0.1, 0.3, 1.0]
# LR_ALPHA = [0.03, 0.1]
BURN_IN = DEFAULT_VALUES["burn_in"] + 1000
NUM_SAMPLES = DEFAULT_VALUES["num_samples"] - 1000
LR0 = [1e-3]
M = [6]
ELL = [0.75]
NOISE = [0.1]
LR_ALPHA = [0.1]
for lr0, m, ell, noise, lr_alpha in itertools.product(LR, M, ELL, NOISE, LR_ALPHA):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "samplers",
        "method": "SGLD",
        "burn_in": BURN_IN,
        "num_samples": NUM_SAMPLES,
        "chain_start": DEFAULT_VALUES['chain_start'],
        "thinning": DEFAULT_VALUES['thinning'],
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "noise": noise,
        "ell": ell,
        "lr0": lr0,
        "lr_gamma": DEFAULT_VALUES["lr_gamma"],
        "lr_t0": DEFAULT_VALUES["lr_t0"],
        "lr_alpha": lr_alpha,
        "id": id_,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1


# pSGLD
id_=1
# LR0 = [1e-3, 3e-3, 1e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# NOISE = [0.03, 0.1, 0.3, 1.0]
# LR_ALPHA = [0.03, 0.1]
# ALPHA = [0.96, 0.99]
BURN_IN = DEFAULT_VALUES["burn_in"] + 1000
NUM_SAMPLES = DEFAULT_VALUES["num_samples"] - 1000
LR0 = [1e-3]
M = [6]
ELL = [0.75]
NOISE = [0.1]
LR_ALPHA = [0.1]
ALPHA = [0.99]

for lr0, m, ell, noise, lr_alpha, alpha in itertools.product(LR, M, ELL, NOISE, LR_ALPHA, ALPHA):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "samplers",
        "method": "pSGLD",
        "burn_in": BURN_IN,
        "num_samples": NUM_SAMPLES,
        "chain_start": DEFAULT_VALUES['chain_start'],
        "thinning": DEFAULT_VALUES['thinning'],
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "noise": noise,
        "ell": ell,
        "lr0": lr0,
        "lr_gamma": DEFAULT_VALUES["lr_gamma"],
        "lr_t0": DEFAULT_VALUES["lr_t0"],
        "lr_alpha": lr_alpha,
        "psgld_alpha": alpha,
        "id": id_,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1


# aSGHMC
id_=1
# LR = [1e-3, 3e-3, 1e-2, 3e-2]
# M = [4,5,6]
# ELL = [0.5, 0.75, 1, 1.25]
# NOISE = [0.03, 0.1, 0.3, 1.0]
BURN_IN = DEFAULT_VALUES["burn_in"]
NUM_SAMPLES = DEFAULT_VALUES["num_samples"]
LR = [1e-2]
M = [6]
ELL = [0.75]
NOISE = [0.1]

for lr, m, ell, noise in itertools.product(LR, M, ELL, NOISE):
    config = {}
    config["output"] = OUT_PATH
    config["data"] = {"pickle_file": DATA_DIR+DATA_FILE}
    config["configs"] = []
    config["configs"].append({
        "inf_type": "samplers",
        "method": "aSGHMC",
        "burn_in": BURN_IN,
        "num_samples": NUM_SAMPLES,
        "chain_start": DEFAULT_VALUES['chain_start'],
        "thinning": DEFAULT_VALUES['thinning'],
        "M": m,
        "sf": DEFAULT_VALUES["sf"],
        "noise": noise,
        "ell": ell,
        "lr": lr,
        "id": id_,
    })
    dir_name = ''
    for k in SENSIBLE_PARAMS:
        if k in config["configs"][0]:
            dir_name += SENSIBLE_PARAMS[k] + str(config["configs"][0][k]) + "_"
    config["configs"][0]["dir_name"] = dir_name
    id_ += 1

    with open(os.path.join(JSON_DIR, str(json_iter)+'.json'), 'w') as f:
        json.dump(config, f)

    json_iter += 1

