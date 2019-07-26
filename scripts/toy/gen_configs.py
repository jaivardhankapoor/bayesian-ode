import os
import itertools
from copy import copy
import numpy
import datetime

CLUSTER_SCRIPT_FILENAME = 'quantitative' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


DEFAULT_VALUES = {
    
}

SENSIBLE_PARAMS = {
    '--inf-tau': '+INF-tau',
    '--inf-alpha-0': '+INF-a',
    '--inf-mu-0': '+INF-mu',
    # '--inf-gamma-0': '+INF-gamma0',
    # '--inf-gamma-1': '+INF-gamma1',
    # '--dataset': '+dataset',
    # '--time-noise': '+time-noise',
    '--dir-cont-factor': '+dir-cont-factor',
    # '--n-particles': '+particles',
    # '--resample-every': '+res_ev',
    '--time-noise': '+time-noise',
    # '--freq-cutoff': '+freq-cutoff',

    # '--n-users': '+users',
    # '--graph-density-factor': '+graph_dens_fac',
    # '--n-patterns-global': '+patterns',
    # '--pattern-sparsity-factor-per-user': '+patt-sparsity',
    # '--vocabulary': '+vocab',
    # '--vocab-sparsity-factor': '+voc-spar',
    # '--n-events': '+n-events',
    # '--doc-min-length': '+doc-min',
    # '--doc-max-length': '+doc-max',
    # '--run-parallel': '+run-parallel',
    # '--mu-update': '+mu-update',
    # '--update-only-when-exo': '+exo_cond',
    # '--pattern-popularity-prior': '+pop_pr',
    # '--is-loglik-joint': '+is_joint_loglik',
    # '--update-all-alphas': '+alpha_update'

}

CLUSTER_SCRIPT_TEMPLATE = """executable = {}
arguments = {}
error = {}.err
output = {}.out
log = {}.log
request_memory = {}
request_cpus = {}
queue
"""


print('\n')



PYTHON_INTERPRETER = 'ipython3'
CLUSTER_PYTHON_INTERPRETER = '/home/avergari/.local/bin/ipython'
# CLUSTER_PYTHON_INTERPRETER = '/home/jkapoor/.local/bin/ipython'
# PYTHON_INTERPRETER = 'python3 '
GEN_BIN_PATH = ' bin/real/quantitative.py '
VERBOSITY_LEVEL = ' -v 2 '
SEED = 12
N_CPUS = 1
MEMORY = 1024 * 6
# OUT_DIR = 'exp/debug_only_when_exo_10k/i_crf-hp/'
OUT_DIR = 'exp/real/enron/quantitative_newgrid'
BID = 30

os.makedirs(OUT_DIR, exist_ok=True)

TIME_NOISE = [10, 20, 60, 100, 150, 300]

MU0 = [(8, 4)]
ALPHA0 = [(10, 5)]
TAU = [120.0, 240.0, 360.0, 600.0, 900.0, 1200.0]
N_PARTICLES = [20]
RESAMPLE_EVERY = [10]
FREQ_CUTOFF = [4000]
DIR_CONT = [1, 2, 4, 0.5]



HYPER_PARAM_COMBOS = [
    # {'--inf-alpha-0': (0.1, 2000),'--inf-tau': 1} # poisson process
]


# for v, vs, (dmin, dmax) in itertools.product(VOC, VOC_SPAR_PRIOR, DOC_LEN):
#     HYPER_PARAM_COMBOS.append({'--vocabulary': v, '--vocab-sparsity-factor': vs,
#                                '--doc-min-length': dmin, '--doc-max-length': dmax})

for alpha0, mu0, tau, n_particles, res_ev, t_noise, dir_cont in itertools.product(ALPHA0, MU0, TAU, N_PARTICLES, RESAMPLE_EVERY, TIME_NOISE, DIR_CONT):
    HYPER_PARAM_COMBOS.append({'--inf-alpha-0': alpha0, '--inf-mu-0': mu0, '--inf-tau': tau, '--n-particles': n_particles,
                               '--resample-every': res_ev, '--dataset': 'data/train/train_250_{}.pklz'.format(t_noise), '--test-file': 'data/test/test_250_{}.pklz'.format(t_noise), '--time-noise': t_noise, '--dir-cont-factor': dir_cont})

MU0 = [(4, 4)]
ALPHA0 = [(4, 5), (2, 6)]
TAU = [240.0, 360.0, 600.0, 900.0, 1200.0]
N_PARTICLES = [20]
RESAMPLE_EVERY = [10]
TIME_NOISE = [4]
# FREQ_CUTOFF = [4000]
DIR_CONT = [1, 2, 4, 0.5]

for alpha0, mu0, tau, n_particles, res_ev, t_noise, dir_cont in itertools.product(ALPHA0, MU0, TAU, N_PARTICLES, RESAMPLE_EVERY, TIME_NOISE, DIR_CONT):
    HYPER_PARAM_COMBOS.append({'--inf-alpha-0': alpha0, '--inf-mu-0': mu0, '--inf-tau': tau, '--n-particles': n_particles,
                               '--resample-every': res_ev, '--dataset': 'data/train/train_250_{}.pklz'.format(t_noise), '--test-file': 'data/test/test_250_{}.pklz'.format(t_noise), '--time-noise': t_noise, '--dir-cont-factor': dir_cont})

for p, config in enumerate(HYPER_PARAM_COMBOS):

    #
    # adding default values
    for k, v in DEFAULT_VALUES.items():
        if k not in config:
            config[k] = v

    exp_id = ''.join('{}={}'.format(v, config[k]) for k, v in SENSIBLE_PARAMS.items())

    #
    # print command
    cmd = '{} -- {}'.format(PYTHON_INTERPRETER, GEN_BIN_PATH)

    cmd += " --exp-id '{}'".format(exp_id)

    cmd += ' -o {} '.format(OUT_DIR)

    for k, v in config.items():
        if v is not None:
            try:
                if len(v) > 0 and not isinstance(v, str):
                    cmd += ' {} {}'.format(k, ' '.join(str(v_i) for v_i in v))
                elif isinstance(v, str):
                    cmd += " {} '{}'".format(k, v)

            except:
                cmd += ' {} {}'.format(k, v)

    cmd += ' {}'.format(VERBOSITY_LEVEL)

    print(cmd)

    #
    # create a new cluster script file
    cluster_script_name = '{}-{}'.format(CLUSTER_SCRIPT_FILENAME, p)
    cluster_cmds = copy(CLUSTER_SCRIPT_TEMPLATE)
    cluster_args = cmd.replace(PYTHON_INTERPRETER, '')
    # cluster_args = cluster_args.replace('"', '\\"')
    cluster_args = '"{}"'.format(cluster_args)
    cluster_cmds = cluster_cmds.format(CLUSTER_PYTHON_INTERPRETER,
                                       cluster_args,
                                       cluster_script_name, cluster_script_name, cluster_script_name,
                                       MEMORY, N_CPUS)
    exp_file_path = os.path.join(OUT_DIR, '{}.sub'.format(cluster_script_name))
    with open(exp_file_path, 'w') as f:
        f.write(cluster_cmds)


genexp_file_path = os.path.join(OUT_DIR, 'gen-exp.sh')
with open(genexp_file_path, 'w') as f:
    for p, _config in enumerate(HYPER_PARAM_COMBOS):
        f.write('condor_submit_bid {} ./{}\n'.format(BID,
                                                     os.path.join(OUT_DIR, '{}-{}.sub'.format(CLUSTER_SCRIPT_FILENAME, p))))
