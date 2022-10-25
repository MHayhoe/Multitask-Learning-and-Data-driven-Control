# Our scripts
import shared
import Initialization as Init
from Optimization import optimize_bootstrap
from SEIRD_clinical import make_data
from SEIRD_bootstrap import bootstrap_data
from Objectives import bootstrap_loss

# For autograd
import autograd.numpy as np

# Helpers
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from os.path import exists
from os import mkdir
import random as rd


def run_bootstrap(num_trials, bootstrap_sample_size, filename, confidence=0.95, validation_days=100):
    # Load parameters and constants
    with open(filename + 'opt_params.pickle','rb') as handle:
        opt_params = pickle.load(handle)
    with open(filename + 'consts.pickle','rb') as handle:
        shared.consts = pickle.load(handle)

    # Create predictions based on learned parameters
    num_counties = len(shared.consts['n'])
    X_est = []
    for c in range(num_counties):
        X_est.append(make_data(opt_params[c], shared.consts, counties=[c], return_all=True))

    # Run the trials to bootstrap
    bootstrap_params = []
    bootstrap_losses = []
    for t in range(num_trials):
        current_params = optimize_bootstrap(X_est, shared.consts, num_iters=2000, step_size=0.01)
        bootstrap_params.append(current_params)
        bootstrap_losses.append(bootstrap_loss(current_params, validation_days))

    # Generate confidence intervals
    sorted_params = bootstrap_params[np.argsort(bootstrap_loss)]
    conf_params = sorted_params[:np.ceil(num_trials * confidence)]
    UCB = []
    LCB = []
    for c in range(num_counties):
        X_region = []
        # Generate trajectories for each set of parameters
        for params in conf_params:
            X_region.append(make_data(params[c], shared.consts, counties=[c], return_all=False))
        # Take upper and lower confidence bounds based on generated trajectories
        UCB.append(np.max(np.asarray(X_region), axis=0))
        LCB.append(np.min(np.asarray(X_region), axis=0))

    return UCB, LCB


if __name__ == '__main__':
    run_bootstrap(10, 20, 'Plots/2020-Jul-16-16-39-56_6_cats/')
