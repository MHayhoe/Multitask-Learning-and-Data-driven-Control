# Our scripts
from SEIRD_clinical import make_data, make_data_parallel
import Initialization as Init
from Objectives import prediction_loss, prediction_loss_sgd, prediction_county
import shared
import random as rd
from Helper import inverse_sigmoid, sig_function

# For autograd
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

# Helpers
import multiprocessing as mp
from functools import partial
from copy import copy
import matplotlib.pyplot as plt
import time


# Makes an initial guess for optimization parameters
def make_guess(batch_num=1):
    num_counties = len(shared.consts['n'])
    return {'rho_EI_coeffs': inverse_sigmoid(1 / 5.1),
            'rho_IR_coeffs': inverse_sigmoid(1 / 15.9),
            'beta_I_coeffs': Init.initialize_beta_coeffs(),
            'beta_I_bias': Init.initialize_beta_bias(),
            'fatality_I': inverse_sigmoid(0.01 * batch_num),
            'ratio_E': np.array([inverse_sigmoid(0.1) for i in range(num_counties)]),
            'c_0': Init.initial_condition()}


# Run the optimization by splitting the counties into batches across epochs, and only updating one batch per step
def optimize_sgd(num_epochs=20, num_batches=1, num_trials=8, step_size=0.01):
    # If number of batches is too large, make it the number of counties
    num_batches = min(num_batches, len(shared.consts['n']))
    shared.consts['num_batches'] = num_batches

    # Multiprocessing
    pool = mp.Pool(mp.cpu_count())
    do_mp = partial(optimize_trial, consts=shared.consts,begin=shared.begin,num_epochs=num_epochs,step_size=step_size)
    mp_params = pool.map(do_mp, range(num_trials))
    opt_params = {}

    # Unpack results for each county, returning parameters which give the lowest error
    for c in range(len(shared.consts['n'])):
        obj_vals = [prediction_county(p,county=c) for p in mp_params]
        county_params = copy(mp_params[np.argmin(obj_vals)])
        for k,v in county_params.items():
            if np.ndim(county_params[k]) >= 1:
                county_params[k] = np.expand_dims(county_params[k][c],axis=0)
        opt_params[c] = county_params

    return opt_params


def optimize_parallel_counties(num_epochs=20, num_batches=1, num_trials=8, step_size=0.01):
    # If number of batches is too large, make it the number of counties
    num_batches = min(num_batches, len(shared.consts['n']))
    shared.consts['num_batches'] = num_batches

    # Multiprocessing
    pool = mp.Pool(mp.cpu_count())
    mp_params = []
    for i in range(num_trials):
        mp_params.append(optimize_trial(i, shared.consts, shared.begin, num_epochs, step_size, pool))
    opt_params = {}

    # Unpack results for each county, returning parameters which give the lowest error
    for c in range(len(shared.consts['n'])):
        obj_vals = [prediction_county(p,county=c) for p in mp_params]
        county_params = copy(mp_params[np.argmin(obj_vals)])
        for k,v in county_params.items():
            if np.ndim(county_params[k]) >= 1:
                county_params[k] = np.expand_dims(county_params[k][c],axis=0)
        opt_params[c] = county_params

    return opt_params


# Picks a random initial starting point for a trial, and runs the optimization
def optimize_trial(trial, consts, begin, num_epochs, step_size, pool=None, show_plots=False):
    grad_predict = grad(partial(prediction_loss_sgd, pool=pool))
    shared.consts = consts
    shared.begin = begin
    guess = make_guess()
    shared.batches = create_batches(range(len(shared.consts['n'])), shared.consts['num_batches'])

    # For plotting
    plot_performance = partial(print_performance, trial=trial, show_plots=show_plots)

    if show_plots:
        setup_plotting()

    # Do the optimization
    opt_params = adam(grad_predict, guess, step_size=step_size, num_iters=num_epochs*shared.consts['num_batches'],
                      callback=plot_performance)

    # Calculate the error on the test data and return the optimized parameters
    obj_val = prediction_loss(opt_params)
    print('Done batch {} with test loss {:.3e}.'.format(trial + 1, obj_val))
    return opt_params


# Splits the counties into the desired number of batches
def create_batches(x, num_batches):
    size_x = len(x)
    items = list(range(size_x))
    rd.shuffle(items)

    # If we don't have enough items, return batches with one county in each of them
    if num_batches >= size_x:
        batches = [[c] for c in items]
    else:
        batch_size, leftover = divmod(size_x, num_batches)
        batches = [items[batch_size*i:batch_size*(i+1)] for i in range(num_batches)]
        for i in range(leftover):
            batches[i].append(items[batch_size*num_batches + i])
    return batches


# Callback function for optimization
def print_performance(params, iteration, gradient, trial, show_plots):
    if iteration % (shared.consts['num_batches'] * 100) == 0:
        print('Trial {}: epoch {}, loss {:.3e}'.format(trial, iteration // shared.consts['num_batches'], prediction_loss(params)))
        # print(gradient)
        if show_plots:
            plot_trajectories(params, gradient, trial, iteration)


# Sets up plotting for optimization callbacks
def setup_plotting():
    shared.plot_values = {}
    num_counties = len(shared.consts['n'])
    plt.ion()
    plt.subplots(num_counties, 3)
    plt.tight_layout()
    plt.show()
    county_range = range(num_counties)
    shared.plot_values['fold'] = None
    shared.plot_values['loss'] = [[] for i in county_range]
    global_keys = set(['rho_EI_coeffs', 'rho_IR_coeffs', 'fatality_I'])
    guess = make_guess()
    county_keys = set(guess.keys()).difference(global_keys)
    for k in global_keys:
        shared.plot_values[k] = []
    for k in county_keys:
        shared.plot_values[k] = [[] for i in county_range]


# Plots trajectories based on current parameters
def plot_trajectories(params, plot_params, batch, iteration):
    num_counties = len(shared.consts['n'])
    num_compartments = 4
    num_real_compartments = 2
    num_plots = 3

    X = Init.get_real_data(shared.consts['T'])
    real_X = (np.asarray(X).T * np.repeat(shared.consts['n'], num_real_compartments)).T

    X_est = make_data(params, shared.consts, return_all=True)
    est_X = (np.asarray(X_est).T * np.repeat(shared.consts['n'], num_compartments)).T

    global_keys = set(['rho_EI_coeffs', 'rho_IR_coeffs', 'fatality_I'])
    county_keys = set(plot_params.keys()).difference(global_keys)

    plt.clf()

    # Update global parameters
    for k in global_keys:
        shared.plot_values[k].append(np.squeeze(plot_params[k]))

    for i in range(num_counties):
        # Plot trajectories
        ax = plt.subplot(num_counties, num_plots, num_plots * i + 1)
        if shared.real_data:
            plt.plot(real_X[i * num_real_compartments:(i + 1) * num_real_compartments, :].T)
            plt.plot(est_X[i * num_compartments:(i + 1) * num_compartments, :].T, '--')
            # plt.legend(['Cases','Deaths','(E) Exposed','(I) Infected','(R) Removed','(D) Dead'])
        else:
            plt.plot(real_X[i * num_compartments:(i + 1) * num_compartments, :].T,
                     label=['Exp.', 'Inf.', 'Rem.', 'Dead'])
            plt.plot(est_X[i * num_compartments:(i + 1) * num_compartments, :].T, '--',
                     label=['Exp.', 'Inf.', 'Rem.', 'Dead'])
            # plt.legend(['(E) Exposed','(I) Infected','(R) Removed','(D) Dead','(E) Exposed','(I) Infected','(R) Removed','(D) Dead'])
        if shared.plot_values['fold']:
            ax.fill_between(shared.plot_values['fold'], 0, ax.get_ylim()[1], facecolor='blue', alpha=0.3)
        ax.yaxis.set_ticklabels([])

        # Plot parameter values
        ax1 = plt.subplot(num_counties, num_plots, num_plots * i + 2)
        plt.title(
            '{} ({:.0f}), iteration {}'.format(shared.consts['county_names'][i], shared.consts['n'][i], iteration))
        for k in global_keys:
            ax1.plot(np.asarray(shared.plot_values[k]), label=k)

        for k in county_keys:
            if not np.shape(plot_params[k][i]):  # This is a scalar
                shared.plot_values[k][i].append(np.squeeze(plot_params[k][i]))
            elif np.shape(shared.plot_values[k][i])[0] == 0:
                shared.plot_values[k][i] = np.expand_dims(np.squeeze(plot_params[k][i]), 0)
            else:
                shared.plot_values[k][i] = np.concatenate(
                    (shared.plot_values[k][i], np.expand_dims(np.squeeze(plot_params[k][i]), 0)))
            # plt.clf()
            ax1.plot(np.asarray(shared.plot_values[k][i]), label=k)

        # Plot loss
        shared.plot_values['loss'][i].append(prediction_loss(params))
        ax2 = ax1.twinx()
        ax2.semilogy(shared.plot_values['loss'][i], '--', label='Loss')
        ax1.yaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        # Plot mobility data
        ax1 = plt.subplot(num_counties, num_plots, num_plots * i + 3)
        beta_I = sig_function(shared.consts['mobility_data'][i][:shared.consts['T'], :], params['beta_I_coeffs'][i],
                              params['beta_I_bias'][i])
        # Line widths based on weights of each mobility category. Normalized to [0,5].
        weights = np.abs(np.squeeze(params['beta_I_coeffs'][i]))
        weights = weights / np.max(weights) * 5
        lines = ax1.plot(shared.consts['mobility_data'][i][:shared.consts['T'], :])
        for j in range(len(lines)):
            lines[j].set_linewidth(weights[j])
        ax2 = ax1.twinx()
        ax2.plot(beta_I, '--')
        ax1.yaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        # Draw the plot
        plt.draw()
        plt.pause(0.001)


# Optimize the parameters using autograd and adam.
# Picks a random initial starting point in each batch, and returns the set of parameters corresponding
# to the batch with the lowest prediction error.
def optimize(num_iters=300, step_size=0.01, num_trials=1, validation_length=30, num_folds=4):
    opt_params = []
    obj_vals = []

    # Run the batches
    for b in range(num_trials):
        opt_par, obj_val = optimize_batch(b, true_params=shared.true_params, consts=shared.consts, begin=shared.begin,
                                          num_iters=num_iters, step_size=step_size, validation_length=validation_length,
                                          num_folds=num_folds)
        opt_params.append(opt_par)
        obj_vals.append(obj_val)

    # Find the batch with lowest error, and return the corresponding parameters
    return opt_params[np.argmin(obj_vals)]


# Optimize the parameters using autograd and adam, parallelized over batches.
# Returns the set of parameters corresponding  to the batch with the lowest prediction error.
def optimize_parallel(num_iters=300, step_size=0.01, num_trials=1, validation_length=30, num_folds=4):
    # Multiprocessing
    pool = mp.Pool(mp.cpu_count())
    do_mp = partial(optimize_batch, true_params=shared.true_params, consts=shared.consts, begin=shared.begin,
                    num_iters=num_iters, step_size=step_size, validation_length=validation_length, num_folds=num_folds)
    mp_result = pool.map(do_mp, range(num_trials))

    # Unpack objective values and parameter values
    obj_vals = [v[-1] for v in mp_result]
    opt_params = [v[0] for v in mp_result]

    # Find the batch with lowest test error, and return the corresponding parameters
    return opt_params[np.argmin(obj_vals)]


# Picks a random initial starting point for a batch, and runs the optimization
def optimize_batch(batch, true_params, consts, begin, num_iters, step_size, validation_length, num_folds):
    grad_predict = grad(prediction_loss)
    shared.true_params = true_params
    shared.consts = consts
    shared.begin = begin
    guess = make_guess()

    # For plotting
    plot_performance = partial(print_performance, batch=batch)
    setup_plotting()

    # Do the optimization
    opt_params = adam(grad_predict, guess, step_size=step_size, num_iters=num_iters, callback=plot_performance)

    # Calculate the error on the test data and return it
    obj_val = prediction_loss(opt_params, length=validation_length)
    print('Done batch {} with test loss {:.3e}.'.format(batch + 1, obj_val))
    plt.close()
    return opt_params, obj_val
