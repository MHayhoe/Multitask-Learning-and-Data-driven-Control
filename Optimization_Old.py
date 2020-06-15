# Our scripts
from SEIRD_clinical import make_data, sig_function
from SEIRD_Markov import make_data_Markov
import Initialization as Init
from Objectives import prediction_loss, prediction_loss_sgd, prediction_loss_cv, prediction_county, error_predict
import shared
import random as rd
from Objectives import log_likelihood_gaussian
from Helper import inverse_sigmoid

# For autograd
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

# Helpers
import multiprocessing as mp
from functools import partial
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


# Makes an initial guess for optimization parameters
def make_guess(batch_num=1):
    num_counties = len(shared.consts['n'])
    return {'rho_EI_coeffs': inverse_sigmoid(1 / 5.1),  # Init.initialize_rho_coeffs(),
            # 'rho_EI_bias':   Init.initialize_rho_bias(),
            # 'rho_IH_coeffs': Init.initialize_rho_coeffs(),
            # 'rho_IH_bias':   Init.initialize_rho_bias(),
            'rho_IR_coeffs': inverse_sigmoid(1 / 15.9),  # Init.initialize_rho_coeffs(),
            # 'rho_IR_bias':   Init.initialize_rho_bias(),
            # 'rho_HR_coeffs': Init.initialize_rho_coeffs(),
            # 'rho_HR_bias':   Init.initialize_rho_bias(),
            # 'rho_ID_coeffs': Init.initialize_rho_coeffs(),
            # 'rho_ID_bias':   Init.initialize_rho_bias(),
            # 'rho_HD_coeffs': Init.initialize_rho_coeffs(),
            # 'rho_HD_bias':   Init.initialize_rho_bias(),
            # 'beta_E_coeffs': Init.initialize_beta_coeffs(),
            # 'beta_E_bias':   Init.initialize_beta_bias(),
            'beta_I_coeffs': Init.initialize_beta_coeffs(),
            'beta_I_bias': Init.initialize_beta_bias(),
            'fatality_I': inverse_sigmoid(0.01 * batch_num),
            # 'fatality_H': 0.01 * batch_num,
            'ratio_E': [inverse_sigmoid(0.1) for i in range(num_counties)],
            'c_0': Init.initial_condition()}


def make_guess_county(county, batch_num=1):
    return {'beta_I_coeffs': Init.initialize_beta_coeffs(county),
            'beta_I_bias': Init.initialize_beta_bias(county),
            'ratio_E': [inverse_sigmoid(0.1)],
            'c_0': Init.initial_condition(county)}


def make_guess_global(batch_num=1):
    return {'rho_EI_coeffs': inverse_sigmoid(1 / 5.1),
            'rho_IR_coeffs': inverse_sigmoid(1 / 15.9),
            'fatality_I': inverse_sigmoid(0.01 * batch_num)}


def optimize_counties(num_rounds, num_trials, num_iters=1000, step_size=0.01, validation_length=40):
    # Multiprocessing
    pool = mp.Pool(mp.cpu_count())
    num_counties = len(shared.consts['n'])
    settings = {'num_trials': num_trials, 'step_size': step_size, 'validation_length': validation_length}

    # Make initial guesses
    global_params = [make_guess_global() for _ in range(num_trials)]
    local_params = [make_guess_county(i) for i in range(num_counties)]
    local_dict = list_to_dict(local_params)

    # Do rounds of alternating local and global optimization
    for r in range(num_rounds):
        # Optimize local parameters among all counties
        settings['num_iters'] = num_iters
        mp_local = partial(optimize_county, consts={**global_params, **shared.consts}, begin=shared.begin,
                           settings=settings)
        local_params = pool.starmap(mp_local, [(c, local_params[c]) for c in range(num_counties)])
        local_dict = list_to_dict(local_params)
        print('Finished optimizing locally for round {}'.format(r+1))

        # Optimize global parameters in a number of batches
        settings['num_iters'] = 300
        mp_global = partial(optimize_global, consts={**local_dict, **shared.consts},
                            begin=shared.begin, settings=settings)
        mp_result = pool.starmap(mp_global, [(b, global_params[b]) for b in range(num_trials)])
        global_params = mp_result[np.argmin([v[-1] for v in mp_result])][0]
        print('Finished optimizing globally for round {}'.format(r+1))

        # After the first round, we do not need to run multiple batches
        settings['num_trials'] = 1

    return {**global_params, **local_dict}


# Transforms a list of dictionaries of parameters into a dictionary of parameters
def list_to_dict(params_list):
    params_dict = {k: np.squeeze([dic[k] for dic in params_list]) for k in params_list[0]}
    params_dict['beta_I_coeffs'] = np.reshape(params_dict['beta_I_coeffs'], (len(shared.consts['n']), 1, 6))
    return params_dict


# Optimize the global parameters
def optimize_global(trial, params, consts, begin, settings):
    grad_predict = grad(prediction_loss)
    shared.consts = consts
    shared.begin = begin

    # Do the optimization
    opt_params = adam(grad_predict, params, step_size=settings['step_size'], num_iters=settings['num_iters'])

    # Calculate the error on the test data and return it
    obj_val = prediction_loss(opt_params, length=settings['validation_length'])
    print('Done global batch {} with test loss {:.3e}.'.format(trial + 1, obj_val))

    return opt_params, obj_val


# Optimize the parameters for a single county
def optimize_county(county, params, consts, begin, settings):
    # Calculate gradient for only this county
    grad_predict = grad(partial(prediction_county, county=county))
    shared.consts = consts
    shared.begin = begin

    num_trials = settings['num_trials']
    obj_vals = np.zeros(num_trials)
    opt_params = []

    for batch in range(num_trials):
        # Do the optimization
        opt_params.append(adam(grad_predict, params, step_size=settings['step_size'], num_iters=settings['num_iters']))

        # Calculate the error on the test data
        obj_vals[batch] = prediction_county(opt_params[-1], county=county, length=settings['validation_length'])

        print('Done batch {} for county {} with test loss {:.3e}.'.format(batch + 1, county, obj_vals[batch]))

    return opt_params[np.argmin(obj_vals)]


# Perform cross-validation, where folds represent windows of time
def cross_validation(batch, step_size, num_iters, num_folds=3):
    test_folds = []
    fold_loss = []
    fold_params = []
    # print_perf = partial(print_performance, batch=batch)
    guess = make_guess()  # One random initial guess used for all folds

    plot_perf = partial(print_performance, batch=batch)
    # setup_plotting()

    # Construct the folds
    fold_length = int(np.ceil(shared.consts['T'] / num_folds))
    for f in range(num_folds):
        test_folds.append(range(f * fold_length, np.min(((f + 1) * fold_length, shared.consts['T']))))

    # Train and test on each fold
    for fold in test_folds:
        shared.plot_values['fold'] = fold
        grad_fold = grad(partial(prediction_loss_cv, fold=fold, train=True))
        fold_params.append(adam(grad_fold, guess, step_size=step_size, num_iters=num_iters, callback=plot_perf))
        loss = prediction_loss_cv(fold_params[-1], fold=fold, train=False)
        # If we encounter an overflow error, make loss high to discard it
        if np.isnan(loss):
            loss = 9e10
        fold_loss.append(loss)
        print('Done fold for batch {} with cross-validation loss {:.3e}.'.format(batch + 1, loss))

    return fold_params[np.argmin(fold_loss)]


def log_transform(x):
    x[np.abs(x) < 1] = 1
    return np.sign(x) * np.log(np.abs(x))


# Plot performance, for callbacks in the optimization
def plot_values(params, batch):
    plt.clf()

    # Plot parameter values
    ax1 = plt.gca()
    for k in params.keys():
        if not np.shape(params[k]):  # This is a scalar
            shared.plot_values[k].append(np.squeeze(params[k]))
        elif np.shape(shared.plot_values[k])[0] == 0:
            shared.plot_values[k] = np.reshape(np.squeeze(params[k]), (1, np.shape(params[k])[-1]))
        else:
            shared.plot_values[k] = np.concatenate(
                (shared.plot_values[k], np.reshape(params[k], (1, np.shape(params[k])[-1]))))
        # plt.clf()
        ax1.plot(np.asarray(shared.plot_values[k]), label=k)
    plt.title('Parameter values and Loss (Batch {})'.format(batch))
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Iteration (100s)')
    plt.legend()

    # Plot loss
    shared.plot_values['loss'].append(prediction_loss(params))
    ax2 = ax1.twinx()
    ax2.plot(shared.plot_values['loss'], '--', label='Loss')
    ax2.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


# Wrap the objective function for autograd
def wrap_log(params, iteration=1):
    return 0  # log_likelihood_gaussian(params, y, consts)


# Maximize the likelihood
def max_ll():
    # Set up
    T_nums = [10, 30, 60]
    num_counties = len(shared.consts['n'])
    estimates = {'beta': [rd.random() * 0.2 + 0.1 for _ in range(num_counties)]}
    maxLL = np.zeros(len(T_nums))
    estLL = np.zeros(len(T_nums))
    err = np.zeros(len(T_nums))

    # Maximize the likelihood
    for jj in range(len(T_nums)):
        shared.consts['T'] = T_nums[jj]
        y, X = make_data(shared.true_params, shared.consts)
        LL_gradient = grad(wrap_log)
        maxLL[jj] = wrap_log(shared.true_params)
        optimized_estimates = adam(LL_gradient, estimates, step_size=0.001, num_iters=300, callback=print_performance)
        err[jj] = np.sum([shared.true_params[k] - v for (k, v) in optimized_estimates.items()])
        estLL[jj] = wrap_log(optimized_estimates)

    # y_est, X_est = make_data(optimized_estimates, consts)


# Perform a grid search over the parameter space
def grid_search(v1_name, min_v1, max_v1, v2_name, min_v2, max_v2, v_step=10):
    # Define all values
    num_counties = len(shared.consts['n'])
    v1 = np.arange(min_v1, max_v1, (max_v1 - min_v1) / v_step)
    v2 = np.arange(min_v2, max_v2, (max_v2 - min_v2) / v_step)
    idx = 1
    plot_split = int(np.ceil(np.sqrt(num_counties)))

    # Simulate data using the true parameters
    y, X = make_data(shared.true_params, shared.consts)

    # Run the grid search
    for c in range(num_counties):
        err_grid = np.zeros((len(v1), len(v2)))
        for i2 in range(len(v2)):
            for i1 in range(len(v1)):
                guess = {
                    v1_name: np.reshape([np.ones(6) * v1[i1] for _ in range(num_counties)], (num_counties, 1, 6)),
                    v2_name: [v2[i2] for _ in range(num_counties)]}
                y_est, X_est = make_data(guess, shared.consts, c)
                err_grid[i2, i1] = error_predict([X[c]], X_est)
        plt.subplot(plot_split, plot_split, idx)
        idx += 1
        ax = sns.heatmap(err_grid, cmap='jet', cbar=False, xticklabels=[], yticklabels=[])
        v1_ind = np.argmin(np.abs(shared.true_params[v1_name][c, 0, 0] - v1))
        v2_ind = np.argmin(np.abs(shared.true_params[v2_name][c] - v2))
        ax.add_patch(Rectangle((v1_ind, v2_ind), 1, 1, fill=False, edgecolor='black', lw=2))
        print(c)
    plt.suptitle("Prediction Error")
    plt.show()
