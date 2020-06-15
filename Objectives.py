import shared
from SEIRD_clinical import make_data
from Initialization import get_real_data, get_real_data_fold, get_real_data_county, train_fold

import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd.scipy.special import gammaln
from copy import copy


# Loss based on prediction error, for cross-validation
def prediction_loss_cv(params, iter=0, fold=None, train=True):
    X_est = make_data(params, shared.consts)
    if fold:
        X = get_real_data_fold(fold,train=train)
        if train:
            X_est_folds = [[x[idx] for idx in train_fold(fold)] for x in X_est]
        else:
            X_est_folds = [[x[idx] for idx in fold] for x in X_est]
    else:
        X = get_real_data(shared.consts['T'])
        X_est_folds = X_est

    return error_predict_loss(X, X_est_folds, fold.stop - fold.start)


# Loss based on prediction error, taking a random batch
def prediction_loss_sgd(params, iteration=0, length=-1):
    counties = shared.batches[iteration % shared.consts['num_batches']]
    X = get_real_data(length, counties=counties)
    X_est = make_data(params, shared.consts, T=length, counties=counties)

    error = error_predict_loss(X, X_est, shared.consts['T'], counties=counties)
    return error


# Returns the relevant parameters and constants for the given counties
def get_county_params(params, counties):
    county_params = {}
    county_consts = copy(shared.consts)
    for k, v in params.items():
        if np.ndim(params[k]) >= 1:
            county_params[k] = np.array(params[k])[counties]
        else:
            county_params[k] = params[k]
    for k, v in county_consts.items():
        if np.ndim(county_consts[k]) >= 1:
            county_consts[k] = np.array(county_consts[k])[counties]
    return county_params, county_consts


# Loss based on prediction error
def prediction_loss(params,iter=0,length=-1):
    X = get_real_data(length)
    X_est = make_data(params, shared.consts, T=length)

    return error_predict_loss(X, X_est, length)  # + penalty_regularization(params)


# Loss based on prediction error
def prediction_county(params,iter=0,county=0,length=-1):
    X = get_real_data(length, counties=[county])
    X_est = make_data(params, shared.consts, T=length, counties=[county])

    return error_predict_loss(X, X_est, shared.consts['T'], counties=[county])


# Calculate any penalties and add regularization
def penalty_regularization(params):
    return np.sum([np.exp(-shared.consts['n'][i] * params['c_0'][i]) / shared.consts['n'][i]**2
                   for i in range(len(shared.consts['n']))])


# Calculate error in prediction
def error_predict_loss(data_true, data_est, length, counties=[]):
    if length <= 0:
        length = shared.consts['T']
    # Mean Square Error
    if counties:
        n = np.repeat(shared.consts['n'][counties], 2)
        gamma = shared.consts['gamma_death'][np.repeat(counties, 2)]
    else:
        n = np.repeat(shared.consts['n'], 2)  # Since we're considering two compartments for each county
        gamma = shared.consts['gamma_death']

    return np.einsum('ij,i->', np.einsum('i,i,ij->ij', n, gamma, data_true - data_est)**2, 1/n**2) / length * 1e9
    # np.sum([np.sum(np.multiply(n * shared.consts['gamma_death'][i], data_true[i] - data_est[i])**2) / n[i]**2
    #               for i in range(len(data_true))]) / length * 1e9


# Calculate error in prediction
def error_predict(data_true, data_est):
    err_total = 0

    for i in range(len(data_true)):
        err_total += np.linalg.norm(data_true[i] - data_est[i])

    return err_total


# Calculate log-likelihood, assuming y~Binomial
def log_likelihood(params, y, c):
    # Define parameters
    c_0 = c['c_0']
    beta = params['beta']
    tau = c['tau']

    # Calculate the log likelihood
    orig_A = np.array([[1 + c['rho']*beta - c['rho_CI'], c['rho']*beta, 0],
                       [c['rho_CI'], 1 - c['rho_IR'], 0],
                       [0, c['rho_IR'], 1]])
    A = orig_A
    e_I = np.array([0, 1, 0])
    LL = 0
    x_0 = np.array([c_0, 0, 0])

    for i in range(c['num_traj']):
        for t in range(1,c['T']):
            eta = c['n']*e_I@A@x_0
            A = A@orig_A
            binom_coeff = gammaln(eta + 1) - gammaln(eta - y[i,t] + 1) - gammaln(y[i,t] + 1)
            logs = y[i,t]*np.log(tau) + (eta - y[i,t])*np.log(1 - tau)
            LL += binom_coeff + logs

    # Normalize
    LL = LL / (c['num_traj'] * (c['T'] - 1))

    return -LL


# Calculate log-likelihood, assuming y~Gaussian (normal approx. to Binomial)
def log_likelihood_gaussian(params, y, consts):
    # Get county information
    num_counties = np.shape(consts['n'])[0]

    # Get constants
    T = consts['T']
    n = consts['n']
    rho = consts['rho']
    rho_CI = consts['rho_CI']
    rho_IR = consts['rho_IR']

    # Define unknown parameters
    c_0 = consts['c_0']
    tau = consts['tau']
    beta_C = params['beta']
    beta_I = params['beta']

    # Calculate the log likelihood
    e_I = np.array([0, 1, 0])
    x_0 = [[c_0[i], 0, 0] for i in range(num_counties)]
    orig_A = np.asarray([[[1 + rho[i] * beta_C[i] - rho_CI[i], rho[i] * beta_I[i], 0],
                          [rho_CI[i], 1 - rho_IR[i], 0],
                          [0, rho_IR[i], 1]]
                         for i in range(num_counties)])
    A = copy(orig_A)
    LL = 0

    for i in range(num_counties):
        for t in range(1, T):
            eta = n[i] * (e_I @ A[i]) @ x_0[i]
            A[i] = A[i] @ orig_A[i]
            logs = np.log(tau[i] * (1 - tau[i]) * eta)
            fraction = 1 / 2 * ((y[i, t] - tau[i] * eta) / (tau[i] * (1 - tau[i]) * eta)) ** 2
            LL -= fraction + logs

    # Normalize
    LL = LL / (num_counties * (consts['T'] - 1))

    return -LL
