import shared
# from SEIRD_clinical import make_data
from SEAIHRD import make_data
from Helper import simple_function

import autograd.numpy as np
import random as rd
from math import pi
import pandas as pd
from datetime import datetime, timedelta
from os.path import exists
from os import mkdir
from astropy.convolution import convolve, Box1DKernel

# For beta parameters with sigmoidal activation
# beta_min = -1
# beta_max = 1

# For beta parameters with posynomial activation
beta_min = 0.01
beta_max = 1
bias_min = np.log(1e-7)
bias_max = np.log(1e-3)
alpha_mob_min = 1.5
alpha_mob_max = 2
c_mob_min = np.log(0.01)
c_mob_max = np.log(0.5)

# For initial conditions
init_min = 1e-5
init_max = 1e-4


# Performs dimensionality reduction via PCA, and standardizes into [0, +inf]
def pca_reduce(mob_data, do_pca=True):
    eps = 0.001
    if not do_pca:
        unscaled_data = mob_data
        return (unscaled_data + 100) / 100
    else:
        first_diff = mob_data[1:] - mob_data[:-1]
        L, W = np.linalg.eig(np.cov(first_diff.T))
        W = W[:,np.argsort(L)[-shared.consts['num_mob_components']:]]
        unscaled_data = np.vstack((W.T@mob_data[0,:], W.T@mob_data[0,:] + np.cumsum(first_diff@W,0)))
        return (unscaled_data - np.min(unscaled_data) + eps) / (np.max(unscaled_data) - np.min(unscaled_data) - eps)
        # return (unscaled_data + 100) / 100


# Smooths out a signal by applying mean filtering
def smooth(x, filter_width=2, length=0):
    x = np.squeeze(x)
    sizes = np.shape(x)
    # Either shorten the list to size, or prepend with zeros
    if sizes[0] >= length:
        x = x[-length:]
    elif len(sizes) == 1:
        x = np.hstack((np.zeros((length - sizes[0])), x))
    else:
        x = np.hstack((np.zeros((length - sizes[0], sizes[1])), x))
    smooth_x = np.zeros(len(x))
    for ii in range(len(x)):
        smooth_x[ii] = np.mean(x[max(0,ii-filter_width):ii+1])
    return smooth_x
    # return convolve(x, Box1DKernel(width=filter_width))


# "Subtracts" two ranges, i.e., finds the range corresponding to the larger one with the smaller removed.
def range_diff(r1, r2):
    return list(set(r1).difference(set(r2)))


# Calculates a training fold based on a given test fold
def train_fold(test_fold):
    return range_diff(range(0,shared.consts['T']),test_fold)


# Returns a training fold of the real data for cross-validation.
# The training fold is the complement of the test fold.
def get_real_data_fold(test_fold,train=True):
    if train:
        fold = [t + shared.consts['begin_cases'] for t in train_fold(test_fold)]
    else:
        fold = [t + shared.consts['begin_cases'] for t in test_fold]
    cases = shared.consts['case_data'][:,fold]
    deaths = shared.consts['death_data'][:,fold]
    # X = deaths
    X = np.empty((cases.shape[0]+deaths.shape[0], cases.shape[1]))
    X[::2,:] = cases
    X[1::2,:] = deaths
    return X


# Returns real/simulated data
def get_real_data(length, counties=[]):
    if shared.real_data:
        if length <= 0:
            length = shared.consts['T']
        if counties:
            cases = shared.consts['case_data'][counties,shared.consts['begin_cases']:(shared.consts['begin_cases'] + length)]
            deaths = shared.consts['death_data'][counties,shared.consts['begin_cases']:(shared.consts['begin_cases'] + length)]
        else:
            cases = shared.consts['case_data'][:,shared.consts['begin_cases']:(shared.consts['begin_cases'] + length)]
            deaths = shared.consts['death_data'][:,shared.consts['begin_cases']:(shared.consts['begin_cases'] + length)]
        if shared.consts['include_cases']:
            X = np.empty((cases.shape[0]+deaths.shape[0], cases.shape[-1]))
            X[::2,:] = cases
            X[1::2,:] = deaths
        else:
            X = deaths
    else:
        _, X = make_data(shared.true_params, shared.consts, T=length)
    return X


# Returns real/simulated data
def get_real_data_county(length, counties):
    if shared.real_data:
        if length <= 0:
            length = shared.consts['T']
        # X = shared.consts['death_data'][:,shared.consts['begin_cases']:(shared.consts['begin_cases'] + length)]
        cases = shared.consts['case_data'][counties,shared.consts['begin_cases']:(shared.consts['begin_cases'] + length)]
        deaths = shared.consts['death_data'][counties,shared.consts['begin_cases']:(shared.consts['begin_cases'] + length)]
        X = np.empty((cases.shape[0]+deaths.shape[0], cases.shape[1]))
        X[::2,:] = cases
        X[1::2,:] = deaths
    else:
        _, X = make_data(shared.true_params, shared.consts, T=length)
    return X


# Compute rho based on population and land area of a county
def calculate_rho(population, area, radius=0.1):
    # Some book-keeping parameters
    rho_min = 0.01
    rho_max = 1

    # Average number of contacts per unit area, assuming contacts occur in a circle of the given radius
    if area == 0:
        rho = rho_min
    else:
        rho = np.max([np.min([population * pi * radius**2 / area, rho_max]), rho_min])

    return rho


# For creating an initial condition
def initial_condition(county=-1):
    if county == -1:
        num_counties = len(shared.consts['n'])
        initial_deaths = shared.consts['death_data'][:, shared.consts['begin_cases']]
        initial_deaths[initial_deaths == 0] = 1e-10
        return np.squeeze([[np.log((rd.random()*(init_max - init_min) + init_min)),  # E
                            np.log((rd.random()*(init_max - init_min) + init_min)),  # I
                            np.log(initial_deaths[i]),                               # R and D
                            np.log((rd.random()*(init_max - init_min) + init_min)),  # A
                            np.log((rd.random()*(init_max - init_min) + init_min))]  # H
                           for i in range(num_counties)])
    else:
        return np.array([np.log((rd.random()*(init_max - init_min) + init_min)),
                         np.log((rd.random()*(init_max - init_min) + init_min)),
                         np.log(shared.consts['death_data'][county, shared.consts['begin_cases']])])


# For initializing a beta bias parameter
def initialize_beta_bias(county=-1):
    if county == -1:
        return np.array([rd.random()*(bias_max - bias_min) + bias_min for _ in range(len(shared.consts['n']))])
    else:
        return np.array([rd.random() * (bias_max - bias_min) + bias_min])


# For initializing a beta coefficient parameter
def initialize_beta_coeffs(county=-1):
    if county == -1:
        num_counties = len(shared.consts['n'])
        return np.array([[np.ones(shared.consts['num_mob_components']) * rd.random() * (alpha_mob_max - alpha_mob_min) + alpha_mob_min,
                        np.ones(shared.consts['num_mob_components']) * rd.random() * (c_mob_max - c_mob_min) + c_mob_min]
                         for _ in range(num_counties)])
    else:
        return np.array([np.ones(shared.consts['num_mob_components']) * rd.random() * (alpha_mob_max - alpha_mob_min) + alpha_mob_min,
                         np.ones(shared.consts['num_mob_components']) * rd.random() * (c_mob_max - c_mob_min) + c_mob_min])


# For initializing a rho bias parameter
def initialize_rho_bias():
    return [rd.random() * 0.6 - 0.3]


# For initializing a rho coefficient parameter
def initialize_rho_coeffs():
    return [rd.random() * -4 - 1]  # [np.ones(3) * rd.random() * -4 - 1]


# Save a dictionary into a pandas dataframe
def params_to_pd(params, dir):
    if not exists(dir):
        mkdir(dir)

    num_counties = len(shared.consts['n'])

    columns = ['state', 'county', 'population',
               'initial_Exposed', 'initial_Infected', 'initial_Recovered', 'initial_Dead',
               'fatality_ratio', 'exposed_infection_ratio', 'rho_EI', 'rho_IR',
               'beta_I_bias', 'beta_I_retail_and_recreation', 'beta_I_grocery_and_pharmacy', 'beta_I_parks',
               'beta_I_transit_stations', 'beta_I_workplaces', 'beta_I_residential']
    df_params = pd.DataFrame(columns=columns)

    date_list = pd.date_range(datetime(2020,1,21) + timedelta(days=shared.consts['begin_cases']), periods=shared.consts['T'])
    date_columns = ['state', 'county'] + ['cases-{}-{:0>2}-{:0>2}'.format(d.year,d.month,d.day) for d in date_list] +\
                   ['deaths-{}-{:0>2}-{:0>2}'.format(d.year, d.month, d.day) for d in date_list]
    df_data = pd.DataFrame(columns=date_columns)

    for c in range(num_counties):
        row_params, row_data = params_to_df_row(params[c], c)
        df_params.loc[c] = row_params
        df_data.loc[c] = row_data

    df_params.to_csv(dir + 'params.csv', index=False)
    df_data.to_csv(dir + 'data.csv', index=False)


def params_to_df_row(p, c):
    sc = shared.consts
    row_params = [sc['county_names'][c][:2], sc['county_names'][c][3:], sc['n'][c]] + \
                  list(np.squeeze(sc['n'][c]*np.exp(p['initial_condition']))) + [simple_function(p['fatality_I']),
                  simple_function(p['ratio_E']), simple_function(p['rho_EI_coeffs']), simple_function(p['rho_IR_coeffs']),
                  p['beta_I_bias']] + list(np.squeeze(p['beta_I_coeffs']))
    X_est = make_data(p, shared.consts, counties=[c])
    row_data = [sc['county_names'][c][:2], sc['county_names'][c][3:]] + [int(sc['n'][c]*x) for x in X_est[0]] + [int(sc['n'][c]*x) for x in X_est[0]]
    return row_params, row_data
