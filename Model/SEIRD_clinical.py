import autograd.numpy as np
from Helper import sigmoid, sig_function, posynomial_function
import multiprocessing as mp
from functools import partial


def make_data(true_params, consts, T=-1, counties=[], return_all=False):
    # Combine the dictionaries
    params = {**true_params, **consts}

    # The simple function we use here is sigmoid
    simple_function = sigmoid

    # Get county information
    if counties:
        num_counties = len(counties)
    else:
        num_counties = np.shape(params['n'])[0]
        counties = range(num_counties)

    # Get parameters
    if T == -1:
        T = params['T']

    # Parametric function we're using is sigmoid(z) for scalar values, sigmoid(w'*x + b) for mobility data x
    ##  Rho rates  ##
    # E -> I
    rho_EI = simple_function(params['rho_EI_coeffs']) * np.ones(num_counties)
    # I -> R
    rho_IR = simple_function(params['rho_IR_coeffs']) * np.ones(num_counties)
    # delay = np.ceil(1/rho_EI + 1/rho_IR)

    ##  Fatality ratios  ##
    # Ratio of I -> D, vs I -> R
    fatality_I = simple_function(params['fatality_I']) * np.ones(num_counties) * params['fatality_I_max']

    # Set parameters
    n = params['n'][counties]
    rho = params['rho'][counties]
    initial_deaths = params['death_data'][counties, params['begin_cases']]

    if len(params['ratio_E']) == 1:
        x_0 = np.reshape(params['initial_condition'], (1,3))
        ##  Beta rates  ##
        # I -> R
        beta_I = posynomial_function(params['mobility_data'][counties], params['beta_I_coeffs'], params['beta_I_bias'],
                                     params['n'][counties]) * params['beta_max']
        # E -> I
        beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E']), beta_I)
    else:
        x_0 = params['initial_condition'][counties]
        ##  Beta rates  ##
        # I -> R
        beta_I = posynomial_function(params['mobility_data'][counties], params['beta_I_coeffs'][counties],
                              params['beta_I_bias'][counties], params['n'][counties]) * params['beta_max']
        # E -> I
        beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E'][counties]), beta_I)

    # Define variables
    X_data = []

    # Run the dynamics and take observations
    for i in range(num_counties):
        E = [np.exp(x_0[i][0])]
        I = [np.exp(x_0[i][1])]
        R = [initial_deaths[i]/params['fatality_I_max'] + np.exp(x_0[i][2])]  # Enforce R(0) >= D(0)/max_fatality_ratio
        D = [initial_deaths[i]]
        for t in range(T - 1):
            # Susceptible compartment
            S = 1 - E[-1] - I[-1] - R[-1] - D[-1]
            # Exposed compartment
            E.append((1 - rho_EI[i]) * E[-1] + rho[i] * S * (beta_E[i, t] * E[-1] + beta_I[i, t] * I[-1]))
            # Infected compartment
            I.append((1 - rho_IR[i]) * I[-1] + rho_EI[i] * E[-2])
            # Recovered compartment
            R.append(R[-1] + (1 - fatality_I[i]) * rho_IR[i] * I[-2])
            # Dead compartment
            D.append(D[-1] + fatality_I[i] * rho_IR[i] * I[-2])
        if return_all:
            X_data.append(output(E, n[i]))
            X_data.append(output(I, n[i]))
            X_data.append(output(R, n[i]))
        X_data.append(output(D, n[i]))

    return np.array(X_data)


# Changes the categories for output. Makes them increase by increments of 1/n
def output(x, n):
    return x  # [np.ceil(x_i*n)/n for x_i in x]


def make_data_parallel(params, consts, pool, T=-1, counties=[], return_all=False):
    # For the multiprocessing
    global_params = {}

    # The simple function we use here is sigmoid
    simple_function = sigmoid

    # Get county information
    if not counties:
        counties = range(np.shape(consts['n'])[0])

    # Get parameters
    if T == -1:
        global_params['T'] = consts['T']
    else:
        global_params['T'] = T

    rho = consts['rho'][counties]
    x_0 = params['initial_condition'][counties]

    ##  Beta rates  ##
    # I -> R
    beta_I = sig_function(consts['mobility_data'][counties], params['beta_I_coeffs'][counties],
                          params['beta_I_bias'][counties])
    # E -> I
    beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E'][counties]), beta_I)

    ##  Rho rates  ##
    # E -> I
    global_params['rho_EI'] = simple_function(params['rho_EI_coeffs'])
    # I -> R
    global_params['rho_IR'] = simple_function(params['rho_IR_coeffs'])

    ##  Fatality ratios  ##
    # Ratio of I -> D, vs I -> R
    global_params['fatality_I'] = simple_function(params['fatality_I'])

    # Run the dynamics and take observations
    do_mp = partial(make_county, global_params=global_params, return_all=return_all)
    args = [[rho[i], x_0[i], beta_E[i, :], beta_I[i, :]] for i in range(len(counties))]
    X_data = pool.starmap(do_mp, args, chunksize=(len(counties) // mp.cpu_count() + 1))

    return np.reshape(X_data, (len(X_data) * len(X_data[0]), global_params['T']))


def make_county(rho, x_0, beta_E, beta_I, global_params, return_all):
    X_data = []

    # Initial conditions
    E = [np.exp(x_0[0])]
    I = [np.exp(x_0[1])]
    R = [np.exp(x_0[2])]
    D = [np.exp(x_0[3])]

    # Run the dynamics
    for t in range(global_params['T'] - 1):
        # Susceptible compartment
        S = 1 - E[-1] - I[-1] - R[-1] - D[-1]
        # Exposed compartment
        E.append((1 - global_params['rho_EI']) * E[-1] + rho * S * (beta_E[t] * E[-1] + beta_I[t] * I[-1]))
        # Infected compartment
        I.append((1 - global_params['rho_IR']) * I[-1] + global_params['rho_EI'] * E[-2])
        # Recovered compartment
        R.append(R[-1] + (1 - global_params['fatality_I']) * global_params['rho_IR'] * I[-2])
        # Dead compartment
        D.append(D[-1] + global_params['fatality_I'] * global_params['rho_IR'] * I[-2])
    if return_all:
        X_data.append(E)
    X_data.append(I)
    if return_all:
        X_data.append(R)
    X_data.append(D)

    return np.array(X_data)


def make_data_county(grad_params, consts, county, T=-1, return_all=False):
    # Combine the dictionaries
    params = {**grad_params, **consts}

    # Get parameters
    if T == -1:
        T = params['T']
    rho = params['rho']
    initial_condition = params['initial_condition']

    # Parametric function we're using is sigmoid(w'*x + b)
    simple_function = sigmoid

    ##  Beta rates  ##
    # S -> E, via E
    # beta_E = sig_function(params['mobility_data'], params['beta_E_coeffs'], params['beta_E_bias'])
    # S -> E, via I
    beta_I = sig_function(np.expand_dims(params['mobility_data'][county, :], axis=0), params['beta_I_coeffs'],
                          params['beta_I_bias'])
    beta_E = (beta_I.T * simple_function(params['ratio_E'])).T

    ##  Rho rates  ##
    # E -> I
    rho_EI = simple_function(params['rho_EI_coeffs'])
    # I -> R
    rho_IR = simple_function(params['rho_IR_coeffs'])

    ##  Fatality ratios  ##
    # Ratio of I -> D, vs I -> R
    fatality_I = simple_function(params['fatality_I'])

    # Define variables
    X_data = []

    # Run the dynamics and take observations
    E = [np.exp(initial_condition[0])]
    I = [np.exp(initial_condition[1])]
    R = [np.exp(initial_condition[2])]
    D = [np.exp(initial_condition[3])]
    for t in range(T - 1):
        # Susceptible compartment
        S = 1 - E[-1] - I[-1] - R[-1] - D[-1]
        # Exposed compartment
        E.append((1 - rho_EI) * E[-1] + rho * S * (beta_E[0, t] * E[-1] + beta_I[0, t] * I[-1]))
        # Infected compartment
        I.append((1 - rho_IR) * I[-1] + rho_EI * E[-2])
        # Recovered compartment
        R.append(R[-1] + (1 - fatality_I) * rho_IR * I[-2])
        # Dead compartment
        D.append(D[-1] + fatality_I * rho_IR * I[-2])
    if return_all:
        X_data.append(E)
    X_data.append(I)
    if return_all:
        X_data.append(R)
    X_data.append(D)

    return X_data
