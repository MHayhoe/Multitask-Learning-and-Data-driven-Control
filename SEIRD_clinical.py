import autograd.numpy as np
import shared
import matplotlib.pyplot as plt
from Helper import sigmoid, sig_function


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

    # Parametric function we're using is sigmoid(w'*x + b)
    if num_counties == 1:
        rho = params['rho']
        x_0 = params['c_0']
        ##  Beta rates  ##
        # I -> R
        beta_I = sig_function(params['mobility_data'][counties], params['beta_I_coeffs'], params['beta_I_bias'])
        # E -> I
        beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E']), beta_I)
    else:
        rho = params['rho'][counties]
        x_0 = params['c_0'][counties]
        ##  Beta rates  ##
        # I -> R
        beta_I = sig_function(params['mobility_data'][counties], params['beta_I_coeffs'][counties],
                              params['beta_I_bias'][counties])
        # E -> I
        beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E'][counties]), beta_I)

    ##  Rho rates  ##
    # E -> I
    rho_EI = simple_function(params['rho_EI_coeffs']) * np.ones(num_counties)
    # I -> R
    rho_IR = simple_function(params['rho_IR_coeffs']) * np.ones(num_counties)

    ##  Fatality ratios  ##
    # Ratio of I -> D, vs I -> R
    fatality_I = simple_function(params['fatality_I']) * np.ones(num_counties)

    # Define variables
    X_data = []

    # Run the dynamics and take observations
    for i in range(num_counties):
        E = [np.exp(x_0[i][0])]
        I = [np.exp(x_0[i][1])]
        R = [np.exp(x_0[i][2])]
        D = [np.exp(x_0[i][3])]
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
    c_0 = params['c_0']

    # Parametric function we're using is sigmoid(w'*x + b)

    ##  Beta rates  ##
    # S -> E, via E
    # beta_E = sig_function(params['mobility_data'], params['beta_E_coeffs'], params['beta_E_bias'])
    # S -> E, via I
    beta_I = sig_function(np.expand_dims(params['mobility_data'][county, :],axis=0), params['beta_I_coeffs'], params['beta_I_bias'])
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
    E = [np.exp(c_0[0])]
    I = [np.exp(c_0[1])]
    R = [np.exp(c_0[2])]
    D = [np.exp(c_0[3])]
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
