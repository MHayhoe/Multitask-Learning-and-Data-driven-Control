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

    # Parametric function we're using is sigmoid(z) for scalar values, c*x^alpha + b for mobility data x
    ##  Rho rates  ##
    # E -> I
    rho_EI = simple_function(params['rho_EI_coeffs']) * np.ones(num_counties)
    # E -> A
    rho_EA = simple_function(params['rho_EA_coeffs']) * np.ones(num_counties)
    # A -> R
    rho_AR = simple_function(params['rho_AR_coeffs']) * np.ones(num_counties)
    # I -> H
    rho_IH = simple_function(params['rho_IH_coeffs']) * np.ones(num_counties)
    # I -> R
    rho_IR = simple_function(params['rho_IR_coeffs']) * np.ones(num_counties)
    # H -> R
    rho_HR = simple_function(params['rho_HR_coeffs']) * np.ones(num_counties)

    ##  Fatality ratios  ##
    # Ratio of H -> D, vs H -> R
    fatality_H = simple_function(params['fatality_H']) * np.ones(num_counties) * params['fatality_max']

    # Set parameters
    n = params['n'][counties]
    rho = params['rho'][counties]
    initial_deaths = params['death_data'][counties, params['begin_cases']]
    initial_cases = params['case_data'][counties, params['begin_cases']]

    if len(params['ratio_A']) == 1:
        x_0 = np.reshape(params['initial_condition'], (1,5))
        if params['include_cases']:
            tau = simple_function(params['tau'])
        ##  Beta rates  ##
        # I -> R
        beta_I = posynomial_function(params['mobility_data'][counties], params['beta_I_coeffs'], params['beta_I_bias'],
                                     params['n'][counties]) * params['beta_max']
        # E -> I
        # beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E']), beta_I)
        # A -> I
        beta_A = np.einsum('i,ij->ij', simple_function(params['ratio_A']), beta_I)
    else:
        x_0 = params['initial_condition'][counties]
        if params['include_cases']:
            tau = simple_function(params['tau'][counties])
        ##  Beta rates  ##
        # I -> R
        beta_I = posynomial_function(params['mobility_data'][counties], params['beta_I_coeffs'][counties],
                              params['beta_I_bias'][counties], params['n'][counties]) * params['beta_max']
        # E -> I
        # beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E'][counties]), beta_I)
        # A -> I
        beta_A = np.einsum('i,ij->ij', simple_function(params['ratio_A'][counties]), beta_I)

    # Define variables
    X_data = []

    # Run the dynamics and take observations
    for i in range(num_counties):
        E = [np.exp(x_0[i][0])]
        if params['include_cases']:
            I = [tau[i]*initial_cases[i]]
        else:
            I = [np.exp(x_0[i][1])]
        R = [initial_deaths[i]/params['fatality_max'] + np.exp(x_0[i][2])]  # Enforce R(0) >= D(0)/max_fatality_ratio
        D = [initial_deaths[i]]
        A = [np.exp(x_0[i][3])]
        H = [np.exp(x_0[i][4])]
        for t in range(T - 1):
            # Susceptible compartment
            S = 1 - E[-1] - A[-1] - I[-1] - H[-1] - R[-1] - D[-1]
            # Exposed compartment
            E.append((1 - rho_EI[i] - rho_EA[i]) * E[-1] + S * (beta_I[i, t] * I[-1] + beta_A[i, t] * A[-1]))
            # Asymptomatic compartment
            A.append((1 - rho_AR[i]) * A[-1] + rho_EA[i] * E[-2])
            # Infected compartment
            I.append((1 - rho_IR[i] - rho_IH[i]) * I[-1] + rho_EI[i] * E[-2])
            # Hospitalized compartment
            H.append((1 - rho_HR[i]) * H[-1] + rho_IH[i] * I[-2])
            # Recovered compartment
            R.append(R[-1] + rho_IR[i] * I[-2] + rho_AR[i] * A[-2] + (1 - fatality_H[i]) * rho_HR[i] * H[-2])
            # Dead compartment
            D.append(D[-1] + fatality_H[i] * rho_HR[i] * H[-2])
        if return_all:
            X_data.append(output(E, n[i]))
            X_data.append(output(A, n[i]))
            X_data.append(output(H, n[i]))
            X_data.append(output(R, n[i]))
        # Output the number of expected cases, i.e., P(record a case) = tau, so E[# cases] = tau*I.
        if params['include_cases']:
            X_data.append([tau[i] * val for val in output(I, n[i])])
        X_data.append(output(D, n[i]))

    return np.array(X_data)


# Changes the categories for output. Makes them increase by increments of 1/n
def output(x, n):
    return x  # [np.ceil(x_i*n)/n for x_i in x]