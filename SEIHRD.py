import autograd.numpy as np
import shared


def make_data(true_params, consts, county=-1, return_all=False):
    # Combine the dictionaries
    params = {**true_params, **consts}

    # Get county information
    num_counties = np.shape(params['n'])[0]
    if county == -1:
        counties = range(num_counties)
    else:
        counties = [county]

    # Get parameters
    T = params['T']
    rho = params['rho']
    c_0 = params['c_0']

    # Parametric function we're using is sigmoid(w'*x + b)
    # S -> E, via E
    beta_E = sig_function(params['mobility_data'], params['beta_E_coeffs'], params['beta_E_bias'])
    # S -> E, via I
    beta_I = sig_function(params['mobility_data'], params['beta_I_coeffs'], params['beta_I_bias'])
    # E -> I
    rho_EI = sig_function(params['age_dist'], params['rho_EI_coeffs'], params['rho_EI_bias']) * np.ones(num_counties)
    # I -> H
    rho_IH = sig_function(params['age_dist'], params['rho_IH_coeffs'], params['rho_IH_bias']) * np.ones(num_counties)
    # I -> R
    rho_IR = sig_function(params['age_dist'], params['rho_IR_coeffs'], params['rho_IR_bias']) * np.ones(num_counties)
    # H -> R
    rho_HR = sig_function(params['age_dist'], params['rho_HR_coeffs'], params['rho_HR_bias']) * np.ones(num_counties)
    # I -> D
    rho_ID = sig_function(params['age_dist'], params['rho_ID_coeffs'], params['rho_ID_bias']) * np.ones(num_counties)
    # H -> D
    rho_HD = sig_function(params['age_dist'], params['rho_HD_coeffs'], params['rho_HD_bias']) * np.ones(num_counties)

    # Define variables
    y = np.zeros((num_counties, T))
    X_data = []

    # Run the dynamics and take observations
    for i in counties:
        E = [c_0[i]]
        I = [0]
        H = [0]
        R = [0]
        D = [0]
        for t in range(T - 1):
            # Susceptible compartment
            S = 1 - E[-1] - I[-1] - H[-1] - R[-1]
            # Exposed compartment
            E.append((1 - rho_EI[i]) * E[-1] + rho[i] * S * (beta_E[i,t] * E[-1] + beta_I[i,t] * I[-1]))
            # Infected compartment
            I.append((1 - rho_IR[i] - rho_IH[i] - rho_HD[i]) * I[-1] + rho_EI[i] * E[-2])
            # Hospitalized compartment
            H.append((1 - rho_HR[i] - rho_HD[i]) * H[-1] + rho_IH[i] * I[-2])
            # Recovered compartment
            R.append(R[-1] + rho_IR[i] * I[-2] + rho_HR[i] * H[-2])
            # Dead compartment
            D.append(D[-1] + rho_ID[i] * I[-2] + rho_HD[i] * H[-2])
        if return_all:
            X_data.append(E)
            X_data.append(H)
            X_data.append(R)
        X_data.append(I)
        X_data.append(D)

    return y, X_data


# Simple linear-sigmoidal function
def simple_function(w):
    return shared.max_rate * sigmoid(w)


# Parametric linear-sigmoidal function
def sig_function(x,w,b=0):
    shapes = np.shape(x)
    num_counties = shapes[0]
    num_vals = shapes[-1]
    if len(shapes) == 3:
        return shared.max_rate * sigmoid(np.sum(x * np.reshape(w,(num_counties,1,num_vals)), axis=2) + np.reshape(b,(num_counties,1)))
    else:
        return shared.max_rate * sigmoid(np.sum(x * np.array(w), axis=1) + np.array(b))


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Soft-max function
def soft_max(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x,axis=1,keepdims=True)
