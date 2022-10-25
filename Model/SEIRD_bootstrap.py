import autograd.numpy as np
from Helper import sigmoid, sig_function


# Sample points at random (with replacement) for use in a bootstrapping procedure
def bootstrap_data(consts, data, num_data_points, counties):
    inds = np.random.choice(consts['T'] - 3, num_data_points, replace=True)

    # Death data, i.e., outputs
    X = {'mobility_data': consts['mobility_data'][counties, inds],
         'initial_condition': np.vstack((data[counties, inds], consts['death_data'][counties, inds]))}
    y = consts['death_data'][counties, inds + 3]

    return X, y


def do_bootstrap(params, data, consts, T=-1, counties=[]):
    # The simple function we use here is sigmoid
    simple_function = sigmoid

    # Get county information
    if counties:
        num_counties = len(counties)
    else:
        num_counties = np.shape(consts['n'])[0]
        counties = range(num_counties)

    # Get parameters
    if T == -1:
        T = consts['T']

    # Parametric function we're using is sigmoid(z) for scalar values, sigmoid(w'*x + b) for mobility data x
    ##  Rho rates  ##
    # E -> I
    rho_EI = simple_function(params['rho_EI_coeffs']) * np.ones(num_counties)
    # I -> R
    rho_IR = simple_function(params['rho_IR_coeffs']) * np.ones(num_counties)

    ##  Fatality ratios  ##
    # Ratio of I -> D, vs I -> R
    fatality_I = simple_function(params['fatality_I']) * np.ones(num_counties) * consts['fatality_I_max']

    # Set parameters
    n = consts['n'][counties]
    rho = consts['rho'][counties]

    if len(params['ratio_E']) == 1:
        x_0 = data['initial_condition']
        ##  Beta rates  ##
        # I -> R
        beta_I = sig_function(data['mobility_data'][counties], params['beta_I_coeffs'], params['beta_I_bias']) \
                 * consts['beta_max']
        # E -> I
        beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E']), beta_I)
    else:
        x_0 = data['initial_condition'][counties]
        ##  Beta rates  ##
        # I -> R
        beta_I = sig_function(consts['mobility_data'][counties], params['beta_I_coeffs'][counties],
                              params['beta_I_bias'][counties]) * consts['beta_max']
        # E -> I
        beta_E = np.einsum('i,ij->ij', simple_function(params['ratio_E'][counties]), beta_I)

    # Define variables
    X_data = []

    # Evolve the dynamics enough for the bootstrap
    for i in range(num_counties):
        D = []
        for t in range(T - 1):
            E_init = np.exp(x_0[i][0, t])
            I_init = np.exp(x_0[i][1, t])
            R_init = np.exp(x_0[i][2, t])
            D_init = np.exp(x_0[i][3, t])
            S_init = 1 - E_init - I_init - R_init - D_init
            # Dead compartment
            D.append(D_init + fatality_I[i] * rho_IR[i] *
                     ((1 + (2 - rho_IR[i]) * (1 - rho_IR[i]) + rho_EI[i] * rho[i] * S_init * beta_I[i,t]) * I_init +
                     rho_EI[i] * (3 - rho_IR[i] - rho_EI[i] + rho[i] * S_init * beta_E[i,t]) * E_init))
        X_data.append(output(D, n[i]))

    return np.array(X_data)


# Changes the categories for output. Makes them increase by increments of 1/n
def output(x, n):
    return x  # [np.ceil(x_i*n)/n for x_i in x]
