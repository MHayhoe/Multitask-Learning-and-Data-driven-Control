import numpy as np
import matplotlib.pyplot as plt


def make_data(true_params, consts, county=-1):
    # Get county information
    num_counties = np.shape(consts['n'])[0]
    if county == -1:
        counties = range(num_counties)
    else:
        counties = [county]
    i = 0

    # Get constants
    T = consts['T']
    n = consts['n']
    rho = consts['rho']
    rho_CI = consts['rho_CI']
    rho_IR = consts['rho_IR']

    # Define unknown parameters
    c_0 = true_params['c_0']
    tau = consts['tau']
    beta_C = true_params['beta']
    beta_I = true_params['beta']

    # Define variables
    y = np.zeros((num_counties, T))
    X_data = []

    # Run the dynamics and take observations
    for i in counties:
        #X[0,0] = 1 - c_0[i]
        X = np.zeros((3, consts['T']))
        X[0,0] = c_0[i]
        for t in range(T - 1):
            y[i, t + 1] = np.random.binomial(n[i] * X[1, t], tau[i])

            # Exposed compartment
            X[0, t + 1] = (1 - rho_CI[i]) * X[0,t] + rho[i] * (1 - X[0,t] - X[1,t] - X[2,t]) * (beta_C[i] * X[0,t] + beta_I[i] * X[1,t])
            # Infected compartment
            X[1, t + 1] = (1 - rho_IR[i]) * X[1,t] + rho_CI[i] * X[0,t]
            # Recovered compartment
            X[2, t + 1] = X[2,t] + rho_IR[i] * X[1,t]
        X_data.append(X)

    return y, X_data


# Plot trajectory and observations
def plot_data(X, y):
    plt.plot(X[0,:], 'b')
    plt.plot(X[1,:], 'r')
    plt.plot(X[2,:], 'g')
    plt.plot(y,'k')
    plt.legend(['C','I','R','y'])
    plt.show()
