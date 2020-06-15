import numpy as np
import matplotlib.pyplot as plt


def make_data(true_params, consts, do_plot=False):
    # Get county information
    num_counties = np.shape(consts['n'])[0]
    i = 0

    # Get constants
    T = consts['T']
    n = consts['n']
    rho = consts['rho']
    rho_CI = consts['rho_CI']
    rho_IR = consts['rho_IR']

    # Define unknown parameters
    c_0 = consts['c_0']
    tau = consts['tau']
    beta_C = true_params['beta']
    beta_I = true_params['beta']

    # Define variables
    y = np.zeros((num_counties, T))
    X = np.zeros((3, consts['T']))
    X_data = []

    # Run the dynamics and take observations
    for i in range(num_counties):
        X[0, 0] = c_0[i]
        A = np.array([[1 + rho[i] * beta_C[i] - rho_CI[i], rho[i] * beta_I[i], 0],
                      [rho_CI[i], 1 - rho_IR[i], 0],
                      [0, rho_IR[i], 1]])
        for t in range(T - 1):
            y[i, t + 1] = np.random.binomial(n[i] * X[1, t], tau[i])
            X[:, t + 1] = A @ X[:, t]
        X_data.append(X)

    if do_plot:
        plot_data(X, y / n)

    return y, X_data


# Plot trajectory and observations
def plot_data(X, y):
    plt.plot(X[0,:], 'b')
    plt.plot(X[1,:], 'r')
    plt.plot(X[2,:], 'g')
    plt.plot(y,'k')
    plt.legend(['C','I','R','y'])
    plt.show()
