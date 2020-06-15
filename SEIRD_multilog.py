import autograd.numpy as np
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
    c_0 = consts['c_0']
    tau = consts['tau']

    # Model we're using is: beta = sigmoid(w'*x + b)
    beta = sigmoid(np.sum(consts['mobility_data'] * np.reshape(true_params['beta_coeffs'],(num_counties,1,6)), axis=2) + np.reshape(true_params['beta_bias'],(num_counties,1)))
    beta_C = beta
    beta_I = beta

    # Define variables
    y = np.zeros((num_counties, T))
    X_data = []

    # Run the dynamics and take observations
    for i in counties:
        X = np.zeros((3, T))
        X[0,0] = c_0[i]
        for t in range(T - 1):
            y[i, t + 1] = np.random.binomial(n[i] * X[1, t], tau[i])

            # Exposed compartment
            X[0, t + 1] = (1 - rho_CI[i]) * X[0,t] + rho[i] * (1 - X[0,t] - X[1,t] - X[2,t]) * (beta_C[i,t] * X[0,t] + beta_I[i,t] * X[1,t])
            # Infected compartment
            X[1, t + 1] = (1 - rho_IR[i]) * X[1,t] + rho_CI[i] * X[0,t]
            # Recovered compartment
            X[2, t + 1] = X[2,t] + rho_IR[i] * X[1,t]
        X_data.append(X[2,:])

    return y, X_data


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Soft-max function
def soft_max(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x,axis=1,keepdims=True)


# Scaled soft-max, so the max index has value 1
def soft_max_scaled(x):
    num_counties, _ = np.shape(x)
    val = soft_max(x)
    max_val = np.reshape(np.max(val,axis=1),(num_counties,1))
    return val / max_val


# Plot trajectory and observations
def plot_data(X, y):
    plt.plot(X[0,:], 'b')
    plt.plot(X[1,:], 'r')
    plt.plot(X[2,:], 'g')
    plt.plot(y,'k')
    plt.legend(['C','I','R','y'])
    plt.show()
