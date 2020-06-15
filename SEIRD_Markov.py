import autograd.numpy as np
import shared
import matplotlib.pyplot as plt
from Helper import simple_function, sig_function
import numpy.random as rd
from scipy.stats import binom


def make_data_Markov(true_params, consts, county=-1, return_all=False):
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

    ##  Beta rates  ##
    # S -> E, via E
    # beta_E = sig_function(params['mobility_data'], params['beta_E_coeffs'], params['beta_E_bias'])
    # S -> E, via I
    beta_I = sig_function(params['mobility_data'], params['beta_I_coeffs'], params['beta_I_bias'])
    beta_E = (beta_I.T * simple_function(params['ratio_E'])).T  # equivalent to multiplying each row of beta_I by corresponding entry of ratio_E

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

    if return_all:
        plt.ion()
        plt.show()

    # Run the dynamics and take observations
    for i in counties:
        E = [np.exp(c_0[i][0])]
        I = [np.exp(c_0[i][1])]
        R = [np.exp(c_0[i][2])]
        D = [np.exp(c_0[i][3])]
        n = params['n'][i]

        for t in range(T - 1):
            # Susceptible compartment
            S = 1 - E[-1] - I[-1] - R[-1] - D[-1]

            # Exposed compartment
            E.append(E[-1] + num_random_transitions(1 / (beta_E[i,t] * E[-1] + beta_I[i,t] * I[-1]), S, n))

            # Infected compartment
            I.append(I[-1] + num_random_transitions(1 / rho_EI[i], E[-1], n))

            # Recovered compartment
            R.append(R[-1] + num_random_transitions(1 / ((1 - fatality_I[i]) * rho_IR[i]), I[-1], n))

            # Dead compartment
            D.append(D[-1] + num_random_transitions(1 / (fatality_I[i] * rho_IR[i]), I[-1], n))

            if return_all:
                plt.clf()
                plt.plot(E,label='Exp.')
                plt.plot(I,label='Inf.')
                plt.plot(R,label='Rec.')
                plt.plot(D,label='Dead')
                plt.legend()
                plt.draw()
                plt.pause(0.5)

        if return_all:
            X_data.append(E)
        X_data.append(I)
        if return_all:
            X_data.append(R)
        X_data.append(D)

    return X_data


# Returns probability of transitions
def num_random_transitions(lam, n_fraction, population_size):
    n = int(np.round(n_fraction * population_size))
    # num_transitions = [j for j in range(n + 1)]
    # transition_probabilities = [(1 - np.exp(-lam))**k * np.exp(-lam)**(n - k) for k in num_transitions]
    # return rd.choice(num_transitions, 1, p=transition_probabilities) / n
    if n > 0:
        return binom.rvs(n, 1 - np.exp(-lam)) / population_size
    else:
        return 0
