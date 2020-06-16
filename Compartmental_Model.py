# Our scripts
import shared
import Initialization as Init
from Optimization import optimize_parallel, optimize, optimize_sgd, optimize_parallel_counties
from Objectives import prediction_county
from SEIRD_clinical import make_data
from SEIRD_Markov import make_data_Markov
from Import_Data import import_data
from Helper import inverse_sigmoid

# For autograd
import autograd.numpy as np

# Helpers
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from os.path import exists
from os import mkdir
import random as rd


# rd.seed(0)


# Set up all variables
def setup(num_counties=1, start_day=0, train_days=10):
    # Set miscellaneous parameters
    gamma_death = 5  # treat loss for death prediction as gamma_death times more important
    shared.begin['mob'] = start_day
    shared.begin['cases'] = 25 + shared.begin['mob']

    # Import mobility data
    if exists('Mobility_US.pickle') and exists('Age_Distribution_US.pickle') and exists('Deaths_US.pickle') \
            and exists('Case_Counts_US.pickle') and exists('Land_Area_US.pickle'):
        with open('Mobility_US.pickle', 'rb') as handle:
            mobility_data = pickle.load(handle)
        with open('Age_Distribution_US.pickle', 'rb') as handle:
            age_distribution_data = pickle.load(handle)
        with open('Deaths_US.pickle', 'rb') as handle:
            deaths_data = pickle.load(handle)
        with open('Case_Counts_US.pickle', 'rb') as handle:
            case_count_data = pickle.load(handle)
        with open('Land_Area_US.pickle', 'rb') as handle:
            land_data = pickle.load(handle)
    else:
        land_data, age_distribution_data, deaths_data, case_count_data, mobility_data = import_data('Global_Mobility_Report.csv')

    # Define constants
    counties = age_distribution_data.keys()
    if num_counties > 0:
        counties = rd.sample(counties,num_counties)
    # counties = ['NH-Strafford County','MI-Midland County','NE-Douglas County','PA-Philadelphia County']
    num_dates, num_categories = np.shape(mobility_data[counties[0]])
    num_nyt_dates = np.shape(deaths_data[counties[0]])[0]
    num_age_categories = 3 # 0-24, 25-64, 65+
    # Age distribution: <5, 5-9, 10-14, 15-17, 18-19, 20, 21, 22-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59,
    #                   60-61, 62-64, 65-66, 67-69, 70-74, 75-79, 80-84, >85.
    i = 0

    # Mobility categories: Retail & recreation, Grocery & pharmacy, Parks, Transit stations, Workplaces, Residential
    county_names = []
    mob_data = np.zeros((num_counties, num_dates - shared.begin['mob'], num_categories))
    n = np.zeros(num_counties)
    age_data = np.zeros((num_counties, num_age_categories))
    death_data = np.zeros((num_counties, num_nyt_dates))
    case_data = np.zeros((num_counties, num_nyt_dates))
    rho = np.zeros(num_counties)
    c_0 = np.zeros(num_counties)
    tau = np.zeros(num_counties)

    # Parametric function values
    rho_EI_bias = np.zeros(num_counties)
    rho_EI_coeffs = np.zeros((num_counties,3))
    rho_IH_bias = np.zeros(num_counties)
    rho_IH_coeffs = np.zeros((num_counties, 3))
    rho_IR_bias = np.zeros(num_counties)
    rho_IR_coeffs = np.zeros((num_counties,3))
    rho_HR_bias = np.zeros(num_counties)
    rho_HR_coeffs = np.zeros((num_counties, 3))
    rho_ID_bias = np.zeros(num_counties)
    rho_ID_coeffs = np.zeros((num_counties, 3))
    rho_HD_bias = np.zeros(num_counties)
    rho_HD_coeffs = np.zeros((num_counties, 3))
    beta_E_bias = np.zeros(num_counties)
    beta_E_coeffs = np.zeros((num_counties,6))
    beta_I_bias = np.zeros(num_counties)
    beta_I_coeffs = np.zeros((num_counties,6))
    fatal_I = np.zeros(num_counties)
    fatal_H = np.zeros(num_counties)
    ratio_inf_E = np.zeros(num_counties)

    for c in counties:
        #### Import data
        # County name
        county_names.append(c)
        # Total Population
        n[i] = np.sum(age_distribution_data[c])
        # Ages 0-24
        age_data[i,0] = (np.sum(age_distribution_data[c][0:8]) + np.sum(age_distribution_data[c][23:31])) / n[i]
        # Ages 25-64
        age_data[i,1] = (np.sum(age_distribution_data[c][8:17]) + np.sum(age_distribution_data[c][31:40])) / n[i]
        # Ages 65+
        age_data[i, 2] = (np.sum(age_distribution_data[c][17:23]) + np.sum(age_distribution_data[c][40:46])) / n[i]
        # Mobility data, via Google's Global Mobility Report
        mob_data[i, :, :] = mobility_data[c][shared.begin['mob']:,:] / 100  # Standardize to be in [-1,1]
        # Deaths from NYT. If we have no data, fill zeros
        if (deaths_data[c] == 0).all():
            death_data[i, :] = np.zeros(num_nyt_dates)
        else:
            death_data[i, :] = np.squeeze(deaths_data[c] / n[i])  # Standardize to be in [0,1]
        # Case counts from NYT. If we have no data, fill zeros
        if (case_count_data[c] == 0).all():
            case_data[i, :] = np.zeros(num_nyt_dates)
        else:
            case_data[i, :] = np.squeeze(case_count_data[c] / n[i])  # Standardize to be in [0,1]

        # Initialize parameters
        rho[i] = Init.calculate_rho(n[i], land_data[c])
        c_0[i] = rd.random()*0.01 + 0.005
        tau[i] = 0.3
        rho_EI_bias[i] = rd.random() * 0.6 - 0.3
        rho_EI_coeffs[i, :] = [(rd.random() * -4 - 1) for i in range(3)]
        rho_IH_bias[i] = rd.random() * 0.6 - 0.3
        rho_IH_coeffs[i, :] = [(rd.random() * -4 - 1) for i in range(3)]
        rho_IR_bias[i] = rd.random() * 0.6 - 0.3
        rho_IR_coeffs[i, :] = [(rd.random() * -4 - 1) for i in range(3)]
        rho_HR_bias[i] = rd.random() * 0.6 - 0.3
        rho_HR_coeffs[i, :] = [(rd.random() * -4 - 1) for i in range(3)]
        rho_ID_bias[i] = rd.random() * 0.6 - 0.3
        rho_ID_coeffs[i, :] = [(rd.random() * -4 - 1) for i in range(3)]
        rho_HD_bias[i] = rd.random() * 0.6 - 0.3
        rho_HD_coeffs[i, :] = [(rd.random() * -4 - 1) for i in range(3)]
        beta_E_bias[i] = rd.random() * 0.9 + 0.1
        beta_E_coeffs[i,:] = [(rd.random() * 0.9 + 0.1) for i in range(6)]
        beta_I_bias[i] = rd.random() * 0.9 + 0.1
        beta_I_coeffs[i, :] = [(rd.random() * 0.9 + 0.1) for i in range(6)]
        fatal_I[i] = inverse_sigmoid(0.01)
        fatal_H[i] = inverse_sigmoid(0.01)
        ratio_inf_E[i] = inverse_sigmoid(0.1)
        i += 1

    # Set all values in the shared dictionaries
    shared.consts['gamma_death'] = np.tile([1, gamma_death], num_counties)
    shared.consts['T'] = train_days
    shared.consts['n'] = n
    shared.consts['age_dist'] = age_data
    shared.consts['rho'] = rho
    shared.consts['mobility_data'] = mob_data
    shared.consts['death_data'] = death_data
    shared.consts['case_data'] = case_data
    shared.consts['county_names'] = county_names
    shared.consts['tau'] = tau

    shared.true_params['c_0'] = c_0
    shared.true_params['rho_EI_bias'] = rd.random() * 0.6 - 0.3  # rho_EI_bias
    shared.true_params['rho_EI_coeffs'] = [(rd.random() * -4 - 1) for i in range(3)]  # rho_EI_coeffs
    shared.true_params['rho_IH_bias'] = rd.random() * 0.6 - 0.3  # rho_IH_bias
    shared.true_params['rho_IH_coeffs'] = [(rd.random() * -4 - 1) for i in range(3)]  # rho_IH_coeffs
    shared.true_params['rho_IR_bias'] = rd.random() * 0.6 - 0.3  # rho_IR_bias
    shared.true_params['rho_IR_coeffs'] = [(rd.random() * -4 - 1) for i in range(3)]  # rho_IR_coeffs
    shared.true_params['rho_HR_bias'] = rd.random() * 0.6 - 0.3  # rho_HR_bias
    shared.true_params['rho_HR_coeffs'] = [(rd.random() * -4 - 1) for i in range(3)]  # rho_HR_coeffs
    shared.true_params['rho_ID_bias'] = rd.random() * 0.6 - 0.3  # rho_ID_bias
    shared.true_params['rho_ID_coeffs'] = [(rd.random() * -4 - 1) for i in range(3)]  # rho_ID_coeffs
    shared.true_params['rho_HD_bias'] = rd.random() * 0.6 - 0.3  # rho_HD_bias
    shared.true_params['rho_HD_coeffs'] = [(rd.random() * -4 - 1) for i in range(3)]  # rho_HD_coeffs
    shared.true_params['beta_E_bias'] = np.reshape(beta_E_bias,(num_counties,1))
    shared.true_params['beta_E_coeffs'] = np.reshape(beta_E_coeffs,(num_counties,1,6))
    shared.true_params['beta_I_bias'] = np.reshape(beta_I_bias,(num_counties,1))
    shared.true_params['beta_I_coeffs'] = np.reshape(beta_I_coeffs,(num_counties,1,6))
    shared.true_params['ratio_E'] = ratio_inf_E
    shared.consts['fatality_H'] = fatal_H


# Plot the predicted and true trajectories (possibly real data)
def plot_prediction(params, length):
    # Make a folder to save plots
    time_dir = 'Plots/' + datetime.now().strftime("%Y-%b-%d-%H-%M-%S") + '/'
    if not exists(time_dir):
        mkdir(time_dir)

    # Save parameters
    with open(time_dir + 'opt_params.pickle', 'wb') as handle:
        pickle.dump(optimized_params, handle, protocol=4)

    # Save data to a csv
    Init.params_to_pd(params, time_dir)

    consts = deepcopy(shared.consts)
    consts['T'] = length
    num_counties = len(shared.consts['n'])
    num_compartments = 2
    num_real_compartments = 2
    plot_split = int(np.ceil(np.sqrt(num_counties)))

    X = Init.get_real_data(length)
    real_X = np.asarray(X)

    X_est = []
    for c in range(num_counties):
        X_est.append(make_data(params[c], consts, counties=[c]))
    est_X = np.reshape(X_est, (num_counties*num_compartments, length))
    # conf_intervals = confidence_intervals(params, length)

    for i in range(num_counties):
        plt.subplot(plot_split, plot_split, i + 1)
        if shared.real_data:
            plt.plot(real_X[i * num_real_compartments:(i + 1) * num_real_compartments, :].T)
        else:
            plt.plot(real_X[i * num_compartments:(i + 1) * num_compartments, :].T)
        plt.plot(est_X[i * num_compartments:(i + 1) * num_compartments, :].T, '--')
        # plt.legend(['Exp.', 'Inf.', 'Hosp.', 'Rec.', 'Dead'])
        plt.title('{} ({:.0f})'.format(consts['county_names'][i], consts['n'][i]))

        fig = plt.figure()
        if shared.real_data:
            plt.plot(real_X[i * num_real_compartments:(i + 1) * num_real_compartments, :].T)
            plt.plot(est_X[i * num_compartments:(i + 1) * num_compartments, :].T, '--')
            # plt.fill_between(range(shared.consts['T'], length), conf_intervals[i * num_compartments,0,shared.consts['T']:length],
            #                  conf_intervals[i * num_compartments,1,shared.consts['T']:length], facecolor='blue', alpha=0.3)
            # plt.fill_between(range(shared.consts['T'], length), conf_intervals[i * num_compartments + 1,0, shared.consts['T']:length],
            #                  conf_intervals[i * num_compartments + 1,1, shared.consts['T']:length], facecolor='orange', alpha=0.3)
            plt.legend(['Cases','Deaths', 'Inf.', 'Dead'])
            # plt.legend(['Deaths', 'Exp.', 'Inf.', 'Rec.', 'Dead'])
        else:
            plt.plot(real_X[i * num_compartments:(i + 1) * num_compartments, :].T)
            plt.plot(est_X[i * num_compartments:(i + 1) * num_compartments, :].T, '--')
            # plt.legend(['Exp.', 'Inf.', 'Hosp.', 'Rec.', 'Dead'])
            plt.legend(['Exp.', 'Inf.', 'Rec.', 'Dead'])
        plt.title('{} ({:.0f})'.format(consts['county_names'][i], consts['n'][i]))
        plt.savefig(time_dir + consts['county_names'][i] + '.png', format='png')
        plt.close(fig)
    plt.tight_layout()
    plt.savefig(time_dir + 'All_counties.png', format='png')
    plt.show()


# Generate confidence intervals based on estimates
def confidence_intervals(params, validation_days, num_trials=100, confidence=0.95):
    X = []
    # Update length of time
    oldT = shared.consts['T']
    shared.consts['T'] = validation_days

    # Run the trials
    for i in range(num_trials):
        X.append(make_data_Markov(params, shared.consts))

    X = np.asarray(X)
    num_trajectories = np.shape(X)[1]

    # Find correct index for what to discard
    ci_index = int(np.round(num_trials * (1 - confidence) / 2))
    ci = np.zeros((num_trajectories,2, shared.consts['T']))

    for d in range(num_trajectories):
        x = np.sort(X[:,d,:], axis=0)
        ci[d,0,:] = np.min(x[ci_index:-ci_index,:], axis=0)
        ci[d,1,:] = np.max(x[ci_index:-ci_index,:], axis=0)

    # Change back length of time
    shared.consts['T'] = oldT

    return ci


if __name__ == '__main__':
    # Define all values
    shared.real_data = True
    num_counties = -1  # use all counties
    start_day = 53
    train_days = 60
    validation_days = train_days + 10
    num_batches = 25
    num_trials = 8

    setup(num_counties=num_counties, start_day=start_day, train_days=train_days)

    optimized_params = optimize_sgd(num_epochs=1000, num_batches=num_batches, num_trials=num_trials, step_size=0.01)
    print(optimized_params)
    plot_prediction(optimized_params, validation_days)
