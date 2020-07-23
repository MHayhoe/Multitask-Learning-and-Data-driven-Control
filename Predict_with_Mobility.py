# Our scripts
import shared
import Initialization as Init
from Optimization import optimize_sgd
from SEIRD_clinical import make_data
from Import_Data import import_data

# For autograd
import autograd.numpy as np

# Helpers
import argparse
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from os.path import exists
from os import mkdir
import random as rd


# Set up all variables
def setup(num_counties=1, start_day=0, train_days=10, validation_days=10, state=None):
    # Set miscellaneous parameters
    gamma_death = 5  # treat loss for death prediction as gamma_death times more important
    num_categories = 6
    shared.consts['num_mob_components'] = num_categories
    shared.consts['begin_mob'] = start_day
    shared.consts['begin_cases'] = 25 + shared.consts['begin_mob']

    # Set maximum allowable values for some parameters
    shared.consts['beta_max'] = 0.5
    shared.consts['fatality_I_max'] = 0.1

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
        land_data, age_distribution_data, deaths_data, case_count_data, mobility_data = import_data()

    # Define constants
    counties = list(age_distribution_data.keys())
    if state:
        state_abbrevs = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
                         "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                         "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                         "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                         "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        # Find the abbreviation, based on state number
        state = state_abbrevs[state]

        # Pick the counties from this state that have some deaths during training period
        counties = [c for c in counties if c.startswith(state + '-') and
                                           deaths_data[c][-(validation_days - train_days)] >= 5]
        #            deaths_data[c][-(validation_days-train_days)] / np.sum(age_distribution_data[c]) > 1e-5]
        counties.append(state)
        num_counties = len(counties)
    elif num_counties == 4:
        counties = ['WY-Carbon County', 'WY-Lincoln County', 'WY-Fremont County', 'WY']
        # counties = ['US', 'AK', 'MT', 'WY']
        # counties = ['US', 'PA', 'SD', 'CO']
        # counties = ['NH-Strafford County','MI-Midland County','NE-Douglas County','PA-Philadelphia County']
    elif num_counties == 52:
        counties = ['US', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    elif num_counties > 0:
        counties = rd.sample(counties, num_counties)
    else:
        num_counties = len(counties)
    num_dates = np.shape(mobility_data[counties[0]])[0]
    num_nyt_dates = np.shape(deaths_data['US'])[0]
    num_age_categories = 3  # 0-24, 25-64, 65+
    # Age distribution: <5, 5-9, 10-14, 15-17, 18-19, 20, 21, 22-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59,
    #                   60-61, 62-64, 65-66, 67-69, 70-74, 75-79, 80-84, >85.

    # Mobility categories: Retail & recreation, Grocery & pharmacy, Parks, Transit stations, Workplaces, Residential
    county_names = []
    mob_data = np.zeros((num_counties, num_dates - shared.consts['begin_mob'], num_categories))
    n = np.zeros(num_counties)
    age_data = np.zeros((num_counties, num_age_categories))
    death_data = np.zeros((num_counties, num_nyt_dates))
    case_data = np.zeros((num_counties, num_nyt_dates))
    rho = np.zeros(num_counties)

    i = 0
    for c in counties:
        #### Import data
        # County name
        county_names.append(c)
        # Total Population
        n[i] = np.sum(age_distribution_data[c])
        # Population density parameter
        rho[i] = Init.calculate_rho(n[i], land_data[c])
        # Ages 0-24
        age_data[i,0] = (np.sum(age_distribution_data[c][0:8]) + np.sum(age_distribution_data[c][23:31])) / n[i]
        # Ages 25-64
        age_data[i,1] = (np.sum(age_distribution_data[c][8:17]) + np.sum(age_distribution_data[c][31:40])) / n[i]
        # Ages 65+
        age_data[i, 2] = (np.sum(age_distribution_data[c][17:23]) + np.sum(age_distribution_data[c][40:46])) / n[i]
        # Mobility data, via Google's Global Mobility Report
        mob_data[i, :, :] = Init.pca_reduce(mobility_data[c][shared.consts['begin_mob']:,:] / 100)  # Standardize to be in [-1,inf]
        # Deaths from NYT.
        death_data[i, :] = Init.smooth(deaths_data[c] / n[i], filter_width=7, length=num_nyt_dates)  # Standardize to be in [0,1]
        # Case counts from NYT
        case_data[i, :] = Init.smooth(case_count_data[c] / n[i], filter_width=7, length=num_nyt_dates)  # Standardize to be in [0,1]

        i += 1

    # Set all values in the shared dictionary
    shared.consts['gamma_death'] = np.tile([1, gamma_death], num_counties)
    shared.consts['T'] = train_days
    shared.consts['n'] = n
    shared.consts['age_dist'] = age_data
    shared.consts['rho'] = rho
    shared.consts['mobility_data'] = mob_data
    shared.consts['death_data'] = death_data
    shared.consts['case_data'] = case_data
    shared.consts['county_names'] = county_names


# Plot the predicted and true trajectories (possibly real data)
def plot_prediction(params, length):
    # Make a folder to save plots
    time_dir = 'Plots/' + datetime.now().strftime("%Y-%b-%d-%H-%M-%S") + '/'
    if not exists(time_dir):
        mkdir(time_dir)

    # Save data to a csv
    # Init.params_to_pd(params, time_dir)

    consts = deepcopy(shared.consts)
    consts['T'] = length
    num_counties = len(shared.consts['n'])
    num_compartments = 1
    num_real_compartments = 1
    plot_split = int(np.ceil(np.sqrt(num_counties)))
    dates = np.arange(validation_days)

    # Save parameters
    with open(time_dir + 'opt_params.pickle', 'wb') as handle:
        pickle.dump(params, handle, protocol=4)

    with open(time_dir + 'consts.pickle', 'wb') as handle:
        pickle.dump(consts, handle, protocol=4)

    # Get real data
    X = Init.get_real_data(length)
    real_X = np.asarray(X).T

    # Get predictions
    X_est = []
    for c in range(num_counties):
        X_est.append(make_data(params[c], consts, counties=[c], return_all=False))
    est_X = np.reshape(X_est, (num_counties*num_compartments, length)).T
    # conf_intervals = confidence_intervals(params, length)

    for i in range(num_counties):
        plt.subplot(plot_split, plot_split, i + 1)
        plt.plot(real_X[:, i * num_real_compartments:(i + 1) * num_real_compartments])
        plt.plot(est_X[:, i * num_compartments:(i + 1) * num_compartments], '--')
        ylims = plt.gca().get_ylim()
        plt.fill_between(dates, ylims[0], ylims[1], where=dates > shared.consts['T'], facecolor='red', alpha=0.2)
        plt.title('{} ({:.0f})'.format(consts['county_names'][i], consts['n'][i]))

        fig = plt.figure()
        plt.plot(real_X[:, i * num_real_compartments:(i + 1) * num_real_compartments])
        plt.plot(est_X[:, i * num_compartments:(i + 1) * num_compartments], '--')
        ylims = plt.gca().get_ylim()
        plt.fill_between(dates, ylims[0], ylims[1], where=dates > shared.consts['T'], facecolor='red', alpha=0.2)
        plt.legend(['Recorded Deaths', 'Predicted Deaths'],loc='upper left')
        plt.title('{} ({:.0f})'.format(consts['county_names'][i], consts['n'][i]))
        plt.savefig(time_dir + consts['county_names'][i] + '.png', format='png')
        plt.close(fig)
    plt.tight_layout()
    plt.savefig(time_dir + 'All_counties.png', format='png')

    # Plot for whole state
    fig = plt.figure()
    plt.plot(Init.get_real_data(length,-1)*consts['n'][-1])
    plt.plot(np.einsum('ij,j->i',est_X[:,:-1],consts['n'][:-1]), '--')
    ylims = plt.gca().get_ylim()
    plt.fill_between(dates, ylims[0], ylims[1], where=dates > shared.consts['T'], facecolor='red', alpha=0.2)
    plt.legend(['Recorded Deaths', 'Predicted Deaths'], loc='upper left')
    plt.title('{} Aggregate'.format(consts['county_names'][-1]))
    plt.savefig(time_dir + 'Statewide.png', format='png')
    plt.close(fig)

    plt.show()


if __name__ == '__main__':
    # For parsing command-line arguments
    parser = argparse.ArgumentParser(description='Train the SEIRD mobility model at the state level.')
    parser.add_argument('--state', dest='state', type=int, default=38)
    args = parser.parse_args()

    # Define all values
    shared.real_data = True
    num_counties = 19  # use all states
    train_days = 30
    validation_days = train_days + 14
    start_day = 134 - validation_days
    num_batches = 1
    num_trials = 8

    setup(num_counties=num_counties, start_day=start_day, train_days=train_days, validation_days=validation_days, state=args.state)
    num_counties = len(shared.consts['n'])

    optimized_params = optimize_sgd(num_epochs=1000, num_batches=num_batches, num_trials=num_trials, step_size=0.01,
                                    show_plots=False)
    #with open('Plots/2020-Jul-20-17-03-54/opt_params.pickle','rb') as handle:
    #    optimized_params = pickle.load(handle)
    print(optimized_params)
    plot_prediction(optimized_params, validation_days)
