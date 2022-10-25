# Our scripts
import shared
import re
import Initialization as Init
from Optimization import optimize_sgd
# from SEIRD_clinical import make_data
from SEAIHRD import make_data
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
    num_categories = 2
    do_PCA = True
    shared.consts['num_lag_days'] = 0
    shared.consts['include_cases'] = False
    shared.consts['num_mob_components'] = num_categories
    shared.consts['begin_mob'] = start_day
    shared.consts['begin_cases'] = 25 + shared.consts['begin_mob'] - shared.consts['num_lag_days']  # For Google data
    # shared.consts['begin_cases'] = shared.consts['begin_mob'] - 15 - shared.consts['num_lag_days']   # For SafeGraph data
    shared.consts['validation_days'] = validation_days

    # Set maximum allowable values for some parameters
    shared.consts['beta_max'] = 0.5
    shared.consts['fatality_max'] = 0.1

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
        # Pick the counties from this state that have some deaths during training period
        counties = [c for c in counties if c.startswith(state + '-') and
                                           len(deaths_data[c]) > validation_days - train_days and
                                           deaths_data[c][-(validation_days - train_days)] >= 5]
        #            deaths_data[c][-(validation_days-train_days)] / np.sum(age_distribution_data[c]) > 1e-5]
        counties.append(state)
        num_counties = len(counties)
    elif num_counties == 4:
        counties = ['WY-Carbon County', 'WY-Lincoln County', 'WY-Fremont County', 'WY']
        # counties = ['US', 'AK', 'MT', 'WY']
        # counties = ['US', 'PA', 'SD', 'CO']
        # counties = ['NH-Strafford County','MI-Midland County','NE-Douglas County','PA-Philadelphia County']
    elif num_counties == 10:
        counties = ['PA-Philadelphia County', 'PA-Montgomery County', 'PA-Bucks County', 'PA-Delaware County',
                    'PA-Chester County',  'DE-New Castle County', 'NJ-Camden County', 'NJ-Gloucester County',
                    'NJ-Burlington County', 'MD-Baltimore County']
    elif num_counties == 52:
        counties = ['US', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    elif num_counties > 0:
        counties = rd.sample(counties, num_counties)
    else:
        num_counties = len(counties)

    print('Learning based on data from {}'.format(counties))
    num_dates = np.shape(mobility_data[counties[0]])[0]
    # num_nyt_dates = np.shape(deaths_data['US'])[0]
    num_nyt_dates = np.shape(deaths_data['PA-Philadelphia County'])[0]
    num_age_categories = 3  # 0-24, 25-64, 65+
    # Age distribution: <5, 5-9, 10-14, 15-17, 18-19, 20, 21, 22-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59,
    #                   60-61, 62-64, 65-66, 67-69, 70-74, 75-79, 80-84, >85.

    # Mobility categories: Retail & recreation, Grocery & pharmacy, Parks, Transit stations, Workplaces, Residential
    county_names = []
    mob_data = np.zeros((num_counties, validation_days, num_categories))
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
        mob_data[i, :, :] = Init.pca_reduce(mobility_data[c][shared.consts['begin_mob']:shared.consts['begin_mob']+validation_days,:], do_PCA)  # Standardize to be in [0,+inf]
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
def plot_prediction(params, length, state=None):
    # Make a folder to save plots
    time_dir = 'Plots/' + datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
    if state:
        time_dir += '_' + state + '/'
    else:
        time_dir += '/'

    if not exists(time_dir):
        mkdir(time_dir)

    # Save data to a csv
    # Init.params_to_pd(params, time_dir)

    consts = deepcopy(shared.consts)
    consts['T'] = length
    #shared.consts['T'] = 60
    num_counties = len(shared.consts['n'])
    if consts['include_cases']:
        num_compartments = 2
        num_real_compartments = 2
    else:
        num_compartments = 1
        num_real_compartments = 1
    plot_split = int(np.ceil(np.sqrt(num_counties)))
    dates = np.arange(length)

    # Save parameters
    with open(time_dir + 'opt_params.pickle', 'wb') as handle:
        pickle.dump(params, handle, protocol=4)

    with open(time_dir + 'consts.pickle', 'wb') as handle:
        pickle.dump(shared.consts, handle, protocol=4)

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

        # Cumulative deaths plot
        fig = plt.figure()
        plt.plot(real_X[:, i * num_real_compartments:(i + 1) * num_real_compartments])
        plt.plot(est_X[:, i * num_compartments:(i + 1) * num_compartments], '--')
        ylims = plt.gca().get_ylim()
        plt.fill_between(dates, ylims[0], ylims[1], where=dates > shared.consts['T'], facecolor='red', alpha=0.2)
        plt.legend(['Recorded Deaths', 'Predicted Deaths'],loc='upper left')
        plt.title('{} ({:.0f})'.format(consts['county_names'][i], consts['n'][i]))
        plt.savefig(time_dir + consts['county_names'][i] + '.png', format='png')
        plt.close(fig)

        # Incident (daily) deaths plot
        fig = plt.figure()
        real_X_incident = real_X[:, i * num_real_compartments:(i + 1) * num_real_compartments]
        real_X_incident = [j-i for i, j in zip(real_X_incident[:-1], real_X_incident[1:])] # Init.smooth([j-i for i, j in zip(real_X_incident[:-1], real_X_incident[1:])], filter_width=7, length=consts['T'])
        plt.plot(real_X_incident)
        est_X_incident = est_X[:, i * num_compartments:(i + 1) * num_compartments]
        est_X_incident = [j-i for i, j in zip(est_X_incident[:-1], est_X_incident[1:])]
        plt.plot(est_X_incident)
        ylims = plt.gca().get_ylim()
        plt.fill_between(dates, ylims[0], ylims[1], where=dates > shared.consts['T'], facecolor='red', alpha=0.2)
        plt.legend(['Recorded Deaths', 'Predicted Deaths'],loc='upper left')
        plt.title('{} ({:.0f})'.format(consts['county_names'][i], consts['n'][i]))
        plt.savefig(time_dir + consts['county_names'][i] + '_incident.png', format='png')
        plt.close(fig)
    plt.tight_layout()
    plt.savefig(time_dir + 'All_counties.png', format='png')

    # Plot for whole state
    # fig = plt.figure()
    # plt.plot(Init.get_real_data(length,-1)*consts['n'][-1])
    # plt.plot(np.einsum('ij,j->i',est_X[:,:-1],consts['n'][:-1]), '--')
    # ylims = plt.gca().get_ylim()
    # plt.fill_between(dates, ylims[0], ylims[1], where=dates > shared.consts['T'], facecolor='red', alpha=0.2)
    # plt.legend(['Recorded Deaths', 'Predicted Deaths'], loc='upper left')
    # plt.title('{} Aggregate'.format(consts['county_names'][-1]))
    # plt.savefig(time_dir + 'Statewide.png', format='png')
    # plt.close(fig)

    plt.show()


def set_state(state_num):
    if state_num:
        state_abbrevs = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
                         "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                         "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                         "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                         "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        # Find the abbreviation, based on state number
        return state_abbrevs[state_num - 1]
    else:
        return None


# Prints a set of parameters to a csv file
def print_params_to_csv(to_print, file_name):
    with open(file_name, 'w') as f:
        for key in to_print.keys():
            item = to_print[key]
            if isinstance(item, dict):
                for kk in item.keys():
                    lst = item[kk].tolist()
                    if isinstance(lst, float):
                        f.write("%s,%s\n" % (kk, lst))
                    else:
                        f.write("%s,%s\n" % (kk, ','.join([re.sub(r"[\[\]]", '', str(x)) for x in lst])))
            else:
                if isinstance(item, float) or isinstance(item, int):
                    f.write("%s,%s\n" % (key, item))
                else:
                    f.write("%s,%s\n" % (key, ','.join([re.sub(r"[\[\]]", '', str(x)) for x in item])))


if __name__ == '__main__':
    # For parsing command-line arguments
    parser = argparse.ArgumentParser(description='Train the SEAIHRD mobility model at the state level.')
    parser.add_argument('--state', dest='state_num', type=int, default=None)
    args = parser.parse_args()
    state = set_state(args.state_num)

    # Define all values
    shared.real_data = True
    num_counties = 10
    train_days = 92
    validation_days = train_days + 30
    start_day = 289 - validation_days   # For Google: Day 320 is Dec 31, 2020. Day 320 - 30 = 290 is December 1, 2020, and Day 320 - 92 = 137 is July 1, 2020
    # start_day = 258 - validation_days   # For SafeGraph: Day 258 is Sept 20, 2020. Day 258 - 81 = 177 is July 1, 2020
    num_batches = 1
    num_trials = 8

    setup(num_counties=10, start_day=start_day, train_days=train_days, validation_days=validation_days, state=state)
    num_counties = len(shared.consts['n'])

    optimized_params = optimize_sgd(num_epochs=1000, num_batches=num_batches, num_trials=num_trials, step_size=0.005,
                                    show_plots=True)

    # with open('Plots/2020-Mar-15-16-31-15/opt_params.pickle','rb') as f:
    #     optimized_params = pickle.load(f)
    #
    # with open('Plots/2020-Mar-15-16-31-15/consts.pickle','rb') as f:
    #     shared.consts = pickle.load(f)

    print(optimized_params)
    # Save data to a CSV so we can import to Matlab
    # print_params_to_csv(shared.consts, 'posynomial_consts.csv')
    # print_params_to_csv(optimized_params, 'posynomial_params.csv')

    # Make plots
    plot_prediction(optimized_params, validation_days, state=state)
