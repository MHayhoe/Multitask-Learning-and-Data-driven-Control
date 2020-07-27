import pandas as pd
import pickle
import numpy as np
import us

from Initialization import get_real_data
from SEIRD_clinical import make_data
import shared


# Create data based on some saved parameters, and save it as a dataframe
def save_dataframe(dir):
    # Import real data
    df_nyt = pd.read_csv('Data/nytimes_infections.csv', encoding="ISO-8859-1", low_memory=False)
    dates = df_nyt.date.unique()
    df_deaths = df_nyt.filter(items=['state', 'date', 'deaths'])
    state_abbrevs = [x.abbr for x in us.states.STATES]
    state_names = [x.name for x in us.states.STATES]
    df_states = pd.DataFrame(columns=['state','abbr'] + ['deaths-' + d for d in dates])

    for (name, abbr) in zip(state_names, state_abbrevs):
        region_deaths = df_deaths[df_deaths['state'] == name]
        df_states.loc[len(df_states)] = [name, abbr] + [np.sum(region_deaths[region_deaths['date'] == d].deaths.to_numpy().T) for d in dates]

    # Make predictions
    with open(dir + '/opt_params.pickle', 'rb') as handle:
        optimized_params = pickle.load(handle)
    with open(dir + '/consts.pickle', 'rb') as handle:
        shared.consts = pickle.load(handle)

    length = shared.consts['T']

    X = get_real_data(length)
    real_X = np.asarray(X).T

    num_counties = len(shared.consts['n'])
    X_est = []
    for c in range(num_counties):
        X_est.append(make_data(optimized_params[c], shared.consts, counties=[c], return_all=False))
    est_X = np.reshape(X_est, (num_counties, length)).T
