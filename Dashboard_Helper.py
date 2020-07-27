import pandas as pd
from urllib.request import urlopen
import json
import plotly.graph_objects as go
from os.path import exists
import pickle
import us
import numpy as np


# Global variables
state_names = [s.name for s in us.states.STATES]
state_abbrevs = [s.abbr for s in us.states.STATES]


# Imports the county JSON
def import_JSON():
    if exists('county_JSON.pickle'):
        with open('county_JSON.pickle', 'rb') as handle:
            counties = pickle.load(handle)
    else:
        with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            counties = json.load(response)
        with open('county_JSON.pickle', 'wb') as handle:
            pickle.dump(counties, handle, protocol=4)

    return counties


counties = import_JSON()


# Import county data
def import_county_data():
    properties = [x['properties'] for x in counties['features']]

    df = pd.DataFrame(properties)
    df.NAME = df.NAME + ' ' + df.LSAD
    df['FIPS'] = df.STATE + df.COUNTY
    df['STATE'] = df.apply(lambda x: us.states.lookup(x['STATE']).abbr, axis=1)
    df = df.filter(items=['STATE', 'FIPS', 'NAME'])
    return df


# Import deaths data
def import_deaths_data():
    with open('deaths_df.pickle', 'rb') as handle:
        deaths_data = pickle.load(handle)
    return deaths_data


# Creates a figure with a state highlighted
def make_deaths_figure(df, selected_state=None):
    if selected_state:
        filt = [x in selected_state for x in df['abbr']]
        real_deaths = df[filt].iloc[:,2:].sum().to_numpy()
    else:
        real_deaths = df.iloc[:,2:].sum().to_numpy()
    dates = np.arange(np.ceil(len(df.iloc[0])/7))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:-4], y=real_deaths[:-28:7], mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dates[-4:], y=real_deaths[-28::7], mode='lines+markers'))
    fig.update_layout(margin={'l':0, 'r':0, 't': 0, 'b': 0}, xaxis_title='Week', yaxis_title='Deaths',
                      showlegend=False)
    return fig


# Creates a figure with a state highlighted
def make_map_figure(df, selected_state=None):
    if selected_state:
        locs = [selected_state]
        deaths = [df[df['abbr'] == x].iloc[:, -1].to_numpy()[0] for x in [selected_state]]
        customdata = np.dstack((deaths,us.states.lookup(selected_state).name))[0]
    else:
        locs = state_abbrevs
        deaths = [df[df['abbr'] == x].iloc[:, -1].to_numpy()[0] for x in state_abbrevs]
        customdata = np.dstack((deaths,state_names))[0]
    fig = go.Figure(data=go.Choropleth(
        locations=locs,
        z=np.log10(deaths),
        customdata=customdata,
        hovertemplate='%{customdata[0]:,}<extra>%{customdata[1]}</extra>',
        locationmode='USA-states',
        colorscale='Reds',
        showscale=False
    ))
    fig.update_layout(margin={'l':0, 'r':0, 't': 0, 'b': 0}, geo_scope='usa')
    if selected_state:
        fig.update_geos(fitbounds='locations')

    return fig
