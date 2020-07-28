import pandas as pd
from urllib.request import urlopen
import json
import plotly.graph_objects as go
from plotly.graph_objs import Layout
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
        fig_title = us.states.lookup(selected_state).name
    else:
        real_deaths = df.iloc[:,2:].sum().to_numpy()
        fig_title = 'US National'
    dates = np.arange(np.ceil(len(df.iloc[0])/7))
    fig = go.Figure(layout=Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
    fig.add_trace(go.Scatter(x=dates[:-4], y=real_deaths[:-28:7], name='True', mode='lines+markers', marker={'color':'lightgrey'},
                             hovertemplate='<b>%{y:,}</b> <i>Actual</i><extra></extra>'))
    fig.add_trace(go.Scatter(x=dates[-4:], y=real_deaths[-28::7], name='Predicted', mode='lines+markers',
                             hovertemplate='<b>%{y:,}</b> <i>Predicted</i><extra></extra>'))
    fig.add_shape(type='rect', xref='x', yref='paper', x0=-2, y0=0, x1=dates[-5], y1=1, line={'width': 0},
                  fillcolor='#edeff2', layer='below')
    fig.update_layout(margin={'l':0, 'r':0, 't': 0, 'b': 0}, xaxis_title='Week', yaxis_title='Deaths',
                      showlegend=False, spikedistance=-1, hovermode='x')
    # Set a vertical spike to show current time period
    fig.update_xaxes(showgrid=False, zeroline=False, spikemode='across', spikesnap='cursor', spikecolor='black', spikedash='solid', spikethickness=1)
    fig.update_yaxes(showgrid=False)
    return fig


# Creates a figure with a state highlighted
def make_map_figure(df, selected_state=None, hover_state=None):
    # For hover tooltip
    deaths = [df[df['abbr'] == x].iloc[:, -1].to_numpy()[0] for x in state_abbrevs]
    custom_data = np.dstack((deaths,state_names))[0]

    # For color scaling
    color_scale = 'Portland'
    z_max = np.max(np.log10(deaths))
    z_min = np.min(np.log10(deaths))

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=state_abbrevs,
        z=np.log10(deaths),
        customdata=custom_data,
        hovertemplate='%{customdata[0]:,}<extra>%{customdata[1]}</extra>',
        locationmode='USA-states',
        marker={'line': {'width': 0}},
        marker_line_color=None,
        #showlakes=False,
        colorscale=color_scale,
        colorbar={'title': 'Deaths', 'tickvals': [1,2,3,4], 'ticktext': ['10','100','1,000','10,000'],
                  'xanchor': 'left', 'x': 0.9, 'yanchor': 'middle'}
    ))
    # If a state is selected, highlight it
    if selected_state:
        selected_data = [[df[df['abbr'] == selected_state].iloc[:, -1].to_numpy()[0],
                          us.states.lookup(selected_state).name]]
        fig.add_trace(go.Choropleth(
            locations=[selected_state],
            z=[np.log10(selected_data[0][0])],
            zmin=z_min,
            zmax=z_max,
            customdata=selected_data,
            hovertemplate='%{customdata[0]:,}<extra>%{customdata[1]}</extra>',
            locationmode='USA-states',
            marker={'line': {'width': 2}},
            marker_line_color='black',
            colorscale=color_scale,
            showscale=False
        ))
    fig.update_layout(margin={'l':0, 'r':0, 't': 0, 'b': 0}, geo_scope='usa', dragmode=False)
    #if selected_state:
    #    fig.update_geos(fitbounds='locations')
    return fig
