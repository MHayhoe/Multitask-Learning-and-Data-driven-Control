import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
from Dashboard_Helper import *

import us

# Global variables
state_abbrevs = [s.abbr for s in us.states.STATES]
state_names = [s.name for s in us.states.STATES]

# Set up the global variables
df = import_county_data()
df_deaths = import_deaths_data()
fig_map = make_map_figure(df_deaths)
fig_series = make_deaths_figure(df_deaths)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': sa, 'value': sn} for (sa, sn) in zip(state_names, state_abbrevs)],
            value='PA')
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(id='county-dropdown')
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='map', figure=fig_map, style={'height': '200px'})
    ], style={'width': '48%', 'height': '250px', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='series-plot', figure=fig_series, style={'height': '200px'})
    ], style={'width': '48%', 'height': '250px', 'display': 'inline-block'})
])


# Set state based on clicking map
@app.callback(Output('state-dropdown', 'value'),
              [Input('map', 'clickData')])
def set_state_on_map_click(click_data):
    if click_data:
        selection = click_data["points"][0]['location']
        if selection in state_abbrevs:
            return selection
    else:
        return None


# Sets available options in the county dropdown when the state dropdown is changed
@app.callback(Output('county-dropdown', 'options'),
              [Input('state-dropdown', 'value')])
def set_county_options(selected_state):
    if selected_state:
        df_state = df[df['STATE'] == selected_state]
        return [{'label': 'Statewide', 'value': us.states.lookup(selected_state).fips}] + \
               [{'label': name, 'value': fips} for (fips, name) in zip(df_state['FIPS'], df_state['NAME'])]
    else:
        return [{'label': 'National', 'value': 'US'}]


# Highlights the map according to current selection
@app.callback(Output('map', 'figure'),
              [Input('state-dropdown', 'value')])
def set_map_highlight(selected_state):
    return make_map_figure(df_deaths, selected_state)


# Updates plot according to current selection
@app.callback(Output('series-plot', 'figure'),
              [Input('state-dropdown', 'value')])
def set_series_highlight(selected_state):
    return make_deaths_figure(df_deaths, selected_state)


# Set the initial value for the county dropdown when a new state is selected
@app.callback(Output('county-dropdown', 'value'),
              [Input('county-dropdown', 'options')])
def set_county_value(available_options):
    return available_options[0]['value']


if __name__ == '__main__':
    app.run_server(debug=True)
