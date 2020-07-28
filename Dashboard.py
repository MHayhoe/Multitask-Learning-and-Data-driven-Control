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
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.0/css/bulma.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(id='hover-data', style={'display': 'none'}),
    html.Div(id='select-data', style={'display': 'none'}),

    html.Header([
        html.Div([html.Div([
            html.Div([
                html.P([html.Img(src='https://branding.web-resources.upenn.edu/sites/default/files/styles/1200x600_image/public/field/image/UniversityofPennsylvania_FullLogo_RGB_card.png?itok=xBxY13TC',
                                 style={'width': '150px'})], className='site-logo'),
            ], className='site-branding'),
        ], className='site-header-inside', style={'padding-left': '20px'})], className='inner')
    ], className='site-header outer'),

    html.Hr(style={'margin': '0px'}),

    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.P(["State"], className='heading'),
                                dcc.Dropdown(
                                    id='state-dropdown', value='', style={'width': '170px'}, className='level-left',
                                    placeholder='US National',
                                    options=[{'label': sa, 'value': sn} for (sa,sn) in zip(state_names, state_abbrevs)])
                            ]),
                        ], className='level is-mobile')
                    ], id='selection-container'),

                    html.Div([
                        dcc.Graph(id='map', figure=fig_map, clear_on_unhover=True, config={'displayModeBar': False}, style={'height': '200px'})
                    ], id='map-figure-container'),
                ], id='map-container', className='column is-4'),

                html.Div([
                    html.Div(['Cumulative observed deaths'], className='heading'),
                    html.Hr(),
                    dcc.Graph(id='series-plot', figure=fig_series, config={'displayModeBar': False}, style={'height': '300px'})
                ], id='chart-container', className='column is-8')
            ], className='columns'),
        ], className='container')
    ], className='section'),

    html.Hr(style={'margin': '0px'}),

    html.Footer([
        html.Div([
            html.Div(['Last updated on Tuesday, July 28th.'], className='content has-text-centered')
        ], className='container')
    ], className='footer')
])


# # Change behaviour based on what is being hovered over
# @app.callback(Output('hover-data', 'children'),
#               [Input('map','hoverData')])
# def update_hover_data(hover_data):
#     if hover_data:
#         return hover_data["points"][0]['location']
#     else:
#         return None


# Set state based on clicking map
@app.callback(Output('state-dropdown', 'value'),
              [Input('map', 'clickData'), Input('select-data', 'children')])
def set_state_on_map_click(click_data, current_selected_state):
    if click_data:
        selection = click_data["points"][0]['location']
        if selection == current_selected_state:
            return None
        elif selection in state_abbrevs:
            return selection
    else:
        return None


# Highlights the map according to current selection
@app.callback(Output('map', 'figure'),
              [Input('state-dropdown', 'value'), Input('hover-data', 'children')])
def set_map_highlight(selected_state, hover_state):
    return make_map_figure(df_deaths, selected_state, hover_state)


# Updates plot according to current selection
@app.callback(Output('series-plot', 'figure'),
              [Input('state-dropdown', 'value')])
def set_series_highlight(selected_state):
    return make_deaths_figure(df_deaths, selected_state)


# Sets available options in the county dropdown when the state dropdown is changed
# @app.callback(Output('county-dropdown', 'options'),
#               [Input('state-dropdown', 'value')])
# def set_county_options(selected_state):
#     if selected_state:
#         df_state = df[df['STATE'] == selected_state]
#         return [{'label': 'Statewide', 'value': us.states.lookup(selected_state).fips}] + \
#                [{'label': name, 'value': fips} for (fips, name) in zip(df_state['FIPS'], df_state['NAME'])]
#     else:
#         return [{'label': 'National', 'value': 'US'}]


# Set the initial value for the county dropdown when a new state is selected
# @app.callback(Output('county-dropdown', 'value'),
#               [Input('county-dropdown', 'options')])
# def set_county_value(available_options):
#     return available_options[0]['value']


if __name__ == '__main__':
    app.run_server(debug=True)
