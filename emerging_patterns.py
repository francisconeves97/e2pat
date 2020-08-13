# - coding: utf-8 --
"""
@info webpage to find patterns in traffic data
@author Francisco Neves
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from pathlib import Path
import pandas as pd
import numpy as np
import json
import dash_table
import plotly.graph_objects as go

from emerging_patterns_utils import TwoStepRegression
import gui_utils
import map_utils

app = dash.Dash(__name__, assets_folder='assets', include_assets_files=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def embed_map(folium_map, prefix):
    return map_utils.embed_map(folium_map, prefix, height='370')


def get_multidrop_options(label_format, lst):
    return [{
        'label': label_format.format(str(el).capitalize()), 'value': el
    } for el in lst]


def two_step_regression_handler(time_series, granularity, locations, csv_file_path, initial_meteo_values=None,
                                initial_restrictions_value=None, initial_restrictions_cache=None):
    method = TwoStepRegression(time_series, granularity, locations)
    results = method.get_visualization()

    columns = [
        {'name': 'time_point', 'type': 'text'},
        {'name': 'location', 'type': 'text'},
        {'name': 'attribute_name', 'type': 'text'},
        {'name': 'is_differentiated', 'type': 'text'},
        {'name': 'score', 'type': 'numeric'},
        {'name': 'r2', 'type': 'numeric'},
        {'name': 'slope', 'type': 'numeric'},
        {'name': 'normalized_slope', 'type': 'numeric'},
        {'name': 'support', 'type': 'numeric'}
    ]

    boxstyle = {'background-color': '#dce7f3', 'width': '25%', 'border-radius': '5px',
                'border': 'none', 'display': 'inline-block', 'vertical-align': 'top', 'padding': 15, 'margin': 5}

    context_inputs = [html.Label('Filtrar por Condicionamentos',
                                 style={'font-weight': 'bold', 'font-style': 'italic', 'marginBottom': '1rem'}),
                      html.Div([
                          html.Label('Restrição de Circulação:', style={'font-weight': 'bold'}),
                          dcc.Dropdown(id='regressions_restrictions', options=get_multidrop_options('{}', [
                              'Estacionamento', 'Corte total', 'Cortes temporarios', 'Estreitamento de via',
                              'Nao previstas', 'Mantem perfil de via']),
                                       value=[] if initial_restrictions_value is None else initial_restrictions_value,
                                       multi=True, style={'width': '100%'})
                      ], style={'marginBottom': '1rem'}),
                      dcc.Input(id='restrictions_cache', style={'display': 'none'}, value=initial_restrictions_cache),
                      html.Label('Filtrar por Meteorologia',
                                 style={'font-weight': 'bold',
                                        'font-style': 'italic',
                                        'marginBottom': '1rem'}), ]
    for meteo_attr in meteo_options:
        value = None
        if initial_meteo_values is not None and meteo_attr in initial_meteo_values:
            value = initial_meteo_values[meteo_attr]
        context_inputs.append(html.Div([
            html.Label(meteo_attr.replace('_', ' ').capitalize() + ':', style={'font-weight': 'bold'}),
            dcc.Input(id=meteo_attr, style={'width': '100%'}, value=value)
        ], style={'marginBottom': '1rem'}))
    submit_style = {'background-color': '#6ABB97', 'border': 'none', 'font-size': '14px', 'width': '100%', 'margin': 5,
                    'marginTop': 20, 'margin-bottom': 25}
    context_inputs.append(html.Button('Filtrar', id='regressions_meteo_button', style=submit_style))

    if len(results) == 0:
        return html.Div([
            html.Div([
                html.Div([
                    dash_table.DataTable(
                        id='patterns-datatable',
                        columns=[
                            {"name": col['name'], "id": col['name'], "type": col['type'], "deletable": False,
                             "selectable": True}
                            for
                            col in columns
                        ],
                        data=[],
                        editable=False,
                        filter_action="custom",
                        sort_action="custom",
                        sort_mode="single",
                        selected_columns=[],
                        selected_rows=[],
                        page_action="custom",
                        page_current=0,
                        page_size=35,
                    ),
                ], style={'width': '70%'}),
                html.Div(context_inputs, style=boxstyle)
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'start'}),
            html.Div(
                [html.Div(
                    [embed_map(map_utils.get_lisbon_map(), prefix='emerging_patterns')],
                    id='emerging-patterns-map'
                )],
                id='emerging-patterns-vis-container'
            ),
            html.Div([dcc.Input(id='regressions_hour_range', style={'width': '100%', 'display': 'none'},
                                value=[])], style={'display': 'block', 'marginTop': '20px'}),
            dcc.Input(id='regressions_cache', style={'width': '100%', 'display': 'none'},
                      value=json.dumps([])),
            dcc.Input(id='regressions_csv_file_path', style={'width': '100%', 'display': 'none'},
                      value=csv_file_path)
        ])

    hours = pd.DataFrame(results).drop_duplicates('time_point')['time_point'].sort_values()
    marks = {i: {'label': value, 'style': {'transform': 'rotate(45deg) translateX(-20%)', 'margin-top': '5px'}}
             for i, value in enumerate(hours)}

    values = [-1, -0.5, 0, 0.5, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values,
        y=values,
        marker=dict(
            color=values,
            colorbar=dict(
                title="Score"
            ),
            colorscale="rdylbu",
            reversescale=True
        ),
        mode="markers"))

    return html.Div([
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='patterns-datatable',
                    columns=[
                        {"name": col['name'], "id": col['name'], "type": col['type'], "deletable": False,
                         "selectable": True}
                        for
                        col in columns
                    ],
                    data=results,
                    editable=False,
                    filter_action="custom",
                    sort_action="custom",
                    sort_mode="single",
                    selected_columns=[],
                    selected_rows=[],
                    page_action="custom",
                    page_current=0,
                    page_size=35,
                ),
            ], style={'width': '70%'}),
            html.Div(context_inputs, style=boxstyle)
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'start'}),
        dbc.Modal([
            dbc.ModalBody("This is the content of the modal", id="patterns-modal-body")
        ],
            id="patterns-modal",
            centered=True,
            size="xl"
        ),
        html.Div([dcc.RadioItems(
            options=[
                {'label': 'Congestionamentos', 'value': True},
                {'label': 'Descongestionamentos', 'value': False}
            ],
            value=True,
            labelStyle={'display': 'inline-block'},
            id='emerging-patterns-congestions-toggle'
        )], style={'margin-top': '48px'}),
        html.Div(
            [html.Div(
                [embed_map(map_utils.get_lisbon_map(), prefix='emerging_patterns')],
                id='emerging-patterns-map'
            ), dcc.Graph(figure=fig)],
            id='emerging-patterns-vis-container'
        ),
        html.Div([dcc.RangeSlider(
            id='regressions_hour_range',
            step=None,
            marks=marks,
            min=0,
            max=len(hours) - 1,
            value=[0, len(hours) - 1]
        )], style={'display': 'block', 'marginTop': '20px'}),
        dcc.Input(id='regressions_cache', style={'width': '100%', 'display': 'none'},
                  value=json.dumps(results, cls=NpEncoder)),
        dcc.Input(id='regressions_csv_file_path', style={'width': '100%', 'display': 'none'},
                  value=csv_file_path)
    ])


DOWNLOADS_PATH = str(Path(__file__).parent) + '/data/'

pagetitle = 'Emerging Patterns from CSV'
prefix = 'padroes_rodovia_from_file'

meteo_options = [
    'temperatura',
    'humidade',
    'intensidade_vento',
    'prec_acumulada',
    'pressao',
    'radiacao'
]
parameters = [
    ('nan', '10', gui_utils.Button.input_hidden),
    ('csv_file_upload', '', gui_utils.Button.upload),
    ('csv_file_path', '', gui_utils.Button.input_hidden),
    ('granularidade_em_minutos', '60', gui_utils.Button.input)
]
charts = [
    ('results_container', gui_utils.get_null_label(), gui_utils.Button.html, True)
]

layout = gui_utils.get_layout(pagetitle, [('parameters', 27, parameters)], charts, prefix=prefix)


def get_graph(fig, title=None):
    children = []

    if title is not None:
        children.append(html.Div([html.H3(title, style={'marginBottom': 0})], style={'textAlign': "center"}))

    children.append(dcc.Graph(figure=fig))

    return html.Div(children)


def get_state_field(field: str, accessor: str = 'value', prefix: str = '', type=None):
    states = dash.callback_context.states
    value = states['{}{}.{}'.format(prefix, field, accessor)]
    if type:
        if type == dict:
            try:
                value = eval(value)
            except:
                return None
        else:
            value = type(value)
    return value


@app.callback([Output(prefix + 'csv_file_upload_output', 'children'), Output(prefix + 'csv_file_path', 'value')],
              [Input(prefix + 'csv_file_upload', 'filename')])
def update_output(csv_file, *args):
    if csv_file is None or len(csv_file) == 0:
        return '', ''
    return html.Span(children=csv_file), DOWNLOADS_PATH + csv_file


@app.callback(
    [Output(prefix + 'results_container', 'children')],
    [Input(prefix + 'button', 'n_clicks')],
    gui_utils.get_states(parameters, False, prefix))
def run(n_clicks, *args):
    granularity = get_state_field('granularidade_em_minutos', prefix=prefix, type=int)

    csv_file = get_state_field('csv_file_path', prefix=prefix, type=str)
    if csv_file == '':
        return []

    state_params = dash.callback_context.states
    # remove prefix and .value from
    params = {}
    for key in state_params:
        params[key.replace(prefix, '').replace('.value', '')] = state_params[key]
    time_series_orig = pd.read_csv(csv_file, parse_dates=True, index_col=[0])
    time_series = time_series_orig

    locations = pd.read_csv(csv_file.split('.')[0] + '-locations.csv', parse_dates=True, index_col=[0])
    res = two_step_regression_handler(time_series, granularity, locations, csv_file)
    res = html.Div(res, id="regressions_bigcontainer")

    return res


if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=False, port=8050)
