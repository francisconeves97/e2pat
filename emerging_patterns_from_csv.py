# - coding: utf-8 --
"""
@info webpage to find patterns in traffic data
@author Francisco Neves
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from pathlib import Path
import pandas as pd

from emerging_patterns import two_step_regression_handler
import gui_utils
from app import app


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
    ('csv_file_path', '', gui_utils.Button.input_hidden)
]
charts = [
    ('results_container', gui_utils.get_null_label(), gui_utils.Button.html, True)
]

layout = gui_utils.get_layout(pagetitle, [('parameters', 55, parameters)], charts, prefix=prefix)


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
    Output(prefix + 'results_container', 'children'),
    [Input(prefix + 'button', 'n_clicks')],
    gui_utils.get_states(parameters, False, prefix))
def run(n_clicks, *args):
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
    res = two_step_regression_handler(time_series, 60, locations, csv_file)
    res = html.Div(res, id="regressions_bigcontainer")

    return res


if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=False, port=8050)
