# - coding: utf-8 --
"""
@info webpage to find patterns in traffic data
@author Francisco Neves
"""

import dash
import numpy as np
from dash.exceptions import PreventUpdate
from folium import GeoJson, CircleMarker
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
from pathlib import Path
import json
import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors

from app import app
import plot_utils
import map_utils
import gui_utils
from folium_draw import Draw
import series_waze
import series_espiras
from emerging_patterns_utils import TwoStepRegression, get_waze_events, concat_meteo_with_series

DOWNLOADS_PATH = str(Path(__file__).parent) + '/data/'


def convert_to_hex(rgba_color):
    red = str(hex(int(rgba_color[0] * 255)))[2:].capitalize()
    green = str(hex(int(rgba_color[1] * 255)))[2:].capitalize()
    blue = str(hex(int(rgba_color[2] * 255)))[2:].capitalize()

    if blue == '0':
        blue = '00'
    if red == '0':
        red = '00'
    if green == '0':
        green = '00'

    return '#' + red + green + blue


def get_multidrop_options(label_format, lst):
    return [{
        'label': label_format.format(str(el).capitalize()), 'value': el
    } for el in lst]


def get_graph(fig, title=None):
    children = []

    if title is not None:
        children.append(html.Div([html.H3(title, style={'marginBottom': 0})], style={'textAlign': "center"}))

    children.append(dcc.Graph(figure=fig))

    return html.Div(children)


def get_map():
    lisbon_map = map_utils.get_lisbon_map()
    Draw(page_prefix='padroes_rodovia',
         position='topleft',
         draw_options={'polyline': True, 'marker': True, 'circlemarker': False, 'circle': False, 'polygon': True,
                       'rectangle': False},
         edit_options={'poly': {'allowIntersection': False}}).add_to(lisbon_map)
    return lisbon_map


def embed_map(folium_map, prefix):
    return map_utils.embed_map(folium_map, prefix, height='370')


pagetitle = 'Road Traffic Emerging Patterns'
prefix = 'padroes_rodovia'

meteo_options = [
    'temperatura',
    'humidade',
    'intensidade_vento',
    'prec_acumulada',
    'pressao',
    'radiacao'
]
parameters = [
    ('date', ['2018-10-17', '2019-01-01'], gui_utils.Button.daterange),
    ('calendario', list(gui_utils.calendar.keys()) + list(gui_utils.week_days.keys()), gui_utils.Button.multidrop),
    ('granularidade_em_minutos', '60', gui_utils.Button.input),
    ('start_hour', '00:00', gui_utils.Button.input),
    ('end_hour', '23:59', gui_utils.Button.input),
    ('dataset', ['waze', 'espiras', 'integrative'], gui_utils.Button.radio),
    ('method', ['two_step_regression'], gui_utils.Button.radio),
    ('attributes', ['all'], gui_utils.Button.multidrop),
    ('geo_json', '', gui_utils.Button.input_hidden),
    ('series_cache', '', gui_utils.Button.input_hidden),
    ('meteo_series_cache', '', gui_utils.Button.input_hidden)
]
charts = [
    ('speed_series', gui_utils.get_null_label(), gui_utils.Button.html, True),
    ('time_point_series', gui_utils.get_null_label(), gui_utils.Button.html, True),
    ('prediction', gui_utils.get_null_label(), gui_utils.Button.html, True)
]

layout = gui_utils.get_layout(pagetitle, [('parameters', 27, parameters),
                                          ('selection_map', 27, [('lisbon_map', embed_map(get_map(), prefix),
                                                                  gui_utils.Button.html)])], charts, prefix=prefix)


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


def draw_score_markers(data, folium_map, congestions_toggle):
    cmap = plt.get_cmap('RdYlBu_r')

    # Sort by score to get higher score markers on top of the map
    data = sorted(data, key=lambda k: k['score'], reverse=congestions_toggle)
    norm = colors.Normalize(vmin=-1, vmax=1)
    placed_markers = set()
    try:
        for row in data:
            color = convert_to_hex(cmap(norm(row['score'])))
            tooltip = 'Location: {}<br />Attribute: {}<br />Time: {}<br />Score: {}'.format(
                row['location'].replace('{', '').replace('`', '').replace('}', ''),
                row['attribute_name'],
                row['time_point'],
                row['score'])
            # Only place a marker per location
            # TODO: change to max/min per location
            if row['location'] in placed_markers:
                continue

            if row['dataset'] == 'espiras':
                pmarker = CircleMarker(location=json.loads(row['location_coor']), radius=8, line_color=color,
                                       color=color,
                                       fill_color=color,
                                       tooltip=tooltip)
                pmarker.add_to(folium_map)
            else:
                if isinstance(row['location_coor'], dict):
                    row['location_coor'] = str(row['location_coor'])
                geojson = GeoJson(
                    row['location_coor'].replace("\'", "\""),
                    style_function=lambda f, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 4,
                        'fillOpacity': 0.7
                    },
                    tooltip=tooltip
                )
                geojson.add_to(folium_map)

            placed_markers.add(row['location'])
    except Exception as e:
        print(e)


def draw_jam_lines(jam_data, folium_map) -> None:
    for i, row in jam_data.iterrows():
        geojson = GeoJson(
            row['path.street_coord'].replace('\'', '\"'),
            style_function=lambda f: {
                'fillColor': '#FF0000',
                'color': '#FF0000',
                'weight': 4,
                'fillOpacity': 0.1,
            }
        )
        geojson.add_to(folium_map)


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


def get_series_plot(time_series, meteo_series, title):
    figs = []
    for attribute in time_series.columns:
        series = concat_meteo_with_series(time_series[attribute], meteo_series,
                                          time_series[attribute].max())
        fig = plot_utils.get_series_plot(series, '{}'.format(attribute.capitalize()),
                                         remove_gaps=True)
        figs.append(get_graph(fig, '{} - {} Time Series with Meteorology'.format(title, attribute.capitalize())))
    return figs


def get_dataset_time_series(dataset, start_date, end_date, days, granularity):
    geojson = get_state_field('geo_json', prefix=prefix, type=dict)
    geojson = geojson['geometry'] if geojson else None

    all_series = []
    time_series = None
    locations = []
    if not geojson:
        return False, 'Selecione um ponto no mapa para obter eventos...'

    if dataset == 'waze' or dataset == 'integrative':
        events_per_street, events_locations = get_waze_events(start_date, end_date, geojson, days)
        events_locations = events_locations.rename(columns={'street_name': 'place_id'})
        events_locations = events_locations.rename(columns={'path.street_coord': 'location'})
        events_locations['dataset'] = 'waze'
        if events_per_street is None or events_per_street.empty:
            return False, 'Não foram encontrados eventos do waze com os filtros selecionados...'

        # Get time series
        time_series, name = series_waze.get_event_series(events_per_street, granularity, geojson)
        all_series.append(time_series)
        locations.append(events_locations)

    if dataset == 'espiras' or dataset == 'integrative':
        time_series, events_locations = series_espiras.get_spatial_series_per_loop(start_date, end_date, granularity,
                                                                                   days, geojson)
        events_locations = events_locations.rename(columns={'espira': 'place_id'})
        events_locations = events_locations.rename(columns={'coordinates': 'location'})
        events_locations['dataset'] = 'espiras'
        events_locations = events_locations.drop_duplicates('place_id')

        locations.append(events_locations)
        all_series.append(time_series)

    if dataset != 'integrative':
        return True, (time_series, locations[0])

    # Integrative
    if len(all_series) > 1:
        time_series = pd.merge(all_series[0], all_series[1], left_index=True, right_index=True)
        locations = pd.concat(locations)
    else:
        time_series = all_series[0]
        locations = locations[0]
    for attr in time_series.columns:
        if attr.startswith('speed'):
            time_series[attr] = time_series[attr].fillna(time_series[attr].max())
        elif attr.startswith('spatial_extension') or attr.startswith('delay'):
            time_series[attr] = time_series[attr].fillna(0)
        else:
            # espiras
            time_series[attr] = time_series[attr].fillna(0)

    return True, (time_series, locations)


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    [Output('regressions_bigcontainer', 'children')],
    [Input('regressions_meteo_button', 'n_clicks')],
    [State(option, 'value') for option in meteo_options] +
    [State('regressions_csv_file_path', 'value')])
def filter_regressions_meteo(n_clicks, *kwargs):
    if not dash.callback_context.triggered or dash.callback_context.triggered[0]['value'] is None:
        raise PreventUpdate

    csv_file = get_state_field('regressions_csv_file_path')
    time_series_orig = pd.read_csv(csv_file, parse_dates=True, index_col=[0])
    csv_file_name = csv_file.split('.')[0]
    locations = pd.read_csv(csv_file_name + '-locations.csv', parse_dates=True, index_col=[0])

    time_series_orig = time_series_orig.dropna()

    # TODO: Get granularity
    res = two_step_regression_handler(time_series_orig, 60, locations, csv_file)
    return [html.Div(res, id="regressions_bigcontainer")]


@app.callback(
    [Output('patterns-datatable', "data"), Output('emerging-patterns-map', 'children')],
    [Input('patterns-datatable', "filter_query"), Input('regressions_cache', 'value'),
     Input('patterns-datatable', "page_current"),
     Input('patterns-datatable', "page_size"),
     Input('patterns-datatable', 'sort_by'),
     Input('regressions_hour_range', 'value'), Input('emerging-patterns-congestions-toggle', 'value')])
def update_table(filter, regressions, page_current, page_size, sort_by, hour_range, congestions_toggle, *kwargs):
    regressions = json.loads(regressions)
    dff = pd.DataFrame(regressions)
    hours = pd.DataFrame(regressions).drop_duplicates('time_point')['time_point'].sort_values()
    value_map = {}
    for i, hour in enumerate(hours):
        value_map[hour] = i

    dff['time_point_values'] = dff['time_point'].map(value_map)
    dff = dff[dff['time_point_values'].between(hour_range[0], hour_range[1])]

    lisbon_map = map_utils.get_lisbon_map()

    if filter is None and sort_by is None:
        draw_score_markers(dff.to_dict('records'), lisbon_map, congestions_toggle)
        map_iframe = embed_map(lisbon_map, prefix='emerging_patterns')
        return [dff.iloc[page_current * page_size:(page_current + 1) * page_size].to_dict('records'), map_iframe]

    if filter is not None:
        filtering_expressions = filter.split(' && ')
        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            inplace=False
        )
    res = dff.iloc[page_current * page_size:(page_current + 1) * page_size].to_dict('records')

    draw_score_markers(dff.to_dict('records'), lisbon_map, congestions_toggle)
    map_iframe = embed_map(lisbon_map, prefix='emerging_patterns')
    return [res, map_iframe]


@app.callback(
    [Output('patterns-modal', 'is_open'), Output('patterns-modal-body', 'children')],
    [Input('patterns-datatable', 'active_cell'), Input('patterns-datatable', 'data')])
def change_dataset(active_cell, data):
    if not active_cell:
        return [False, '']

    row = data[active_cell['row']]
    graph = TwoStepRegression.get_prediction_graph(row['x_names'], row['x'], row['y'], row['attribute'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=row['x_names'], y=row['y'], mode='markers', name='data'))
    fig.add_trace(graph['fig'])
    fig.update_layout(xaxis_title='Days', yaxis_title=row['attribute'])
    return [True, get_graph(fig, '')]


@app.callback(
    [Output(prefix + 'charts', 'children'),
     Output(prefix + 'attributes', 'options'),
     Output(prefix + 'series_cache', 'value'),
     Output(prefix + 'meteo_series_cache', 'value')],
    [Input(prefix + 'button', 'n_clicks'), Input(prefix + 'attributes', 'value')],
    gui_utils.get_states(parameters, False, prefix))
def run_discovery(n_clicks, attributes, *args):
    attributes_opts = []
    if not n_clicks:
        return [[], attributes_opts, '', '']

    trigger = dash.callback_context.triggered[0]
    data_cached = False
    if 'button' in trigger['prop_id']:
        attributes = None
    else:
        data_cached = True

    # Date range
    start_date = pd.to_datetime(get_state_field('date', 'start_date', prefix))
    end_date = pd.to_datetime(get_state_field('date', 'end_date', prefix))

    # Days
    calendar = get_state_field('calendario', prefix=prefix)
    days = [gui_utils.get_calendar_days(calendar)]

    # Granularity
    granularity = get_state_field('granularidade_em_minutos', prefix=prefix, type=int)

    pattern_discovery_method = get_state_field('method', prefix=prefix, type=str)
    dataset = get_state_field('dataset', prefix=prefix, type=str)

    if not data_cached:
        params_ok, res = get_dataset_time_series(dataset, start_date, end_date, days, granularity)

        if not params_ok:
            res = html.Span(res)
            return [[res], attributes_opts, '', '']

        time_series, locations = res
        time_series_orig = time_series
    else:
        # Read stuff from cached fields
        time_series_orig = pd.read_json(get_state_field('series_cache', prefix=prefix, type=str), orient='split')

        # Select only the columns of selected attributes
        if len(attributes) != 0:
            time_series = time_series_orig[attributes]
        else:
            time_series = time_series_orig

    start_hour = get_state_field('start_hour', prefix=prefix, type=str)
    end_hour = get_state_field('end_hour', prefix=prefix, type=str)

    time_series = time_series.between_time(start_hour, end_hour)
    filename = 'dataset_{}{}-{}{}-{}'.format(start_date, start_hour, end_date, end_hour, dataset)
    file_path = '{}/{}'.format(DOWNLOADS_PATH, filename)
    csv_file_path = '{}.csv'.format(file_path)
    time_series_orig.between_time(start_hour, end_hour).to_csv('{}.csv'.format(file_path))

    if pattern_discovery_method == 'two_step_regression':
        res = two_step_regression_handler(time_series, granularity, locations, csv_file_path)
        res = html.Div(res, id="regressions_bigcontainer")
    else:
        res = [html.Span('Abordagem nao implementada')]

    time_series_attrs = list(time_series_orig.columns)
    time_series_attrs = get_multidrop_options('{}', time_series_attrs)

    attributes_opts += time_series_attrs

    return [res, attributes_opts, time_series_orig.to_json(orient='split')]


if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=False, port=8051)
