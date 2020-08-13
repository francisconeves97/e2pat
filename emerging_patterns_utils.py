from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import dash_html_components as html
from sklearn.metrics import r2_score
from pathlib import Path
from arff2pandas import a2p
import subprocess
import re
import hashlib
import pandas as pd
from datetime import timedelta, datetime

DOWNLOADS_PATH = str(Path(__file__).parent.parent.parent.parent) + '/data/temp'
JAR_DIRECTORY = str(Path(__file__).parent)


def replace_missing_values(series, attribute):
    if attribute.startswith('speed'):
        val_to_replace = series[attribute].max()
    elif attribute.startswith('spatial_extension'):
        val_to_replace = 0
    elif attribute.startswith('delay'):
        val_to_replace = 0
    else:
        val_to_replace = 0

    return series[attribute].replace(np.nan, val_to_replace)


def get_time_point_series(time_series, time_point):
    for attr in time_series.columns:
        time_series[attr] = replace_missing_values(time_series, attr)
    time_series = time_series.at_time(time_point)
    return time_series


def get_x_y(data, attribute):
    time_points = (data.index - data.index[0]).days.values.reshape(-1, 1)
    y = data[attribute].values

    return time_points, y


def differentiate(array):
    return np.concatenate([[0], np.diff(array)])


class TwoStepRegression:
    # Bounds calculating by analyzing slopes obtained and defining bounds for outliers
    SPEED_BOUND = 0.13
    SPATIAL_EXTENSION_BOUND = 50
    DELAY_BOUND = 28
    LOOP_DETECTORS_BOUND = 120

    def __init__(self, series, granularity, locations):
        self.series_original = series
        self.granularity = granularity
        self.locations = locations
        self.max_support = {'waze': 0, 'espiras': 0}

    @staticmethod
    def get_prediction_graph(cols, x, y, name):
        model = LinearRegression()
        model.fit(x, y)

        prediction = model.predict(x)
        r2_squared = r2_score(y, prediction)
        slope = model.coef_[0]
        template = '<b>Day: </b>%{x}<br />' + \
                   '<b>' + name + ': </b>%{y}<br />' + \
                   '<br />' \
                   '<b>R-Squared:</b>' + str(r2_squared) + \
                   '<br />' \
                   '<b>Slope:</b>' + str(slope)

        return {'fig': go.Scatter(x=cols, y=prediction, mode='lines',
                                  hovertemplate=template,
                                  name=name), 'r2': r2_squared, 'slope': slope}

    @staticmethod
    def remove_uncongested_rows(series, attribute):
        if attribute.startswith('speed'):
            return series[series[attribute] != series[attribute].max()]
        elif attribute.startswith('spatial_extension'):
            return series[series[attribute] != 0]
        elif attribute.startswith('delay'):
            return series[series[attribute] != 0]
        else:
            return series[series[attribute] != 0]

    @staticmethod
    def calculate_score(prediction, relative_support, is_differentiated, attribute):
        a1 = 3 / 10
        a2 = 5 / 10
        a3 = 2 / 10

        cong_factor = -1 if attribute.startswith('speed') else 1
        emerging_factor = 1 if is_differentiated else 2
        if prediction['slope'] > 0:
            multiplier = 1
        elif prediction['slope'] < 0:
            multiplier = -1
        else:
            multiplier = 0

        return (((abs(TwoStepRegression.normalize_slope(prediction['slope'], attribute)) * a1) +
                 (prediction['r2'] ** 2 * a2) + (relative_support * a3)) ** emerging_factor) * multiplier * cong_factor

    @staticmethod
    def normalize_slope(slope, attribute):
        if attribute.startswith('speed'):
            return slope / TwoStepRegression.SPEED_BOUND
        elif attribute.startswith('spatial_extension'):
            return slope / TwoStepRegression.SPATIAL_EXTENSION_BOUND
        elif attribute.startswith('delay'):
            return slope / TwoStepRegression.DELAY_BOUND
        # Loop detectors
        else:
            return slope / TwoStepRegression.LOOP_DETECTORS_BOUND

    @staticmethod
    def bound_slope(slope, attribute):
        abs_slope = abs(slope)
        multiplier = 1
        if slope < 0:
            multiplier = -1
        if attribute.startswith('speed') and abs_slope > TwoStepRegression.SPEED_BOUND:
            return TwoStepRegression.SPEED_BOUND * multiplier
        elif attribute.startswith('spatial_extension') and abs_slope > TwoStepRegression.SPATIAL_EXTENSION_BOUND:
            return TwoStepRegression.SPATIAL_EXTENSION_BOUND * multiplier
        elif attribute.startswith('delay') and abs_slope > TwoStepRegression.DELAY_BOUND:
            return TwoStepRegression.DELAY_BOUND * multiplier
        # Loop detectors
        elif abs_slope > TwoStepRegression.LOOP_DETECTORS_BOUND:
            return TwoStepRegression.LOOP_DETECTORS_BOUND * multiplier

        return slope

    def get_visualization(self):
        figs = []

        freq = "%dmin" % self.granularity
        if self.granularity % 60 == 0:
            freq = "%dH" % (self.granularity / 60)
        start_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        time_range = pd.DataFrame(index=pd.date_range(start_date, start_date + timedelta(hours=23, minutes=59),
                                                      freq=freq))

        for time_point in time_range.index:
            series = get_time_point_series(self.series_original, time_point)
            for attribute in self.series_original.columns:
                location_row = self.locations.loc[
                    self.locations.apply(lambda x: x['place_id'].lower() in attribute.lower(), axis=1)].iloc[0]
                data = self.remove_uncongested_rows(series, attribute)
                support = len(data)
                dataset = location_row['dataset']
                if self.max_support[dataset] < support:
                    self.max_support[dataset] = support

        for time_point in time_range.index:
            time_point_str = time_point.strftime('%H:%M')
            series = get_time_point_series(self.series_original, time_point)
            for attribute in self.series_original.columns:
                data = self.remove_uncongested_rows(series, attribute)

                if data.empty or len(data) < 5:
                    continue

                for differentiate_y in [False, True]:
                    x, y = get_x_y(data, attribute)
                    if differentiate_y:
                        y = differentiate(y)

                    if differentiate_y:
                        attribute_name = 'prediction_{}_diff'.format(attribute)

                    else:
                        attribute_name = 'prediction_{}'.format(attribute)

                    prediction = self.get_prediction_graph(data.index, x, y, attribute_name)

                    location_row = self.locations.loc[
                        self.locations.apply(lambda x: x['place_id'].lower() in attribute.lower(), axis=1)].iloc[0]

                    if location_row['dataset'] == 'waze':
                        attribute_name = attribute.lower().replace('_' + location_row['place_id'].lower(), '')
                    else:
                        attribute_name = attribute

                    support = len(data)
                    prediction['slope'] = self.bound_slope(prediction['slope'], attribute)
                    figs.append({'attribute_name': attribute_name,
                                 'attribute': attribute,
                                 'location': location_row['place_id'],
                                 'r2': round(prediction['r2'], 4),
                                 'slope': round(prediction['slope'], 4),
                                 'normalized_slope': round(
                                     TwoStepRegression.normalize_slope(prediction['slope'], attribute), 4),
                                 'time_point': time_point_str,
                                 'title': '{} at {}'.format(attribute.capitalize(), time_point_str),
                                 'score': round(
                                     self.calculate_score(prediction,
                                                          support / self.max_support[location_row['dataset']],
                                                          differentiate_y,
                                                          attribute), 4),
                                 'is_differentiated': str(differentiate_y),
                                 'support': support,
                                 'x_names': [str(x) for x in data.index],
                                 'x': [list(lst) for lst in x],
                                 'y': list(y),
                                 'location_coor': location_row['location'],
                                 'dataset': location_row['dataset']
                                 })

        figs = sorted(figs, key=lambda k: (abs(k['slope']), abs(k['r2'])), reverse=True)
        return figs

    @staticmethod
    def discover_patterns(data, attribute, differentiate_y=False):
        time_points, y = get_x_y(data, attribute)
        if differentiate_y:
            y = differentiate(y)

        model = LinearRegression()
        model.fit(time_points, y)
        return model
