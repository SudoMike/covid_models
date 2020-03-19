#!/usr/bin/env python
import csv
from datetime import date, datetime, timedelta
from io import StringIO
import os
from os import path
import sys
import tempfile
from typing import Callable, Dict, List, Tuple
import zipfile

import click
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
import requests
from tabulate import tabulate



DATA_FILENAME = 'data.pickle'

FILE_DATE_KEY = 'SourceCsvDate'


# CLI boilerplate
@click.group()
def cli():
    pass


@cli.command()
def fetch():
    '''
    Pull in data locally.
    '''
    print('Pulling all the CSVs..')
    start_date = date(2020, 1, 22)
    today = datetime.now().date()
    
    data: Optional[pd.DataFrame] = None

    cur_date = start_date
    while cur_date <= today:
        date_str = f'{cur_date.month:02}-{cur_date.day:02}-{cur_date.year}'
        sys.stdout.write(f'\r{date_str}...    ')

        # Pull down the csv
        url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{date_str}.csv'
        r = requests.get(url)

        # If it's too early in the day today, then there won't be data for today, and that's ok.
        if r.status_code != 200 and cur_date == today:
            break
        assert r.status_code == 200

        # Read it into a pandas dataframe
        with StringIO(r.text) as file:
            loaded_data: pd.DataFrame = pd.read_csv(file)
            loaded_data[FILE_DATE_KEY] = cur_date
            if data is None:
                data = loaded_data
            else:
                data = loaded_data.append(data)
        
        cur_date += timedelta(days=1)

    with open(DATA_FILENAME, 'wb') as f:
        pickle.dump(data, f)



def ensure_strictly_increasing(x: np.array, min_increment: float=0.01) -> np.array:
    '''
    Ensure that the input is strictly increasing.

    Args:
        x: The array to ensure is strictly increasing. This function will raise an exception
            if any value is LESS than the previous value. If it's equal to the previous value,
            that's fine, and it will add min_increment.

        min_increment: The minimum difference between consecutive values.

    Return:
        An array of the same (1-D) shape as x, but with values strictly increasing.
    '''
    assert len(x.shape) == 1 # must be 1-D

    ret = np.zeros(len(x))
    prev_val_truth = None
    for i, val in enumerate(x):
        if i == 0:
            ret[i] = val
        else:
            diff = val - prev_val_truth
            assert diff >= 0

            if diff == 0:
                ret[i] = prev_val_modified + min_increment
            else:
                ret[i] = val

        prev_val_modified = ret[i]
        prev_val_truth = val

    return ret


class ReferenceCurve:
    def __init__(self, data: pd.DataFrame, use_spline: bool = False):
        '''
        '''
        # Filter to the country we're gonna use to build our model.
        the_filter = data['Country/Region'].isin(['Italy'])
        country = data[the_filter]
        country = country[['Confirmed', FILE_DATE_KEY]].sort_values(by=[FILE_DATE_KEY])

        # Aggregate across all the provinces.
        country = country.groupby(FILE_DATE_KEY).sum()
        country.reset_index(level=0, inplace=True) # Convert the index (of the date) to a column.

        # Easier access to confirmed # cases and date.
        confirmed = country['Confirmed']
        file_dates = country[FILE_DATE_KEY]

        # Setup X (day offset) and Y (confirmed cases) to fit a model on.
        first_day: date = file_dates.iloc[0]
        last_day: date = file_dates.iloc[-1]
        num_days = (last_day - first_day).days + 1
        x = np.array(range(0, num_days))
        y = np.array(confirmed)

        if use_spline:
            self.spline = InterpolatedUnivariateSpline(x, y)
            self.spline_inverse = InterpolatedUnivariateSpline(ensure_strictly_increasing(y), x, k=1)
        else:
            # We're modeling this as an exponential, so we're gonna fit a line to log(y).
            model = np.polyfit(x, np.log(y), 1)
            self.estimation_func = lambda day_offset: np.exp(day_offset * model[0] + model[1])

            # Scale the estimation.
            self.scale = confirmed.iloc[-1] / self.estimation_func(x[-1])
            self.model = model

        self.use_spline = use_spline
        self.first_day = first_day
        self.country = country
        self.x = x
        self.y = y

    def estimate_confirmed_cases(self, day_offset: float) -> float:
        '''
        Estimate the number of confirmed cases in the model.

        Args:
            day_offset: The day, starting with 0=the first day in the dataset.

        Return:
            Estimate of the number of confirmed cases. The curve is rescaled so it touches the
            last data sample.
        '''
        if self.use_spline:
            return self.spline(day_offset)
        else:
            return self.estimation_func(day_offset) * self.scale

    def estimate_confirmed_cases_no_scale(self, day_offset: float) -> float:
        '''
        Args:
            day_offset: The day, starting with 0=the first day in the dataset.

        Return:
            Estimate of the number of confirmed cases. In this case, the curve is not rescaled
            to touch the last data sample.
        '''
        if self.use_spline:
            return self.spline(day_offset)
        else:
            return self.estimation_func(day_offset)

    def cases_to_day(self, cases: float) -> float:
        '''
        Do the inverse mapping - map a number of cases to a day.

        Args:
            cases: The number of cases.

        Return:
            The estimated day. If the function were perfectly accurate, then for example
            if `cases` == the number of cases on the last day of the dataset, this would
            return the number of the last day.
        '''
        if self.use_spline:
            return self.spline_inverse(cases)
        else:
            # Invert the exponential we would've used to generate the # of cases.
            inverted = (cases / self.scale)
            inverted = np.log(inverted)
            inverted = (inverted - self.model[1]) / self.model[0]
            return inverted



@cli.command()
@click.option('--use_spline/--no_use_spline', is_flag=True, default=True,
    help='Interpolate with a spline if True, otherwise an exponential.', show_default=True)
def show_reference_curve(use_spline: bool):
    '''
    Display the confirmed cases curve and an estimate using an exponential.
    '''
    # Load in the data.
    with open(DATA_FILENAME, 'rb') as f:
        data: pd.DataFrame = pickle.load(f)

    curve = ReferenceCurve(data, use_spline=use_spline)

    df = curve.country

    # Apply our function 
    df['Estimate'] = list(map(curve.estimate_confirmed_cases_no_scale, curve.x))
    df['EstimateScaled'] = list(map(curve.estimate_confirmed_cases, curve.x))
    df['DayOffset'] = [(x - curve.first_day).days for x in df[FILE_DATE_KEY]]

    # Plot.
    df.plot(x='DayOffset', y=['Confirmed', 'Estimate', 'EstimateScaled'], kind='line')
    plt.show()


@cli.command()
@click.option('--target_cases', default=20000, type=float,
              help='The number of cases to calculate # of days until.',
              show_default=True)
@click.option('--show_map/--no_show_map', is_flag=True, default=True,
    help='Render a map. Otherwise, just print a table.', show_default=True)
@click.option('--country', default='US', 
    help='The country to show info for. If not US, then you should use --no_show_map', show_default=True)
def show_map(target_cases: float, show_map: bool, country: str):
    '''
    Show a map of the US with T-XXX (to N cases) shown on the states.
    '''
    # Load the dataset.
    with open(DATA_FILENAME, 'rb') as f:
        data: pd.DataFrame = pickle.load(f)

    # Setup the reference curve.
    curve = ReferenceCurve(data, use_spline=True)

    # Load the geo data.
    usa = gpd.read_file(path.join('data', 'states.shp'))
    usa = usa[~usa.STATE_ABBR.isin(['AK', 'HI'])]

    # Restrict to US data, and confirmed cases only.
    data = data[data['Country/Region'] == country]

    # Calculate days until the target for each state.
    estimated_target_cases_day = int(curve.cases_to_day(target_cases))

    data['Province/State'] = data['Province/State'].fillna('ALL')
    if show_map:
        state_names = [x.STATE_NAME for _, x in usa.iterrows()]
    else:
        state_names = data['Province/State'].unique()

    state_to_data: Dict[str, pd.DataFrame] = {}
    state_to_days_until: Dict[str, int] = {}
    for state_name in state_names:
        # Get confirmed cases by day
        state_data = data[data['Province/State'] == state_name]
        state_data = state_data[['Confirmed', FILE_DATE_KEY]].sort_values(by=[FILE_DATE_KEY])
        state_data = state_data.groupby(FILE_DATE_KEY).sum()
        state_data.reset_index(level=0, inplace=True) # Convert the index (of the date) to a column.

        if len(state_data) == 0:
            continue

        cur_cases = int(state_data.iloc[-1]['Confirmed'])

        estimated_cur_day = int(curve.cases_to_day(cur_cases))
        estimated_cur_day = max(0, estimated_cur_day) # Don't let it get < 0 or else it throws ranges off.
        
        state_to_data[state_name] = state_data
        state_to_days_until[state_name] = estimated_target_cases_day - estimated_cur_day

    # If not show
    if not show_map:
        table = [['State', 'Current confirmed', f'Days until {int(target_cases)} cases']]
        for state in sorted(state_to_days_until.keys()):
            table.append([state, state_to_data[state].iloc[-1]['Confirmed'], state_to_days_until[state]])

        print(tabulate(table[1:], headers=table[0]))
        return

    # Set the 'days_until' column.
    usa['days_until'] = usa['STATE_NAME'].map(lambda x: state_to_days_until[x])

    def remap01(x, start, end):
        return start + (end - start) * x

    # Draw the whole map.
    cmap = plt.get_cmap('jet')
    cmap = [cmap(int(remap01(x, 0, 1))) for x in range(256)]
    cmap_reversed = ListedColormap(cmap[::-1])
    usa.plot(column='days_until', cmap=cmap_reversed, edgecolor='black')

    days_until_min = min(state_to_days_until.values())
    days_until_max = max(state_to_days_until.values())

    # Draw stuff on each state.
    for _, state_geo in usa.iterrows():
        # Get confirmed cases by day
        state_data = state_to_data[state_geo.STATE_NAME]
        cur_cases = int(state_data.iloc[-1]['Confirmed'])
        days_until = state_to_days_until[state_geo.STATE_NAME]

        # Figure out a text color.
        t = (days_until - days_until_min) / (days_until_max - days_until_min)
        color = cmap_reversed(int(t * 255))

        #x = usa[usa.STATE_NAME == state_geo.STATE_NAME]
        #x.plot()

        pt = state_geo.geometry.representative_point()
        txt = plt.text(
            pt.x,
            pt.y,
            f'{state_geo.STATE_ABBR}\nT-{days_until} days',
            horizontalalignment='center',
            color='white')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    highest_date = data[FILE_DATE_KEY].max()
    plt.title(f'From {highest_date}, days until {int(target_cases):,} (known) cases')
    plt.show()



# CLI boilerplate
if __name__ == '__main__':
    cli()
