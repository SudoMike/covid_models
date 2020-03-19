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
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
import requests


# From https://www.arcgis.com/home/item.html?id=f7f805eb65eb4ab787a0a3e1116ca7e5
GEO_DATA_URL = 'https://ago-item-storage.s3.us-east-1.amazonaws.com/f7f805eb65eb4ab787a0a3e1116ca7e5/states_21basic.zip?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCMaOiSKuOpfvNEuBBw%2BJoqI7InQBgJHsk2cOT%2Brne%2BmAIgKqcCrNzYyYl%2BbF2uljj7WL10YQsrIcjzCwTOEZRNziQqvQMI2P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw2MDQ3NTgxMDI2NjUiDFdrXs2MP3n0PrZZZCqRA%2BbH%2F0oiCh5GwGGZddtM4j1bopoYRcJIZPOiaD7joOEDkWmVN5iV2gZgp8NAKMXLUlMLZVcN8jpwZ9F%2BjRwTMwPtkx55m1lvkjxmgxb4COW4dIYf1Ywh3%2FwOggCd1BmQHPIbF3%2Bpvk7wf6610wbM9APLkfZYfXHsJZf14HtdPsTjU0FsBjMdr6cWIakiNdrzdUG84y%2F0oZBYgBUhWgTwwdpEqicOXAih9dLhJ2etxzNw8Ir%2FSTPNRqtGRwczp8QjNnCZHkqzfr3Tbq6eCMSHiT%2Bfym80hBlD0ku2xZydP7%2BECn948vMdhc6RDPDoQwCkM3kYwRjtCAdFDqtO7nGsEjULUSo6dmblp3dcs%2BgHGRdYw%2BrwSeRYX%2FFJbrcqUeaNtmJSZqTcjyyAepmtqfRbuFNEQQVOVHWq0IT1uq%2FpfzZwSE5NRj3n%2Blp1%2FwFl7VmbhWovPlj1n1EYVCmXmUQWjMhxudhdLXe5Ep04wiYItAhcBeowCGUKOezL9Sc%2BwMBtykF8f5D6dmW6JiR6KdF6PFrAMJONzvMFOusBJhONKhjppLEnbECm4yRtM9WS6%2Fp41NyaUuIOd44SqHrZo1cifahGsxNf5jMffS1DPDqHTetbSgvtgYHcz1BJXlGw6koxoPmy39f3S%2BTuI5V%2Fl%2F9IKOA28M32lkYgXVotHVAAQu9Wm%2FcQHEoJJo32gG%2FedFnxH8ilDZe77Do0WiGuDz6ETLu0eS8UNKp2Hrs562iXsqRXz8iLdpoDHBwf7vy6scmrj7wNm7sifYHOdovSVkHLJBv2dSCe144hTyYAqVjq5libqpAuiYZ9l7SYpyYnC8UxXpkkAMcoArf1urFOtW3NhDu4rjmErA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200319T152420Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAYZTTEKKEX7WFHP2F%2F20200319%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a030c52888812246d6c50c403f806c7467ee465358bcf85877fb60b5fe5ea6e0'
GEO_DATA_FILENAME = 'states_21basic.zip'

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
    print(f'Pulling the geo data into {GEO_DATA_FILENAME}..')
    with open(GEO_DATA_FILENAME, 'wb') as f:
        r = requests.get(GEO_DATA_URL, allow_redirects=True)
        f.write(r.content)

    return

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
def show_map(target_cases: float):
    '''
    Show a map of the US with T-XXX (to N cases) shown on the states.
    '''
    # Load the dataset.
    with open(DATA_FILENAME, 'rb') as f:
        data: pd.DataFrame = pickle.load(f)

    # Setup the reference curve.
    curve = ReferenceCurve(data, use_spline=True)

    # Load the geo data.
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(GEO_DATA_FILENAME) as geo_files:
            geo_files.extractall(tempdir)

        usa = gpd.read_file(path.join(tempdir, 'states.shp'))
        usa = usa[~usa.STATE_ABBR.isin(['AK', 'HI'])]


    # Restrict to US data, and confirmed cases only.
    data = data[data['Country/Region'].isin(['US'])]

    # Calculate days until the target for each state.
    estimated_target_cases_day = int(curve.cases_to_day(target_cases))

    state_to_data: Dict[str, pd.DataFrame] = {}
    state_to_days_until: Dict[str, int] = {}
    for _, state_geo in usa.iterrows():
        # Get confirmed cases by day
        state_data = data[data['Province/State'] == state_geo.STATE_NAME]
        state_data = state_data[['Confirmed', FILE_DATE_KEY]].sort_values(by=[FILE_DATE_KEY])
        state_data = state_data.groupby(FILE_DATE_KEY).sum()
        state_data.reset_index(level=0, inplace=True) # Convert the index (of the date) to a column.

        cur_cases = int(state_data.iloc[-1]['Confirmed'])

        estimated_cur_day = int(curve.cases_to_day(cur_cases))
        estimated_cur_day = max(0, estimated_cur_day) # Don't let it get < 0 or else it throws ranges off.
        
        state_to_data[state_geo.STATE_NAME] = state_data
        state_to_days_until[state_geo.STATE_NAME] = estimated_target_cases_day - estimated_cur_day


    # Draw the whole map.
    usa.plot()

    days_until_min = min(state_to_days_until.values())
    days_until_max = max(state_to_days_until.values())

    cmap = plt.get_cmap('jet') # 0-255 maps to red->orange

    # Draw stuff on each state.
    for _, state_geo in usa.iterrows():
        # Get confirmed cases by day
        state_data = state_to_data[state_geo.STATE_NAME]
        cur_cases = int(state_data.iloc[-1]['Confirmed'])
        days_until = state_to_days_until[state_geo.STATE_NAME]

        # Figure out a text color.
        t = (days_until - days_until_min) / (days_until_max - days_until_min)
        color = cmap(255 - int(t * 255))

        pt = state_geo.geometry.representative_point()
        txt = plt.text(
            pt.x,
            pt.y,
            f'{state_geo.STATE_ABBR}\ncases: {cur_cases}\nT-{days_until} days',
            horizontalalignment='center',
            color=color,
            weight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    highest_date = data[FILE_DATE_KEY].max()
    plt.title(f'From {highest_date}, days until {int(target_cases):,} cases')
    plt.show()



# CLI boilerplate
if __name__ == '__main__':
    cli()
