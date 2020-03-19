#!/usr/bin/env python
import csv
from datetime import date, datetime, timedelta
from io import StringIO
import sys
from typing import Dict, List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests


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
    cur_date = date(2020, 1, 22)
    today = datetime.now().date()
    
    data: Optional[pd.DataFrame] = None

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


@cli.command()
def show_reference_curve():
    '''
    Display the confirmed cases curve and an estimate using an exponential.
    '''
    with open(DATA_FILENAME, 'rb') as f:
        data: pd.DataFrame = pickle.load(f)

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

    # We're modeling this as an exponential, so we're gonna fit a line to log(y).
    model = np.polyfit(x, np.log(y), 1)
    estimation_func = lambda day_offset: np.exp(day_offset * model[0] + model[1])

    # Apply our function 
    estimations = list(map(estimation_func, x))
    country['Estimate'] = estimations
    country['DayOffset'] = [(x - first_day).days for x in file_dates]

    # Scale all the estimates so the most recent date's estimate matches truth.
    scale = confirmed.iloc[-1] / estimations[-1]
    country['ScaledEstimate'] = country['Estimate'] * scale

    # Plot.
    country.plot(x='DayOffset', y=['Confirmed', 'Estimate', 'ScaledEstimate'], kind='line')
    plt.show()


# CLI boilerplate
if __name__ == '__main__':
    cli()
