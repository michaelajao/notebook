import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import plotly
# import plotly.figure_factory as ff
# import scipy.stats
# import pymc3 as pm
# import arviz as az
# import sunode
# import sunode.wrappers.as_theano
# from pymc3.ode import DifferentialEquation
# import theano.tensor as tt
# import theano
# import datetime
# import shelve
from datetime import datetime as dt
import time


# -------- Usage --------#
# covid_obj = COVID_data('US', Population=328.2e6)
# covid_obj.get_dates(data_begin='7/11/20', data_end='7/20/20')
# sir_model = SIR_model(covid_obj)
# likelihood = {'distribution': 'lognormal', 'sigma': 2}
# prior= {'lam': 0.4, 'mu': 1/8, lambda_std', 0.5 'mu_std': 0.5 }
# sir_model.run_SIR_model(n_samples=20, n_tune=10, likelihood=likelihood)
np.random.seed(0)


class COVID_data():

    def __init__(self, country='US', Population=328.2e6):

        confirmed_cases_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        self.confirmed_cases = pd.read_csv(confirmed_cases_url, sep=',')
        deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        self.deaths = pd.read_csv(deaths_url, sep=',')
        path_to_save = ''

        # ------------------------- Country for inference -------------------

        self.country = country
        self.N = Population   # Population of the country
        # Germany - 83.7e6
        # US - 328.2e6

    def get_dates(self, data_begin='7/11/20', data_end='7/20/20'):

        # ------------------------- Date for inference ----------------------#
        self.data_begin = data_begin  # Take the data until yesterday
        self.data_end = data_end
        self.num_days_to_predict = 14
        confirmed_cases = self.confirmed_cases
        country = self.country
        self.cases_country = confirmed_cases.loc[confirmed_cases["Country/Region"] == country]
        self.cases_obs = np.array(
            confirmed_cases.loc[confirmed_cases["Country/Region"] == country, data_begin:data_end])[0]

        print("------------ Cases for selected period ----------- ", self.cases_obs)

        date_data_end = confirmed_cases.loc[confirmed_cases["Country/Region"]
                                            == self.country, data_begin:data_end].columns[-1]
        month, day, year = map(int, date_data_end.split('/'))
        date_data_end = datetime.date(year+2000, month, day)
        date_today = date_data_end + datetime.timedelta(days=1)
        print("------------- Cases yesterday ({}): {} and day before yesterday: {} ------------".format(
            date_data_end.isoformat(), *self.cases_obs[:-3:-1]))
        self.num_days = len(self.cases_obs)

        day_before_start = dt.strptime(
            data_end, '%m/%d/%y') + datetime.timedelta(days=-1)
        day_before_start_cases = np.array(
            self.cases_country.loc[:, day_before_start.strftime('%-m/%-d/%-y')])
        print("------------ Day before start and cases for that date ------------",
              day_before_start, day_before_start_cases)
        future_days_begin = dt.strptime(
            data_end, '%m/%d/%y') + datetime.timedelta(days=1)
        future_days_end = future_days_begin + \
            datetime.timedelta(days=self.num_days_to_predict)
        self.future_days_begin_s = future_days_begin.strftime('%-m/%-d/%-y')
        self.future_days_end_s = future_days_end.strftime('%-m/%-d/%-y')
        print("------------- Future date begin and end -------------",
              self.future_days_begin_s, self.future_days_end_s)
        self.future_days = np.array(
            self.cases_country.loc[:, self.future_days_begin_s: self.future_days_end_s])[0]
        print("------------- Future days cases ------------", self.future_days)
