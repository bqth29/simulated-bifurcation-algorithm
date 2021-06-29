import numpy as np
import pandas as pd

class Markowitz():

    def __init__(
        self, 
        covariance, 
        expected_return, 
        risk_coefficient, 
        number_of_bits,
        date,
        assets_list
    ):

        self.covariance = covariance
        self.expected_return = expected_return
        self.number_of_bits = number_of_bits
        self.risk_coefficient = risk_coefficient
        self.date = date,
        self.assets_list = assets_list
        self.portfolio = []

        try:

            self.number_of_assets = np.shape(self.covariance)[0]

        except:

            pass    

    def from_csv(
        self,
        covariance_filename,
        expected_return_filename,
        date, 
        assets_list
    ):

        """
        Retrieves the data for the Markowitz model from .csv files.
        """

        if not assets_list:

            complete_monthly_returns = pd.read_csv(expected_return_filename)
            complete_monthly_returns.set_index('Date', inplace = True)

            cov = pd.read_csv(covariance_filename)
            cov.set_index('Unnamed: 0', inplace = True)

            mu = np.expand_dims(complete_monthly_returns.loc[date].to_numpy(),1)
            sigma = cov.to_numpy()

            self.covariance = sigma
            self.expected_return = mu
            self.number_of_assets = np.shape(self.covariance)[0]

        else:

            complete_monthly_returns = pd.read_csv(expected_return_filename)
            complete_monthly_returns.set_index('Date', inplace = True)

            cov = pd.read_csv(covariance_filename)
            cov.set_index('Unnamed: 0', inplace = True)

            mu = np.expand_dims(complete_monthly_returns[assets_list].loc[date].to_numpy(),1)
            sigma = cov[assets_list].loc[assets_list].to_numpy()

            self.covariance = sigma
            self.expected_return = mu
            self.number_of_assets = np.shape(self.covariance)[0]