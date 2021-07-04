import numpy as np
import pandas as pd
from models.Ising import Ising

class Markowitz():

    def __init__(
        self, 
        covariance = None, 
        expected_return = None, 
        risk_coefficient = 1, 
        number_of_bits = 1,
        date = '2021-03-01',
        assets_list = []
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
        expected_return_filename
    ):

        """
        Retrieves the data for the Markowitz model from .csv files.
        """

        if not self.assets_list:

            complete_monthly_returns = pd.read_csv(expected_return_filename)
            complete_monthly_returns.set_index('Date', inplace = True)

            cov = pd.read_csv(covariance_filename)
            cov.set_index('Unnamed: 0', inplace = True)

            mu = np.expand_dims(complete_monthly_returns.loc[self.date].to_numpy(),1)
            sigma = cov.to_numpy()

            self.covariance = sigma
            self.expected_return = mu
            self.number_of_assets = np.shape(self.covariance)[0]

        else:

            complete_monthly_returns = pd.read_csv(expected_return_filename)
            complete_monthly_returns.set_index('Date', inplace = True)

            cov = pd.read_csv(covariance_filename)
            cov.set_index('Unnamed: 0', inplace = True)

            mu = np.expand_dims(complete_monthly_returns[self.assets_list].loc[self.date].to_numpy(),1)
            sigma = cov[self.assets_list].loc[self.assets_list].to_numpy()

            self.covariance = sigma
            self.expected_return = mu
            self.number_of_assets = np.shape(self.covariance)[0]

    def spin_matrix(self):

        matrix = np.zeros((self.number_of_assets * self.number_of_bits, self.number_of_assets))

        for a in range(self.number_of_assets):
            for b in range(self.number_of_bits):

                matrix[a*self.number_of_bits+b][a] = 2**b

        return matrix   

    def to_Ising(self):

        sigma = np.block(
            [
                [2**(i+j)*self.covariance for i in range(self.number_of_bits)] for j in range(self.number_of_bits)
            ]
        )

        mu = self.spin_matrix() @ self.expected_return

        J = -self.risk_coefficient/2 * sigma
        h = self.risk_coefficient/2 * sigma @ np.ones((self.number_of_assets * self.number_of_bits, 1)) - mu 

        return Ising(J, h)

    def optimize(self, hamiltonian, parameters):

        ising = self.to_Ising()  
        ising.optimize(hamiltonian, parameters)
        
        self.portfolio = pd.DataFrame(
            {
                'assets': self.assets_list,
                'stocks': ((self.spin_matrix()).T @ ((ising.ground_state + 1)/2)).T[0] 
            }
        )