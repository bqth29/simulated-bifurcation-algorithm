import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objs as go
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
        self.portfolio = None

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

        optimized_portfolio = ((self.spin_matrix()).T @ ((ising.ground_state + 1)/2)).T[0]
        assets_to_purchase = [self.assets_list[ind] for ind in range(len(self.assets_list)) if optimized_portfolio[ind] > 0]
        stocks_to_purchase = [optimized_portfolio[ind] for ind in range(len(optimized_portfolio)) if optimized_portfolio[ind] > 0]
        total_stocks = sum(stocks_to_purchase)

        self.portfolio = pd.DataFrame(
            {
                'assets': assets_to_purchase,
                'stocks': stocks_to_purchase,
                'ratios': [round(stock/total_stocks*10000)/100 for stock in stocks_to_purchase]
            }
        ).sort_values(by=['assets'])

    ############################
    # Graphical representation #
    ############################

    def pie(self):

        if self.portfolio is not None:

            fig = px.pie(self.portfolio, values='stocks', names='assets', title='Optimal portfolio')
            fig.show()

    def table(self):

        if self.portfolio is not None:

            trace = go.Table(
            header=dict(values=["Assets","Stocks to purchase",'Percentage of capital invested'],
                        fill = dict(color='#C2D4FF'),
                        align = ['left'] * 5),
            cells=dict(values=[self.portfolio.assets,self.portfolio.stocks,self.portfolio.ratios],
                    fill = dict(color='#F5F8FF'),
                    align = ['left'] * 5))

            data = [trace]
            iplot(data, filename = 'pandas_table')  