import torch
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objs as go

import simulated_bifurcation as sb

from data.data import assets, dates

class Markowitz(sb.SBModel):

    """
    Implementation of an integer Markowitz model.
    """

    @classmethod
    def from_csv(
        cls,
        risk_coefficient : float = 1, 
        number_of_bits : int = 1,
        date : str = dates[-1],
        assets_list : list = assets[:]
    ):

        """
        Retrieves the data for the Markowitz model from .csv files.
        Only works with the files in the ./data folder.
        """
        
        covariance_filename = "./data/cov.csv"
        expected_return_filename = "./data/mu.csv"

        complete_monthly_returns = pd.read_csv(expected_return_filename)
        complete_monthly_returns.set_index('Date', inplace = True)

        cov = pd.read_csv(covariance_filename)
        cov.set_index('Unnamed: 0', inplace = True)

        mu = np.expand_dims(complete_monthly_returns[assets_list].loc[date].to_numpy(),1)
        sigma = cov[assets_list].loc[assets_list].to_numpy()

        covariance = torch.from_numpy(sigma)
        expected_return = torch.from_numpy(mu)

        return Markowitz(
            covariance,
            expected_return,
            risk_coefficient = risk_coefficient,
            number_of_bits = number_of_bits,
            assets_list = assets_list
        ) 

    def __init__(
        self, 
        covariance: torch.Tensor, 
        expected_return: torch.Tensor, 
        risk_coefficient: float = 1, 
        number_of_bits: int = 1,
        assets_list: list = assets[:],
        assert_parameters: bool = True,
    ) -> None:

        # Data
        self.covariance       = covariance
        self.expected_return  = expected_return
        self.risk_coefficient = risk_coefficient

        self.assets_list      = assets_list

        self.number_of_assets = covariance.shape[0]
        self.number_of_bits   = number_of_bits

        # Parameters to optimize

        self.portfolio        = None

        # Conversion matrix and vector

        self.M                = self.__conversion_matrix__()         
        self.U                = torch.ones([self.number_of_assets * self.number_of_bits, 1], dtype=torch.float64)

        self.assert_parameters = assert_parameters

    def __repr__(self) -> str:

        return f"""Utility gain: {self.utlity_function()}
        {self.as_dataframe()}
        """

    def __conversion_matrix__(self) -> torch.Tensor:

        matrix = torch.zeros(
            [self.number_of_assets * self.number_of_bits, self.number_of_assets],
            dtype=torch.float64
        )

        for a in range(self.number_of_assets):
            for b in range(self.number_of_bits):

                matrix[a*self.number_of_bits+b][a] = 2.0**b

        return matrix         

    # Inherited methods         

    def __to_Ising__(self) -> sb.Ising:

        """
        Generates the equivalent Ising model.
        """

        sigma = self.M @ self.covariance @ self.M.t()
        mu = self.M @ self.expected_return

        J = - .5 * self.risk_coefficient * sigma
        h = .5 * self.risk_coefficient * sigma @ self.U - mu 
        
        return sb.Ising(J, h, self.assert_parameters)

    def __from_Ising__(self, ising: sb.Ising) -> None:
        self.portfolio = .5 * self.M.t() @ (ising.ground_state + self.U)

    # Data extraction
        
    def as_dataframe(self) -> pd.DataFrame:

        """
        Formats the portfolio data in a dataframe.
        """

        if self.portfolio is None:

            return None

        else:

            optimized_portfolio = self.portfolio.T[0]

            assets_to_purchase = [self.assets_list[ind] for ind in range(len(self.assets_list)) if optimized_portfolio[ind] > 0]
            stocks_to_purchase = [optimized_portfolio[ind].item() for ind in range(len(optimized_portfolio)) if optimized_portfolio[ind] > 0]
            total_stocks = np.sum(stocks_to_purchase)

            df = pd.DataFrame(
                {
                    'assets': assets_to_purchase,
                    'stocks': stocks_to_purchase,
                    'ratios': [round(100 * stock/total_stocks, 3) for stock in stocks_to_purchase]
                }
            ).sort_values(by=['assets'])

            return df

    def utlity_function(self) -> float:

        if self.portfolio is None:

            return 0

        else:

            gain = - .5 * self.risk_coefficient * self.portfolio.t() @ self.covariance @ self.portfolio + self.expected_return.t() @ self.portfolio
            return gain.item()      

    # Graphical representation

    def pie(self) -> None:

        df = self.as_dataframe()

        if df is not None:

            fig = px.pie(
                df,
                values = 'stocks',
                names = 'assets',
                title = 'Optimal portfolio'
            )
            fig.show()

    def table(self) -> None:

        df = self.as_dataframe()

        if df is not None:

            trace = go.Table(
            header = dict(
                values = [
                    "Assets",
                    "Stocks to purchase",
                    'Percentage of capital invested'
                ],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5
            ),
            cells = dict(
                values = [
                    df.assets,
                    df.stocks,
                    df.ratios
                ],
                fill = dict(color='#F5F8FF'),
                align = ['left'] * 5)
            )

            data = [trace]
            iplot(data, filename = 'pandas_table')  