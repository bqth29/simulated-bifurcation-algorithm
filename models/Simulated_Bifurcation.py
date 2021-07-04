import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objs as go

from models.Simulation import Simulation
from models.Markowitz import Markowitz
from models.Ising import Ising
from models.Hamiltionian import Hamiltonian

from data.assets import assets

class Simulated_Bifurcation():

    def __init__(
        self,
        covariance_filename = "./data/cov.csv", 
        expected_return_filename = "./data/mu.csv", 
        number_of_bits = 1,
        time_step = 0.01,
        simulation_time = 600,
        kerr_constant = 1,
        detuning_frequency = 1,
        risk_coefficient = 1,
        pressure = lambda t: 0.01 * t,
        symplectic_parameter = 2,
        date = '2021-03-01', 
        assets_list = assets
    ):

        # Simulation parameters

        self.parameters = Simulation(
            time_step,
            simulation_time,
            symplectic_parameter
            )

        # Hamiltonian

        self.Hamiltonian = Hamiltonian(
            kerr_constant,
            detuning_frequency,
            pressure
            )

        # Markowitz model

        self.Markowitz = Markowitz(
            None, 
            None, 
            risk_coefficient,
            number_of_bits,
            date,
            assets_list
        )
        self.Markowitz.from_csv(
            covariance_filename,
            expected_return_filename,
            date,
            assets_list
        )

        # Ising model

        self.Ising = Ising()
        self.Ising.from_Markowitz(self.Markowitz)

    # Optimal portfolio    

    def optimize(self):

        self.Ising.optimize(self.Hamiltonian, self.parameters)
        self.Markowitz.portfolio = (self.Ising.matrix.T @ (1 + np.sign(self.Ising.ground_state))/2).T[0]  

    def get_optimal_portfolio(self):

        if self.Markowitz.portfolio:

            return self.Markowitz.portfolio

        else:

            self.optimize()  
            return self.Markowitz.portfolio 

    # Charts

    def pie(self):

        """
        Draws a pie chart to show the assets allocation.
        """

        indexes = [i for i in range(self.Markowitz.number_of_assets) if self.Markowitz.portfolio[i] > 10**(-3)]
        assets = [self.Markowitz.portfolio[i] for i in indexes]
        names = [self.Markowitz.assets_list[i] for i in indexes]

        df = pd.DataFrame(
            {
                'value' : assets,
                'names': names
            }
        )

        fig = px.pie(df, values='value', names='names', title='Optimal portfolio')
        fig.show()

    def table(self):

        """
        Draws a comprehensive table gathering all the data regarding the assets allocation.
        """

        number_of_assets = len(self.Markowitz.portfolio)

        indexes = [i for i in range(number_of_assets) if self.Markowitz.portfolio[i] > 10**(-3)]
        assets = [self.Markowitz.portfolio[i] for i in indexes]
        names = [self.Markowitz.assets_list[i] for i in indexes]

        df = pd.DataFrame(
            {
                'value' : [round(100*asset/sum(assets),5) for asset in assets],
                'stocks' : assets,
                'names': names
            }
        ).sort_values(by=['names'])
        
        trace = go.Table(
        header=dict(values=["Assets","Stocks to purchase",'Percentage of capital invested'],
                    fill = dict(color='#C2D4FF'),
                    align = ['left'] * 5),
        cells=dict(values=[df.names,df.stocks,df.value],
                fill = dict(color='#F5F8FF'),
                align = ['left'] * 5))

        data = [trace]
        iplot(data, filename = 'pandas_table')    

    def draw_chart(self):

        if not self.Markowitz.portfolio:

            self.optimize()  

        if len([i for i in range(len(self.Markowitz.portfolio)) if self.Markowitz.portfolio[i] > 10**(-3)]) > 50:
            
            self.table()

        else:

            self.pie()             