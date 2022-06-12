from typing import Tuple, overload
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objs as go
import simulated_bifurcation as sb

from data.data import assets, dates

class Markowitz(sb.SBModel):

    """
    A representation of the Markowitz model for portolio optimization.
    Portfolio only takes integer stocks.

    ...

    Attributes
    ----------
    covariance : numpy.ndarray
        the correlation matrix between the assets
    expected_return : numpy.ndarray
        expected return on the investment
    risk_coefficient : float
        risk aversion on the investment
    assets_list : list 
        list of the assets to invest in
    number_of_assets : int 
        number of different assets 
    number_of_bits : int 
        number of bits for the binary decomposition of the assets stocks
    portfolio : numpy.ndarray 
        array of stocks to purchase per asset  
    M : numpy.ndarray 
        integer-binary conversion matrix  
    ones : numpy.ndarray 
        ones vector      
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

        covariance = sigma
        expected_return = mu

        return Markowitz(
            covariance,
            expected_return,
            risk_coefficient = risk_coefficient,
            number_of_bits = number_of_bits,
            assets_list = assets_list
        ) 

    def __init__(
        self, 
        covariance: np.ndarray, 
        expected_return: np.ndarray, 
        risk_coefficient: float = 1, 
        number_of_bits: int = 1,
        assets_list: list = assets[:],
        assert_parameters: bool = True,
    ) -> None:

        """
        Constructs all the necessary attributes for the Markowitz object.

        Parameters
        ----------
            covariance : numpy.ndarray
                the correlation matrix between the assets
            expected_return : numpy.ndarray
                expected return on the investment
            risk_coefficient : float
                risk aversion on the investment
            assets_list : list 
                list of the assets to invest in
            number_of_bits : int 
                number of bits for the binary decomposition of the assets stocks
            assert_parameters : bool, optional
                check the format of the inputs (default is True)
        """

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
        self.ones                = np.ones((self.number_of_assets * self.number_of_bits, 1), dtype = np.float64)

        # Check parameters

        if assert_parameters:
            self.__assert__()

    def __repr__(self) -> str:

        return f"""Utility gain: {self.utlity_function}
        {self.as_dataframe()}
        """

    def __assert__(self) -> None:

        """
        Checks the format of the attributes.

        Returns
        -------
        float
        """    
        
        # Checking types
        assert isinstance(self.covariance, np.ndarray), f"WARNING: The type of the covariance matrix must be a numpy array, instead got {type(self.covariance)}"
        assert isinstance(self.expected_return, np.ndarray), f"WARNING: The type of h must be a numpy array, instead got {type(self.expected_return)}"

        # Checking dimensions
        assert self.covariance.shape[0] == self.covariance.shape[1], f"WARNING: The covariance matrix must be a square matrix, instead got {self.covariance.shape}"
        assert self.expected_return.shape[0] == self.expected_return.shape[0], f"WARNING: The dimension of h must fits the covariance matrix's, instead of {self.covariance.shape[0]} got {self.expected_return.shape[0]}"
        assert self.expected_return.shape[1] == 1, f"WARNING: h must be a column vector with dimensions of this pattern: (n,1), instead got {self.expected_return.shape}"

        # Checking covariance matrix's properties
        assert np.allclose(self.covariance, self.covariance.T), "WARNING: The covariance matrix must be symmetric"
        assert min(np.linalg.eig(self.covariance)[0]) > 0, "WARNING: The covariance matrix must be postive definite"

    def __conversion_matrix__(self) -> np.ndarray:

        """
        Generates the integer-binary conversion matrix with the model's dimensions.

        Returns
        -------
        numpy.ndarray
        """  

        matrix = np.zeros(
            (self.number_of_assets * self.number_of_bits, self.number_of_assets),
            dtype = np.float64
        )

        for a in range(self.number_of_assets):
            for b in range(self.number_of_bits):

                matrix[a*self.number_of_bits+b][a] = 2.0**b

        return matrix              

    def __to_Ising__(self) -> sb.Ising:

        sigma = self.M @ self.covariance @ self.M.T
        mu = self.M @ self.expected_return

        J = - .5 * self.risk_coefficient * sigma
        h = .5 * self.risk_coefficient * sigma @ self.ones - mu 
        
        return sb.Ising(J, h)

    def __from_Ising__(self, ising: sb.Ising) -> None:
        self.portfolio = .5 * self.M.T @ (ising.ground_state + self.ones)

    # Data extraction
        
    def as_dataframe(self) -> pd.DataFrame:

        """
        Formats the portfolio data in a dataframe.

        Returns
        -------
        pandas.DataFrame
        """

        if self.portfolio is None:

            return None

        else:

            optimized_portfolio = self.portfolio.T[0]

            assets_to_purchase = [self.assets_list[ind] for ind in range(len(self.assets_list)) if optimized_portfolio[ind] > 0]
            stocks_to_purchase = [optimized_portfolio[ind] for ind in range(len(optimized_portfolio)) if optimized_portfolio[ind] > 0]
            total_stocks = np.sum(stocks_to_purchase)

            df = pd.DataFrame(
                {
                    'assets': assets_to_purchase,
                    'stocks': stocks_to_purchase,
                    'ratios': [round(100 * stock/total_stocks, 3) for stock in stocks_to_purchase]
                }
            ).sort_values(by=['assets'])

            return df

    @property
    def utlity_function(self) -> float:

        """
        Computes the Markowitz utility function given the portfolio.
        Default is 0 is case there is no portfolio.

        Returns
        -------
        float
        """

        if self.portfolio is None:

            return 0

        else:

            gain = - .5 * self.risk_coefficient * self.portfolio.T @ self.covariance @ self.portfolio + self.expected_return.T @ self.portfolio
            return gain[0][0]      

    # Graphical representation

    def pie(self) -> None:

        """
        Plots a pie chart to visualize the investments' data.

        Returns
        -------
        None
        """

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

        """
        Draws a data table to visualize the investments' data.

        Returns
        -------
        None
        """

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
            
def recursive_subportfolio_optimization(
    assets_list : list = assets[:],
    number_of_bits : int = 1,
    risk_coefficient : float = 1, 
    date : str = dates[-1],
    time_step: float = .01,
    symplectic_parameter: int = 2,
    convergence_threshold: int = 60,
    sampling_period: int = 35,
    max_steps: int = 60000,
    agents: int = 20,
    detuning_frequency: float = 1.,
    pressure_slope: float = .01,
    final_pressure: float = 1.,
    xi0: float = None,
    heat_parameter: float = 0.06,
    use_window: bool = True,
    ballistic: bool = False,
    heated: bool = True,
    print_evolution: bool = True
):

    # Initialization
    assets_kept = assets_list[:]
    previous, current = -1, 0
    investment = None

    while previous < current:

        previous = current

        # Retrieve previous step's portfolio
        try:
            investment = markowitz.as_dataframe()
        except:
            pass   

        # Optimize with the sub assets list    
        markowitz = Markowitz.from_csv(
            assets_list = assets_kept[:],
            number_of_bits = number_of_bits,
            date = date,
            risk_coefficient = risk_coefficient
        )

        markowitz.optimize(
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period,
            max_steps,
            agents,
            detuning_frequency,
            pressure_slope,
            final_pressure,
            xi0,
            heat_parameter,
            use_window,
            ballistic,
            heated
        )

        current = markowitz.utlity_function
        assets_kept = list(markowitz.as_dataframe()['assets'])

        if print_evolution and previous < current: print(current, len(assets_kept))

    return previous, investment