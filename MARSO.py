from typing import List, Tuple
import random as rd
import pandas as pd

from data.data import assets, dates
from models.Markowitz import Markowitz

class MultiAgentRecursiveSubportfolioOptimizer():

    def __init__(
        self,
        agents: List[int] = [0],
        assets_list : list = assets[:],
        risk_coefficient : float = 1, 
        number_of_bits : int = 1,
        date : str = dates[-1],
    ) -> None:

        assert len(agents) > 0
        
        self.root = Agent(agents[:], assets_list[:], risk_coefficient, number_of_bits, date)
        self.layers = len(agents)
        self.utility_function = 0
        self.portfolio = None

    def solve(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 35,
        sampling_period: int = 50,
    ) -> None:

        self.root.build_relative_tree(
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period
        )

        self.utility_function, self.portfolio = self.root.explore_relative_tree()

class Agent(Markowitz):

    def __init__(
        self,
        relative_tree: List[int],
        assets_list : list = assets[:],
        risk_coefficient : float = 1, 
        number_of_bits : int = 1,
        date : str = dates[-1],
    ) -> None:

        aux_markowitz = Markowitz.from_csv(risk_coefficient, number_of_bits, date, assets_list)
        super().__init__(aux_markowitz.covariance, aux_markowitz.expected_return, risk_coefficient, number_of_bits, assets_list)
        self.date = date
        
        self.children = list()
        self.relative_tree = relative_tree[:]
        self.kept_assets = list()

    def remove_outliers(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 35,
        sampling_period: int = 50,
    ) -> None:

        self.optimize(
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period
        )    

        self.kept_assets = list(self.as_dataframe()['assets'])

    def create_children(self):

        # If the portfolio is not optimzed yet or if it is a leaf in the tree
        # then the agent's children is an empty list 
        if self.portfolio is None or len(self.relative_tree) == 0: self.children = list()
        else: 

            # Reset the list
            self.children = list()

            # Aux list to shuffle
            aux_kept_assets = self.kept_assets[:]
            n_children = self.relative_tree[0]

            for _ in range(n_children):

                self.children.append(
                    Agent(
                        self.relative_tree[1:],
                        aux_kept_assets[:],
                        self.risk_coefficient,
                        self.number_of_bits,
                        self.date
                    )
                )

                # Shuffle the assets list
                rd.shuffle(aux_kept_assets)

    def build_relative_tree(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 35,
        sampling_period: int = 50,
    ) -> None:           

        self.remove_outliers(
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period
        )
        self.create_children()

        for child in self.children:
            child.build_relative_tree()

    def explore_relative_tree(self):

        agent_data = (self.utlity_function(), self.as_dataframe())
        
        if len(self.relative_tree) == 0: return agent_data
        else: return Agent.max([agent_data] + [child.explore_relative_tree() for child in self.children])

    @classmethod
    def max(cls, list: List) -> Tuple[float, pd.DataFrame]:
        numerical = [x for x, y in list]
        return list[numerical.index(max(numerical))]
