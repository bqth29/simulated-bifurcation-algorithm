"""
Implementation of various *classical* optimization problems that can be
solved using the Simulated Bifurcation algorithm. This module is a
pre-defined models library to extend the use of SB.

It also comes with an abstract class called `ABCModel` which acts as a
basis for defining custom optimization models that are not implemented
yet.

"""

from .abc_model import ABCModel
from .ising import Ising
from .knapsack import Knapsack
from .markowitz import Markowitz, SequentialMarkowitz
from .number_partitioning import NumberPartitioning
from .qubo import QUBO
