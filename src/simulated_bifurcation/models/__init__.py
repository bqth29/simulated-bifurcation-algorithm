"""
Implementation of various *classical* optimization problems that can be
solved using the Simulated Bifurcation algorithm. This module is a
pre-defined models library to extend the use of SB.

It also comes with an abstract class called `ABCQuadraticModel` which acts as a
basis for defining custom optimization models that are not implemented
yet.

"""

from .knapsack import knapsack
from .markowitz import markowitz, sequential_markowitz
from .number_partitioning import number_partitioning
