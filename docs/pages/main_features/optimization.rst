Optimization
============

The package provides three functions to easily optimize quadratic polynomial
instances using the Simulated Bifurcation algorithm. This instance can be passed
as a SymPy polynomial expression or as a sequence of coefficient tensors. An optimization
domain shall also be specified.

This functions also include a list of optional parameters to customize the call to SB by
setting the stopping strategy and the computation dtype and device.

.. autofunction:: simulated_bifurcation.optimize
.. autofunction:: simulated_bifurcation.minimize
.. autofunction:: simulated_bifurcation.maximize
