# Knapsack problem

$$\text{Minimize } - \sum_{i = 1}^{N} c_{i}x_{i} + A_{\text{totalWeight}} \left ( \sum_{i = 1}^{N} w_{i}x_{i} - \sum_{k = 1}^{W} ky_{k} \right )^{2} + A_{\text{onlyOneWeight}} \left ( 1 - \sum_{k = 0}^{W} y_{k} \right )^{2}$$

with:
- $A_{\text{totalWeight}} = sum_{i = 1}^{N} c_{i} + 1$
- $A_{\text{onlyOneWeight}} = \left ( sum_{i = 1}^{N} c_{i} + 1 \right ) \left ( W + sum_{i = 1}^{N} w_{i} + 1 \right)$

```{eval-rst}
.. autoclass:: simulated_bifurcation.models.Knapsack
    :members:
```
