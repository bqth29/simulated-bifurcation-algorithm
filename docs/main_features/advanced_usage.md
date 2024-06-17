# Advanced Usage

The Simulated Bifurcation algorithm has a set of hyperparameters corresponding to physical constants derived from quantum theory, which have been fine-tuned (Goto *et al.*[^1][^2][^3]) to give the best results most of the time. Nevertheless, the relevance of specific hyperparameters may vary depending on the properties of the instances. For this purpose, if is possible to manually configure these hyperparameters (reading the Simulated Bifurcation theory is highly recommended).

The pre-set hyperparameters and their respective default values are gathered in the following table:

| Name             | Default value | Definition                                                                                      |
| ---------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| Time step        | 0.1           | Temporal discretization step for the evolution of the oscillators in the symplectic integrator. |
| Pressure slope   | 0.01          | Adiabatic system evolution rate (slope of the linear pumping function).                         |
| Heat coefficient | 0.06          | Influence of heating for HbSB or HdSB.                                                          |

```{eval-rst}
.. autofunction:: simulated_bifurcation.set_env
.. autofunction:: simulated_bifurcation.get_env
.. autofunction:: simulated_bifurcation.reset_env
```

[^1]: Hayato Goto et al., "Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems". Sci. Adv.5, eaav2372(2019). DOI:10.1126/sciadv.aav2372

[^2]: Hayato Goto et al., "High-performance combinatorial optimization based on classical mechanics". Sci. Adv.7, eabe7953(2021). DOI:10.1126/sciadv.abe7953

[^3]: Kanao, T., Goto, H. "Simulated bifurcation assisted by thermal fluctuation". Commun Phys 5, 153 (2022). https://doi.org/10.1038/s42005-022-00929-9
