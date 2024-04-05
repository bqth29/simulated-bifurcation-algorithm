# Advanced usages

The Simulated Bifurcation algorithm has a set of hyperparameters corresponding to physical
constants derived from quantum theory, which have been fine-tuned (Goto *et al.*)
to give the best results most of the time. Nevertheless, the relevance of specific
hyperparameters may vary depending on the properties of the instances. For this purpose,
if is possible to manually configure these hyperparameters (reading the Simulated Bifurcation
theory is highly recommended).

- Goto, H., Tatsumura, K., & Dixon, A. R. (2019). Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems. *Science advances, 5* (4), eaav2372.
- Kanao, T., & Goto, H. (2022). Simulated bifurcation assisted by thermal fluctuation. *Communications Physics, 5* (1), 153.
- Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., ... & Tatsumura, K. (2021). High-performance combinatorial optimization based on classical mechanics. *Science Advances, 7* (6), eabe7953.

```{eval-rst}
.. autofunction:: simulated_bifurcation.set_env
.. autofunction:: simulated_bifurcation.get_env
.. autofunction:: simulated_bifurcation.reset_env
```
