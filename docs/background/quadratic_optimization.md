# Quadratic optimization

**Unconstrained quadratic optimization**, is a type of mathematical optimization problem that deals with optimizing a quadratic objective function. In computer science, these problems arise in various fields such as machine learning, finance, engineering, and operations research. They are often solved using specialized algorithms and software packages.

## Mathematical definition

In mathematical terms, a quadratic optimization problem can be formulated as follows:

$$\text{Minimize } x^T Q x + l^T x + c$$

Where:
- $x$ is a vector of decision variables that we want to find,
- $Q$ is a matrix of coefficients of the quadratic terms,
- $l$ is a vector of coefficients of the linear terms,
- $c$ is a constant term called the *bias*.

The goal is to find the values of $x$ that minimize the objective function while satisfying the given constraints.

> **Remarks**
> - it is also possible to maximize the objective function and there is a straightforward equivalence between maximization and minimization:
> 
> $$\text{Minimize } x^T Q x + l^T x + c \Leftrightarrow \text{Maximize } - \left ( x^T Q x + l^T x + c \right )$$
> - the objective function can also be seen as a **multivariate quadratic polynomial**, i.e. a polynomial with multiple variables with monoms that individually have a degree lower or equal to 2:
> 
> $$P(x_{1}, ..., x_{n}) = \sum_{i=1}^{n} \sum_{j=1}^{n} Q_{ij}x_{i}x_{j} + \sum_{i=1}^{n} l_{i}x_{i} + c$$

## Discrete optimization

When the decision variables are continuous, solving quadratic optimization problems becomes a convex optimization problem, and efficient algorithms such as simplex or interior-point methods can be used to find the global optimum in polynomial time. However, when the decision variables are integer or binary, the problem becomes significantly harder to solve and enters the frame of NP-hard combinatorial optimization.

Discrete quadratic optimization must be tackled using heuristic algorithms in order to find sub-optimal solutions in a reasonable amount of time. The [Simulated Bifurcation algorithm](simulated_bifurcation_algorithm.md) is such an algorithm that can be adapted to approach any kind of quadratic optimization problem.

## Common quadratic models

### Quadratic Unconstrained Binary Optimization (QUBO)

[QUBO](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) stands for Quadratic Unconstrained Binary Optimization. It is a specific form of quadratic optimization problem where the decision variables are binary, meaning they can only take on values of 0 or 1. The general form of a QUBO problem is:

$$\text{Minimize } \sum_{i=1}^{N} \sum_{j=1}^{N} Q_{ij} x_i x_j = x^{T}Qx$$

Where:
- $x_i \in \{0, 1\}$ represents the binary decision variables,
- $Q_{ij}$ represents the coefficients of the quadratic terms.

> By convention, QUBO problems are often written without bias and with an upper-triangular matrix in which the terms of the diagonal account for the linear coefficients and the terms above the diagonal account for the quadratic coefficients.

QUBO problems have applications in various fields, including combinatorial optimization, machine learning, cryptography, and quantum computing. They are particularly relevant in the context of optimization problems that involve binary decisions, such as graph partitioning, maximum clique, Boolean satisfiability problems (SAT), and many others.

Thanks to its simple, compact form, QUBO is often seen as a central model for modeling a wide range of combinatorial quadratic problems and tackling numerous real-world optimization challenges. Finally, using base 2 representation of integers, it is also very easy to convert integer quadratic optimization problems into QUBO format.

Thus, solving QUBO problems efficiently is important and there are specialized heuristic algorithms and techniques designed to address them. Additionally, with the emergence of quantum computing, there is growing interest in using quantum annealing and other quantum optimization techniques, including the [Simulated Bifurcation algorithm](simulated_bifurcation_algorithm.md), to solve QUBO problems more efficiently.

### Ising model

An Ising problem is a type of mathematical model used to describe interactions between spins in a physical system, such as a collection of magnetic dipoles. It's named after the physicist Ernst Ising who first introduced it in the context of ferromagnetism.

In the Ising model, each spin can be in one of two states, typically represented as +1 (spin up) or -1 (spin down). The energy of the system is determined by the interactions between neighboring spins. The total energy of the system can be expressed as a quadratic function of the spin variables.

The Ising model can be represented mathematically using the following energy function:

$$E(\mathbf{s}) = -\frac{1}{2}\sum_{i,j} J_{ij} s_i s_j + \sum_{i} h_i s_i$$

Where:
- $\mathbf{s} = (s_1, s_2, ..., s_n)$ represents the spin configuration,
- $J_{ij}$ are the coupling coefficients between spins $s_i$ and $s_j$ (with $J_{ii} = 0, \forall i$),
- $h_i$ are the local magnetic field coefficients,
- $s_i$ takes values in {-1, 1}.

The goal in solving an Ising problem is often to find the spin configuration, called **ground state**, that minimizes the global energy of the system.

The [Simulated Bifurcation algorithm](simulated_bifurcation_algorithm.md), first introduced by Toshiba experts, takes its inspiration the quantum theory to specifically tackle Ising problems and find local minima for its objective function in a polynomial time.

Mapping an Ising problem to a [QUBO](#quadratic-unconstrained-binary-optimization-qubo) problem involves translating the Ising model into an equivalent quadratic unconstrained binary optimization (QUBO) problem. This is done by converting the spin variables $s_i$ into binary variables $x_i$ that can take values of 0 or 1. Of course, the reverse conversion from QUBO to Ising is just as straightforward. Thus, solving an Ising problem or a QUBO problem are two completely equivalent tasks, making the Simulated Bifurcation algorithm relevant for a wider range of quadratic optimization problems.
