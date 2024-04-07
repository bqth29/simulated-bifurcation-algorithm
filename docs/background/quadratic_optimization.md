# Quadratic optimization

Quadratic optimization, also known as quadratic programming (QP), is a type of mathematical optimization problem that deals with optimizing a quadratic objective function subject to linear equality and/or inequality constraints. In computer science, these problems arise in various fields such as machine learning, finance, engineering, and operations research. They are often solved using specialized algorithms and software packages.

## Mathematical definition

In mathematical terms, a quadratic optimization problem can be formulated as follows:

$$\text{Minimize } \frac{1}{2} x^T Q x + c^T x$$

$$\text{Subject to: } Ax = b \text{ and } Gx \leq h$$

Where:
- $x$ is a vector of decision variables that we want to find,
- $Q$ is a symmetric matrix of coefficients of the quadratic terms,
- $c$ is a vector of coefficients of the linear terms,
- $A$ is a matrix representing the coefficients of the equality constraints,
- $b$ is a vector representing the right-hand side of the equality constraints,
- $G$ is a matrix representing the coefficients of the inequality constraints,
- $h$ is a vector representing the right-hand side of the inequality constraints.

The goal is to find the values of $x$ that minimize the objective function while satisfying the given constraints.

## Unconstrained combinatorial quadratic optimization

In a quadratic programming problem, the decision variables can be continuous, integer, or binary. When the decision variables are continuous, solving QP becomes a convex optimization problem, and efficient algorithms such as simplex or interior-point methods can be used to find the global optimum in polynomial time. However, when the decision variables are integer or binary, the problem becomes significantly harder to solve and enters the frame of combinatorial optimization.

$$\text{Minimize } \frac{1}{2} x^T Q x + c^T x$$

### NP-Hardness

Quadratic programming is generally regarded as a type of NP (nondeterministic polynomial time) problem due to its computational complexity under certain circumstances. Here's why:

1. Decision Variables: In a quadratic programming problem, the decision variables can be continuous, integer, or binary. When the decision variables are continuous, solving QP becomes a convex optimization problem, and efficient algorithms such as interior-point methods can be used to find the global optimum in polynomial time. However, when the decision variables are integer or binary, the problem becomes significantly harder to solve.

2. Integer and Binary Variables: When decision variables are constrained to be integer or binary, as is often the case in combinatorial optimization problems, QP becomes a more complex problem. This is because the feasible solution space becomes discrete, leading to a combinatorial explosion in the number of potential solutions to be explored.

3. NP-Hardness: Many combinatorial optimization problems can be formulated as QP with integer or binary variables. Examples include the traveling salesman problem, the knapsack problem, and graph partitioning problems. These problems are known to be NP-hard, meaning that no known algorithm can solve them optimally in polynomial time. Instead, approximate algorithms or heuristics are often used to find near-optimal solutions.

4. Transformation: Some NP-hard problems can be transformed into QP instances, making QP a subclass of NP problems. For example, QUBO (Quadratic Unconstrained Binary Optimization) problems, which are a special case of QP with binary variables, can be formulated to represent various NP-hard problems.

While QP itself may not always be NP-hard, its variants with integer or binary variables often are. Therefore, quadratic programming is generally regarded as an NP problem when considering its more general formulations and applications, especially in the context of combinatorial optimization.

### Common quadratic models

Introduce with integer values.

#### Quadratic Unconstrained Binary Optimization (QUBO)

[QUBO](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) stands for Quadratic Unconstrained Binary Optimization. It is a specific form of quadratic optimization problem where the decision variables are binary, meaning they can only take on values of 0 or 1. QUBO problems have a quadratic objective function and may include linear equality and/or inequality constraints.

The general form of a QUBO problem is:

$$\text{Minimize } \sum_{i=1}^{N} \sum_{j=1}^{N} Q_{ij} x_i x_j = x^{T}Qx$$

Where:
- $x_i \in \{0, 1\}$ represents the binary decision variables,
- $Q_{ij}$ represents the coefficients of the quadratic terms.

QUBO problems have applications in various fields, including combinatorial optimization, machine learning, cryptography, and quantum computing. They are particularly relevant in the context of optimization problems that involve binary decisions, such as graph partitioning, maximum clique, Boolean satisfiability problems (SAT), and many others.

Thanks to its simple, compact form, QUBO is often seen as a central model for modeling a wide range of combinatorial quadratic problems and tackling numerous real-world optimization challenges.

Thus, solving QUBO problems efficiently is important and there are specialized heuristic algorithms and techniques designed to address them. Additionally, with the emergence of quantum computing, there is growing interest in using quantum annealing and other quantum optimization techniques, including the [Simulated Bifurcation algorithm](simulated_bifurcation_algorithm.md), to solve QUBO problems more efficiently.

#### Ising model

An Ising problem is a type of mathematical model used to describe interactions between spins in a physical system, such as a collection of magnetic dipoles. It's named after the physicist Ernst Ising who first introduced it in the context of ferromagnetism.

In the Ising model, each spin can be in one of two states, typically represented as +1 (spin up) or -1 (spin down). The energy of the system is determined by the interactions between neighboring spins. The total energy of the system can be expressed as a quadratic function of the spin variables.

The Ising model can be represented mathematically using the following energy function:

$$E(\mathbf{s}) = -\frac{1}{2}\sum_{i,j} J_{ij} s_i s_j + \sum_{i} h_i s_i$$

Where:
- $\mathbf{s} = (s_1, s_2, ..., s_n)$ represents the spin configuration,
- $J_{ij}$ are the coupling coefficients between spins $s_i$ and $s_j$,
- $h_i$ are the local magnetic field coefficients,
- $s_i$ takes values in {-1, 1}.

The goal in solving an Ising problem is often to find the spin configuration that minimizes the energy of the system.

Mapping an Ising problem to a [QUBO](#quadratic-unconstrained-binary-optimization-qubo) problem involves translating the Ising model into an equivalent quadratic unconstrained binary optimization (QUBO) problem. This is done by converting the spin variables $s_i$ into binary variables $x_i$ that can take values of 0 or 1. Of course, the reverse conversion from QUBO to Ising is just as straightforward.

Thus, solving an Ising problem or a QUBO problem are two completely equivalent tasks.

#### Integer Quadratic Optimization (IQP)

Using base 2 -> map to QUBO
