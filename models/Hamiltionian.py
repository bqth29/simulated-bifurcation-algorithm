class Hamiltonian():

    """
    Hamiltonian used to solve an Ising problem with a symplectic
    Euler's scheme.
    """

    def __init__(
        self,
        kerr_constant : float = 1,
        detuning_frequency : float = 1,
        pressure = lambda t: 0.01 * t,
        time_step : float = 0.01,
        simulation_time : int = 600,
        symplectic_parameter : int = 2
    ):

        self.kerr_constant = kerr_constant
        self.detuning_frequency = detuning_frequency
        self.pressure = pressure
        self.time_step = time_step
        self.simulation_time = simulation_time
        self.symplectic_parameter = symplectic_parameter