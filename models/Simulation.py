class Simulation():

    def __init__(
        self,
        time_step,
        simulation_time,
        symplectic_parameter
    ):

        self.time_step = time_step
        self.simulation_time = simulation_time
        self.symplectic_parameter = symplectic_parameter