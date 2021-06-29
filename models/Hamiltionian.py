class Hamiltonian():

    def __init__(
        self,
        kerr_constant,
        detuning_frequency,
        risk_coefficient,
        pressure
    ):

        self.kerr_constant = kerr_constant
        self.detuning_frequency = detuning_frequency
        self.risk_coefficient = risk_coefficient
        self.pressure = pressure