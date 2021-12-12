import torch
from typing import Tuple, final
from time import time

# OBJECTS

class Ising():

    def __init__(self, J: torch.Tensor, h: torch.Tensor, assert_parameters: bool = True) -> None:

        """
        Class constructor.
        """
        
        self.J                  = J
        self.h                  = h

        self.dimension, _       = J.size()
        
        self.ground_state       = None 

        # Check parameters

        if assert_parameters:
            self.__assert__()

    def __assert__(self) -> None:

        """
        Asserts that the parameters of the object follow the right pattern.
        """  

        Jx, Jy = self.J.size()
        hx, hy = self.h.size()

        # Checking types
        assert isinstance(self.J, torch.Tensor), f"WARNING: The type of J must be a numpy array, instead got {type(self.J)}"
        assert isinstance(self.h, torch.Tensor), f"WARNING: The type of h must be a numpy array, instead got {type(self.h)}"

        # Checking dimensions
        assert Jx == Jy, f"WARNING: J must be a square matrix, instead got {(Jx, Jy)}"
        assert hx == Jx, f"WARNING: The dimension of h must fits J's, instead of {Jx} got {hx}"
        assert hy == 1, f"WARNING: h must be a column vector with dimensions of this pattern: (n,1), instead got {(hx, hy)}"

        # Checking J's properties
        assert torch.allclose(self.J, self.J.t()), "WARNING: J must be symmetric"
        assert not torch.any(self.J == torch.zeros([Jx, Jy])), "WARNING: J must not have null elements"   

    def energy(self) -> float:

        if self.ground_state is None:
            return None

        else:
            energy = -0.5 * self.ground_state.t() @ self.J @ self.ground_state + self.ground_state.t() @ self.h
            return energy.item()     

    def get_ground_state(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        window_size: int = 35,
        sampling_period: int = 50,
        display_time: bool = True,
    ) -> None:

        """
        Finds the optimal ground state with a symplectic Euler's scheme.
        """  
        
        start_time = time()

        X, _ = symplectic_euler_scheme(
            self,
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            window_size,
            sampling_period
        )

        end_time = time()

        simulation_time = round(end_time - start_time, 3)

        if display_time:    

            print(f"Run in {simulation_time} seconds.")

        self.ground_state = torch.sign(X)

class SBModel():

    def __to_Ising__(self) -> Ising:
        pass

    def __from_Ising__(self, ising: Ising) -> None:
        pass
    
    @final
    def optimize(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        window_size: int = 35,
        sampling_period: int = 50,
        display_time: bool = True,
    ) -> None:

        ising_equivalent = self.__to_Ising__()
        ising_equivalent.get_ground_state(
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            window_size,
            sampling_period,
            display_time,
        )
        self.__from_Ising__(ising_equivalent)     

# FUNCTIONS

def stop_criterion(window: torch.Tensor, spins: torch.Tensor) -> Tuple[bool, torch.Tensor]:

    """
    Determines whether the Euler's scheme must stop using a rolling window.
    """  
    
    window = torch.roll(window, -1, dims = 1) # Shift all columns to the left
    window[:, -1] = spins                  # Replace the last column with the spins
    mean = torch.mean(window, dim = 1)    # Computes the mean

    return not torch.allclose(mean, spins, atol = 1e-8), window     

def symplectic_euler_scheme(
    ising: Ising,
    detuning_frequency: float = 1,
    kerr_constant: float = 1,
    pressure = lambda t: 0.01 * t,
    time_step: float = 0.01,
    symplectic_parameter: int = 2,
    window_size: int = 35,
    sampling_period: int = 60,
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Symplectic Euler scheme computing the evolution of the oscillators of a KNPO network.
    Ends using a rolling-window stop criterion.
    """

    # Parameters initialization

    X = torch.zeros([ising.dimension, 1], dtype=torch.float64) 
    Y = torch.zeros([ising.dimension, 1], dtype=torch.float64) 

    xi0 = 0.7 * detuning_frequency / (torch.std(ising.J) * (ising.dimension)**(1/2))

    window = torch.zeros([ising.dimension, window_size], dtype=torch.float64)
    symplectic_time_step = time_step / symplectic_parameter

    run = True
    step = 0

    # Simulation

    while run:

        factor = pressure(step * time_step) - detuning_frequency

        if factor > 0:

            # Symplectic loops

            for _ in range(symplectic_parameter):

                X += symplectic_time_step * detuning_frequency * Y
                Y -= symplectic_time_step * (kerr_constant * X**3 - factor * X)  

            Y += time_step * xi0 * (ising.J @ X - 2 * pow(factor / kerr_constant, .5) * ising.h)

            # Check the stop criterion

            if step % sampling_period == 0:

                run, window = stop_criterion(window, torch.sign(X).t()[0])

        step += 1      

    return X, Y