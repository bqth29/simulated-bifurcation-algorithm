from typing import Tuple, final
import numpy as np
from time import time

# Classes

class Ising():

    """
    A class to implement Ising models.

    ...

    Attributes
    ----------
    J : numpy.ndarray
        spin interactions matrix (must be semi-definite positive)
    h : numpy.ndarray
        magnectic field effect vector
    dimension : int
        number of spins
    ground_state : numpy.ndarray   
        vector of spins orientation to minimize the energy
    optimization_logs : dict   
        data about the optimization of the model    
    """

    def __init__(self, J: np.ndarray, h: np.ndarray, assert_parameters: bool = True) -> None:

        """
        Constructs all the necessary attributes for the Ising object.

        Parameters
        ----------
            J : numpy.ndarray
                spin interactions matrix (must be semi-definite positive)
            h : numpy.ndarray
                magnectic field effect vector
            assert_parameters : bool, optional
                check the format of the inputs (default is True)
        """
        
        self.J                 = J
        self.h                 = h
        
        self.dimension         = J.shape[0]
        
        self.ground_state      = None
        self.optimization_logs = dict() 

        # Check parameters

        if assert_parameters:
            self.__assert__()

    def __assert__(self) -> None:

        """
        Checks the format of the attributes.

        Returns
        -------
        float
        """  

        # Checking types
        assert isinstance(self.J, np.ndarray), f"WARNING: The type of J must be a numpy array, instead got {type(self.J)}"
        assert isinstance(self.h, np.ndarray), f"WARNING: The type of h must be a numpy array, instead got {type(self.h)}"

        # Checking dimensions
        assert self.J.shape[0] == self.J.shape[1], f"WARNING: J must be a square matrix, instead got {self.J.shape}"
        assert self.h.shape[0] == self.J.shape[0], f"WARNING: The dimension of h must fits J's, instead of {self.J.shape[0]} got {self.h.shape[0]}"
        assert self.h.shape[1] == 1, f"WARNING: h must be a column vector with dimensions of this pattern: (n,1), instead got {self.h.shape}"

        # Checking J's properties
        assert np.allclose(self.J, self.J.T), "WARNING: J must be symmetric"
        assert not np.any(self.J == np.zeros(self.J.shape)), "WARNING: J must not have null elements"   

    def energy(self) -> float:

        """
        Computes the Ising energy of the model.

        Returns
        -------
        float
        """

        if self.ground_state is None:
            return None

        else:
            energy = -0.5 * self.ground_state.T @ self.J @ self.ground_state + self.ground_state.T @ self.h
            return energy[0][0]     

    def get_ground_state(
        self,
        detuning_frequency: float = 1.0,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 35,
        sampling_period: int = 50,
    ) -> None:

        """
        Determines the ground state of the Ising model using a symplectic Euler's scheme.
        The computation is based on the quantum mechanics of Kerr-nonlinear parametric oscillators (KPO).
        The `ground_state` attribute is modified in place.

        Parameters
        ----------
            detuning_frequency : float, optional
                detuning frequency of the KPO (default is 1.0)
            kerr_constant : float, optional
                value of the Kerr constant (default is 1.0)
            pressure : function, optional
                pumping pressure allowing adiabatic evolution (default is t -> 0.01 * t)
            time_step : float, optional
                step size for the time discretization (default is 0.01)    
            symplectic_parameter : int, optional
                symplectic parameter for the Euler's scheme (default is 2)    
            convergence_threshold : int, optional
                number of consecutive identical spin sampling considered as a proof of convergence (default is 35) 
            sampling_period : int, optional
                number of time steps between two spin sampling (default is 50)          

        Returns
        -------
        None        
        """

        X, _, data = symplectic_euler_scheme(
            self,
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period
        )

        self.optimization_logs = data

        self.ground_state = np.sign(X)

class SBModel():

    """
    A class to implement Ising problems.
    """

    def __to_Ising__(self) -> Ising:

        """
        Generate the equivalent Ising model of the problem.

        Returns
        -------
        Ising
        """

        pass

    def __from_Ising__(self, ising: Ising) -> None:

        """
        Retrieves information from the optimized equivalent Ising model.
        Modifies the object's attributes in place.

        Parameters
        ----------
            ising : Ising
                equivalent Ising model of the problem

        Returns
        -------
        None
        """

        pass
    
    @final
    def optimize(
        self,
        detuning_frequency: float = 1,
        kerr_constant: float = 1,
        pressure = lambda t: 0.01 * t,
        time_step: float = 0.01,
        symplectic_parameter: int = 2,
        convergence_threshold: int = 35,
        sampling_period: int = 50,
    ) -> None:

        """
        Optimizes the problem by determining the ground state of the equivalent Ising model.
        The ground state is found using a symplectic Euler's scheme.
        The computation is based on the quantum mechanics of Kerr-nonlinear parametric oscillators (KPO).
        Retrieves information linked to thi soptimization and updates the object's attributes.

        Parameters
        ----------
            detuning_frequency : float, optional
                detuning frequency of the KPO (default is 1.0)
            kerr_constant : float, optional
                value of the Kerr constant (default is 1.0)
            pressure : function, optional
                pumping pressure allowing adiabatic evolution (default is t -> 0.01 * t)
            time_step : float, optional
                step size for the time discretization (default is 0.01)    
            symplectic_parameter : int, optional
                symplectic parameter for the Euler's scheme (default is 2)    
            convergence_threshold : int, optional
                number of consecutive identical spin sampling considered as a proof of convergence (default is 35) 
            sampling_period : int, optional
                number of time steps between two spin sampling (default is 50)         

        Returns
        -------
        None        
        """

        ising_equivalent = self.__to_Ising__()
        ising_equivalent.get_ground_state(
            detuning_frequency,
            kerr_constant,
            pressure,
            time_step,
            symplectic_parameter,
            convergence_threshold,
            sampling_period
        )
        self.__from_Ising__(ising_equivalent)              

# Euler Scheme

def symplectic_euler_scheme(
    ising: Ising,
    detuning_frequency: float = 1,
    kerr_constant: float = 1,
    pressure = lambda t: 0.01 * t,
    time_step: float = 0.01,
    symplectic_parameter: int = 2,
    convergence_threshold: int = 35,
    sampling_period: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Use of a symplectic Euler's scheme to simulate the evolution of Kerr-nonlinear parametric oscillators (KPO) network.
    The oscillators are initialized to zero.

    Parameters
    ----------
        ising : Ising
            the Ising model to optimize 
        detuning_frequency : float, optional
            detuning frequency of the KPO (default is 1.0)
        kerr_constant : float, optional
            value of the Kerr constant (default is 1.0)
        pressure : function, optional
            pumping pressure allowing adiabatic evolution (default is t -> 0.01 * t)
        time_step : float, optional
            step size for the time discretization (default is 0.01)    
        symplectic_parameter : int, optional
            symplectic parameter for the Euler's scheme (default is 2)    
        convergence_threshold : int, optional
            number of consecutive identical spin sampling considered as a proof of convergence (default is 35) 
        sampling_period : int, optional
            number of time steps between two spin sampling (default is 50)              

    Returns
    -------
        X : numpy.ndarray  
        Y : numpy.ndarray   
        data : dict    
    """

    # Parameters initialization

    X = np.zeros((ising.dimension, 1)) 
    Y = np.zeros((ising.dimension, 1)) 

    xi0 = 0.7 * detuning_frequency / (np.std(ising.J) * (ising.dimension)**(1/2))

    current_spins = np.zeros((ising.dimension, 1))
    equal_streak  = 0

    symplectic_time_step = time_step / symplectic_parameter

    step = 0
    zero_step = 0

    # Simulation

    start_time = time()

    while equal_streak < convergence_threshold - 1:

        factor = pressure(step * time_step) - detuning_frequency

        if factor > 0:

            # Symplectic loops

            for _ in range(symplectic_parameter):

                X += symplectic_time_step * detuning_frequency * Y
                Y -= symplectic_time_step * (kerr_constant * X**3 - factor * X)  

            Y += time_step * xi0 * (ising.J @ X - 2 * pow(factor / kerr_constant, .5) * ising.h)

            # Check the stop criterion

            if step % sampling_period == 0:

                spins = np.sign(X)
                
                if np.allclose(spins, current_spins):
                    equal_streak += 1
                else:
                    equal_streak = 0
                    current_spins = spins.copy()

        else: zero_step +=1
        step += 1  

    end_time = time()

    data = dict(
        time = round(end_time - start_time, 3),
        steps = step,
        non_zero_steps = step - zero_step,
        detuning_frequency = detuning_frequency,
        kerr_constant = kerr_constant,
        pressure_slope = pressure(1),
        time_step = time_step,
        symplectic_parameter = symplectic_parameter,
        convergence_threshold = convergence_threshold,
        sampling_period = sampling_period,
    )        

    return X, Y, data