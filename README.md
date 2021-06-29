# simulated-bifurcation-algorithm
Python implementation of a Simulated Bifurcation algorithm in order to appromize the optimal assets allocation for a portfolio.

## Install required packages
This algorithm relies on several Python packages. To install them all, execute the following command : ```python -m pip install -r requirements.txt```.

## Run a simulation
To run a simulaton, you simply need to run the ```__main__.py``` file.

### How does it work ?

#### Object
For every simulation, we create a ```Simulated_Bifurcation``` object which contains all the necessary data to optimize the portfolio. All the parameters are set by default but you can choose your own ones. These parameters are : 

*Data*

- ```covariance_filename``` (```str```): the filename of the .csv file containing the covariance data. 
- ```expected_return_filename``` (```str```): the filename of the .csv file containing the exected return vector data. 
- ```risk_coefficient``` (```int / float```, default: ```1```): the risk aversion coefficient of the user with respect to the volatility of its portfolio. 
- ```number_of_bits``` (```int```, default: ```1```): the number of bits with which the value of every single asset can be written. This number of bits is set by default to one as it has the best accuracy (over 99%) in comparison with the real optimal portfolio. This very situation represents an evenly spread sub-portfolio.
- ```date``` (```str```,  default: ```2021-03-01```): the date of the last *S&P500* data update. To be sure that the date you provide works, verify it appears in the first columns of the ```mu.csv``` file in the ```data``` folder.
- ```assets_list``` (```str list```): the list of the *S&P500* assets taken in account for the simulation. You can see which ones exist by checking the ```assets.py``` file in the ```data``` folder.

*Simulation and Hamiltonian* (the following parameters were chosen as a very good trade-off between the time of the simulation and its efficiency so you should *not* change them to keep these performances)

- ```time_step``` (```float```, default: ```0.01```): the time (in seconds) of a time step of the time-discretized Hamiltonian problem. 
- ```simulation_time``` (```int```, default: ```600```): the time span (in seconds) of time-discretized Hamiltonian problem simulation. 
- ```kerr_constant``` (```int / float```, default: ```1```): the Kerr constant of the Hamiltonian. 
- ```detuning_frequency``` (```int / float```, default: ```1```): the detuning frequency constant of the Hamiltonian. 
- ```pressure``` (```(float -> float) function```, default: ```lambda t: 0.01 * t```): the pumping function with respect to time. It must use a very slow evolution.
- ```symplectic_parameter``` (```int```, default: ```2```): the symplectic parameter used for the Euler's scheme. It is recommanded to keep its value between 2 and 5.

#### Data
The data used comes from the latest update of the *S&P500*. 
