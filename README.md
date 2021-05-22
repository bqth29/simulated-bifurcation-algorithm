# simulated-bifurcation-algorithm
A Python implementation of a Simulated Bifurcation algorithm in order to appromize the optimal assets allocation for a portfolio.

## Requirements
You will need some Python packages to run the _Similated Bifurcation_ algorithm. They are: ```numpy```, ```pandas```, ```plotly```, ```statistics``` and ```time```. To add a package to your Python environment, simply execute the following command: ```pip install [package_name]```.

## Simulation
To run a simulation, you must execute the ```__main__.py``` file. By default, the whole asset list will be considered. If you want to only select some assets, just add ```assets_list = [list_of_your_assets]``` in the arguments of the ```simulated_bifurcation``` function, still in ```__main__.py```.

Besides, the value of the different parameters id defined by default but can also be set manually as arguments of the same ```simulated_bifurcation``` function.
