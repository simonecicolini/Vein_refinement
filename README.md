# Vein_refinement
This repository contains python codes to simulate the signalling network involved in vein refinement and described in the paper "Signalling-dependent refinement of cell fate choice during tissue remodelling".
***
**Simulation on a wing tissue template**

The scripts  ```Run_sim_WT_wing.py``` and ```Run_sim_Dumpy_wing.py``` perform a simulation on segmented wild-type and dumpy mutant wings tissue template. To run ```Run_sim_WT_wing.py``` one has to import the cell class ```cell_wing.py```, ```tissue_miner.py``` and ```tissue_miner_tools.py```, which contain classes and functions to extract data from segmented wing movies. Data from wing movies can be found in the folder "wing_movies" at https://zenodo.org/records/14064618 , which contains wild-type and Dumpy mutant wings. There are lines of code that can be commented/uncommented to simulate optogenetic veins. To run the simulation it is sufficient to save the folder containing the segmented movie, ```tissue_miner_tools.py```, ```cell_wing.py```, and ```tissue_miner.py``` in the same directory as ```Run_sim_WT_wing.py``` or ```Run_sim_Dumpy_wing.py```.
***
**Simulation on hexagonal lattice**

The scripts ```cell.py```,  ```run_sim.py```,  ```plot_functions.py```, and ```generate_graphs.py```, are used to simulate of vein refinement on a hexagonal lattice.  ```run_sim.py``` contains functions to update cell varibales at each iteration of the simulation. ```cell.py``` defines the cell class. ```plot_functions.py``` contains functions to display hexagonal packings, animations, and profiles of different quantities. ```generate_graphs.py``` calls functions defined in these files to run simulations of Wild-Type, optogenetic cross veins, Delta DN, Nintra, and TkvCA conditions.

Analogously, the scripts ```run_sim_NDtrans_degradation.py```,  ```plot_functions_NDtrans_degradation.py```, and ```generate_graphs_NDtrans_degradation.py``` can be used to simulate the version of the model with degradation of intracellular Notch and Delta-Notch complexes after cleavage.

