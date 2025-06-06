# Vein_refinement
This repository contains python codes to simulate the signalling network involved in vein refinement and described in the paper "Signalling-dependent refinement of cell fate choice during tissue remodelling".
***
**Simulation on a wing tissue template**

The scripts  ```Run_sim_WT_wing.py``` and ```Run_sim_Dumpy_wing.py``` perform a simulation on segmented wild-type (Fig.5B-C) and dumpy mutant (Fig.5F) wings  tissue template. To run ```Run_sim_WT_wing.py``` one has to import the cell class ```cell_wing.py```, ```tissue_miner.py``` and ```tissue_miner_tools.py```, which contain classes and functions to extract data from segmented wing movies. Data from wild-type and Dumpy mutant wing movies can be found  at https://zenodo.org/records/14064618. There are lines of code that can be commented/uncommented to simulate optogenetic veins (Fig.7A) and different ways to assign Delta and Notch to daughter cells at division (Fig.S5H). To run the simulation it is sufficient to save the folder containing the segmented movie, ```tissue_miner_tools.py```, ```cell_wing.py```, and ```tissue_miner.py``` in the same directory as ```Run_sim_WT_wing.py``` or ```Run_sim_Dumpy_wing.py```.
***
**Simulation on hexagonal lattice**

The scripts ```cell.py```,  ```run_sim.py```,  ```plot_functions.py```, and ```generate_graphs.py```, are used to simulate of vein refinement on a hexagonal lattice.  ```run_sim.py``` contains functions to update cell variables at each iteration of the simulation. ```cell.py``` defines the cell class. ```plot_functions.py``` contains functions to display hexagonal packings, animations, and profiles of different quantities. ```generate_graphs.py``` calls functions defined in these files to run simulations of Wild-Type (Fig.4B,6B,S6A-B), optogenetic cross veins (Fig.4F), Delta DN (Fig.6D,S6C-D), Nintra (Fig.6F,S6E-F), TkvCA (Fig.6H,S6G-H) conditions, and the various phase diagrams (Fig.4C-D, S4H-I'). It also generates Wild-Type simulations with inhomogeneous initial conditions (Fig.4E), with a fixed Delta pattern (Fig.S4A), and with different values of Kcis and Ktrans (Fig.S4B).

Analogously, the scripts ```run_sim_NDtrans_degradation.py```,  ```plot_functions_NDtrans_degradation.py```, and ```generate_graphs_NDtrans_degradation.py``` can be used to simulate the version of the model with degradation of intracellular Notch and Delta-Notch complexes after cleavage (Fig.S4K-L).

The script ```activation_2nd_neighbours.py``` run simulations of the version of the model with activation extended to second nearest neighbours (Fig.S4J-J').

The script ```run_sim_u_prime.py```run simulations of the version of the model where DSRF (represented by u') is inhibited by vein cells (Fig.S4C).
