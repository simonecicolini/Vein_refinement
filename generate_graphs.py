from cell import cell
import matplotlib.pyplot as plt
from run_sim import run_sim
from plot_functions import save_snapshots
from plot_functions import save_profiles_u_notch_reporter
# from plot_functions import save_spatial_profiles
# from plot_functions import save_temporal_profiles
import numpy as np

import os
from pathlib import Path

# =============================================================================
# GENERATE CELL ARRAY (HEXAGONAL PACKING)
# =============================================================================
#number of cells in both distances
N_x=32
N_y=24
#distance between cells in the y direction, in um
Delta_y = 4
#size of one side of an he
side_length = Delta_y/np.sqrt(3)

# positions of the cells on a hexagonal lattice
indices = np.array([  [i, j]
             for i in range(N_x)
               for j in range(N_y)
              ])
positions = side_length * np.array([  [3*i/2, np.sqrt(3)/2.0*((i%2) +2*j)] 
                                    for i in range(N_x)
                                    for j in range(N_y)
                                    ])
u_array = np.zeros((N_x, N_y))

cells = {k: cell(pos, u, index)
         for k, (pos, u, index)
         in enumerate(zip(positions, u_array.flatten(), indices))}

# construct list of nearest neighbours
for k, c in cells.items():
    if (k//N_y)%2==1:
        c.neighbours=[(N_y*(k//N_y)+(k-1)%N_y)%(N_x*N_y),
                      (k+N_y)%(N_x*N_y),
                      (N_y*((k+N_y)//N_y)+(k+1+N_y)%N_y)%(N_x*N_y),
                      (N_y*(k//N_y)+(k+1)%N_y)%(N_x*N_y),
                      ( N_y*((k-N_y)//N_y)+(k+1-N_y)%N_y)%(N_x*N_y),
                      (k-N_y)%(N_x*N_y)]
    else:
        c.neighbours=[(N_y*(k//N_y)+(k-1)%N_y)%(N_x*N_y),
                      (N_y*((k+N_y)//N_y)+(k+N_y-1)%N_y)%(N_x*N_y),
                      (N_y*((k+N_y)//N_y)+(k+1+N_y-1)%N_y)%(N_x*N_y),
                      (N_y*(k//N_y)+(k+1)%N_y)%(N_x*N_y),
                      ( N_y*((k-N_y)//N_y)+(k-N_y)%N_y)%(N_x*N_y),
                      (N_y*((k-N_y)//N_y)+(k-1-N_y)%N_y)%(N_x*N_y)]
 
dt = 0.0167 # 1 minute
t_0 = 21 #initial time  in hours - for comparison with experiments
N_times = 1500
fraction_saved = 10
print('total simulation time: ', dt*N_times)
# timescales
tau_Notch=6#time scale of Notch protein relaxation in hours
tau_u = 2 #time scale of u dynamics in hours
tau_reporter = 8 #degradation time scale of Notch signalling reporter
#Model parameters
JA = 0.015
JI = 0.35
r = 0.5
Kcis= 1500
Ktrans = 1800
N_low = 0.3
N_high = 2.2
D_high = 1 
Sstar = 0.25
alphaS = 0.025
S0 = 0.4
           
# # =============================================================================
# # WILD-TYPE REFERENCE SIMULATION
# # =============================================================================
plt.rcParams.update({'font.size': 12})
print("running WT")
## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=10
sigma_rep=7
for k, c in cells.items():
    c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_free = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0

## run simulation       
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved)

#generate simulation snapshots
dir_path = os.getcwd()
Path(dir_path+"/WT/snapshots/").mkdir(parents=True, exist_ok=True)  
        
times_to_plot=[i*60//fraction_saved for i in range(24)] #60 minutes is 1 hour                    
save_snapshots(cells,
                 fraction_saved,
                 side_length,
                 delta_saved,
                 notch_saved,
                 signal_saved,
                 u_saved,
                 notch_reporter_saved,
                 times_to_plot,
                 t_0, 
                dir_path+"/WT/snapshots/")

#generate plots
Path(dir_path+"/WT/plots_FigS6/").mkdir(parents=True, exist_ok=True)                                        
 
y_positions_all = [c.position[1] for c in cells.values()]
y_positions = np.unique(y_positions_all)
initial_time=21
final_time=43
one_hour=60//fraction_saved
times=[0,
        3*one_hour,
        8*one_hour,
        13*one_hour,
        18*one_hour,
        22*one_hour]

save_profiles_u_notch_reporter(cells,
                u_saved,
                notch_reporter_saved,
                y_positions,
                y_positions_all,
                center_y,
                times,
                fraction_saved,
                dt,
                initial_time,
                final_time,
                t_0,
                dir_path+"/WT/plots_FigS6/")       
                      
# # =============================================================================
# # WILD-TYPE REFERENCE SIMULATION - WITH CROSS-VEIN
# # =============================================================================
 
print("running WT with cross vein")
## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
for k, c in cells.items():
    c.u=0
    c.delta_tot=0
    c.delta_free=0
    if np.abs(c.position[1]-center_y-20)<10:
        c.u = 1
        c.delta_tot = 1
        c.delta_free = 1
    if np.abs(c.position[1]-center_y+20)<10:
        c.u = 1
        c.delta_tot = 1
        c.delta_free = 1
    if ((c.position[1]<center_y+20) and (c.position[1]>center_y-20)
        and np.abs(c.position[0]-center_x)<12):
        c.u = 1
        c.delta_tot = 1
        c.delta_free= 1
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0


N_times = 2000
## run simulation       
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved)
                        
## generate simulation snapshots
dir_path = os.getcwd()
Path(dir_path+"/WT_crossvein/snapshots").mkdir(parents=True, exist_ok=True)

times_to_plot=[i*60//fraction_saved for i in range(33)]
save_snapshots(cells,
                  fraction_saved,
                  side_length,
                  delta_saved,
                  notch_saved,
                  signal_saved,
                  u_saved,
                  notch_reporter_saved,
                  times_to_plot,
                  t_0,
                  dir_path+"/WT_crossvein/snapshots/")


# # =============================================================================
# # PERTURBATION - DELTA DN
# # =============================================================================

print("running DeltaDN")

## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=15
for k, c in cells.items():
    c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot =0
    c.delta_free = 0
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0


## run simulation    
N_times=1500   
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times,
                              fraction_saved,
                              noDelta=True)
                              
# generate simualation snapshots
dir_path = os.getcwd()
Path(dir_path+"/DeltaDN/snapshots/").mkdir(parents=True, exist_ok=True)
times_to_plot=[i*60//fraction_saved for i in range(24)] #60 minutes is 1 hour                    
save_snapshots(cells,
                 fraction_saved,
                 side_length,
                 delta_saved,
                 notch_saved,
                 signal_saved,
                 u_saved,
                 notch_reporter_saved,
                 times_to_plot,
                 t_0, 
                dir_path+"/DeltaDN/snapshots/")

#generate plots
Path(dir_path+"/DeltaDN/plots_FigS6/").mkdir(parents=True, exist_ok=True)                                         
y_positions_all = [c.position[1] for c in cells.values()]
y_positions = np.unique(y_positions_all)
initial_time=16.5
final_time=42
one_hour=60//fraction_saved
times=[0,
        5*one_hour,
        10*one_hour,
        15*one_hour,
        21*one_hour]

save_profiles_u_notch_reporter(cells,
                u_saved,
                notch_reporter_saved,
                y_positions,
                y_positions_all,
                center_y,
                times,
                fraction_saved,
                dt,
                initial_time,
                final_time,
                t_0,
                dir_path+"/DeltaDN/plots_FigS6/")       

# # =============================================================================
# # PERTURBATION - Nintra
# # =============================================================================

print("running Nintra")

## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=15
sigma_rep=7
for k, c in cells.items():
    c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_free = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0

## run simulation       
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved,
                              extraSignal=True,
                              extraSignalValue=0.45)
                              
## generate simulation snapshots
dir_path = os.getcwd()
Path(dir_path+"/Nintra/snapshots/").mkdir(parents=True, exist_ok=True)
times_to_plot=[i*60//fraction_saved for i in range(24)] #60 minutes is 1 hour                    
save_snapshots(cells,
                 fraction_saved,
                 side_length,
                 delta_saved,
                 notch_saved,
                 signal_saved,
                 u_saved,
                 notch_reporter_saved,
                 times_to_plot,
                 t_0, 
                dir_path+"/Nintra/snapshots/")

#generate plots
Path(dir_path+"/Nintra/plots_FigS6/").mkdir(parents=True, exist_ok=True)                                         
y_positions_all = [c.position[1] for c in cells.values()]
y_positions = np.unique(y_positions_all)
initial_time=18.6
final_time=33
one_hour=60//fraction_saved
times=[0,
        4*one_hour,
        8*one_hour,
        12*one_hour]

save_profiles_u_notch_reporter(cells,
                u_saved,
                notch_reporter_saved,
                y_positions,
                y_positions_all,
                center_y,
                times,
                fraction_saved,
                dt,
                initial_time,
                final_time,
                t_0,
                dir_path+"/Nintra/plots_FigS6/")       



# # =============================================================================
# # PERTURBATION - TkvCA
# # =============================================================================

print("running TkvCA")

## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=15
sigma_rep=7
for k, c in cells.items():
    c.u= np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_free = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0


## run simulation       
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved,
                              extraVeinActivation=True,
                              extraVeinActivationValue=0.5)
                              
## generate simulation snapshots
dir_path = os.getcwd()
Path(dir_path+"/TkvCA/snapshots/").mkdir(parents=True, exist_ok=True)
times_to_plot=[i*60//fraction_saved for i in range(24)] #60 minutes is 1 hour                    
save_snapshots(cells,
                 fraction_saved,
                 side_length,
                 delta_saved,
                 notch_saved,
                 signal_saved,
                 u_saved,
                 notch_reporter_saved,
                 times_to_plot,
                 t_0, 
                dir_path+"/TkvCA/snapshots/")

#generate plots
Path(dir_path+"/TkvCA/plots_FigS6/").mkdir(parents=True, exist_ok=True)                                         
y_positions_all = [c.position[1] for c in cells.values()]
y_positions = np.unique(y_positions_all)
initial_time=17.5
final_time=37
one_hour=60//fraction_saved
times=[0,
        4*one_hour,
        8*one_hour,
        12*one_hour,
        16*one_hour]

save_profiles_u_notch_reporter(cells,
                u_saved,
                notch_reporter_saved,
                y_positions,
                y_positions_all,
                center_y,
                times,
                fraction_saved,
                dt,
                initial_time,
                final_time,
                t_0,
                dir_path+"/TkvCA/plots_FigS6/")       

# # =============================================================================
# # WILD-TYPE - FIXED DELTA (FigS4A)
# # =============================================================================
 
print("running WT fixed Delta")
## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=10
sigma_rep=7
for k, c in cells.items():
    c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_free = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0

## run simulation       
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved, DeltaFixed= True)

#generate simulation snapshots
dir_path = os.getcwd()
Path(dir_path+"/WT_fixed_Delta/snapshots/").mkdir(parents=True, exist_ok=True)  
        
times_to_plot=[i*60//fraction_saved for i in range(24)] #60 minutes is 1 hour                    
save_snapshots(cells,
                 fraction_saved,
                 side_length,
                 delta_saved,
                 notch_saved,
                 signal_saved,
                 u_saved,
                 notch_reporter_saved,
                 times_to_plot,
                 t_0, 
                dir_path+"/WT_fixed_Delta/snapshots/")

# # =============================================================================
# # WILD-TYPE - partial restoration of straightness (Fig4E)
# # =============================================================================
 
print("running WT rough initial condition")
## initial condition fig 4E top
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=10
sigma_rep=7

deltax = 4
#side length of a cell in um
side_length = deltax/np.sqrt(3)
for k, c in cells.items():
    if np.abs(c.position[1]-center_y)<side_length*(7+ np.cos(3*c.position[0]*2.0*np.pi/(deltax*N_x*np.cos(np.pi/6)))):
        c.u=1
        c.delta_tot = 1
        c.delta_free = 1    
    else:
        c.u=0
        c.delta_tot = 0
        c.delta_free = 0
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0


## run simulation       
N_times=2500
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved)

#generate simulation snapshots
dir_path = os.getcwd()
Path(dir_path+"/WT_rough_initial_condition/snapshots_Fig4E_top/").mkdir(parents=True, exist_ok=True)  
        
times_to_plot=[i*60//fraction_saved for i in range(40)] #60 minutes is 1 hour                    
save_snapshots(cells,
                 fraction_saved,
                 side_length,
                 delta_saved,
                 notch_saved,
                 signal_saved,
                 u_saved,
                 notch_reporter_saved,
                 times_to_plot,
                 t_0, 
                dir_path+"/WT_rough_initial_condition/snapshots_Fig4E_top/")



## initial condition fig 4E bottom 
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=10
sigma_rep=7

deltax = 4
#side length of a cell in um
side_length = deltax/np.sqrt(3)
for k, c in cells.items():
    if np.abs(c.position[1]-center_y)<side_length*(7+2*np.cos(2*c.position[0]*2.0*np.pi/(deltax*N_x*np.cos(np.pi/6)))):
        c.u=1
        c.delta_tot = 1
        c.delta_free = 1    
    else:
        c.u=0
        c.delta_tot = 0
        c.delta_free = 0
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0


## run simulation       
N_times=2500
[delta_saved,
notch_saved,
signal_saved,
u_saved,
notch_reporter_saved]=run_sim(cells,
                              r, JI, JA, 
                              Kcis, Ktrans, 
                              tau_u, tau_Notch, tau_reporter,
                              N_low, N_high, D_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved)

#generate simulation snapshots
dir_path = os.getcwd()
Path(dir_path+"/WT_rough_initial_condition/snapshots_Fig4E_bottom/").mkdir(parents=True, exist_ok=True)  
        
times_to_plot=[i*60//fraction_saved for i in range(40)] #60 minutes is 1 hour                    
save_snapshots(cells,
                 fraction_saved,
                 side_length,
                 delta_saved,
                 notch_saved,
                 signal_saved,
                 u_saved,
                 notch_reporter_saved,
                 times_to_plot,
                 t_0, 
                dir_path+"/WT_rough_initial_condition/snapshots_Fig4E_bottom/")

