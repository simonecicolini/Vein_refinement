from cell import cell
import matplotlib.pyplot as plt
from run_sim import run_sim
from plot_functions import save_animation
from plot_functions import save_packings
from plot_functions import save_spatial_profiles
from plot_functions import save_temporal_profiles
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
Sstar = 0.25
alphaS = 0.025
S0 = 0.4
           
# =============================================================================
# WILD-TYPE REFERENCE SIMULATION
# =============================================================================
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
                              N_low, N_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved)
                        
## generate graphs and plots
dir_path = os.getcwd()
Path(dir_path+"/WT/").mkdir(parents=True, exist_ok=True)

time_indices = [time_index for time_index in range(len(u_saved))]
times=t_0+dt*fraction_saved*np.array(time_indices)

save_animation(cells,
                N_x,
                N_y,
                N_times,
                fraction_saved,
                side_length,
                delta_saved,
                notch_saved,
                signal_saved,
                u_saved,
                notch_reporter_saved,
                dir_path +'/WT/')

frames=[i for i in range(N_times//(fraction_saved))] #every hour
for frame in frames:
    time=times[frame]
    relative_time=time/tau_u
    save_packings(cells,
                  N_x,
                  N_y,
                  N_times,
                  fraction_saved,
                  side_length,
                  delta_saved,
                  notch_saved,
                  signal_saved,
                  u_saved,
                  notch_reporter_saved,
                  frame,
                  time,
                  relative_time,
                  dir_path +'/WT/')

frames_for_plot = [0, 3*6, 8*6, 13*6, 18*6, 22*6]   
times_for_plot=[times[frame] for frame in frames_for_plot]
save_spatial_profiles(cells,
                      delta_saved,
                      notch_saved,
                      signal_saved,
                      u_saved,
                      notch_reporter_saved,
                      frames_for_plot,
                      times_for_plot,
                      dir_path +'/WT/')

spatial_boundaries = side_length*np.array([0, 1, 2, 4, 6, 8, 10])
save_temporal_profiles(cells,
                        delta_saved,
                        notch_saved,
                        signal_saved,
                        u_saved,
                        notch_reporter_saved,
                        spatial_boundaries,
                        time_indices,
                        times,
                        dir_path +'/WT/')

# # =============================================================================
# # WILD-TYPE REFERENCE SIMULATION - WITH CROSS-VEIN
# # =============================================================================
 
print("running WT with cross vein")
## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=10
for k, c in cells.items():
    c.u = 0
    c.delta_tot = 0
    c.delta_free= 0
    if np.abs(c.position[1]-center_y-18)<10:
        c.u = 1
        c.delta_tot = 1
        c.delta_free = 1
    if np.abs(c.position[1]-center_y+18)<10:
        c.u = 1
        c.delta_tot = 1
        c.delta_free = 1
    if ((c.position[1]<center_y+18) and (c.position[1]>center_y-18)
        and np.abs(c.position[0]-center_x)<8):
        c.u = 1
        c.delta_tot = 1
        c.delta_free= 1
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0
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
                              N_low, N_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved)
                        
## generate graphs and plots
dir_path = os.getcwd()
Path(dir_path+"/WT_crossvein/").mkdir(parents=True, exist_ok=True)

time_indices = [time_index for time_index in range(len(u_saved))]
times=t_0+dt*fraction_saved*np.array(time_indices)

save_animation(cells,
                N_x,
                N_y,
                N_times,
                fraction_saved,
                side_length,
                delta_saved,
                notch_saved,
                signal_saved,
                u_saved,
                notch_reporter_saved,
                dir_path +'/WT_crossvein/')

for frame in [0, 10, 20, 30, 40, 50]:
    time=times[frame]
    relative_time=time/tau_u
    save_packings(cells,
                  N_x,
                  N_y,
                  N_times,
                  fraction_saved,
                  side_length,
                  delta_saved,
                  notch_saved,
                  signal_saved,
                  u_saved,
                  notch_reporter_saved,
                  frame,
                  time,
                  relative_time,
                  dir_path +'/WT_crossvein/')


# =============================================================================
# PERTURBATION - DELTA DN
# =============================================================================

print("running DeltaDN")

## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=10
for k, c in cells.items():
    c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot =0* np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_free = 0*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.notch_tot=N_low
    c.notch_free=N_low
    c.notch_reporter=0
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
                              N_low, N_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times,
                              fraction_saved,
                              noDelta=True)
                              
## generate graphs and plots
dir_path = os.getcwd()
Path(dir_path+"/DeltaDN/").mkdir(parents=True, exist_ok=True)

time_indices = [time_index for time_index in range(len(u_saved))]
times=t_0+dt*fraction_saved*np.array(time_indices)

save_animation(cells,
                N_x,
                N_y,
                N_times,
                fraction_saved,
                side_length,
                delta_saved,
                notch_saved,
                signal_saved,
                u_saved,
                notch_reporter_saved,
                dir_path +'/DeltaDN/')

for frame in [0, 10, 20, 30, 40, 50]:
    time=times[frame]
    relative_time=time/tau_u
    save_packings(cells,
                  N_x,
                  N_y,
                  N_times,
                  fraction_saved,
                  side_length,
                  delta_saved,
                  notch_saved,
                  signal_saved,
                  u_saved,
                  notch_reporter_saved,
                  frame,
                  time,
                  relative_time,
                  dir_path +'/DeltaDN/')
frames_for_plot = [0, 10, 20, 30, 40, 50]    
times_for_plot=[times[frame] for frame in frames_for_plot]
save_spatial_profiles(cells,
                      delta_saved,
                      notch_saved,
                      signal_saved,
                      u_saved,
                      notch_reporter_saved,
                      frames_for_plot,
                      times_for_plot,
                      dir_path +'/DeltaDN/')
spatial_boundaries = side_length*np.array([0, 1, 2, 4, 6, 8, 10])
save_temporal_profiles(cells,
                        delta_saved,
                        notch_saved,
                        signal_saved,
                        u_saved,
                        notch_reporter_saved,
                        spatial_boundaries,
                        time_indices,
                        times,
                        dir_path +'/DeltaDN/')    
# =============================================================================
# PERTURBATION - Nintra
# =============================================================================

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
                              N_low, N_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved,
                              extraSignal=True,
                              extraSignalValue=0.45)
                              
## generate graphs and plots
dir_path = os.getcwd()
Path(dir_path+"/Nintra/").mkdir(parents=True, exist_ok=True)

time_indices = [time_index for time_index in range(len(u_saved))]
times=t_0+dt*fraction_saved*np.array(time_indices)

save_animation(cells,
                N_x,
                N_y,
                N_times,
                fraction_saved,
                side_length,
                delta_saved,
                notch_saved,
                signal_saved,
                u_saved,
                notch_reporter_saved,
                dir_path +'/Nintra/')

for frame in [0, 10, 20, 30, 40, 50]:
    time=times[frame]
    relative_time=time/tau_u
    save_packings(cells,
                  N_x,
                  N_y,
                  N_times,
                  fraction_saved,
                  side_length,
                  delta_saved,
                  notch_saved,
                  signal_saved,
                  u_saved,
                  notch_reporter_saved,
                  frame,
                  time,
                  relative_time,
                  dir_path +'/Nintra/')
frames_for_plot = [0, 4*6, 8*6, 12*6]    
times_for_plot=[times[frame] for frame in frames_for_plot]
save_spatial_profiles(cells,
                      delta_saved,
                      notch_saved,
                      signal_saved,
                      u_saved,
                      notch_reporter_saved,
                      frames_for_plot,
                      times_for_plot,
                      dir_path +'/Nintra/')   
spatial_boundaries = side_length*np.array([0, 1, 2, 4, 6, 8, 10])
save_temporal_profiles(cells,
                        delta_saved,
                        notch_saved,
                        signal_saved,
                        u_saved,
                        notch_reporter_saved,
                        spatial_boundaries,
                        time_indices,
                        times,
                        dir_path +'/Nintra/') 
# # # =============================================================================
# # # PERTURBATION - TkvCA
# # # =============================================================================

print("running TkvCA")

## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=15
sigma_rep=7
for k, c in cells.items():
    c.u=1*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot = 1*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_free =1* np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
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
                              N_low, N_high,
                              Sstar, alphaS, S0,
                              t_0, dt, N_times, fraction_saved,
                              extraVeinActivation=True,
                              extraVeinActivationValue=0.5)
                              
## generate graphs and plots
dir_path = os.getcwd()
Path(dir_path+"/TkvCA/").mkdir(parents=True, exist_ok=True)

time_indices = [time_index for time_index in range(len(u_saved))]
times=t_0+dt*fraction_saved*np.array(time_indices)

save_animation(cells,
                N_x,
                N_y,
                N_times,
                fraction_saved,
                side_length,
                delta_saved,
                notch_saved,
                signal_saved,
                u_saved,
                notch_reporter_saved,
                dir_path +'/TkvCA/')

for frame in [0, 10, 20, 30, 40, 50]:
    time=times[frame]
    relative_time=time/tau_u
    save_packings(cells,
                  N_x,
                  N_y,
                  N_times,
                  fraction_saved,
                  side_length,
                  delta_saved,
                  notch_saved,
                  signal_saved,
                  u_saved,
                  notch_reporter_saved,
                  frame,
                  time,
                  relative_time,
                  dir_path +'/TkvCA/')
frames_for_plot = [0, 4*6, 8*6, 12*6, 16*6]    
times_for_plot=[times[frame] for frame in frames_for_plot]
save_spatial_profiles(cells,
                      delta_saved,
                      notch_saved,
                      signal_saved,
                      u_saved,
                      notch_reporter_saved,
                      frames_for_plot,
                      times_for_plot,
                      dir_path +'/TkvCA/')  
spatial_boundaries = side_length*np.array([0, 1, 2, 4, 6, 8, 10])
save_temporal_profiles(cells,
                        delta_saved,
                        notch_saved,
                        signal_saved,
                        u_saved,
                        notch_reporter_saved,
                        spatial_boundaries,
                        time_indices,
                        times,
                        dir_path +'/TkvCA/')  

# Extended range of activation

# =============================================================================
# Delay of DSRF relative to u
# =============================================================================