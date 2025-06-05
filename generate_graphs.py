#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:28:01 2025

@author: simonecicolini
"""

import matplotlib.cm as cm
import matplotlib as mpl
from cell import cell
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
N_times = 2000
fraction_saved = 10
print('total simulation time: ', dt*N_times)
# timescales
tau_Notch=6#time scale of Notch protein relaxation in hours
tau_u = 2 #time scale of u dynamics in hours
tau_reporter = 8 #degradation time scale of Notch signalling reporter
#Model parameters
JA = 0.015
JAprime = 0.015
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
# # WILD-TYPE REFERENCE SIMULATION (Fig.4B, Fig.6B,S6A-B)
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
                              r, JI, JA, JAprime,
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
                  1.1,
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
# # WILD-TYPE REFERENCE SIMULATION - WITH CROSS-VEIN (Fig.4F)
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
                              r, JI, JA, JAprime,
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
                  1.1,
                  times_to_plot,
                  t_0,
                  dir_path+"/WT_crossvein/snapshots/")


# # =============================================================================
# # PERTURBATION - DELTA DN (Fig.6D, Fig.S6C-D)
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
                              r, JI, JA, JAprime,
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
                  1.1,
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
# # PERTURBATION - Nintra (Fig.6F, Fig.S6E-F)
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
                              r, JI, JA, JAprime,
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
                  1.1,
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
# # PERTURBATION - TkvCA (Fig.6H, Fig.S6G-H)
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
                              r, JI, JA, JAprime,
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
                  1.1,
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
                              r, JI, JA, JAprime,
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
                  1.1,
                  times_to_plot,
                  t_0, 
                dir_path+"/WT_fixed_Delta/snapshots/")

# # =============================================================================
# # WILD-TYPE - partial restoration of straightness (Fig4E)
# # =============================================================================
 
print("running WT rough initial condition (Fig4E)")
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
                              r, JI, JA, JAprime,
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
                  1.1,
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
                              r, JI, JA, JAprime,
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
                  1.1,
                  times_to_plot,
                  t_0, 
                dir_path+"/WT_rough_initial_condition/snapshots_Fig4E_bottom/")


# # =============================================================================
# # PHASE DIAGRAM Dhigh, Nlow-Nhigh (Fig.4C)
# # =============================================================================
N_times = 16001
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

for D_high in [0., 0.5, 1., 1.5, 2.]:
    print('Dhigh= '+str(D_high))
    for NlowOverNlowRef in [0., 0.5, 0.75, 1., 1.25, 1.5]:
        N_low = NlowOverNlowRef*0.3
        N_high = NlowOverNlowRef*2.2
        parameters = '/Phase_diagram_Fig4C/JA='+str(JA)+'JI='+str(JI)+'r='+str(r)+'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/NlowOverNlowRef='+str(NlowOverNlowRef)+'/D_high='+str(D_high)

        #generate simulation snapshots
        dir_path = os.getcwd()
        Path(dir_path+parameters).mkdir(parents=True, exist_ok=True)               
                
        # initialize values 
        center_x = np.mean([cell.position[0] for cell in cells.values()])
        center_y = np.mean([cell.position[1] for cell in cells.values()])
        sigma=10
        sigma_rep=7
        for k, c in cells.items():
            c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_tot = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_free = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.notch_tot= N_low
            c.notch_free= N_low
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
                                      r, JI, JA, JAprime,
                                      Kcis, Ktrans, 
                                      tau_u, tau_Notch, tau_reporter,
                                      N_low, N_high, D_high,
                                      Sstar, alphaS, S0,
                                      t_0, dt, N_times, fraction_saved)

        # save packings final frame
        patches = [] 
        norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
        cmap = cm.Blues
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for c, delta in zip(cells.values(), delta_saved[0]):
              color_to_plot = m.to_rgba(delta)
              hexagon = RegularPolygon((c.position[0], c.position[1]), 
                                      numVertices=6,
                                      radius=0.9*side_length, 
                                      orientation=np.pi/2.0,
                                      facecolor = color_to_plot,
                                      alpha=0.4,
                                      edgecolor='grey')
              patches.append(hexagon)        
        
        #u final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[-1]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

        #signal final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[-1]))
        p.set_clim([0, N_high])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

#generate phase diagram with u and Notch signalling (Fig4C)
fig=plt.figure(figsize=(60,50))
columns=6
rows=5
Dhigh_values=[0., 0.5, 1., 1.5, 2.]
Dhigh_values.reverse()
NlowOverNlowRef_values=[0., 0.5, 0.75, 1., 1.25, 1.5]
i=0
for D_high  in Dhigh_values:
    for NlowOverNlowRef in NlowOverNlowRef_values:
        i=i+1
        parameters = '/Phase_diagram_Fig4C/JA='+str(JA)+'JI='+str(JI)+'r='+str(r)+'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/NlowOverNlowRef='+str(NlowOverNlowRef)+'/D_high='+str(D_high)
        try:
            img=mpimg.imread('./'+parameters+'/u_final.png')
            #rotated_img = ndimage.rotate(img, 90)
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_Fig4C/JA='+str(JA)+'JI='+str(JI)+'r='+str(r)+'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/u_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


#generate phase diagram with u and Notch signalling (Fig4C)
fig=plt.figure(figsize=(60,50))
columns=6
rows=5
Dhigh_values=[0., 0.5, 1., 1.5, 2.]
Dhigh_values.reverse()
NlowOverNlowRef_values=[0., 0.5, 0.75, 1., 1.25, 1.5]
i=0
for D_high  in Dhigh_values:
    for NlowOverNlowRef in NlowOverNlowRef_values:
        i=i+1
        parameters = '/Phase_diagram_Fig4C/JA='+str(JA)+'JI='+str(JI)+'r='+str(r)+'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/NlowOverNlowRef='+str(NlowOverNlowRef)+'/D_high='+str(D_high)
        try:
            img=mpimg.imread('./'+parameters+'/signal_final.png')
            #rotated_img = ndimage.rotate(img, 90)
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_Fig4C/JA='+str(JA)+'JI='+str(JI)+'r='+str(r)+'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/signal_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)
 
# # =============================================================================
# # PHASE DIAGRAM J^I-r (Fig.4D)
# # =============================================================================
N_times = 66001
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

for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print('JI= '+str(JI))
    for JI in [0, 0.2, 0.35, 0.5, 0.65]:
        parameters = '/Phase_diagram_Fig4D/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/JI='+str(JI)+'/r='+str(r)

        #generate simulation snapshots
        dir_path = os.getcwd()
        Path(dir_path+parameters).mkdir(parents=True, exist_ok=True)               
                
        # initialize values 
        center_x = np.mean([cell.position[0] for cell in cells.values()])
        center_y = np.mean([cell.position[1] for cell in cells.values()])
        sigma=10
        sigma_rep=7
        for k, c in cells.items():
            c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_tot = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_free = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.notch_tot= N_low
            c.notch_free= N_low
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
                                      r, JI, JA, JAprime,
                                      Kcis, Ktrans, 
                                      tau_u, tau_Notch, tau_reporter,
                                      N_low, N_high, D_high,
                                      Sstar, alphaS, S0,
                                      t_0, dt, N_times, fraction_saved)

        # save packings final frame
        patches = [] 
        norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
        cmap = cm.Blues
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for c, delta in zip(cells.values(), delta_saved[0]):
              color_to_plot = m.to_rgba(delta)
              hexagon = RegularPolygon((c.position[0], c.position[1]), 
                                      numVertices=6,
                                      radius=0.9*side_length, 
                                      orientation=np.pi/2.0,
                                      facecolor = color_to_plot,
                                      alpha=0.4,
                                      edgecolor='grey')
              patches.append(hexagon)
        
        
        #u final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[-1]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

        #signal final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[-1]))
        p.set_clim([0, N_high])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)


fig=plt.figure(figsize=(60,60))
columns=5
rows=5
JI_values=[0, 0.2, 0.35, 0.5, 0.65]
JI_values.reverse()
r_values=[0.1, 0.3, 0.5, 0.7, 0.9]
i=0
for JI in JI_values:
    for r in r_values:
        i=i+1
        parameters = '/Phase_diagram_Fig4D/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/JI='+str(JI)+'/r='+str(r)

        try:
            img=mpimg.imread('./'+parameters+'/u_final.png')
            #rotated_img = ndimage.rotate(img, 90)
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_Fig4D/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/u_phase_diag.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)

fig=plt.figure(figsize=(60,60))
columns=5
rows=5
JI_values=[0, 0.2, 0.35, 0.5, 0.65]
JI_values.reverse()
r_values=[0.1, 0.3, 0.5, 0.7, 0.9]
i=0
for JI in JI_values:
    for r in r_values:
        i=i+1
        parameters = '/Phase_diagram_Fig4D/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/JI='+str(JI)+'/r='+str(r)
        try:
            img=mpimg.imread('./'+parameters+'/signal_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_Fig4D/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_Sstar='+str(Sstar)+'_alphaS='+str(alphaS)+'/signal_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


# # =============================================================================
# # PHASE DIAGRAM S^* - N^low (Fig.S4H)
# # =============================================================================
N_times = 16001
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

for Sstar in [0.05, 0.25, 0.45, 0.65, 0.85, 1]:
    print('JI= '+str(JI))
    for N_low in [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]:
        parameters = '/Phase_diagram_FigS4H/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/Sstar='+str(Sstar)+'/Nlow='+str(N_low)

        #generate simulation snapshots
        dir_path = os.getcwd()
        Path(dir_path+parameters).mkdir(parents=True, exist_ok=True)               
                
        # initialize values 
        center_x = np.mean([cell.position[0] for cell in cells.values()])
        center_y = np.mean([cell.position[1] for cell in cells.values()])
        sigma=10
        sigma_rep=7
        for k, c in cells.items():
            c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_tot = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_free = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.notch_tot= N_low
            c.notch_free= N_low
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
                                      r, JI, JA, JAprime,
                                      Kcis, Ktrans, 
                                      tau_u, tau_Notch, tau_reporter,
                                      N_low, N_high, D_high,
                                      Sstar, alphaS, S0,
                                      t_0, dt, N_times, fraction_saved)

        # save packings final frame
        patches = [] 
        norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
        cmap = cm.Blues
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for c, delta in zip(cells.values(), delta_saved[0]):
              color_to_plot = m.to_rgba(delta)
              hexagon = RegularPolygon((c.position[0], c.position[1]), 
                                      numVertices=6,
                                      radius=0.9*side_length, 
                                      orientation=np.pi/2.0,
                                      facecolor = color_to_plot,
                                      alpha=0.4,
                                      edgecolor='grey')
              patches.append(hexagon)
        
        
        #u final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[-1]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

        #signal final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[-1]))
        p.set_clim([0, N_high])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

#generate phase diagram with u and Notch signalling
fig=plt.figure(figsize=(60,50))
columns=7
rows=6
Sstar_values=[0.05, 0.25, 0.45, 0.65, 0.85, 1]
Sstar_values.reverse()
Nlow_values=[0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
i=0
for Sstar  in Sstar_values:
    for N_low in Nlow_values:
        i=i+1
        parameters = '/Phase_diagram_FigS4H/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/Sstar='+str(Sstar)+'/Nlow='+str(N_low)
        try:
            img=mpimg.imread('./'+parameters+'/u_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4H/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/u_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


fig=plt.figure(figsize=(60,50))
columns=7
rows=6
Sstar_values=[0.05, 0.25, 0.45, 0.65, 0.85, 1]
Sstar_values.reverse()
Nlow_values=[0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
i=0
for Sstar  in Sstar_values:
    for N_low in Nlow_values:
        i=i+1
        parameters = '/Phase_diagram_FigS4H/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/Sstar='+str(Sstar)+'/Nlow='+str(N_low)
        try:
            img=mpimg.imread('./'+parameters+'/signal_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4H/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/signal_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


# =============================================================================
# # PHASE DIAGRAM S^* - N^high (Fig.S4H')
# # =============================================================================
N_times = 16001
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

for Sstar in [0.05, 0.25, 0.45, 0.65, 0.85, 1]:
    print('JI= '+str(JI))
    for N_high in [0, 0.4, 1, 1.6, 2.2, 2.8]:
        parameters = '/Phase_diagram_FigS4H\'/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/Sstar='+str(Sstar)+'/Nhigh='+str(N_high)

        #generate simulation snapshots
        dir_path = os.getcwd()
        Path(dir_path+parameters).mkdir(parents=True, exist_ok=True)               
                
        # initialize values 
        center_x = np.mean([cell.position[0] for cell in cells.values()])
        center_y = np.mean([cell.position[1] for cell in cells.values()])
        sigma=10
        sigma_rep=7
        for k, c in cells.items():
            c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_tot = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.delta_free = D_high*np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
            c.notch_tot= N_low
            c.notch_free= N_low
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
                                      r, JI, JA, JAprime,
                                      Kcis, Ktrans, 
                                      tau_u, tau_Notch, tau_reporter,
                                      N_low, N_high, D_high,
                                      Sstar, alphaS, S0,
                                      t_0, dt, N_times, fraction_saved)

        # save packings final frame
        patches = [] 
        norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
        cmap = cm.Blues
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for c, delta in zip(cells.values(), delta_saved[0]):
              color_to_plot = m.to_rgba(delta)
              hexagon = RegularPolygon((c.position[0], c.position[1]), 
                                      numVertices=6,
                                      radius=0.9*side_length, 
                                      orientation=np.pi/2.0,
                                      facecolor = color_to_plot,
                                      alpha=0.4,
                                      edgecolor='grey')
              patches.append(hexagon)
        
        
        #u final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[-1]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

        #signal final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[-1]))
        p.set_clim([0, N_high])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

#generate phase diagram with u and Notch signalling
fig=plt.figure(figsize=(60,60))
columns=6
rows=6
Sstar_values=[0.05, 0.25, 0.45, 0.65, 0.85, 1]
Sstar_values.reverse()
Nhigh_values=[0, 0.4, 1, 1.6, 2.2, 2.8]
i=0
for Sstar  in Sstar_values:
    for N_high in Nhigh_values:
        i=i+1
        parameters = '/Phase_diagram_FigS4H\'/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/Sstar='+str(Sstar)+'/Nhigh='+str(N_high)
        try:
            img=mpimg.imread('./'+parameters+'/u_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4H\'/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/u_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


fig=plt.figure(figsize=(60,60))
columns=6
rows=6
Sstar_values=[0.05, 0.25, 0.45, 0.65, 0.85, 1]
Sstar_values.reverse()
Nhigh_values=[0, 0.4, 1, 1.6, 2.2, 2.8]
i=0
for Sstar  in Sstar_values:
    for N_high in Nhigh_values:
        i=i+1
        parameters = '/Phase_diagram_FigS4H\'/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/Sstar='+str(Sstar)+'/Nhigh='+str(N_high)
        try:
            img=mpimg.imread('./'+parameters+'/signal_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4H\'/JA='+str(JA)+'Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'_r='+str(r)+'/signal_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


# =============================================================================
# # PHASE DIAGRAM JA,JAprime - r (Fig.S4I)
# # =============================================================================
N_times = 16001
tau_Notch=6#time scale of Notch protein relaxation in hours
tau_u = 2 #time scale of u dynamics in hours
tau_reporter = 8 #degradation time scale of Notch signalling reporter
#Model parameters
JA = 0.015
JAprime =0.015
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

for JA in [0, 0.015, 0.03, 0.045, 0.06, 0.15]:
    JAprime=JA
    print('JA= '+str(JA))
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        parameters = '/Phase_diagram_FigS4I/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/JA=JAprime='+str(JA)+'/r='+str(r)

        #generate simulation snapshots
        dir_path = os.getcwd()
        Path(dir_path+parameters).mkdir(parents=True, exist_ok=True)               
                
        # initialize values of u
        center_x = np.mean([cell.position[0] for cell in cells.values()])
        center_y = np.mean([cell.position[1] for cell in cells.values()])
        sigma=10
        sigma_rep=6
        deltax = 4
        for k, c in cells.items():
            rn=np.random.normal(0,0.1)
            if np.abs(c.position[1]-center_y)<side_length*(7+1*np.cos(3*c.position[0]*2.0*np.pi/(deltax*N_x*np.cos(np.pi/6)))):
                c.u=1+rn 
                c.delta_tot = 1+rn
                c.delta_free = 1+rn 
            else:
                c.u=rn
                c.delta_tot = rn*np.heaviside(rn,0)
                c.delta_free = rn*np.heaviside(rn,0)
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
                                      r, JI, JA, JAprime,
                                      Kcis, Ktrans, 
                                      tau_u, tau_Notch, tau_reporter,
                                      N_low, N_high, D_high,
                                      Sstar, alphaS, S0,
                                      t_0, dt, N_times, fraction_saved)

        # save packings final frame
        patches = [] 
        norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
        cmap = cm.Blues
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for c, delta in zip(cells.values(), delta_saved[0]):
              color_to_plot = m.to_rgba(delta)
              hexagon = RegularPolygon((c.position[0], c.position[1]), 
                                      numVertices=6,
                                      radius=0.9*side_length, 
                                      orientation=np.pi/2.0,
                                      facecolor = color_to_plot,
                                      alpha=0.4,
                                      edgecolor='grey')
              patches.append(hexagon)
        
        
        #u final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[-1]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

        
        #u initial frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[0]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_initial.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)



        #signal final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[-1]))
        p.set_clim([0, N_high])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

#generate phase diagram with u and notch signalling
fig=plt.figure(figsize=(60,60))
columns=5
rows=6
JA_values=[0, 0.015, 0.03, 0.045, 0.06, 0.15] # equal to JAprime values
JA_values.reverse()
r_values=[0.1, 0.2, 0.3, 0.4, 0.5]
i=0
for JA in JA_values:
    for r in r_values:
        i=i+1
        parameters= '/Phase_diagram_FigS4I/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/JA=JAprime='+str(JA)+'/r='+str(r)
        try:
            img=mpimg.imread('./'+parameters+'/u_final.png')
            #rotated_img = ndimage.rotate(img, 90)
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4I/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/u_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


fig=plt.figure(figsize=(60,60))
columns=5
rows=6
JA_values=[0, 0.015, 0.03, 0.045, 0.06, 0.15] # equal to JAprime values
JA_values.reverse()
r_values=[0.1, 0.2, 0.3, 0.4, 0.5]
i=0
for JA in JA_values:
    for r in r_values:
        i=i+1
        parameters= '/Phase_diagram_FigS4I/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/JA=JAprime='+str(JA)+'/r='+str(r)
        try:
            img=mpimg.imread('./'+parameters+'/signal_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4I/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/signal_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


# =============================================================================
# # PHASE DIAGRAM JA - r (Fig.S4I')
# # =============================================================================
N_times = 16001
tau_Notch=6#time scale of Notch protein relaxation in hours
tau_u = 2 #time scale of u dynamics in hours
tau_reporter = 8 #degradation time scale of Notch signalling reporter
#Model parameters
JA = 0.015
JAprime =0.015
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

for JA in [0, 0.015, 0.03, 0.045, 0.06, 0.15]:
    print('JA= '+str(JA))
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        parameters = '/Phase_diagram_FigS4I\'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/JA=JAprime='+str(JA)+'/r='+str(r)

        #generate simulation snapshots
        dir_path = os.getcwd()
        Path(dir_path+parameters).mkdir(parents=True, exist_ok=True)               
                
        # initialize values of u
        center_x = np.mean([cell.position[0] for cell in cells.values()])
        center_y = np.mean([cell.position[1] for cell in cells.values()])
        sigma=10
        sigma_rep=7
        deltax = 4
        for k, c in cells.items():
            rn=np.random.normal(0,0.1)
            if np.abs(c.position[1]-center_y)<side_length*(7+1*np.cos(3*c.position[0]*2.0*np.pi/(deltax*N_x*np.cos(np.pi/6)))):
                c.u=1+rn 
                c.delta_tot = 1+rn
                c.delta_free = 1+rn 
            else:
                c.u=rn
                c.delta_tot = rn*np.heaviside(rn,0)
                c.delta_free = rn*np.heaviside(rn,0)
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
                                      r, JI, JA, JAprime,
                                      Kcis, Ktrans, 
                                      tau_u, tau_Notch, tau_reporter,
                                      N_low, N_high, D_high,
                                      Sstar, alphaS, S0,
                                      t_0, dt, N_times, fraction_saved)

        # save packings final frame
        patches = [] 
        norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
        cmap = cm.Blues
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for c, delta in zip(cells.values(), delta_saved[0]):
              color_to_plot = m.to_rgba(delta)
              hexagon = RegularPolygon((c.position[0], c.position[1]), 
                                      numVertices=6,
                                      radius=0.9*side_length, 
                                      orientation=np.pi/2.0,
                                      facecolor = color_to_plot,
                                      alpha=0.4,
                                      edgecolor='grey')
              patches.append(hexagon)
        
        
        #u final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[-1]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

        
        #u initial frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[0]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/u_initial.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)



        #signal final frame
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[-1]))
        p.set_clim([0, N_high])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

#generate phase diagram with u and notch signalling
fig=plt.figure(figsize=(60,60))
columns=5
rows=6
JA_values=[0, 0.015, 0.03, 0.045, 0.06, 0.15] # equal to JAprime values
JA_values.reverse()
r_values=[0.1, 0.2, 0.3, 0.4, 0.5]
i=0
for JA in JA_values:
    for r in r_values:
        i=i+1
        parameters= '/Phase_diagram_FigS4I\'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/JA=JAprime='+str(JA)+'/r='+str(r)
        try:
            img=mpimg.imread('./'+parameters+'/u_final.png')
            #rotated_img = ndimage.rotate(img, 90)
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4I\'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/u_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)


fig=plt.figure(figsize=(60,60))
columns=5
rows=6
JA_values=[0, 0.015, 0.03, 0.045, 0.06, 0.15] # equal to JAprime values
JA_values.reverse()
r_values=[0.1, 0.2, 0.3, 0.4, 0.5]
i=0
for JA in JA_values:
    for r in r_values:
        i=i+1
        parameters= '/Phase_diagram_FigS4I\'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/JA=JAprime='+str(JA)+'/r='+str(r)
        try:
            img=mpimg.imread('./'+parameters+'/signal_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4I\'/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'tau_u='+str(tau_u)+'tauDN='+str(tau_Notch)+'totTime='+str(dt*N_times)+'h_JI='+str(JI)+'Sstar='+str(Sstar)+'Nhigh='+str(N_high)+'/signal_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)



# # =============================================================================
# # WILD-TYPE  VARYING Kcis Ktrans (Fig.S4B)
# # =============================================================================
print("running Fig S4B")
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
Sstar = 0.2
alphaS = 0.025
S0 = 0.4


Kcis_Ktrans_values=[[150, 180], [1500, 1000], [1,1]]
norms_color_signal=[0.65, 0.4, 0.4]
for i in range(len(Kcis_Ktrans_values)):
    Kcis, Ktrans = Kcis_Ktrans_values[i]
    norm_color_signal=norms_color_signal[i]
    
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
                                  r, JI, JA, JAprime,
                                  Kcis, Ktrans, 
                                  tau_u, tau_Notch, tau_reporter,
                                  N_low, N_high, D_high,
                                  Sstar, alphaS, S0,
                                  t_0, dt, N_times, fraction_saved)
    
    #generate simulation snapshots
    dir_path = os.getcwd()
    Path(dir_path+"/Fig_S4B/Kcis="+str(Kcis)+"Ktrans"+str(Ktrans)+"/").mkdir(parents=True, exist_ok=True)  
            
    times_to_plot=[i*60//fraction_saved for i in range(24)] #60 minutes is 1 hour                    
    save_snapshots(cells,
                      fraction_saved,
                      side_length,
                      delta_saved,
                      notch_saved,
                      signal_saved,
                      u_saved,
                      notch_reporter_saved,
                      norm_color_signal,
                      times_to_plot,
                      t_0, 
                      dir_path+"/Fig_S4B/Kcis="+str(Kcis)+"Ktrans"+str(Ktrans)+"/")



