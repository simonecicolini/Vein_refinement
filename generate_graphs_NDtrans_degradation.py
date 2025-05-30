from cell import cell

from run_sim_NDtrans_degradation import run_sim
# from plot_functions_NDtrans_degradation import save_animation
# from plot_functions_NDtrans_degradation import save_packings
# from plot_functions_NDtrans_degradation import save_spatial_profiles
# from plot_functions_NDtrans_degradation import save_spatial_profiles_2
# from plot_functions_NDtrans_degradation import save_temporal_profiles
from plot_functions_NDtrans_degradation import save_snapshots
from plot_functions_NDtrans_degradation import save_Nicd_reporter
import numpy as np

# import os
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
 
dt = 0.0167
t_0 = 21 #initial time  in hours - for comparison with experiments
N_times = 3001
fraction_saved = 10
print('total simulation time: ', dt*N_times)


# timescales
tau_Notch = 1 #time scale of Notch protein relaxation in hours
tau_u = 2 #time scale of u dynamics in hours
tau_signal = 1 #time scale of signal in hours
tau_reporter = 8 #degradation time scale of Notch signalling reporter
#other model parameters
JA = 0.015
JI = 0.35
r = 0.5
Kcis= 1500
Ktrans = 1800
N_low = 0.3
N_high = 1.5
Sstar = 0.2
alphaS = 0.025
S0 = 0.1
k_signal = 2 

Dtotinit=0.4
dir_path='NDtrans_degradation/tau='+str(tau_u)+'_tauNotch=' +str(tau_Notch)+'_tauSig='+str(tau_signal)+'_Nlow='+str(N_low)+'_Nhigh='+str(N_high)+'_Sstar='+str(Sstar)+'_S0='+str(S0)+'_alphaS='+str(alphaS)+'_ks='+str(k_signal)+'/JA='+str(JA)+'JI='+str(JI)+'r='+str(r)
# =============================================================================
# WILD-TYPE NDtrans degradation
# =============================================================================

print("running WT NDtrans degradation")
## initial condition
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma=10
sigma_rep=7
for k, c in cells.items():
    c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_tot = Dtotinit * np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.delta_free = Dtotinit * np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
    c.notch_tot=N_low
    c.notch_free=N_low
    c.signal=0
    c.notch_reporter=0.1*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )

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
                              tau_u, tau_signal, tau_Notch, tau_reporter,
                              N_low, N_high,
                              Sstar, alphaS, S0, k_signal,
                              t_0, dt, N_times, fraction_saved)
                        
## generate graphs and plots
Path(dir_path+"/WT/snapshots/").mkdir(parents=True, exist_ok=True)

time_indices = [time_index for time_index in range(len(u_saved))]
times=t_0+dt*fraction_saved*np.array(time_indices)

#save simulation snapshots (FigS4K)
times_to_plot=[i*6 for i in range(40)] #60 minutes is 1 hour    
save_snapshots(cells,
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
                    t_0,
                    times_to_plot,
                    dir_path +'/WT/snapshots/')
   
#save plots (FigS4L)
Path(dir_path+"/WT/plot_FigS4L/").mkdir(parents=True, exist_ok=True)
save_Nicd_reporter(cells,
                    delta_saved,
                    notch_saved,
                    signal_saved,
                    u_saved,
                    notch_reporter_saved,
                    times,
                    t_0,
                    dt,
                    fraction_saved,
                    dir_path +'/WT/plot_FigS4L/')


