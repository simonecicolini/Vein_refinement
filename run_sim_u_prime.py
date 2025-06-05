
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from cell import cell
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib import animation
from numba import njit
import os

# =============================================================================
# INITIALISATION
# =============================================================================

#whether to save results on disk
save_results = 1

norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
cmap = cm.Blues

m = cm.ScalarMappable(norm=norm, cmap=cmap)

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
        c.neighbours=[#(k-1),
                  (N_y*(k//N_y)+(k-1)%N_y)%(N_x*N_y),
                  (k+N_y)%(N_x*N_y),
                 # k+1+N_y,
                  (N_y*((k+N_y)//N_y)+(k+1+N_y)%N_y)%(N_x*N_y),
                  (N_y*(k//N_y)+(k+1)%N_y)%(N_x*N_y),
                  #k+1-N_y,
                  ( N_y*((k-N_y)//N_y)+(k+1-N_y)%N_y)%(N_x*N_y),
                  (k-N_y)%(N_x*N_y)]
    else:
        c.neighbours=[#(k-1),
                  (N_y*(k//N_y)+(k-1)%N_y)%(N_x*N_y),
                  (N_y*((k+N_y)//N_y)+(k+N_y-1)%N_y)%(N_x*N_y),
                 # k+1+N_y,
                  (N_y*((k+N_y)//N_y)+(k+1+N_y-1)%N_y)%(N_x*N_y),
                  (N_y*(k//N_y)+(k+1)%N_y)%(N_x*N_y),
                  #k+1-N_y,
                 ( N_y*((k-N_y)//N_y)+(k-N_y)%N_y)%(N_x*N_y),
                  (N_y*((k-N_y)//N_y)+(k-1-N_y)%N_y)%(N_x*N_y)]

dt = 0.0167
t_0 = 21 #initial time  in hours - for comparison with experiments
N_times = 1501
fraction_saved = 10
print('total simulation time: ', dt*N_times)
#distance between cells in um
deltax = 4
#side length of a cell in um
side_length = deltax/np.sqrt(3)
# timescales
tau_Notch=6 #time scale of Notch protein relaxation in hours
tau_u = 2 #time scale of u dynamics in hours
tau_reporter = 8 #degradation time scale of Notch signalling reporter
#binding constants
Kcis= 1500
Ktrans = 300
#JA, JI, r
JA = 0.015
JI = 0.35
r = 0.5
# Notch production
N_low = 0.3
N_high=2.2

Juprime=0.3
tauDSRF=1

#other parameters
Sstar = 0.25 #signal level necessary for notch activation
alphaS = 0.025

parameters = 'FigS4C/tauDSRF='+str(tauDSRF)+'_Juprime='+str(Juprime)

if not os.path.exists('./'+parameters):
    os.makedirs('./'+parameters)


# initialize values of u
center_x = np.mean([cell.position[0] for cell in cells.values()])
center_y = np.mean([cell.position[1] for cell in cells.values()])
sigma_u=7
sigma_rep=7
for k, c in cells.items():
    c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma_u**2)
    c.uprime=1-np.exp(-(c.position[1]-center_y)**2/2/sigma_u**2)
    c.delta_tot = np.exp(-(c.position[1]-center_y)**2/2/sigma_u**2)
    c.delta_free =np.exp(-(c.position[1]-center_y)**2/2/sigma_u**2)
    c.notch_tot=N_low
    c.notch_free=N_high
    c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
    c.delta_notch_cis=0
    c.delta_notch_trans=0
    c.notch_delta_trans=0

           
         
# =============================================================================
# RUNNING
# =============================================================================

#for saving
delta_saved = []
notch_saved = []
signal_saved = []
u_saved = []
uprime_saved=[]
notch_reporter_saved = []

#to gain speed
ffact_Notch=dt/tau_Notch
ffact_u=dt/tau_u
ffact_uprime= dt/tauDSRF
ffact_reporter=dt/tau_reporter
#function to perform iteration
#accerelated with numba
@njit
def next_iter_delta_notch_free(delta_free_values,
                               notch_free_values,
                               delta_tot_values,
                               notch_tot_values,
                               nearest_neighbour_delta_free_values,
                               nearest_neighbour_notch_free_values):
    new_delta_free_values = []
    new_notch_free_values = []

    for (delta_free,
         notch_free,
         delta_tot,
         notch_tot,
         nearest_neighbours_delta_free,
         nearest_neighbours_notch_free) in zip(delta_free_values,
                                               notch_free_values,
                                               delta_tot_values,
                                               notch_tot_values,
                                               nearest_neighbour_delta_free_values,
                                               nearest_neighbour_notch_free_values):
        nearest_delta_free_sum = 0
        nearest_notch_free_sum = 0                      
        for (neighbour_delta_free,
             neighbour_notch_free) in zip(nearest_neighbours_delta_free,
                                          nearest_neighbours_notch_free):
            nearest_delta_free_sum += neighbour_delta_free
            nearest_notch_free_sum += neighbour_notch_free    
        # complexes (assumed to be balanced)
        #short-hand notation
        Kc=Kcis
        Kt=Ktrans
        Dtot=delta_tot
        Ntot=notch_tot
        XD =nearest_delta_free_sum
        XN =nearest_notch_free_sum    
        #solve the chemical balance problem of finding the free notch and delta
        #given the total delta, total notch and neighbouring concentration
        delta_free =-((1/(2* (Kc + Kc* Kt* XN)))*(1 - Dtot* Kc + Kc* Ntot + Kt *XD + Kt *XN + 
                     Kt**2 * XD* XN - np.sqrt(4 *(Dtot + Dtot* Kt *XD) *(Kc + Kc *Kt *XN) + (-Dtot* Kc + 
                                                                                             Kc* Ntot 
                                                                                             + (1 + Kt* XD)* (1 + Kt* XN))**2)))
        notch_free=-((1/(2 *(Kc+Kc* Kt* XD)))*
                     (1+Dtot* Kc-Kc* Ntot+Kt *XD+Kt* XN+Kt**2 * XD* XN
                      -np.sqrt(4 *(Dtot+Dtot* Kt* XD)* (Kc+Kc *Kt* XN)+(-Dtot* Kc+Kc* Ntot+(1+Kt* XD) *(1+Kt* XN))**2)))
        new_delta_free_values.append(delta_free)
        new_notch_free_values.append(notch_free )
    return (new_delta_free_values,
            new_notch_free_values)

@njit
def next_iter_uprime(u_values,
              uprime_values,
              nearest_neighbour_u_values):

    new_uprime_values =[]
    for (u,
         uprime,
         nearest_neighbours_u) in zip(u_values,
                                uprime_values,
                                nearest_neighbour_u_values):

        #addition of other terms
        nearest_u_sum = 0
        for neighbour_u in nearest_neighbours_u:
            nearest_u_sum += neighbour_u


        new_uprime = uprime + ffact_uprime*( 1-u -uprime  -Juprime *(nearest_u_sum)*( uprime>0) )
        #Notch signalling reporter
        new_uprime_values.append(new_uprime)
        
    return  new_uprime_values


@njit
def next_iter(u_values,
              uprime_values,
              notch_free_values,
              delta_tot_values,
              notch_tot_values,
              nearest_neighbour_u_values,
              nearest_neighbour_uprime_values,
              nearest_neighbour_delta_free_values,
              notch_reporter_values):
    new_delta_tot_values = []
    new_notch_tot_values = []
    new_signal_values = []
    new_u_values = []
    new_uprime_values =[]
    new_notch_reporter_values = []
    for (u,
         uprime,
         notch_free,
         delta_tot,
         notch_tot,
         nearest_neighbours_u,
         nearest_neighbours_uprime,
         nearest_neighbours_delta_free,
         notch_reporter) in zip(u_values,
                                uprime_values,
                                notch_free_values,
                                delta_tot_values,
                                notch_tot_values,
                                nearest_neighbour_u_values,
                                nearest_neighbour_uprime_values,
                                nearest_neighbour_delta_free_values,
                                notch_reporter_values):

        #addition of other terms
        nearest_u_sum = 0
        nearest_uprime_sum=0
        nearest_delta_free_sum =0 
        for (neighbour_u, neighbour_uprime, neighbour_delta_free) in zip(nearest_neighbours_u, nearest_neighbours_uprime,
                                                       nearest_neighbours_delta_free):
            nearest_u_sum += neighbour_u
            nearest_uprime_sum += neighbour_uprime
            nearest_delta_free_sum+= neighbour_delta_free
        notch_delta_trans = Ktrans*notch_free*nearest_delta_free_sum 
        #total delta and total notch
        #basic behaviour
        new_delta_tot = delta_tot + ffact_Notch*(u*(u>0)-delta_tot)
        new_notch_tot = notch_tot + ffact_Notch*(N_low-notch_tot)
            
        signal =  notch_delta_trans 
        #extra production if high notch signalling
        new_notch_tot += ffact_Notch *(N_high-N_low)* (1+np.tanh((signal-Sstar)/alphaS))/2.0
        new_u= u+ffact_u*(-u*(u-1)*(u-r)-JI * signal *(signal>0.4)*(u>0)+ JA  *(nearest_u_sum-6*u))
        new_uprime = uprime + ffact_uprime*(1-u -uprime  -Juprime *(nearest_u_sum)*( uprime>0) )
        #Notch signalling reporter
        new_notch_reporter = notch_reporter + ffact_reporter*(signal-notch_reporter)
        #store result
        new_delta_tot_values.append(new_delta_tot)
        new_notch_tot_values.append(new_notch_tot )
        new_signal_values.append(signal)
        new_u_values.append(new_u)
        new_uprime_values.append(new_uprime)
        new_notch_reporter_values.append(new_notch_reporter)
    return (new_delta_tot_values,
            new_notch_tot_values,
            new_signal_values,
            new_u_values,
            new_uprime_values,
            new_notch_reporter_values)

u_values = np.array([c.u for c in cells.values()])
nearest_neighbour_u_values = np.array([[cells[c2].u for c2 in c.neighbours]
                                     for c in cells.values()] )

#realx initial state
count=0
max_iter_uprime=1
while (count<1.0e6):
    uprime_values = np.array([c.uprime for c in cells.values()])
    new_uprime_values= next_iter_uprime(u_values, uprime_values, nearest_neighbour_u_values)
    
    for (c, new_uprime) in zip(cells.values(), new_uprime_values):
        c.uprime= new_uprime
    max_iter_uprime=np.max(np.abs(new_uprime_values-uprime_values))
    count+=1
if max_iter_uprime>1.0e-2:
    print("convergence failed")

   
for t in range(N_times):
   if t%fraction_saved ==0:
       print(np.round(t/N_times*100,2), '%executed')
   u_values = np.array([c.u for c in cells.values()])
   uprime_values = np.array([c.uprime for c in cells.values()])
   delta_tot_values = np.array([c.delta_tot for c in cells.values()])
   notch_tot_values = np.array([c.notch_tot for c in cells.values()])
   notch_reporter_values = np.array([c.notch_reporter for c in cells.values()])
   

#first update notch free and delta free (this dynamics is instantaneous)  
   count=0
   max_iter_delta = 1
   max_iter_notch=1
   while((max_iter_delta>1.0e-4 or max_iter_notch>1.0e-4) and count<100):
       delta_free_values = np.array([c.delta_free for c in cells.values()])
       notch_free_values = np.array([c.notch_free for c in cells.values()])
       nearest_neighbour_delta_free_values = np.array([[cells[c2].delta_free for c2 in c.neighbours]
                                            for c in cells.values()] )
       nearest_neighbour_notch_free_values = np.array([[cells[c2].notch_free for c2 in c.neighbours]
                                            for c in cells.values()] )
       (new_delta_free_values,
        new_notch_free_values)= next_iter_delta_notch_free(delta_free_values,
                                                           notch_free_values,
                                                           delta_tot_values,
                                                           notch_tot_values,
                                                           nearest_neighbour_delta_free_values,
                                                           nearest_neighbour_notch_free_values) 
       for (c, 
            new_delta_free,
            new_notch_free) in zip(cells.values(),
                                   new_delta_free_values,
                                   new_notch_free_values,):
           c.delta_free=new_delta_free
           c.notch_free=new_notch_free
       max_iter_delta = np.max(np.abs(new_delta_free_values-delta_free_values))
       max_iter_notch = np.max(np.abs(new_notch_free_values-notch_free_values))
       count+=1
   if (max_iter_delta>1.0e-4 or max_iter_notch>1.0e-4):
       print('convergence failed')
   #then update other quantities.
   notch_free_values = np.array([c.notch_free for c in cells.values()])
   nearest_neighbour_delta_free_values = np.array([[cells[c2].delta_free
                                                     for
                                                     c2 in c.neighbours]
                                         for c in cells.values()] )
   nearest_neighbour_u_values = np.array([[cells[c2].u for c2 in c.neighbours]
                                        for c in cells.values()] )
   
   nearest_neighbour_uprime_values = np.array([[cells[c2].uprime for c2 in c.neighbours]
                                        for c in cells.values()] )

   (new_delta_tot_values,
    new_notch_tot_values,
    new_signal_values,
    new_u_values,
    new_uprime_values,
    new_notch_reporter_values)= next_iter(u_values,
                                          uprime_values,
                                          notch_free_values,
                                          delta_tot_values,
                                          notch_tot_values,
                                          nearest_neighbour_u_values,
                                          nearest_neighbour_uprime_values,
                                          nearest_neighbour_delta_free_values,
                                          notch_reporter_values)
   for (c, 
        new_delta_tot,
        new_notch_tot,
        new_signal,
        new_u,
        new_uprime,
        new_notch_reporter) in zip(cells.values(),
                                      new_delta_tot_values,
                                      new_notch_tot_values,
                                      new_signal_values,
                                      new_u_values,
                                      new_uprime_values,
                                      new_notch_reporter_values):
       c.delta_tot=new_delta_tot
       c.notch_tot=new_notch_tot
       c.signal = new_signal
       c.u = new_u
       c.uprime = new_uprime
       c.notch_reporter=new_notch_reporter
       
   if t%fraction_saved ==0:
       delta_saved.append([c.delta_tot for c in cells.values()])
       notch_saved.append([c.notch_tot for c in cells.values()])
       signal_saved.append([c.signal for c in cells.values()])
       u_saved.append([c.u for c in cells.values()])
       uprime_saved.append([c.uprime for c in cells.values()])
       notch_reporter_saved.append([c.notch_reporter for c in cells.values()])

       

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#SNAPOSHOTS OF THE SIMULATION EVERY  HOUR
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
patches = [] 
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

##u     
times_to_plot=[i*6 for i in range(24)] #60 minutes is 1 hour
for i in times_to_plot:
    fig,ax=plt.subplots()
    p= PatchCollection(patches, cmap = mpl.cm.get_cmap('plasma') , match_original=True)
    ax.add_collection(p)
    p.set_array(np.array(u_saved[i]))
    p.set_clim([-0.1, 1.1])
    ax.axis('equal')
    plt.axis('off')
    plt.savefig('./'+parameters+'/u_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
    plt.close(fig)

#u final frame
plt.close(fig)
fig,ax=plt.subplots()
p= PatchCollection(patches, cmap = mpl.cm.get_cmap('plasma') , match_original=True)
ax.add_collection(p)
p.set_array(np.array(u_saved[-1]))
p.set_clim([-0.1, 1.1])
ax.axis('equal')
plt.axis('off')
plt.savefig('./'+parameters+'/u_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
plt.close(fig)

#uprime final frame
plt.close(fig)
fig,ax=plt.subplots()
p= PatchCollection(patches, cmap =cm.Oranges , match_original=True)
ax.add_collection(p)
p.set_array(np.array(uprime_saved[-1]))
p.set_clim([0, np.max(uprime_saved)])
ax.axis('equal')
plt.axis('off')
plt.savefig('./'+parameters+'/uprime_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
plt.close(fig)


##u_prime
for i in times_to_plot:
    fig,ax=plt.subplots()
    p= PatchCollection(patches, cmap = cm.Oranges , match_original=True)
    ax.add_collection(p)
    p.set_array(np.array(uprime_saved[i]))
    p.set_clim([0, np.max(uprime_saved)])
    ax.axis('equal')
    plt.axis('off')
    plt.savefig('./'+parameters+'/uprime_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
    plt.close(fig)


# ##notch tot
# for i in times_to_plot:
#     fig,ax=plt.subplots()
#     p= PatchCollection(patches, cmap = cm.Purples , match_original=True)
#     ax.add_collection(p)
#     p.set_array(np.array(notch_saved[i]))
#     p.set_clim([0, 2.2])
#     ax.axis('equal')
#     plt.axis('off')
#     plt.savefig('./'+parameters+'/notch_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
#     plt.close(fig)

# ##delta tot
# for i in times_to_plot:
#     fig,ax=plt.subplots()
#     p= PatchCollection(patches, cmap = cm.Blues , match_original=True)
#     ax.add_collection(p)
#     p.set_array(np.array(delta_saved[i]))
#     p.set_clim([0, 1.1])
#     ax.axis('equal')
#     plt.axis('off')
#     plt.savefig('./'+parameters+'/delta_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
#     plt.close(fig)

# ##notch signal
# for i in times_to_plot:
#     fig,ax=plt.subplots()
#     p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
#     ax.add_collection(p)
#     p.set_array(np.array(signal_saved[i]))
#     p.set_clim([0, 1.1])
#     ax.axis('equal')
#     plt.axis('off')
#     plt.savefig('./'+parameters+'/signal_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
#     plt.close(fig)

# #signal final frame
# fig,ax=plt.subplots()
# p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
# ax.add_collection(p)
# p.set_array(np.array(signal_saved[-1]))
# p.set_clim([0, 0.9])
# ax.axis('equal')
# plt.axis('off')
# plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
# plt.close(fig)


# ##notch reporter
# for i in times_to_plot:
#     fig,ax=plt.subplots()
#     p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
#     ax.add_collection(p)
#     p.set_array(np.array(notch_reporter_saved[i]))
#     p.set_clim([0, 0.4])
#     ax.axis('equal')
#     plt.axis('off')
#     plt.savefig('./'+parameters+'/reporter_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
#     plt.close(fig)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PLOT PROFILES OF 1-u, u, and Notch signalling reporter
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
one_hour=6
times=[0,
        3*one_hour,
        8*one_hour,
        13*one_hour,
        18*one_hour,
        22*one_hour]

plt.rcParams.update({'font.size': 16})
colormap = plt.get_cmap("plasma_r")
y_positions_all = [c.position[1] for c in cells.values()]
y_positions = np.unique(y_positions_all)

plt.figure()
for i, time_index in enumerate(times):
    time_hours = np.round(int(t_0+dt*fraction_saved*time_index))
    mean_u_values = []

    for y_pos in y_positions:
        indices =np.where(y_positions_all==y_pos)
        mean_u_values.append(1-np.mean(np.take(u_saved[time_index],indices)))

    plt.plot(y_positions-center_y, mean_u_values, label=str(time_hours)+' h',color=colormap((int(t_0+dt*fraction_saved*time_index)-21) /(43-21)))
plt.xlabel('Distance from vein center (\u03bcm)', fontsize=16)
#plt.ylabel('Signal Intensty (a. u.)', fontsize=16)
plt.xlim(-30,30)
plt.ylim(-0.1,1.1)
plt.xticks([-30,-20,-10,0,10,20,30])
plt.legend(loc='upper right', bbox_to_anchor=(1.32,1))   
plt.savefig('./'+parameters+'/1-u_profile.svg', bbox_inches='tight',format='svg')  
plt.savefig('./'+parameters+'/1-u_profile.png', bbox_inches='tight',format='png')  


plt.figure()
for i, time_index in enumerate(times):
    time_hours = np.round(int(t_0+dt*fraction_saved*time_index))
    mean_u_values = []

    for y_pos in y_positions:
        indices =np.where(y_positions_all==y_pos)
        mean_u_values.append(np.mean(np.take(uprime_saved[time_index],indices)))

    plt.plot(y_positions-center_y, mean_u_values, label=str(time_hours)+' h',color=colormap((int(t_0+dt*fraction_saved*time_index)-21) /(43-21)))
plt.xlabel('Distance from vein center (\u03bcm)', fontsize=16)
#plt.ylabel('Signal Intensty (a. u.)', fontsize=16)
plt.xlim(-30,30)
#plt.ylim(-0.5,1.1)
plt.xticks([-30,-20,-10,0,10,20,30])
plt.legend(loc='upper right', bbox_to_anchor=(1.32,1))   
plt.savefig('./'+parameters+'/uprime_profile.svg', bbox_inches='tight',format='svg')  
plt.savefig('./'+parameters+'/uprime_profile.png', bbox_inches='tight',format='png')  


plt.figure()
for i, time_index in enumerate(times):
    time_hours = np.round(int(t_0+dt*fraction_saved*time_index))
    mean_notch_reporter_values = []

    for y_pos in y_positions:
        indices =np.where(y_positions_all==y_pos)
        mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[time_index],indices)))

    plt.plot(y_positions-center_y,mean_notch_reporter_values, label=str(time_hours)+' h',color=colormap((int(t_0+dt*fraction_saved*time_index)-21) /(43-21)))
plt.xlabel('Distance from vein center (\u03bcm)', fontsize=16)
#plt.ylabel('Signal Intensty (a. u.)', fontsize=16)
plt.xlim(-30,30)
plt.ylim(-0.02,0.6)
plt.xticks([-30,-20,-10,0,10,20,30])
plt.legend(loc='upper right', bbox_to_anchor=(1.32,1))   
plt.savefig('./'+parameters+'/reporter_profile.svg', bbox_inches='tight',format='svg')  
plt.savefig('./'+parameters+'/reporter_profile.png', bbox_inches='tight',format='png')  


