import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from cell import cell
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
import matplotlib.image as mpimg
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
        
   
# construct list of second neighbours
for k, c in cells.items():
   first_neighbours=c.neighbours
   second_neighbours=[]
   for neighbour_index in first_neighbours:
       neighbours_of_the_neighbour = cells[neighbour_index].neighbours
       for neighbour_of_the_neighbour in neighbours_of_the_neighbour:
           if (neighbour_of_the_neighbour != k) and (neighbour_of_the_neighbour not in first_neighbours):
               second_neighbours.append(neighbour_of_the_neighbour)
   c.second_neighbours= np.unique(second_neighbours)
               
    
 
# =============================================================================
# RUNNING
# =============================================================================

dt = 0.0167 #time step in hours (~ 1 minute)
t_0 = 21 #initial time  in hours
N_times = 50001
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
Ktrans = 1800
#JA, JI, r
#JA = 0.015
JI = 0.35
r = 0.5

# Notch production
Nlow = 0.3
Nhigh=2.2

#other parameters
Sstar = 0.25 #signal level necessary for notch activation
alpha_S= 0.025 #signal sensitivity
S0=0.4

#comment/uncomment one of the two following lines to set the initial condition
condition='regular_stripe' #smooth initial condition of the provein domain (FigS4J)
#condition='rough_stripe'  #noisy initial condition of the provein domain (FigS4J')

#the parameter alpha (0 ≤ alpha ≤ 1) is the relative weight of signalling to second vs first nearest neighbours (called "range" in FigS4J-J')
#the parameter J is the is the overall magnitude of signalling, defined such that 6J is the overall contribution of the sum on first and second nearest neighbours, when all cells have u = 1
for alpha in [0, 0.25, 0.5, 0.75, 1 ]:
    for J in [0, 0.015, 0.03, 0.045, 0.06, 0.15]:
        JA1=J/(1+2*alpha) #interaction strength among nearest neighbours (alpha*J1 is the interaction strength among second neighbours)
        JAprime=JA1
        parameters = 'Phase_diagram_FigS4J-J\'/'+condition+'_JI='+str(JI)+'r='+str(r)+'tau_u='+str(tau_u)+'_tauDN='+str(tau_Notch)+'_totTime='+str(dt*N_times)+'h/Kcis='+str(Kcis)+'Ktrans='+str(Ktrans)+'_Sstar='+str(Sstar)+'alphaS='+str(alpha_S)+'Nlow='+str(Nlow)+'Nhigh='+str(Nhigh)+'/J='+str(J)+'/range='+str(alpha)
        
        if not os.path.exists('./'+parameters):
            os.makedirs('./'+parameters)
        
        
        # initialize values of u
        center_x = np.mean([cell.position[0] for cell in cells.values()])
        center_y = np.mean([cell.position[1] for cell in cells.values()])
        sigma=10
        sigma_rep=7
        if condition=='regular_stripe':
            for k, c in cells.items():
                c.u=np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
                c.delta_tot = np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
                c.delta_free =np.exp(-(c.position[1]-center_y)**2/2/sigma**2)
                c.notch_tot=Nlow
                c.notch_free=Nlow
                c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
                c.delta_notch_cis=0
                c.delta_notch_trans=0
                c.notch_delta_trans=0
        if condition =='rough_stripe'   :
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
                c.notch_tot=Nlow
                c.notch_free=Nlow
                c.notch_reporter=0.2*(np.exp(-(c.position[1]-(center_y-13))**2/2/sigma_rep**2) + np.exp(-(c.position[1]-(center_y+13))**2/2/sigma_rep**2) )
                c.delta_notch_cis=0
                c.delta_notch_trans=0
                c.notch_delta_trans=0

        
        #for saving
        delta_saved = []
        notch_saved = []
        signal_saved = []
        u_saved = []
        notch_reporter_saved = []
        
        
        #to gain speed
        ffact_Notch=dt/tau_Notch
        ffact_u=dt/tau_u
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
                Kt=Ktrans/6
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
        def next_iter(u_values,
                      notch_free_values,
                      delta_tot_values,
                      notch_tot_values,
                      nearest_neighbour_u_values,
                      second_neighbour_u_values,
                      nearest_neighbour_delta_free_values,
                      notch_reporter_values):
            new_delta_tot_values = []
            new_notch_tot_values = []
            new_signal_values = []
            new_u_values = []
            new_notch_reporter_values = []
            for (u,
                 notch_free,
                 delta_tot,
                 notch_tot,
                 nearest_neighbours_u,
                 second_neighbours_u,
                 nearest_neighbours_delta_free,
                 notch_reporter) in zip(u_values,
                                        notch_free_values,
                                        delta_tot_values,
                                        notch_tot_values,
                                        nearest_neighbour_u_values,
                                        second_neighbour_u_values,
                                        nearest_neighbour_delta_free_values,
                                        notch_reporter_values):
        
                #addition of other terms
                nearest_u_sum = 0
                nearest_delta_free_sum =0 
                for (neighbour_u, neighbour_delta_free) in zip(nearest_neighbours_u,
                                                               nearest_neighbours_delta_free):
                    nearest_u_sum += neighbour_u
                    nearest_delta_free_sum+= neighbour_delta_free
                    
                second_u_sum =0 
                for second_neighbour_u in   second_neighbours_u:
                    second_u_sum+= second_neighbour_u
                    
                    
                notch_delta_trans = Ktrans*notch_free*nearest_delta_free_sum/6 
                #total delta and total notch
                #basic behaviour
                new_delta_tot = delta_tot + ffact_Notch*(u*(u>0)-delta_tot)
                new_notch_tot = notch_tot + ffact_Notch*(Nlow-notch_tot)
                    
                signal =  notch_delta_trans 
                #extra production if high notch signalling
                new_notch_tot += ffact_Notch *(Nhigh-Nlow)* (1+np.tanh((signal-Sstar)/alpha_S))/2.0
                new_u= u+ffact_u*(-u*(u-1)*(u-r)-JI * signal *(signal>S0)*(u>0) - JAprime*6*u +JA1* ( nearest_u_sum + alpha * second_u_sum))
                #Notch signalling reporter
                new_notch_reporter = notch_reporter + ffact_reporter*(signal-notch_reporter)
                #store result
                new_delta_tot_values.append(new_delta_tot)
                new_notch_tot_values.append(new_notch_tot )
                new_signal_values.append(signal)
                new_u_values.append(new_u)
                new_notch_reporter_values.append(new_notch_reporter)
            return (new_delta_tot_values,
                    new_notch_tot_values,
                    new_signal_values,
                    new_u_values,
                    new_notch_reporter_values)
        
           
        for t in range(N_times):
           if t%fraction_saved ==0:
               print(np.round(t/N_times*100,2), '%executed')
           u_values = np.array([c.u for c in cells.values()])
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
           
           second_neighbour_u_values = np.array([[cells[c2].u for c2 in c.second_neighbours]
                                                 for c in cells.values()] )
           (new_delta_tot_values,
            new_notch_tot_values,
            new_signal_values,
            new_u_values,
            new_notch_reporter_values)= next_iter(u_values,
                                                  notch_free_values,
                                                  delta_tot_values,
                                                  notch_tot_values,
                                                  nearest_neighbour_u_values,
                                                  second_neighbour_u_values ,
                                                  nearest_neighbour_delta_free_values,
                                                  notch_reporter_values)
           for (c, 
                new_delta_tot,
                new_notch_tot,
                new_signal,
                new_u,
                new_notch_reporter) in zip(cells.values(),
                                              new_delta_tot_values,
                                              new_notch_tot_values,
                                              new_signal_values,
                                              new_u_values,
                                              new_notch_reporter_values):
               c.delta_tot=new_delta_tot
               c.notch_tot=new_notch_tot
               c.signal = new_signal
               c.u = new_u
               c.notch_reporter=new_notch_reporter
               
           if t%fraction_saved ==0:
               delta_saved.append([c.delta_tot for c in cells.values()])
               notch_saved.append([c.notch_tot for c in cells.values()])
               signal_saved.append([c.signal for c in cells.values()])
               u_saved.append([c.u for c in cells.values()])
               notch_reporter_saved.append([c.notch_reporter for c in cells.values()])
        
               
        # =============================================================================
        # save packings final frame
        # =============================================================================        
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
        p.set_clim([0, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters+'/signal_final.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

#assemble final snapshots together to generate the phase diagram of u and Notch signalling (Fig.S4J-J')
fig=plt.figure(figsize=(25,25))
columns=5
rows=6
J_values=[0, 0.015, 0.03, 0.045, 0.06, 0.15]
J_values.reverse()
alpha_values=[0, 0.25, 0.5, 0.75, 1]
i=0
for J in J_values:
    for alpha in alpha_values:
        i=i+1
        try:
            img=mpimg.imread('./Phase_diagram_FigS4J-J\'/'+condition+'_JI=0.35r=0.5tau_u=2_tauDN=6_totTime='+str(dt*N_times)+'h/Kcis=1500Ktrans=1800_Sstar=0.25alphaS=0.025Nlow=0.3Nhigh=2.2/J='+str(J)+'/range='+str(alpha)+'/u_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
fig1.savefig('./Phase_diagram_FigS4J-J\'/'+condition+'_JI=0.35r=0.5tau_u=2_tauDN=6_totTime='+str(dt*N_times)+'h/Kcis=1500Ktrans=1800_Sstar=0.25alphaS=0.025Nlow=0.3Nhigh=2.2/u_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)

fig=plt.figure(figsize=(25,25))
columns=5
rows=6
J_values=[0, 0.015, 0.03, 0.045, 0.06, 0.15]
J_values.reverse()
alpha_values=[0, 0.25, 0.5, 0.75, 1]
i=0
for J in J_values:
    for alpha in alpha_values:
        i=i+1
        try:
            img=mpimg.imread('./Phase_diagram_FigS4J-J\'/'+condition+'_JI=0.35r=0.5tau_u=2_tauDN=6_totTime='+str(dt*N_times)+'h/Kcis=1500Ktrans=1800_Sstar=0.25alphaS=0.025Nlow=0.3Nhigh=2.2/J='+str(J)+'/range='+str(alpha)+'/signal_final.png')
            fig.add_subplot(rows,columns,i)
            plt.axis('off')
            plt.gca().set_axis_off()
    
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img,interpolation="nearest")
        except FileNotFoundError:
            print('file not present')
fig1=plt.gcf()        
#plt.show()
fig1.savefig('./Phase_diagram_FigS4J-J\'/'+condition+'_JI=0.35r=0.5tau_u=2_tauDN=6_totTime='+str(dt*N_times)+'h/Kcis=1500Ktrans=1800_Sstar=0.25alphaS=0.025Nlow=0.3Nhigh=2.2/signal_phase_diagram.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=130)

        
 
