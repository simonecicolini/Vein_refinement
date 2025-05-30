
import numpy as np
from cell import cell
from numba import njit


#function to perform iteration
#accerelated with numba
@njit
def next_iter_delta_notch_free(delta_free_values,
                               notch_free_values,
                               delta_tot_values,
                               notch_tot_values,
                               nearest_neighbour_delta_free_values,
                               nearest_neighbour_notch_free_values,
                               Kcis,
                               Ktrans):
    """perform an iteration in pseudotime to calculate free Delta/Notch concentrations
    and signal, given total values of Delta and Notch
    """
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
              nearest_neighbour_delta_free_values,
              notch_reporter_values,
              ffact_u,
              ffact_Notch,
              ffact_reporter,
              Ktrans,
              r,
              JI,
              JA,
              N_low,
              N_high,
              D_high,
              Sstar,
              alphaS,
              S0,
              noDelta,
              DeltaFixed,
              extraSignal,
              extraSignalValue,
              extraVeinActivation,
              extraVeinActivationValue):
    """perform an iteration to evolve u, Deltatot, Ntot
    """
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
         nearest_neighbours_delta_free,
         notch_reporter) in zip(u_values,
                                notch_free_values,
                                delta_tot_values,
                                notch_tot_values,
                                nearest_neighbour_u_values,
                                nearest_neighbour_delta_free_values,
                                notch_reporter_values):

        #addition of other terms
        nearest_u_sum = 0
        nearest_delta_free_sum =0 
        for (neighbour_u, neighbour_delta_free) in zip(nearest_neighbours_u,
                                                       nearest_neighbours_delta_free):
            nearest_u_sum += neighbour_u
            nearest_delta_free_sum+= neighbour_delta_free
        notch_delta_trans = Ktrans/6*notch_free*nearest_delta_free_sum
        #total delta and total notch
        #basic behaviour
        if (not noDelta) and (not DeltaFixed):
            new_delta_tot = delta_tot + ffact_Notch*(D_high* u*(u>0)-delta_tot)
        if noDelta:
            new_delta_tot = delta_tot + ffact_Notch*(0-delta_tot)
        if DeltaFixed:
            new_delta_tot = delta_tot 
            
        new_notch_tot = notch_tot + ffact_Notch*(N_low-notch_tot)
        signal =  notch_delta_trans 
        if extraSignal:
            signal += extraSignalValue 
        #extra production if high notch signalling
        new_notch_tot += ffact_Notch *(N_high-N_low)* (1+np.tanh((signal-Sstar)/alphaS))/2.0
        new_u= u+ffact_u*(-u*(u-1)*(u-r)-JI *(signal>S0)*(u>0)* signal + JA  *(nearest_u_sum-6*u))
        if extraVeinActivation:
            new_u+= -ffact_u* extraVeinActivationValue * (u-1)
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

# =============================================================================
# INITIALISATION
# ============================================================================
def run_sim(cells, 
            r, JI, JA, 
            Kcis, Ktrans, 
            tau_u, tau_Notch, tau_reporter,
            N_low, N_high, D_high,
            Sstar, alphaS, S0,
            t_0, dt, N_times, fraction_saved,
            noDelta=False,
            DeltaFixed=False,
            extraSignal=False,
            extraSignalValue=0,
            extraVeinActivation=False,
            extraVeinActivationValue=0
            ):
    """
    Run simulation
    cells: dictionary of cells, containing values which are used for initial condition
    r to S0: model parameters
    t_0: initial time in hours - for comparison with experiment
    dt: time increment in hours
    N_times: total times simulated
    fraction_saved: which fraction of simulation steps to save and return
    """
    #for saving
    delta_saved = []
    notch_saved = []
    signal_saved = []
    u_saved = []
    notch_reporter_saved = []
    
    
    #to gain speed, define normalised time increments
    ffact_Notch=dt/tau_Notch
    ffact_u=dt/tau_u
    ffact_reporter=dt/tau_reporter
       
    # =============================================================================
    # RUNNING
    # =============================================================================
         
       
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
                                                               nearest_neighbour_notch_free_values,
                                                               Kcis,
                                                               Ktrans) 
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
       (new_delta_tot_values,
        new_notch_tot_values,
        new_signal_values,
        new_u_values,
        new_notch_reporter_values)= next_iter(u_values,
                                              notch_free_values,
                                              delta_tot_values,
                                              notch_tot_values,
                                              nearest_neighbour_u_values,
                                              nearest_neighbour_delta_free_values,
                                              notch_reporter_values,
                                              ffact_u,
                                              ffact_Notch,
                                              ffact_reporter,
                                              Ktrans,
                                              r,
                                              JI,
                                              JA,
                                              N_low,
                                              N_high,
                                              D_high,
                                              Sstar,
                                              alphaS,
                                              S0,
                                              noDelta,
                                              DeltaFixed,
                                              extraSignal,
                                              extraSignalValue,
                                              extraVeinActivation,
                                              extraVeinActivationValue)
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
    return delta_saved, notch_saved, signal_saved, u_saved, notch_reporter_saved
