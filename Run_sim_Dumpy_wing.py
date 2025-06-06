#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 19:17:04 2025

@author: simonecicolini
"""

import pandas as pd
import numpy as np
import tissue_miner_tools as tml
import tissue_miner as tm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
import os 
from tqdm import tqdm
from scipy.optimize import curve_fit

from cell_wing import cell_wing


pd.options.mode.chained_assignment = None
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

#paths to databases 
Neighbors_info='./Dumpy_DSRF_1/200924_wing1/cellNeighbors.csv' #neighbours relationships
data_base='./Dumpy_DSRF_1/200924_wing1/DBcells2.csv' #contains the relevant quantities for each cell at each time frame, such as DSRF level, cell centroids, lineages,...
DB=pd.read_csv(data_base);DB=DB.drop(['Unnamed: 0'],axis=1)

##import movie info using the tissue_miner class
movieDatabaseDir='./Dumpy_DSRF_1/' 
name='200924_wing1' 
movieRoiPath= 'roi_bt/' 
movieRoiFile= 'lgRoiSmoothed'            
movie=tm.Movie(name, path= movieDatabaseDir, ROI_path= movieRoiPath, ROI_filename= movieRoiFile)
movie.load_cellshapes()
Neighbors=pd.read_csv(Neighbors_info)
#The cell 10 000 is not an actual cell but the border of the tissue:
DB=DB[DB.cell_id!=10000]
##--------------------------------------------------------------------------------------------------------
#parameters (time scales are expressed in  hours and length scales in microns)
#tau_reporter half-time mcherry (reporter of Notch signalling activity)
#tau_Notch degradation time of Dtot and Ntot
tau_Notch=6; tau_u=2;   tau_reporter=8
starting_frame=53 #corresponding to 21 hAPF
# u dynamics
r=0.5; JI=0.35; JA=0.015; S0=0.4  
#binding affinities  in cis and trans
Kcis=1500; Ktrans=1800

#response of Nocth production to Nothc signalling
signal_activation_threshold= 0.25 #Sstar
signal_activation_sensitivity= 0.025 #alpha_S
Nlow=0.3 
N0=Nlow
Nhigh=2.2

#inital conditons for the notch signalling reporter
sigma_reporter=0
reporter_amplitude=0

dx=4 # average distance between cells np.sqrt(DB.area.mean()*2/(np.sqrt(3)));
dt=0.0167 #time step (1 minute in hours)
StepsPerFrame=5# time interval between frames = 5 minutes

ffact_Notch=dt/tau_Notch
ffact_u=dt/tau_u
ffact_reporter=dt/tau_reporter

folder='Results_Dumpy'
param='JA='+str(JA)+'r='+str(r)+'Nlow='+str(Nlow)+'Nhigh='+str(Nhigh)+'_Sstar='+str(signal_activation_threshold)+'_alphaS='+str(signal_activation_sensitivity)+'tauDN='+str(tau_Notch)+'tau_u='+str(tau_u)
if folder not in os.listdir('./'):
    os.mkdir(folder)
if param not in os.listdir('./'+folder):
    os.mkdir('./'+folder+'/'+param)
    os.mkdir('./'+folder+'/'+param+'/u')
    os.mkdir('./'+folder+'/'+param+'/Dtot')
    os.mkdir('./'+folder+'/'+param+'/Ntot')
    os.mkdir('./'+folder+'/'+param+'/NDtrans')
    os.mkdir('./'+folder+'/'+param+'/reporter')


#--------------------------------------------------------------------------------------------
#function to generate snapshots of the simulation on the wing template
def plot_frame_cells(MOVIE, frame, location, coll_df, color_column, c_min= 0., c_max= 1., n_ticks= 5, figsize= (6, 6), polygon_lw= .1, color_map= cm.afmhot, title= ''):
        """
        Plots a collection of polygons provided in coll_df DataFrame in 'plot_vertices' column.
        Color is assigned based on values in color_column column of the coll_df DataFrame.
        c_min and c_max control the range of the colormap.
        Colormap can be provided by user and is set to afmhot by default.
        """
        plt.figure(figsize= figsize);
        plt.title(title, fontsize= 25)
        MOVIE.show_image(frame)
        plt.gca().autoscale_view()
        plt.gca().set_aspect('equal')
        colors= color_map((coll_df[color_column].values-c_min)/(c_max - c_min)) 
        coll= mc.PolyCollection(coll_df['plot_vertices'].values, lw= polygon_lw)
        coll.set(facecolors= colors)
        plt.gca().add_collection(coll)
        plt.xlim(0, 1200)
        plt.ylim(0, 1200) 
        plt.gca().invert_yaxis()
        plt.axis('off')
        #divider= make_axes_locatable(plt.gca())
        #cax= divider.append_axes('right', size= '5%', pad= 0.05)
        mm= cm.ScalarMappable(cmap= color_map)
        mm.set_array(colors)
        #cbar= plt.colorbar(mm, cax= cax, cmap= color_map, ticks= np.linspace(0, 1, n_ticks + 1))
        #cbar.ax.set_yticklabels(np.linspace(c_min, c_max, n_ticks + 1))
        plt.savefig(location+str(frame)+'.tif',format='tif',dpi=120, bbox_inches='tight') 
        plt.close()

#below functions to allocate quantities to boundary cells
def find_u_largest_x(CELL, NEIGHBORS, DataB_f, FRAME): 
    this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
    this_y=DataB_f[DataB_f.cell_id==CELL].center_y.values[0]    
    if FRAME>=starting_frame:
        try:
            if (this_x>=0.8*max_x_fr)or(this_y<=0.7*max_y_fr):
                if this_x<= midx:
                    max_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.max()-0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x>=max_x)].cell_id.values[0]
                    u_return=DataB_f[DataB_f.cell_id==neigh_id].u.values[0]
                else:
                    min_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.min()+0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x<=min_x)].cell_id.values[0] 
                    u_return=DataB_f[DataB_f.cell_id==neigh_id].u.values[0]
                    
            else:                
                min_y=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_y.min()+0.00001
                neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_y<=min_y)].cell_id.values[0] 
                u_return=DataB_f[DataB_f.cell_id==neigh_id].u.values[0]
                
        except IndexError: 
            u_return=np.heaviside(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m),0)*(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m))
    else:
        u_return=np.heaviside(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m),0)*(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m))
 
    return u_return

def find_Dfree_largest_x(CELL, NEIGHBORS, DataB_f, FRAME):  
    this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
    this_y=DataB_f[DataB_f.cell_id==CELL].center_y.values[0]    
    if FRAME>=starting_frame:
        this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
        try:
            if (this_x>=0.8*max_x_fr)or(this_y<=0.7*max_y_fr):
                if this_x<= midx:
                    max_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.max()-0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x>=max_x)].cell_id.values[0]
                    Dfree_return=DataB_f[DataB_f.cell_id==neigh_id].Dfree.values[0]
                else:
                    min_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.min()+0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x<=min_x)].cell_id.values[0] 
                    Dfree_return=DataB_f[DataB_f.cell_id==neigh_id].Dfree.values[0]
                    
            else:                
                min_y=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_y.min()+0.00001
                neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_y<=min_y)].cell_id.values[0] 
                Dfree_return=DataB_f[DataB_f.cell_id==neigh_id].Dfree.values[0]
        except IndexError: 
            Dfree_return=np.heaviside(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m),0)*(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m))   
    else:
        Dfree_return=np.heaviside(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m),0)*(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m))           
    return Dfree_return

def find_Dtot_largest_x(CELL, NEIGHBORS, DataB_f, FRAME):  
    this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
    this_y=DataB_f[DataB_f.cell_id==CELL].center_y.values[0]    
    if FRAME >=starting_frame:
        this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
        try:
            if (this_x>=0.8*max_x_fr)or(this_y<=0.7*max_y_fr):
                if this_x<= midx:
                    max_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.max()-0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x>=max_x)].cell_id.values[0]
                    Dtot_return=DataB_f[DataB_f.cell_id==neigh_id].Dtot.values[0]
                else:
                    min_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.min()+0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x<=min_x)].cell_id.values[0] 
                    Dtot_return=DataB_f[DataB_f.cell_id==neigh_id].Dtot.values[0]
                    
            else:                
                min_y=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_y.min()+0.00001
                neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_y<=min_y)].cell_id.values[0] 
                Dtot_return=DataB_f[DataB_f.cell_id==neigh_id].Dtot.values[0]
        except IndexError: 
            Dtot_return=np.heaviside(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m),0)*(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m))
    else:
        Dtot_return=np.heaviside(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m),0)*(1-(DataB_f[DataB_f.cell_id==CELL].DSRF_Conc.values[0]-m)/(M-m))           
            
    return Dtot_return

def find_Nfree_largest_x(CELL, NEIGHBORS, DataB_f, FRAME): 
    this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
    this_y=DataB_f[DataB_f.cell_id==CELL].center_y.values[0]    
    if FRAME>=starting_frame:
        this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
        try:
            if (this_x>=0.8*max_x_fr)or(this_y<=0.7*max_y_fr):
                if this_x<= midx:
                    max_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.max()-0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x>=max_x)].cell_id.values[0]
                    Nfree_return=DataB_f[DataB_f.cell_id==neigh_id].Nfree.values[0]
                else:
                    min_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.min()+0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x<=min_x)].cell_id.values[0] 
                    Nfree_return=DataB_f[DataB_f.cell_id==neigh_id].Nfree.values[0]
                    
            else:                
                min_y=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_y.min()+0.00001
                neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_y<=min_y)].cell_id.values[0] 
                Nfree_return=DataB_f[DataB_f.cell_id==neigh_id].Nfree.values[0]
        except IndexError: 
            Nfree_return=Nlow
    else:
        Nfree_return=Nlow
    return Nfree_return

def find_Ntot_largest_x(CELL, NEIGHBORS, DataB_f, FRAME):  
    this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
    this_y=DataB_f[DataB_f.cell_id==CELL].center_y.values[0]    
    if FRAME >=starting_frame:
        this_x=DataB_f[DataB_f.cell_id==CELL].center_x.values[0]
        try:
            if (this_x>=0.8*max_x_fr)or(this_y<=0.7*max_y_fr):
                if this_x<= midx:
                    max_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.max()-0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x>=max_x)].cell_id.values[0]
                    Ntot_return=DataB_f[DataB_f.cell_id==neigh_id].Ntot.values[0]
                else:
                    min_x=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_x.min()+0.00001
                    neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_x<=min_x)].cell_id.values[0] 
                    Ntot_return=DataB_f[DataB_f.cell_id==neigh_id].Ntot.values[0]
                    
            else:                
                min_y=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].center_y.min()+0.00001
                neigh_id=DataB_f[(DataB_f.cell_id.isin(NEIGHBORS))&(DataB_f.center_y<=min_y)].cell_id.values[0] 
                Ntot_return=DataB_f[DataB_f.cell_id==neigh_id].Ntot.values[0]
        except IndexError: 
            Ntot_return=Nlow
    else:
        Ntot_return=Nlow
    return Ntot_return

#iteratively calculate free Delta and free Notch 
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
        if len(nearest_neighbours_delta_free)>0:
            XD =nearest_delta_free_sum/len(nearest_neighbours_delta_free)
            XN =nearest_notch_free_sum/len(nearest_neighbours_delta_free)    
        else: 
            XD =0
            XN =0   
            
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

#function that updates cell variables at each iteration of the simulation
def next_iter(u_values,
              notch_free_values,
              delta_tot_values,
              notch_tot_values,
              nearest_neighbour_u_values,
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
         nearest_neighbours_delta_free,
         notch_reporter) in zip(u_values,
                                notch_free_values,
                                delta_tot_values,
                                notch_tot_values,
                                nearest_neighbour_u_values,
                                nearest_neighbour_delta_free_values,
                                notch_reporter_values):

        #addition of interaction terms
        nearest_u_sum = 0
        nearest_delta_free_sum =0 
        for (neighbour_u, neighbour_delta_free) in zip(nearest_neighbours_u,
                                                       nearest_neighbours_delta_free):
            nearest_u_sum += neighbour_u #activation term
            nearest_delta_free_sum+= neighbour_delta_free
            
        if len(nearest_neighbours_delta_free)>0:
            notch_delta_trans = Ktrans*notch_free*nearest_delta_free_sum/len(nearest_neighbours_delta_free)
        else: 
            notch_delta_trans=0
        #total delta and total notch
        new_delta_tot = delta_tot + ffact_Notch*(u*np.heaviside(u,0)-delta_tot)
        new_notch_tot = notch_tot + ffact_Notch*(Nlow-notch_tot)
            
        #notch signalling
        signal =  notch_delta_trans
        #extra production if high notch signalling
        new_notch_tot += ffact_Notch * (Nhigh-Nlow)*(1+np.tanh((signal-signal_activation_threshold)/signal_activation_sensitivity))/2.0
        new_u= u+ffact_u*(-u*(u-1)*(u-r)-JI * signal *(signal>S0)*(u>0)+ JA  *(nearest_u_sum-len(nearest_neighbours_u)*u))
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


#_______________________________________________________________________________________________________
#initialization:
DB['Ntot']=Nlow;  DB['Nfree']=Nlow;  DB['NDtrans']=0.; DB['Dtot']=0.; DB['u']=0.; DB['Dfree']=0.;  DB['reporter']=0.
DB['Border']=0; DB['neighbors']=pd.Series([[]]*len(DB), dtype=object)
#functions to fit the peaks in the histogram of DSRF concentration
m=448. ##position first DSRF peak
M=1128. ##position second DSRF peak at initial time
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, sigma1, A1, mu2, sigma2, A2): 
    return gauss(x,m,sigma1,A1)+gauss(x,mu2,sigma2,A2)
#initial guess for fitting M (second peak in DSRF histogram)
expected=(300,20,1600,200,50)

#Initialise all cells  with u_exp (experimental value of u) and Dtot=u_exp \theta(u_exp)
for FR in tqdm(range(186)):
    data=DB[DB.frame==FR].DSRF_Conc
    y,x,_=plt.hist(data,bins=50); 
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    params, cov = curve_fit(bimodal, x, y, expected)
    M=params[2]
    plt.close()

    #map DSRF concentration to u
    DB.loc[DB.frame==FR,'u']=1-(DB[DB.frame==FR].DSRF_Conc-m)/(M-m)
    DB.loc[DB.frame==FR,'Dtot']=np.heaviside((1-(DB[DB.frame==FR].DSRF_Conc-m)/(M-m)),0)*(1-(DB[DB.frame==FR].DSRF_Conc-m)/(M-m))
    DB.loc[DB.frame==FR,'Dfree']=np.heaviside((1-(DB[DB.frame==FR].DSRF_Conc-m)/(M-m)),0)*(1-(DB[DB.frame==FR].DSRF_Conc-m)/(M-m))


#=============================================================
#RUN SIMULATION
#=============================================================
for FRAME in tqdm(range(starting_frame,186)):
    DB_f=DB[DB.frame==FRAME] #selct subset of the dataframe for this frame
    Neighbors_f=Neighbors.loc[Neighbors.frame==FRAME]
    #add neighbours relationships to the database
    DB_f['neighbors']=DB_f['cell_id'].transform(lambda x: Neighbors_f[(Neighbors_f.cell_id==x)].neighbor_cell_id.unique())
    DB_f['NeighborsCount']=DB_f['neighbors'].apply(lambda x: len(x))
    #find boundary cells 
    #a cell is part of the border if the number od edges is different from the number of first neighbours  
    frame_cellshapes=tml.subset_dataframe(movie.cellshapes, ['frame'], [FRAME])    
    Differences=[]
    for cell in DB_f.cell_id.unique():
        Differences.append((-DB_f[DB_f.cell_id==cell].NeighborsCount.values+frame_cellshapes[frame_cellshapes.cell_id==cell].bond_order.max())[0])
    DB_f['Border']=Differences
    Border_cells=DB_f[DB_f.Border!=0].cell_id.unique()    
    DB[DB.frame==FRAME]=DB_f
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    #allocate u, Delta, and Notch to boundary cells the value of their neighboor along the direction of the vein
    max_x_fr=DB_f.center_x.max()
    max_y_fr=DB_f.center_y.max()
    min_x_fr=DB_f.center_x.min()
    midx=0.5*(max_x_fr+min_x_fr)
    
    #fit second peak of DSRF histogram 
    data=DB[DB.frame==FRAME].DSRF_Conc
    y,x,_=plt.hist(data,bins=50); 
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    params, cov = curve_fit(bimodal, x, y, expected)
    M=params[2]
    plt.close()
    
    DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'u']=DB_f.apply(lambda x: find_u_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)
    
    DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Dfree']=DB_f.apply(lambda x: find_Dfree_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)

    DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Dtot']=DB_f.apply(lambda x: find_Dtot_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)

    DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Ntot']=DB_f.apply(lambda x: find_Ntot_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)

    DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Nfree']=DB_f.apply(lambda x: find_Nfree_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)
    
    # #++++++++++++++++++++++++++++++++++++++++++++++   
    DB_f=DB[DB.frame==FRAME] #redefine DB_f so that contains the value u, Dfree, Dtot, Ntot, Nfree, calculated above for border cells 
    #this part plots the simulated tissue for this frame: 
    frame_polygons_0=frame_cellshapes.groupby('cell_id').apply(lambda x: list(zip(x['x_pos'].values, x['y_pos'].values))).reset_index().rename(columns= {0: 'plot_vertices'})
    frame_polygons=DB[DB.frame==FRAME][[u'frame', u'cell_id', u'center_x', u'center_y',u'Dfree', 
                                        u'u', u'Dtot', u'Ntot', u'NDtrans' ,u'Nfree', u'reporter' ]].merge(frame_polygons_0, on= 'cell_id')
    plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/u/_', frame_polygons, title= 'u', color_column= 'u', c_min= -0.1, c_max= 1.1, color_map=cm.plasma) #cm.gist_rainbow)
    plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/Dtot/_', frame_polygons, title= 'Dtot', color_column= 'Dtot', c_min= 0, c_max= 1.1, color_map=cm.Blues) #cm.gist_rainbow)
    plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/Ntot/_', frame_polygons, title= 'Ntot', color_column= 'Ntot', c_min= 0, c_max= 1.2, color_map=cm.Purples) #cm.gist_rainbow)
    plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/NDtrans/_', frame_polygons, title= 'NDtrans', color_column= 'NDtrans', c_min= 0, c_max= 1.2, color_map=cm.Reds) #cm.gist_rainbow)
    plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/reporter/_', frame_polygons, title= 'reporter', color_column= 'reporter', c_min= 0, c_max=1, color_map=cm.Reds) #cm.gist_rainbow)   
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #create a dictionary of cells with their state at this time frame
    cells = {k: cell_wing(u_, D_tot, N_tot, D_free, N_free, Signal, Reporter, index_, border_)
             for k, (u_, D_tot, N_tot, D_free, N_free, Signal, Reporter, index_, border_)
             in enumerate(zip( np.array(DB_f.u.values), np.array(DB_f.Dtot.values), np.array(DB_f.Ntot.values), 
                              np.array(DB_f.Dfree.values), np.array(DB_f.Nfree.values) ,np.array(DB_f.NDtrans.values) ,np.array(DB_f.reporter.values), np.array(DB_f.cell_id.values), np.array(DB_f.Border.values)) )}

    these_keys=list(cells.keys())
    for k1 in these_keys:
        cells[cells[k1].index]=cells[k1]
        del cells[k1]
            
    if FRAME>=starting_frame:
        for k, c in cells.items():
            c.neighbours=DB_f[DB_f.cell_id==c.index].neighbors.values[0]
        
        for step in  range(StepsPerFrame):
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
                                                     for c in cells.values()], dtype=object )
                nearest_neighbour_notch_free_values = np.array([[cells[c2].notch_free for c2 in c.neighbours]
                                                     for c in cells.values()], dtype=object )
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
                                                  for c in cells.values()] , dtype=object)
            nearest_neighbour_u_values = np.array([[cells[c2].u for c2 in c.neighbours]
                                                 for c in cells.values()] , dtype=object)
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

    # store new quantities in the database         
    DB_f['u']=np.array([c.u for c in cells.values()])
    DB_f['Dtot']=np.array([c.delta_tot for c in cells.values()])
    DB_f['Ntot']=np.array([c.notch_tot for c in cells.values()])
    DB_f['NDtrans']=np.array([c.signal for c in cells.values()])
    DB_f['reporter']=np.array([c.notch_reporter for c in cells.values()])
    DB_f['Dfree']=np.array([c.delta_free for c in cells.values()])
    DB_f['Nfree']=np.array([c.notch_free for c in cells.values()])
    # deal with the case of cells that disappear between this frame and the next one:
    disappearing_cells=DB_f[(DB_f.last_occ==FRAME)].cell_id.unique()
    disappearing_cells_division=DB_f[(DB_f.disappears_by=='Division')&(DB_f.last_occ==FRAME)].cell_id.unique()
    #this part treats with an error happening when the number of cells between two frames don't match
    list_cells_t=DB[(DB.frame==FRAME)&(~DB.cell_id.isin(disappearing_cells))].cell_id.unique()
    try : 
        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id.isin(list_cells_t)),['u','Dfree',
                                                                    'Nfree','Dtot','Ntot','NDtrans','reporter']]=DB_f[~DB_f.cell_id.isin(disappearing_cells)][['u','Dfree',
                                                                                                                                          'Nfree','Dtot','Ntot','NDtrans','reporter']].values
    except ValueError :
        list_next_time=DB.loc[(DB.frame==FRAME+1)&(DB.cell_id.isin(list_cells_t))].cell_id.unique()
        list_this_time=DB_f[~DB_f.cell_id.isin(disappearing_cells)].cell_id.unique()
        list_difference = [item for item in list_this_time if item not in list_next_time]
        DB_f=DB_f[~DB_f.cell_id.isin(list_difference)]
        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id.isin(list_cells_t)),['u','Dfree',
                                                                    'Nfree','Dtot','Ntot','NDtrans','reporter']]=DB_f[~DB_f.cell_id.isin(disappearing_cells)][['u','Dfree',
                                                                                                                                          'Nfree','Dtot','Ntot','NDtrans','reporter']].values
    #deal with cells appearing by division
    #allocate to daughter cells the same state of the mother
    for CELL in disappearing_cells_division :         
        daughter1,daughter2=DB_f[DB_f.cell_id==CELL].left_daughter_cell_id.unique()[0],DB_f[DB_f.cell_id==CELL].right_daughter_cell_id.unique()[0]
        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id==daughter1),['Dfree',
                                                            'Nfree','Dtot','Ntot','NDtrans','reporter']]=DB_f[DB_f.cell_id==CELL][['Dfree',
                                                                                                                         'Nfree','Dtot','Ntot','NDtrans','reporter']].values[0]
        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id==daughter2),['Dfree',
                                                            'Nfree','Dtot','Ntot','NDtrans','reporter']]=DB_f[DB_f.cell_id==CELL][['Dfree',
                                                                                                                         'Nfree','Dtot','Ntot','NDtrans','reporter']].values[0]

        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id==daughter1),'u']=DB_f[DB_f.cell_id==CELL]['u'].values[0]
        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id==daughter2),'u']=DB_f[DB_f.cell_id==CELL]['u'].values[0]



                                                                                                                                   
DB.to_csv('./'+folder+'/'+param+'/DB.csv')       
## uncomment to continue running simulations after the final time frame                                                                                                                                          
# FRAME=185
# frame_cellshapes=tml.subset_dataframe(movie.cellshapes, ['frame'], [FRAME])    
# DB_f=DB.loc[DB.frame==FRAME] #subset of the database for this frame
# Neighbors_f=Neighbors.loc[Neighbors.frame==FRAME]
# DB_f['neighbors']=DB_f['cell_id'].transform(lambda x: Neighbors_f[(Neighbors_f.cell_id==x)].neighbor_cell_id.unique())
# DB_f['NeighborsCount']=DB_f['neighbors'].apply(lambda x: len(x))
# #This next part selects the cells at the border of the tissue (within 3 cell layers) to set the boundary condition:
# Differences=[]
# for cell in DB_f.cell_id.unique():
#     Differences.append((-DB_f[DB_f.cell_id==cell].NeighborsCount.values+frame_cellshapes[frame_cellshapes.cell_id==cell].bond_order.max())[0])
# DB_f['Border']=Differences
# Border_cells=DB_f[DB_f.Border!=0].cell_id.unique()
# #subset of the database for this frame
# DB[DB.frame==FRAME]=DB_f #allocate DB_f to DB so that it cpontains the column neighbors
# #perform the simulation for cells inside the tissue (not in the border)
# #DB_f_in=DB_f[~DB_f.cell_id.isin(Border_cells)]

# save_times=[60*j for j in range(100)]
# for time in range(60*100): 
#     # length_u=len(DB_f_in['u'])
#     # noise=np.random.normal(0,np.sqrt(sigma*dt),length_u)
#     #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           
#     DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'u']=DB_f.apply(lambda x: find_u_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)
    
#     DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Dfree']=DB_f.apply(lambda x: find_Dfree_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)

#     DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Dtot']=DB_f.apply(lambda x: find_Dtot_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)

#     DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Ntot']=DB_f.apply(lambda x: find_Ntot_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)

#     DB.loc[(DB.frame==FRAME)&(DB.cell_id.isin(Border_cells)),'Nfree']=DB_f.apply(lambda x: find_Nfree_largest_x(x.cell_id, x.neighbors, DB_f, FRAME),axis=1)

#     DB_f=DB[DB.frame==FRAME]

#     cells = {k: cell_wing(u_, D_tot, N_tot, D_free, N_free, Signal, Reporter, index_, border_)
#              for k, (u_, D_tot, N_tot, D_free, N_free, Signal, Reporter, index_, border_)
#              in enumerate(zip( np.array(DB_f.u.values), np.array(DB_f.Dtot.values), np.array(DB_f.Ntot.values), 
#                               np.array(DB_f.Dfree.values), np.array(DB_f.Nfree.values) ,np.array(DB_f.NDtrans.values) ,np.array(DB_f.reporter.values), np.array(DB_f.cell_id.values), np.array(DB_f.Border.values)) )}
#     #++++++++++++++++++++
#     #DB_f_border=DB_f[DB_f]
#     #++++++++++++++++
#     these_keys=list(cells.keys())
#     for k1 in these_keys:
#         cells[cells[k1].index]=cells[k1]
#         del cells[k1]

#     for k, c in cells.items():
#         c.neighbours=DB_f[DB_f.cell_id==c.index].neighbors.values[0]


#     u_values = np.array([c.u for c in cells.values()])
#     delta_tot_values = np.array([c.delta_tot for c in cells.values()])
#     notch_tot_values = np.array([c.notch_tot for c in cells.values()])
#     notch_reporter_values = np.array([c.notch_reporter for c in cells.values()])
    
#     #first update notch free and delta free (this dynamics is instantaneous)  
#     count=0
#     max_iter_delta = 1
#     max_iter_notch=1
#     while((max_iter_delta>1.0e-4 or max_iter_notch>1.0e-4) and count<100):
#         delta_free_values = np.array([c.delta_free for c in cells.values()])
#         notch_free_values = np.array([c.notch_free for c in cells.values()])
#         #print('yo')
#         nearest_neighbour_delta_free_values = np.array([[cells[c2].delta_free for c2 in c.neighbours]
#                                              for c in cells.values()], dtype=object )
#         nearest_neighbour_notch_free_values = np.array([[cells[c2].notch_free for c2 in c.neighbours]
#                                              for c in cells.values()], dtype=object )
#         (new_delta_free_values,
#          new_notch_free_values)= next_iter_delta_notch_free(delta_free_values,
#                                                             notch_free_values,
#                                                             delta_tot_values,
#                                                             notch_tot_values,
#                                                             nearest_neighbour_delta_free_values,
#                                                             nearest_neighbour_notch_free_values) 
#         for (c, 
#              new_delta_free,
#              new_notch_free) in zip(cells.values(),
#                                     new_delta_free_values,
#                                     new_notch_free_values,):
#             c.delta_free=new_delta_free
#             c.notch_free=new_notch_free
#         max_iter_delta = np.max(np.abs(new_delta_free_values-delta_free_values))
#         max_iter_notch = np.max(np.abs(new_notch_free_values-notch_free_values))
#         count+=1
#     if (max_iter_delta>1.0e-4 or max_iter_notch>1.0e-4):
#         print('convergence failed')
  
#     #then update other quantities.
#     notch_free_values = np.array([c.notch_free for c in cells.values()])
#     nearest_neighbour_delta_free_values = np.array([[cells[c2].delta_free
#                                                       for
#                                                       c2 in c.neighbours]
#                                           for c in cells.values()] , dtype=object)
#     nearest_neighbour_u_values = np.array([[cells[c2].u for c2 in c.neighbours]
#                                          for c in cells.values()] , dtype=object)
#     (new_delta_tot_values,
#      new_notch_tot_values,
#      new_signal_values,
#      new_u_values,
#      new_notch_reporter_values)= next_iter(u_values,
#                                            notch_free_values,
#                                            delta_tot_values,
#                                            notch_tot_values,
#                                            nearest_neighbour_u_values,
#                                            nearest_neighbour_delta_free_values,
#                                            notch_reporter_values)
#     for (c, 
#          new_delta_tot,
#          new_notch_tot,
#          new_signal,
#          new_u,
#          new_notch_reporter) in zip(cells.values(),
#                                        new_delta_tot_values,
#                                        new_notch_tot_values,
#                                        new_signal_values,
#                                        new_u_values,
#                                        new_notch_reporter_values):
#         c.delta_tot=new_delta_tot
#         c.notch_tot=new_notch_tot
#         c.signal = new_signal
#         c.u = new_u
#         c.notch_reporter=new_notch_reporter
    
#     DB_f['u']=np.array([c.u for c in cells.values()])
#     DB_f['Dtot']=np.array([c.delta_tot for c in cells.values()])
#     DB_f['Ntot']=np.array([c.notch_tot for c in cells.values()])
#     DB_f['NDtrans']=np.array([c.signal for c in cells.values()])
#     DB_f['reporter']=np.array([c.notch_reporter for c in cells.values()])
#     DB_f['Dfree']=np.array([c.delta_free for c in cells.values()])
#     DB_f['Nfree']=np.array([c.notch_free for c in cells.values()])
#     DB.loc[DB.frame==FRAME]=DB_f
#     if time in save_times:
#         print(time)
#         frame_cellshapes=tml.subset_dataframe(movie.cellshapes, ['frame'], [FRAME])    
#         frame_polygons_0=frame_cellshapes.groupby('cell_id').apply(lambda x: list(zip(x['x_pos'].values, x['y_pos'].values))).reset_index().rename(columns= {0: 'plot_vertices'})
#         frame_polygons=DB[DB.frame==FRAME][[u'frame', u'cell_id', u'center_x', u'center_y',u'DSRF_Conc', 
#                                             u'u', u'Dtot', u'Ntot', u'NDtrans' ,u'reporter']].merge(frame_polygons_0, on= 'cell_id')
#         plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/u/_step_'+str(time), frame_polygons, title= 'u', color_column= 'u', c_min= -0.1, c_max= 1.1, color_map=cm.plasma) #cm.gist_rainbow)
#         plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/Dtot/_step'+str(time), frame_polygons, title= 'Dtot', color_column= 'Dtot', c_min= 0, c_max= 1.1, color_map=cm.Blues) #cm.gist_rainbow)
#         plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/Ntot/_step'+str(time), frame_polygons, title= 'Ntot', color_column= 'Ntot', c_min= 0, c_max= 1.2, color_map=cm.Purples) #cm.gist_rainbow)
#         plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/NDtrans/_step'+str(time), frame_polygons, title= 'NDtrans', color_column= 'NDtrans', c_min= 0, c_max= 1.2, color_map=cm.Reds) #cm.gist_rainbow)
#         plot_frame_cells(movie,FRAME,'./'+folder+'/'+param+'/reporter/_step'+str(time), frame_polygons, title= 'reporter', color_column= 'reporter', c_min= 0, c_max= 0.8, color_map=cm.Reds) #cm.gist_rainbow)



