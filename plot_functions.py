#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:57:22 2024

@author: salbreux
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib import animation
import numpy as np
    
#save simulations snapshots
def save_snapshots(cells,
                   fraction_saved,
                   side_length,
                   delta_saved,
                   notch_saved,
                   signal_saved,
                   u_saved,
                   notch_reporter_saved,
                   norm_color_signal,
                   times_to_plot,
                   t_0, saving_path):
    
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
    
    ##u     
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.plasma , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[i]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'u_'+str(int(t_0+i*fraction_saved/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)    
    
    ##notch tot
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Purples , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(notch_saved[i]))
        p.set_clim([0, 2.2])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'notch_'+str(int(t_0+i*fraction_saved/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)
    
    ##delta tot
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Blues , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(delta_saved[i]))
        p.set_clim([0, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'delta_'+str(int(t_0+i*fraction_saved/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)
    
    ##notch signal
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[i]))
        p.set_clim([0, norm_color_signal])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'signal_'+str(int(t_0+i*fraction_saved/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)


    ##notch reporter
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(notch_reporter_saved[i]))
        p.set_clim([0, 0.72])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'reporter_'+str(int(t_0+i*fraction_saved/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

#plot profiles of 1-u and nocth signalling reporter
def save_profiles_u_notch_reporter(cells,
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
                saving_path):
    plt.rcParams.update({'font.size': 16})
    colormap = cm.plasma_r #plt.get_cmap("plasma_r")
    plt.figure()
    for i, time_index in enumerate(times):
        time_hours = np.round(int(t_0+dt*fraction_saved*time_index))
        mean_u_values = []
    
        for y_pos in y_positions:
            indices =np.where(y_positions_all==y_pos)
            mean_u_values.append(1-np.mean(np.take(u_saved[time_index],indices)))
    
        plt.plot(y_positions-center_y, mean_u_values, label=str(time_hours)+' h',color=colormap((int(t_0+dt*fraction_saved*time_index)-initial_time) /(final_time-initial_time)))
    plt.xlabel('Distance from vein center (\u03bcm)', fontsize=16)
    plt.xlim(-30,30)
    plt.ylim(-0.1,1.1)
    plt.xticks([-30,-20,-10,0,10,20,30])
    plt.legend(loc='upper right', bbox_to_anchor=(1.32,1))   
    plt.savefig(saving_path+'1-u_profile.svg', bbox_inches='tight',format='svg')  
    plt.savefig(saving_path+'1-u_profile.png', bbox_inches='tight',format='png')  
    
    plt.figure()
    for i, time_index in enumerate(times):
        time_hours = np.round(int(t_0+dt*fraction_saved*time_index))
        mean_notch_reporter_values = []
    
        for y_pos in y_positions:
            indices =np.where(y_positions_all==y_pos)
            mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[time_index],indices)))
    
        plt.plot(y_positions-center_y,mean_notch_reporter_values, label=str(time_hours)+' h',color=colormap((int(t_0+dt*fraction_saved*time_index)-initial_time) /(final_time-initial_time)))
    plt.xlabel('Distance from vein center (\u03bcm)', fontsize=16)
    #plt.ylabel('Signal Intensty (a. u.)', fontsize=16)
    plt.xlim(-30,30)
    plt.ylim(-0.02,0.6)
    plt.xticks([-30,-20,-10,0,10,20,30])
    plt.legend(loc='upper right', bbox_to_anchor=(1.32,1))   
    plt.savefig(saving_path+'reporter_profile.svg', bbox_inches='tight',format='svg')  
    plt.savefig(saving_path+'reporter_profile.png', bbox_inches='tight',format='png')  


    
# def save_spatial_profiles(cells,
#                          delta_saved,
#                          notch_saved,
#                          signal_saved,
#                          u_saved,
#                          notch_reporter_saved,
#                          frames,
#                          times,
#                          saving_path):
#     #plot some profiles at different times - quantity by quantity
#     fig, ((ax1, ax2, ax3),( ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(25, 15))
#     axes_list=[ax1,ax2,ax3,ax4, ax5, ax6]
#     y_positions_all = [c.position[1] for c in cells.values()]
#     y_positions = np.unique(y_positions_all)
#     center_y = np.mean([cell.position[1] for cell in cells.values()])
#     list_integers = range(0,10)
#     cmap=cm.plasma_r(np.array(list_integers)/np.mean(list_integers))
    
#     for i, (frame_index, time_hours) in enumerate(zip(frames, times)):
#         mean_u_values = []
#         mean_signal_values = []
#         mean_notch_values = []
#         mean_delta_values = []
#         mean_notch_reporter_values = []
    
#         for y_pos in y_positions:
#             indices =np.where(y_positions_all==y_pos)
#             mean_u_values.append(np.mean(np.take(u_saved[frame_index],indices)))
#             mean_signal_values.append(np.mean(np.take(signal_saved[frame_index],indices)))
#             mean_notch_values.append(np.mean(np.take(notch_saved[frame_index],indices)))
#             mean_delta_values.append(np.mean(np.take(delta_saved[frame_index],indices)))
#             mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[frame_index],indices)))
#         axes_list[0].plot(y_positions-center_y,
#                           mean_u_values,
#                           color=cmap[i],
#                           lw=3, 
#                           label=time_hours)
#         axes_list[1].plot(y_positions-center_y,
#                           mean_signal_values,
#                           color=cmap[i],
#                           lw=3,
#                           label=time_hours)
#         axes_list[2].plot(y_positions-center_y,
#                           mean_notch_reporter_values,
#                           color=cmap[i],
#                           lw=3,
#                           label=time_hours)
#         axes_list[3].plot(y_positions-center_y,
#                           mean_notch_values,
#                           color=cmap[i],
#                           lw=3, 
#                           label=time_hours)
#         axes_list[4].plot(y_positions-center_y,
#                           mean_delta_values,
#                           color=cmap[i],
#                           lw=3, 
#                           label=time_hours)
    
#     for i in range(4):
#         axes_list[i].set_xlabel('position (y) in um', fontsize=15)
#         axes_list[i].tick_params(axis='both', which='major', labelsize=15)
#         axes_list[i].set_ylabel('Values', fontsize=15)
#         axes_list[i].set_xlim([-30,30])
#         axes_list[i].legend()
        
#     axes_list[0].set_title('profiles, u', fontsize=15)
#     axes_list[1].set_title('profiles, signal', fontsize=15)
#     axes_list[2].set_title('profiles, Notch reporter', fontsize=15)
#     axes_list[3].set_title('profiles, total Notch', fontsize=15)
#     axes_list[4].set_title('profiles, total delta', fontsize=15)
#     plt.tight_layout()
#     plt.savefig(saving_path+'spatial_profiles'+'.eps')

# def save_temporal_profiles(cells,
#                            delta_saved,
#                            notch_saved,
#                            signal_saved,
#                            u_saved,
#                            notch_reporter_saved,
#                            spatial_boundaries,
#                            frames,
#                            times,
#                            saving_path):
    
#     y_positions_all = [c.position[1] for c in cells.values()]
#     center_y = np.mean([cell.position[1] for cell in cells.values()])                      
#     fig, ((ax1_temp, ax2_temp, ax3_temp),( ax4_temp, ax5_temp, ax6_temp)) = plt.subplots(2, 3,figsize=(25, 15))
#     axes_list=[ax1_temp,ax2_temp,ax3_temp,ax4_temp, ax5_temp, ax6_temp]
    
#     for i, pos_1 in enumerate(spatial_boundaries[:-1]):
#         pos_2 = spatial_boundaries[i+1]
#         indices = np.where((np.abs(y_positions_all-center_y)<pos_2) 
#                               &(np.abs(y_positions_all-center_y)>=pos_1))
#         mean_u_values = []
#         mean_signal_values = []
#         mean_notch_values = []
#         mean_delta_values = []
#         mean_notch_reporter_values = []
#         for time_index in frames:
        
#             mean_u_values.append(np.mean(np.take(u_saved[time_index],
#                                                  indices)))
#             mean_signal_values.append(np.mean(np.take(signal_saved[time_index],
#                                                       indices)))
#             mean_notch_values.append(np.mean(np.take(notch_saved[time_index],
#                                                      indices)))
#             mean_delta_values.append(np.mean(np.take(delta_saved[time_index],
#                                                      indices)))
#             mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[time_index],
#                                                               indices)))
    
#         axes_list[0].plot(times,
#                           mean_u_values,
#                           lw=3, 
#                           label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
#         axes_list[1].plot(times,
#                           mean_signal_values,
#                           lw=3,
#                           label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
#         axes_list[2].plot(times,
#                           mean_notch_values,
#                           lw=3, 
#                           label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
#         axes_list[3].plot(times,
#                           mean_delta_values,
#                           lw=3, 
#                           label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
#         axes_list[4].plot(times,
#                           mean_notch_reporter_values,
#                           lw=3, 
#                           label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
#     names = ['u', 'signal', 'total notch', 'total delta', 'reporter','']
#     max_range = [1.2, 
#                  1,
#                  3,
#                  1.5,
#                  0.7,
#                  0]
#     for i in range(6):
#         axes_list[i].set_xlabel('Time (hours)', fontsize=15)
#         axes_list[i].tick_params(axis='both', which='major', labelsize=15)
#         axes_list[i].set_ylabel(names[i], fontsize=15)
#         axes_list[i].set_title(names[i],
#                                fontsize=15)
#         axes_list[i].set_xlim([times[0], 43])
#         axes_list[i].set_ylim([-0.2, max_range[i]])
#         axes_list[i].legend()
#     plt.tight_layout()
#     plt.savefig(saving_path+'temporal_profiles'+'.eps')
