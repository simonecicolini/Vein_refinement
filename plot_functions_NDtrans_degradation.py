import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib import animation
import numpy as np

def save_animation(cells,
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
                   saving_path):
    plt.close('all')
    fig, ((ax1, ax2, ax3),( ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(10, 10))
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
                                  alpha=1.0,
                                  edgecolor='k')
         patches.append(hexagon)
    p1 = PatchCollection(patches, cmap=cm.Greens, alpha=1.0)
    p1.set_array(np.array(u_saved[0]))
    p1.set_clim([-0.1, 1.1])
    
    p2 = PatchCollection(patches, cmap=cm.Reds, alpha=1.0)
    p2.set_array(np.array(signal_saved[0]))
    p2.set_clim([-0.1, 1.1])
    
    
    p3 = PatchCollection(patches, cmap=cm.Reds, alpha=1.0)
    p3.set_array(np.array(notch_reporter_saved[0]))
    p3.set_clim([-0.1, 1.1])
    
    
    p4 = PatchCollection(patches, cmap=cm.Blues, alpha=1.0)
    p4.set_array(np.array(delta_saved[0]))
    p4.set_clim([-0.1, 1.1])
    
    p5 = PatchCollection(patches, cmap=cm.Purples, alpha=1.0)
    p5.set_array(np.array(notch_saved[0]))
    p5.set_clim([-0.1, 3.1])
    
    p6 = PatchCollection(patches, cmap=cm.Purples, alpha=1.0)
    p6.set_array(np.array(notch_saved[0]))
    p6.set_clim([-0.1, 3.1])
    
    ax1.add_collection(p1)
    ax2.add_collection(p2)
    ax3.add_collection(p3)
    ax4.add_collection(p4)
    ax5.add_collection(p5)
    ax6.add_collection(p6)
    # Add some coloured hexagons
    
    def animate(i):
        p1.set_array(np.array(u_saved[i]))
        p2.set_array(np.array(signal_saved[i]))
        p3.set_array(np.array(notch_reporter_saved[i]))
        p4.set_array(np.array(delta_saved[i]))
        p5.set_array(np.array(notch_saved[i]))
        p6.set_array(np.array(np.array(notch_saved[i])+np.array(signal_saved[i])))
        return p1, p2, p3, p4, p5, p6
            
    ani = animation.FuncAnimation(fig, animate, frames=N_times//fraction_saved , interval=1)
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    ax4.set_aspect('equal', adjustable='box')
    ax5.set_aspect('equal', adjustable='box')
    ax6.set_aspect('equal', adjustable='box')
    ax1.set_title('u (vein state)')
    ax2.set_title('Notch signalling')
    ax3.set_title('Notch signalling reporter')
    ax4.set_title('total Delta')
    ax5.set_title('total Notch')
    ax6.set_title('total Notch + signal')
    
    ax1.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax1.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax2.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax2.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax3.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax3.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax4.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax4.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax5.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax5.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax6.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax6.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    Writer=animation.writers['ffmpeg']
    movie_writer=Writer(codec='mpeg4', bitrate=1e6, fps=24)
    ani.save(saving_path+'sim_movie.mp4', writer=movie_writer)



def save_snapshots(cells,
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
                   saving_path):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #SNAPOSHOTS OF THE SIMULATION EVERY 2 HOURS
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    times_to_plot=[i*6 for i in range(40)] #60 minutes is 1 hour
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = mpl.cm.get_cmap('plasma') , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(u_saved[i]))
        p.set_clim([-0.1, 1.1])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'/u_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
    
    ##notch tot
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Purples , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(notch_saved[i]))
        p.set_clim([0, 1.3])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'/notch_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)
    
    ##delta tot
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Blues , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(delta_saved[i]))
        p.set_clim([0, 0.7])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'/delta_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)
    
    ##notch signal
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[i]))
        p.set_clim([0, 0.55])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'/signal_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

    ##notch tot + signal
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Purples , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(signal_saved[i])+np.array(notch_saved[i]))
        p.set_clim([0, 1.5])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'/Ntot+signal_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)

    ##notch signalling reporter
    for i in times_to_plot:
        fig,ax=plt.subplots()
        p= PatchCollection(patches, cmap = cm.Reds , match_original=True)
        ax.add_collection(p)
        p.set_array(np.array(notch_reporter_saved[i]))
        p.set_clim([0, 0.45])
        ax.axis('equal')
        plt.axis('off')
        plt.savefig(saving_path+'/reporter_'+str(int(t_0+i*10/60))+'h.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=120)
        plt.close(fig)



    
def save_packings(cells,
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
                   saving_path):
    plt.close('all')
    fig, ((ax1, ax2, ax3),( ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 10))
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
                                  alpha=1.0,
                                  edgecolor='k')
         patches.append(hexagon)
         
    p1 = PatchCollection(patches, cmap=cm.Greens, alpha=1.0)
    p1.set_array(np.array(u_saved[frame]))
    p1.set_clim([-0.1, 1.1])
    
    p2 = PatchCollection(patches, cmap=cm.Reds, alpha=1.0)
    p2.set_array(np.array(signal_saved[frame]))
    p2.set_clim([-0.1, 1.1])
    
    
    p3 = PatchCollection(patches, cmap=cm.Reds, alpha=1.0)
    p3.set_array(np.array(notch_reporter_saved[frame]))
    p3.set_clim([-0.1, 1.1])
    
    
    p4 = PatchCollection(patches, cmap=cm.Blues, alpha=1.0)
    p4.set_array(np.array(delta_saved[frame]))
    p4.set_clim([-0.1, 1.1])
    
    p5 = PatchCollection(patches, cmap=cm.Purples, alpha=1.0)
    p5.set_array(np.array(notch_saved[frame]))
    p5.set_clim([-0.1, 3.1])
    
    p6 = PatchCollection(patches, cmap=cm.Purples, alpha=1.0)
    p6.set_array(np.array(np.array(notch_saved[frame])+np.array(signal_saved[frame])))
    p6.set_clim([-0.1, 3.1])
    
    ax1.add_collection(p1)
    ax2.add_collection(p2)
    ax3.add_collection(p3)
    ax4.add_collection(p4)
    ax5.add_collection(p5)
    ax6.add_collection(p6)
    
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    ax4.set_aspect('equal', adjustable='box')
    ax5.set_aspect('equal', adjustable='box')
    ax6.set_aspect('equal', adjustable='box')
    ax1.set_title('u (vein state)')
    ax2.set_title('Notch signalling')
    ax3.set_title('Notch signalling reporter')
    ax4.set_title('total Delta')
    ax5.set_title('total Notch')
    ax6.set_title('total Notch + signal')
    
    ax1.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax1.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax2.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax2.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax3.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax3.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax4.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax4.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax5.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax5.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    ax6.set_xlim(side_length*np.array([-1,3*N_x/2-1/2]))
    ax6.set_ylim(side_length*np.array([-1,np.sqrt(3)*N_y]))
    
    plt.suptitle('Packings for time '+str(time) + ' hAPF, t/$tau$='+str(relative_time))
    plt.savefig(saving_path+'packings_'+str(frame)+'.pdf')
    
    
def save_spatial_profiles(cells,
                         delta_saved,
                         notch_saved,
                         signal_saved,
                         u_saved,
                         notch_reporter_saved,
                         t_0,
                         dt,
                         fraction_saved,
                         saving_path):
    #plot some profiles at different times - quantity by quantity
    fig, ((ax1, ax2, ax3),( ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(25, 15))
    axes_list=[ax1,ax2,ax3,ax4, ax5, ax6]
    y_positions_all = [c.position[1] for c in cells.values()]
    y_positions = np.unique(y_positions_all)
    center_y = np.mean([cell.position[1] for cell in cells.values()])
    
    increment_frames = 18
    times=[0, increment_frames,2*increment_frames,3*increment_frames, 4 *increment_frames, 5*increment_frames]
 
    # list_integers = range(0,10)
    cmap=plt.get_cmap("plasma_r")
    
    for  frame_index in times:
        mean_u_values = []
        mean_signal_values = []
        mean_notch_values = []
        mean_delta_values = []
        mean_notch_reporter_values = []
    
        for y_pos in y_positions:
            indices =np.where(y_positions_all==y_pos)
            mean_u_values.append(np.mean(np.take(u_saved[frame_index],indices)))
            mean_signal_values.append(np.mean(np.take(signal_saved[frame_index],indices)))
            mean_notch_values.append(np.mean(np.take(notch_saved[frame_index],indices)))
            mean_delta_values.append(np.mean(np.take(delta_saved[frame_index],indices)))
            mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[frame_index],indices)))
        axes_list[0].plot(y_positions-center_y,
                          mean_u_values,
                          color=cmap((t_0+dt*fraction_saved*frame_index-21) /(36-21)),
                          lw=3, 
                          label=int(t_0+dt*fraction_saved*frame_index))
        axes_list[1].plot(y_positions-center_y,
                          mean_signal_values,
                          color=cmap(((t_0+dt*fraction_saved*frame_index-21) /(36-21))),
                          lw=3,
                          label=round(t_0+dt*fraction_saved*frame_index))
        axes_list[2].plot(y_positions-center_y,
                          mean_notch_reporter_values,
                          color=cmap((t_0+dt*fraction_saved*frame_index-21) /(36-21)),
                          lw=3,
                          label=round(t_0+dt*fraction_saved*frame_index))
        axes_list[3].plot(y_positions-center_y,
                          mean_notch_values,
                          color=cmap(((t_0+dt*fraction_saved*frame_index-21) /(36-21))),
                          lw=3, 
                          label=round(t_0+dt*fraction_saved*frame_index))
        axes_list[4].plot(y_positions-center_y,
                          mean_delta_values,
                          color=cmap((t_0+dt*fraction_saved*frame_index-21) /(36-21)),
                          lw=3, 
                          label=round(t_0+dt*fraction_saved*frame_index))
        axes_list[5].plot(y_positions-center_y,
                          np.array(mean_notch_values)+np.array(mean_signal_values),
                          color=cmap((t_0+dt*fraction_saved*frame_index-21) /(36-21)),
                          lw=3, 
                          label=round(t_0+dt*fraction_saved*frame_index))
    
    for i in range(5):
        axes_list[i].set_xlabel('position (y) in um', fontsize=15)
        axes_list[i].tick_params(axis='both', which='major', labelsize=15)
        axes_list[i].set_ylabel('Values', fontsize=15)
        axes_list[i].set_xlim([-40,40])
        axes_list[i].legend()
        
    axes_list[0].set_title('profiles, u', fontsize=15)
    axes_list[1].set_title('profiles, signal', fontsize=15)
    axes_list[2].set_title('profiles, Notch reporter', fontsize=15)
    axes_list[3].set_title('profiles, total Notch', fontsize=15)
    axes_list[4].set_title('profiles, total delta', fontsize=15)
    axes_list[5].set_title('profiles, total Notch+signal', fontsize=15)
    plt.tight_layout()
    plt.savefig(saving_path+'spatial_profiles'+'.pdf')

def save_spatial_profiles_2(cells,
                         delta_saved,
                         notch_saved,
                         signal_saved,
                         u_saved,
                         notch_reporter_saved,
                         times,
                         t_0,
                         dt,
                         fraction_saved,
                         saving_path):
    
    fig, ((ax1, ax2, ax3),( ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(25, 15))
    axes_list=[ax1,ax2,ax3,ax4, ax5, ax6]
    y_positions_all = [c.position[1] for c in cells.values()]
    y_positions = np.unique(y_positions_all)
    center_y = np.mean([cell.position[1] for cell in cells.values()])
    # number_frames=N_times//fraction_saved
    increment_frames = 18
    times=[0, increment_frames,2*increment_frames,3*increment_frames, 4 *increment_frames, 5*increment_frames]
    
    for i, time_index in enumerate(times):
        mean_u_values = []
        #mean_signal_values = []
        mean_notch_plus_signal_values = []
        mean_delta_values = []
        mean_notch_reporter_values = []
    
        for y_pos in y_positions:
            indices =np.where(y_positions_all==y_pos)
            mean_u_values.append(np.mean(np.take(u_saved[time_index],indices)))
            #mean_signal_values.append(np.mean(np.take(signal_saved[time_index],indices)))
            mean_notch_plus_signal_values.append(np.mean(np.take(notch_saved[time_index],indices))+np.mean(np.take(signal_saved[time_index],indices)))
            mean_delta_values.append(np.mean(np.take(delta_saved[time_index],indices)))
            mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[time_index],indices)))
            
        # norm_signal=(mean_signal_values-np.min(mean_signal_values))/(np.max(mean_signal_values)-np.min(mean_signal_values))
        axes_list[i].plot(y_positions-center_y,
                          mean_u_values,
                          color='g',
                          lw=3, 
                          label='u')
        # axes_list[i].plot(y_positions-center_y,
        #                   mean_signal_values,
        #                   color='orange',
        #                   lw=3,
        #                   label='signal')
        axes_list[i].plot(y_positions-center_y,
                          mean_notch_plus_signal_values,
                          color='purple',
                          lw=3, 
                          label='total notch + signal')
        axes_list[i].plot(y_positions-center_y,
                          mean_delta_values,
                          color='blue',
                          lw=3, 
                          label='total delta')
        axes_list[i].plot(y_positions-center_y,
                          mean_notch_reporter_values,
                          color='r',
                          lw=3,
                          label='notch reporter')
    
        axes_list[i].set_xlabel('position (y) in um', fontsize=15)
        axes_list[i].tick_params(axis='both', which='major', labelsize=15)
        axes_list[i].set_ylabel('Values normalised to maximum', fontsize=15)
        axes_list[i].set_xlim([-30,30])
        axes_list[i].set_title('profiles, t='+str(np.round(t_0+dt*fraction_saved*time_index,3))+' hours', fontsize=15)
    axes_list[0].legend()
    plt.tight_layout()
    plt.savefig(saving_path+'spatial_profiles_v2'+'.pdf')

def save_Nicd_reporter(cells,
                         delta_saved,
                         notch_saved,
                         signal_saved,
                         u_saved,
                         notch_reporter_saved,
                         times,
                         t_0,
                         dt,
                         fraction_saved,
                         saving_path):
    

    y_positions_all = [c.position[1] for c in cells.values()]
    y_positions = np.unique(y_positions_all)
    center_y = np.mean([cell.position[1] for cell in cells.values()])
    # number_frames=N_times//fraction_saved
    increment_frames = 18
    times=[2*increment_frames]
    
    for i, time_index in enumerate(times):
        fig, ax = plt.subplots()
        ax2=ax.twinx()
        mean_notch_plus_signal_values = []
        mean_notch_reporter_values = []
    
        for y_pos in y_positions:
            indices =np.where(y_positions_all==y_pos)
            mean_notch_plus_signal_values.append(np.mean(np.take(notch_saved[time_index],indices))+np.mean(np.take(signal_saved[time_index],indices)))
            mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[time_index],indices)))
            
      
        ax.plot(y_positions-center_y,
                          mean_notch_plus_signal_values,
                          color='purple',
                          lw=3, 
                          label='total notch + signal')
        ax2.plot(y_positions-center_y,
                          mean_notch_reporter_values,
                          color='r',
                          lw=3,
                          label='notch reporter')
    
        ax.set_xlabel('position (y) in um', fontsize=16)
        ax2.set_ylabel('Notch reporter', color='r', fontsize=16)
        ax.set_ylabel('Notch tot + signal', color='purple', fontsize=16)
        #ax.tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        #axes_list[i].set_ylabel('Val', fontsize=15)
        ax.set_xlim([-30,30])
        ax.set_title(' t='+str(np.round(t_0+dt*fraction_saved*time_index,3))+' hours', fontsize=16)
        #ax.legend()
        plt.tight_layout()
        plt.savefig(saving_path+'spatial_profiles_Nicd_reporter'+str(np.round(t_0+dt*fraction_saved*time_index,3))+'_h.pdf')


def save_temporal_profiles(cells,
                           delta_saved,
                           notch_saved,
                           signal_saved,
                           u_saved,
                           notch_reporter_saved,
                           spatial_boundaries,
                           frames,
                           times,
                           saving_path):
    
    y_positions_all = [c.position[1] for c in cells.values()]
    center_y = np.mean([cell.position[1] for cell in cells.values()])                      
    fig, ((ax1_temp, ax2_temp, ax3_temp),( ax4_temp, ax5_temp, ax6_temp)) = plt.subplots(2, 3,figsize=(25, 15))
    axes_list=[ax1_temp,ax2_temp,ax3_temp,ax4_temp, ax5_temp, ax6_temp]
    
    for i, pos_1 in enumerate(spatial_boundaries[:-1]):
        pos_2 = spatial_boundaries[i+1]
        indices = np.where((np.abs(y_positions_all-center_y)<pos_2) 
                              &(np.abs(y_positions_all-center_y)>=pos_1))
        mean_u_values = []
        mean_signal_values = []
        mean_notch_values = []
        mean_delta_values = []
        mean_notch_reporter_values = []
        for time_index in frames:
        
            mean_u_values.append(np.mean(np.take(u_saved[time_index],
                                                 indices)))
            mean_signal_values.append(np.mean(np.take(signal_saved[time_index],
                                                      indices)))
            mean_notch_values.append(np.mean(np.take(notch_saved[time_index],
                                                     indices)))
            mean_delta_values.append(np.mean(np.take(delta_saved[time_index],
                                                     indices)))
            mean_notch_reporter_values.append(np.mean(np.take(notch_reporter_saved[time_index],
                                                              indices)))
    
        axes_list[0].plot(times,
                          mean_u_values,
                          lw=3, 
                          label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
        axes_list[1].plot(times,
                          mean_signal_values,
                          lw=3,
                          label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
        axes_list[2].plot(times,
                          mean_notch_values,
                          lw=3, 
                          label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
        axes_list[3].plot(times,
                          mean_delta_values,
                          lw=3, 
                          label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
        axes_list[4].plot(times,
                          mean_notch_reporter_values,
                          lw=3, 
                          label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
        axes_list[5].plot(times,
                          np.array(mean_notch_values)+np.array(mean_signal_values),
                          lw=3, 
                          label='y>'+str(np.round(pos_1,2))+' and y<'+str(np.round(pos_2,2)))
    names = ['u', 'signal', 'total notch', 'total delta', 'reporter','total notch+signal']
    max_range = [1.2, 
                 1.2,
                 3,
                 1.5,
                 1.0,
                 3.0]
    for i in range(6):
        axes_list[i].set_xlabel('Time (hours)', fontsize=15)
        axes_list[i].tick_params(axis='both', which='major', labelsize=15)
        axes_list[i].set_ylabel(names[i], fontsize=15)
        axes_list[i].set_title(names[i],
                               fontsize=15)
        axes_list[i].set_xlim([times[0], 35])
        axes_list[i].set_ylim([-0.2, max_range[i]])
        axes_list[i].legend()
    plt.tight_layout()
    plt.savefig(saving_path+'temporal_profiles'+'.pdf')
