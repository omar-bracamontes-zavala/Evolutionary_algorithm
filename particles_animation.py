import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from itertools import combinations

def plotting_plane_style(ax_object):
    ax_object.xaxis.pane.fill = False 
    ax_object.yaxis.pane.fill = False
    ax_object.zaxis.pane.fill = False

    #ax_object.w_xaxis.set_pane_color((1.0, 1.0, 0.5, 1.0))

    ax_object.set_xticks([])  
    ax_object.set_yticks([]) 
    ax_object.set_zticks([]) 
    ax_object.xaxis.line.set_color((1.0, 1.0, 1.0, 0.9))
    ax_object.yaxis.line.set_color((1.0, 1.0, 1.0, 0.9))
    ax_object.zaxis.line.set_color((1.0, 1.0, 1.0, 0.9))
    return ax_object

def set_box_limits(ax_object,final_configuration,limits_scheme = 'fit_final'):
    '''
    Sets the limits for the final configuration
    input:
        ax_object(Axis): ax object of the plot that will be adjusted.
        final_configuration(array): Vector of positions of the last configuration
        limits_scheme (str) ['fit_final', 'loose']: "fit_final" will set the limits to fit the final configuration.
                                                    "loose" will let the plot adjust with the configuration, scale of dimensions can be lost.
    '''

    if limits_scheme=='loose':
        return ax_object
    N = len(final_configuration)//3
    final_config_matrix = np.reshape(final_configuration, (N,3))
    ax_object.set_xlim(min(final_config_matrix[:,0]),max(final_config_matrix[:,0]))
    ax_object.set_ylim(min(final_config_matrix[:,1]),max(final_config_matrix[:,1]))
    ax_object.set_zlim(min(final_config_matrix[:,2]),max(final_config_matrix[:,2]))
    return ax_object

def create_colormap(energies, vmin=None, vmax=None, cmap='hot'):
    if not vmin:
        vmin=min(energies)
    if not vmax:
        vmax=max(energies)
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    return norm, colormap

def initialize_plotting_elements(axes_object,vectors,dimensions,number_of_particles,particle_size=100):
    ### Create the elements on the plot that will be updated
    first_frame = np.array(vectors[0]).reshape(number_of_particles, dimensions)
    scatter_object = axes_object.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], s=particle_size)

    # Just dummy lines for later updating
    number_of_lines = number_of_particles*(number_of_particles-1)//2
    lines = [axes_object.plot([0, 0], [0, 0], [0, 0], 'gray', lw=0.5, alpha=0.3)[0] for _ in range(number_of_lines)]

    # Dummy text and formatting
    time_text = axes_object.text2D(0.00, 0.95, '', transform=axes_object.transAxes)
    return scatter_object, lines, time_text

def visualize_particle_system(vectors, energies, number_of_particles, filename='particle_animation.gif',dimensions=3):
    ### Create the plot canvas
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Best individual configuration')
    normalizer,colormap = create_colormap(energies,vmin= None,vmax=10) 
    
    # some styling
    ax = plotting_plane_style(ax)
    ax = set_box_limits(ax, vectors[-1],limits_scheme='fit_final')

    scatter_object, lines, time_text = initialize_plotting_elements(ax,vectors,dimensions,number_of_particles)

    def update(frame_number):
        positions = np.array(vectors[frame_number]).reshape(number_of_particles, dimensions)
        scatter_object._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        color = colormap(normalizer(energies[frame_number]))
        scatter_object.set_color(color)

        for line, (i, j) in zip(lines, combinations(range(number_of_particles), 2)):
            line.set_data(np.array([positions[i, 0], positions[j, 0]]), np.array([positions[i, 1], positions[j, 1]]))
            line.set_3d_properties(np.array([positions[i, 2], positions[j, 2]]))

        time_text.set_text(f'Generation: {frame_number}. Energy: {round(energies[frame_number],3)}')

        return [scatter_object, *lines, time_text]

    anim = FuncAnimation(fig, update, frames=len(vectors), interval=50, blit=False)
    plt.show()
    #anim.save('tst.gif', writer='pillow', fps=24)
