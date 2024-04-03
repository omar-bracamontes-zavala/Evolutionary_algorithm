import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from itertools import combinations

def visualize_particle_system(vectors, number_of_particles, box_limits = (-1.5,1.5), filename='particle_animation.gif',dimensions=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    ax.set_title('Best genome configuration')

    ax.grid(False)  # alv el grid
    ax.xaxis.pane.fill = False 
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])  
    ax.set_yticks([]) 
    ax.set_zticks([]) 
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.9))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.9))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.9))

    lower_box_limit,upper_box_limit = box_limits

    ## To get consistent limits across the system evolution
    ax.set_xlim(lower_box_limit,upper_box_limit)
    ax.set_ylim(lower_box_limit,upper_box_limit)
    ax.set_zlim(lower_box_limit,upper_box_limit)

    first_frame = np.array(vectors[0]).reshape(number_of_particles, dimensions)
    scatter_object = ax.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], s=100)
    lines = [ax.plot([0, 0], [0, 0], [0, 0], 'gray', lw=0.5, alpha=0.5)[0] for _ in range(number_of_particles*(number_of_particles-1)//2)]

    time_text = ax.text2D(0.00, 0.95, '', transform=ax.transAxes)

    def update(frame_number):
        data = np.array(vectors[frame_number]).reshape(number_of_particles, dimensions)
        scatter_object._offsets3d = (data[:, 0], data[:, 1], data[:, 2])

        
        for line, (i, j) in zip(lines, combinations(range(number_of_particles), 2)):
            line.set_data(np.array([data[i, 0], data[j, 0]]), np.array([data[i, 1], data[j, 1]]))
            line.set_3d_properties(np.array([data[i, 2], data[j, 2]]))

        time_text.set_text(f'Generation: {frame_number}')

        return [scatter_object, *lines, time_text]

    anim = FuncAnimation(fig, update, frames=len(vectors), interval=100, blit=False)
    plt.show()
    #anim.save('tst.gif', writer='pillow', fps=24)



if __name__=='__main__':
    visualize_particle_system('')