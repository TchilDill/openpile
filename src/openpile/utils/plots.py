""" general plots for openfile

"""

# import libraries
import matplotlib.pyplot as plt

def connectivity_plot(mesh):
    
        support_color = '#F97306'
        #create 4 subplots with (deflectiom, normal force, shear force, bending moment)
        fig1, ax = plt.subplots()
        ax.set_ylabel('x [m]')
        ax.set_xlabel('y [m]')
        ax.grid(which='both')
        ax.plot(mesh.nodes_coordinates['y [m]'],mesh.nodes_coordinates['x [m]'],'-k',marker='+')
        #name support boolean
        support_along_z = mesh.global_restrained['Tx'].values
        support_along_z_down = np.copy(support_along_z)
        support_along_z_down[-1] = False
        support_along_z_up = np.copy(support_along_z)
        support_along_z_up[:-1] = False
        support_along_y = mesh.global_restrained['Ty'].values
        support_along_x = mesh.global_restrained['Rz'].values
        ax.scatter(mesh.nodes_coordinates['y [m]'][support_along_y],mesh.nodes_coordinates['x [m]'][support_along_y], color=support_color, marker=5, s=100)
        ax.scatter(mesh.nodes_coordinates['y [m]'][support_along_z_down],mesh.nodes_coordinates['x [m]'][support_along_z_down], color=support_color, marker=7, s=100)
        ax.scatter(mesh.nodes_coordinates['y [m]'][support_along_z_up],mesh.nodes_coordinates['x [m]'][support_along_z_up], color=support_color, marker=6, s=100)
        ax.scatter(mesh.nodes_coordinates['y [m]'][support_along_x],mesh.nodes_coordinates['x [m]'][support_along_x], color=support_color, marker='s', s=35)

