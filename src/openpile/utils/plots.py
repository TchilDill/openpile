""" general plots for openfile

"""

# import libraries
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

def connectivity_plot(mesh):

    support_color = '#F97306'
    #create 4 subplots with (deflectiom, normal force, shear force, bending moment)
    fig, ax = plt.subplots()
    ax.set_ylabel('x [m]')
    ax.set_xlabel('y [m]')
    ax.axis('equal')
    ax.grid(which='both')
    
    # plot mesh with + scatter points to see nodes.
    x = mesh.nodes_coordinates['x [m]']
    y = mesh.nodes_coordinates['y [m]']
    ax.plot(y,x,'-k',marker='+')
    
    total_length = max( (mesh.nodes_coordinates['x [m]'].max() - mesh.nodes_coordinates['x [m]'].min()) , (mesh.nodes_coordinates['y [m]'].max() - mesh.nodes_coordinates['y [m]'].min() ) )

    ylim = ax.get_ylim()
    
    # plots SUPPORTS
    # Plot supports along x
    support_along_x = mesh.global_restrained['Tx'].values
    support_along_x_down = np.copy(support_along_x)
    support_along_x_down[-1] = False
    support_along_x_up = np.copy(support_along_x)
    support_along_x_up[:-1] = False
    ax.scatter(y[support_along_x_down],x[support_along_x_down], color=support_color, marker=7, s=100)
    ax.scatter(y[support_along_x_up],x[support_along_x_up], color=support_color, marker=6, s=100)
    
    # Plot supports along y    
    support_along_y = mesh.global_restrained['Ty'].values
    ax.scatter(y[support_along_y],x[support_along_y], color=support_color, marker=5, s=100)
    
    # Plot supports along z
    support_along_z = mesh.global_restrained['Rz'].values
    ax.scatter(y[support_along_z],x[support_along_z], color=support_color, marker='s', s=35)

    # plot LOADS
    arrows = []    
    
    normalized_arrow_size = 0.10*total_length #max arrow length will be 20% of the total structure length
    
    load_max =  mesh.global_forces['Py [kN]'].max()   
    for yval, xval, load in zip(x, y, mesh.global_forces['Py [kN]']):
        if load == 0:
            pass
        else:
            style = "Simple, tail_width=0.5, head_width=5, head_length=3"
            kw = dict(arrowstyle=style, color="r")
            arrow_length = normalized_arrow_size*load/load_max
            if load > 0:
                arrows.append(FancyArrowPatch((-arrow_length, yval),(xval, yval),**kw))
            elif load < 0:
                arrows.append(FancyArrowPatch(( arrow_length, yval),(xval, yval),**kw))
    
    load_max =  mesh.global_forces['Px [kN]'].max()    
    for yval, xval, load in zip(x, y, mesh.global_forces['Px [kN]']):
        if load == 0:
            pass
        else:
            style = "Simple, tail_width=0.5, head_width=5, head_length=3"
            kw = dict(arrowstyle=style, color="b")
            arrow_length = normalized_arrow_size*load/load_max
            if load > 0:
                arrows.append(FancyArrowPatch((xval,-arrow_length),(xval, yval),**kw))
            elif load < 0:
                arrows.append(FancyArrowPatch((xval, arrow_length),(xval, yval),**kw))
    
    load_max =  mesh.global_forces['Mz [kNm]'].abs().max()     
    for idx, (yval, xval, load) in enumerate(zip(x, y, mesh.global_forces['Mz [kNm]'])):
        if load == 0:
            pass
        else:
            kw = dict(arrowstyle=style, color="g")
            arrow_length = normalized_arrow_size*load/load_max
            style = "Simple, tail_width=0.5, head_width=5, head_length=3"
            if load > 0:
                if idx == len(x):
                    arrows.append(FancyArrowPatch((arrow_length/3, yval), (-arrow_length/3, yval),connectionstyle="arc3,rad=0.5", **kw))
                else:
                    arrows.append(FancyArrowPatch((-arrow_length/3, yval), (arrow_length/3, yval),connectionstyle="arc3,rad=0.5", **kw))
            elif load < 0:
                if idx == len(x):
                    arrows.append(FancyArrowPatch((arrow_length/3, yval), (-arrow_length/3, yval),connectionstyle="arc3,rad=-0.5", **kw))
                else:
               
                    arrows.append(FancyArrowPatch((-arrow_length/3, yval), (arrow_length/3, yval),connectionstyle="arc3,rad=-0.5", **kw))

    for arrow in arrows:
        plt.gca().add_patch(arrow)
        
    ax.set_ylim(ylim[0]-0.11*total_length, ylim[1]+0.11*total_length)
    
    return fig
        