"""
`Compute` module
==================

The `compute` module is used to run various simulations. 

Every function from this module returns an `openpile.compute.Result` object. 

"""

import openpile.utils.kernel as kernel 
import openpile.utils.validation as validate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from pydantic import BaseModel, Field, root_validator
from pydantic.dataclasses import dataclass


class PydanticConfig:
    arbitrary_types_allowed = True

@dataclass(config=PydanticConfig)
class Results:
    coordinates: pd.DataFrame
    displacements: pd.DataFrame
    forces: pd.DataFrame
    
    @property
    def x_coordinates(self):
        return self.coordinates['x [m]']

    @property
    def y_coordinates(self):    
        
        return self.coordinates['y [m]']
    
    @property
    def deflection(self):
        return pd.Series(data = {'Deflection [m]':self.displacements[1::3]})
    
    @property
    def rotation(self):
        return pd.Series(data = {'Deflection [m]':self.displacements[2::3]})
    
    @property
    def deflection(self):
        return self.displacements[1::3]

def simple_beam_analysis(mesh):
    """
    Function where loading or displacement defined in the mesh boundary conditions 
    are used to solve the system of equations, .
    
    Parameters
    ----------
    mesh : `openpile.construct.Mesh` object
        mesh where structure and boundary conditions are defined.

    Returns
    -------
    results : `openpile.compute.Result` object
        Results of the analysis where
    """
    
    def repeat_inner(arr):
        
        arr = arr.reshape(-1,1)
        
        arr_inner = arr[1:-1]
        arr_inner = np.tile(arr_inner,(2)).reshape(-1)
        
        return np.hstack([arr[0],arr_inner,arr[-1]])
    
    def structural_forces_to_df(mesh,q):
            x = mesh.nodes_coordinates['x [m]'].values
            x = repeat_inner(x)
            L = kernel.mesh_to_element_length(mesh).reshape(-1)

            N = np.vstack( (-q[0::6], q[3::6]) ).reshape(-1,order='F')
            V = np.vstack( (-q[1::6],q[4::6]) ).reshape(-1,order='F')
            M = np.vstack( (-q[2::6], -q[2::6]+L*q[1::6]) ).reshape(-1,order='F')
            
            structural_forces_to_DataFrame = pd.DataFrame(data={
                'Elevation [m]': x,
                'N [kN]': N ,
                'V [kN]': V,
                'M [kNm]': M,
                }
            )
        
            return structural_forces_to_DataFrame
    
    if mesh.soil is None:
        F = kernel.mesh_to_global_force_dof_vector(mesh.global_forces)
        K = kernel.build_stiffness_matrix(mesh, soil_flag = False)
        U = kernel.mesh_to_global_disp_dof_vector(mesh.global_disp)
        supports = kernel.mesh_to_global_restrained_dof_vector(mesh.global_restrained)
        
        
        # validate boundary conditions
        validate.check_boundary_conditions(mesh)
        
        u, _ = kernel.solve_equations(K, F, U, restraints=supports)
        
        #internal forces
        q_int = kernel.struct_internal_force(mesh, u)
        
        NVM = structural_forces_to_df(mesh,q_int)
        
        # Define plot colors
        force_facecolor = '#E6DAA6' #beige
        force_edgecolor = '#AAA662' #khaki
        
        #create 4 subplots with (deflectiom, normal force, shear force, bending moment)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
        ax1.set_ylabel('Elevation [m VREF]',fontsize=8)
        ax1.set_xlabel('Deflection [m]',fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        ax1.grid(which='both')
        ax1.plot(mesh.nodes_coordinates['y [m]'],mesh.nodes_coordinates['x [m]'],color='0.4')
        ax1.plot(u[1::3].reshape(-1),mesh.nodes_coordinates['x [m]'],color='0.0', lw=2)
        
        ax2.set_xlabel('Normal [kN]',fontsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(which='both')
        ax2.fill_betweenx(NVM['Elevation [m]'].values,NVM['N [kN]'].values,edgecolor=force_edgecolor,facecolor=force_facecolor)
        ax2.plot(mesh.nodes_coordinates['y [m]'],mesh.nodes_coordinates['x [m]'],color='0.4')
        ax2.set_xlim([NVM['N [kN]'].values.min()-0.1, NVM['N [kN]'].values.max()+0.1])
        ax2.set_yticklabels('')

        ax3.set_xlabel('Shear [kN]',fontsize=8)
        ax3.tick_params(axis='both', labelsize=8)
        ax3.grid(which='both')
        ax3.fill_betweenx(NVM['Elevation [m]'].values,NVM['V [kN]'].values,edgecolor=force_edgecolor,facecolor=force_facecolor)
        ax3.plot(mesh.nodes_coordinates['y [m]'],mesh.nodes_coordinates['x [m]'],color='0.4')
        ax3.set_xlim([NVM['V [kN]'].values.min()-0.1, NVM['V [kN]'].values.max()+0.1])
        ax3.set_yticklabels('')

        ax4.set_xlabel('Moment [kNm]',fontsize=8)
        ax4.tick_params(axis='both', labelsize=8)
        ax4.grid(which='both')
        ax4.fill_betweenx(NVM['Elevation [m]'].values,NVM['M [kNm]'].values,edgecolor=force_edgecolor,facecolor=force_facecolor)
        ax4.plot(mesh.nodes_coordinates['y [m]'],mesh.nodes_coordinates['x [m]'],color='0.4')
        ax4.set_xlim([NVM['M [kNm]'].values.min()-0.1, NVM['M [kNm]'].values.max()+0.1])
        ax4.set_yticklabels('')
                
        return u, NVM

def simple_winkler_analysis(mesh):
    pass


