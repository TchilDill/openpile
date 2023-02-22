"""
`Analyses` module
==================

The `analyses` module is used to run various simulations. 

Every function from this module returns an `openpile.compute.Result` object. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pydantic import BaseModel, Field, root_validator
from pydantic.dataclasses import dataclass
import openpile.utils.kernel as kernel 
import openpile.utils.validation as validation
import openpile.utils.graphics as graphics


class PydanticConfig:
    arbitrary_types_allowed = True

@dataclass(config=PydanticConfig)
class Result:
    displacements: pd.DataFrame
    forces: pd.DataFrame
    
    class Config:
        frozen = True
    
    @property
    def settlement(self):
        return self.displacements[['Elevation [m]','Settlement [m]']]

    @property
    def deflection(self):
        return self.displacements[['Elevation [m]','Deflection [m]']]
    
    @property
    def rotation(self):
        return self.displacements[['Elevation [m]','Rotation [rad]']]
    
    def plot_deflection(self, assign = False):
        fig = graphics.plot_deflection(self)
        return fig if assign else None

    def plot_forces(self, assign = False):
        fig = graphics.plot_forces(self)
        return fig if assign else None
    
    def plot(self, assign = False):
        fig = graphics.plot_results(self)
        return fig if assign else None


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
    
    def disp_to_df(mesh,u):
        x = mesh.nodes_coordinates['x [m]'].values

        Tx = u[::3].reshape(-1)
        Ty = u[1::3].reshape(-1)
        Rx = u[2::3].reshape(-1)
        
        disp_to_DataFrame = pd.DataFrame(data={
            'Elevation [m]': x,
            'Settlement [m]': Tx ,
            'Deflection [m]': Ty,
            'Rotation [rad]': Rx,
            }
        )
    
        return disp_to_DataFrame
    
    if mesh.soil is None:
        F = kernel.mesh_to_global_force_dof_vector(mesh.global_forces)
        K = kernel.build_stiffness_matrix(mesh, soil_flag = False)
        U = kernel.mesh_to_global_disp_dof_vector(mesh.global_disp)
        supports = kernel.mesh_to_global_restrained_dof_vector(mesh.global_restrained)
        
        
        # validate boundary conditions
        validation.check_boundary_conditions(mesh)
        
        u, _ = kernel.solve_equations(K, F, U, restraints=supports)
        
        #internal forces
        q_int = kernel.struct_internal_force(mesh, u)
        
        NVM = structural_forces_to_df(mesh,q_int)
        df_u = disp_to_df(mesh, u)
        
        results = Result(coordinates=mesh.nodes_coordinates, displacements=df_u, forces=NVM)
  
        return results

def simple_winkler_analysis(mesh):
    pass


