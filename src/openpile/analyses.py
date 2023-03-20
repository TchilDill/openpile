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

from openpile.utils import kernel 
import openpile.utils.validation as validation
import openpile.utils.graphics as graphics
import openpile.utils.misc as misc

class PydanticConfig:
    arbitrary_types_allowed = True

def structural_forces_to_df(model,q):
    x = model.nodes_coordinates['x [m]'].values
    x = misc.repeat_inner(x)
    L = kernel.mesh_to_element_length(model).reshape(-1)

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


def disp_to_df(model,u):
    x = model.nodes_coordinates['x [m]'].values

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


def simple_beam_analysis(model):
    """
    Function where loading or displacement defined in the model boundary conditions 
    are used to solve the system of equations, .
    
    Parameters
    ----------
    model : `openpile.construct.Model` object
        Model where structure and boundary conditions are defined.

    Returns
    -------
    results : `openpile.compute.Result` object
        Results of the analysis 
    """
    
    if model.soil is None:
        # initialise global force and displacement vectors 
        F = kernel.mesh_to_global_force_dof_vector(model.global_forces)
        U = kernel.mesh_to_global_disp_dof_vector(model.global_disp)
        # initialise global stiffness matrix
        K = kernel.build_stiffness_matrix(model)
        # initialise global supports vector 
        supports = kernel.mesh_to_global_restrained_dof_vector(model.global_restrained)
        
        # validate boundary conditions
        validation.check_boundary_conditions(model)
        
        u, _ = kernel.solve_equations(K, F, U, restraints=supports)
        
        #internal forces
        q_int = kernel.struct_internal_force(model, u)
        
        NVM = structural_forces_to_df(model,q_int)
        df_u = disp_to_df(model, u)
        
        results = Result(displacements=df_u, forces=NVM)
  
        return results

def simple_winkler_analysis(model, solver='MNR', max_iter:int=100):
    """
    Function where loading or displacement defined in the model boundary conditions 
    are used to solve the system of equations, .
    
    #TODO
    
    Parameters
    ----------
    model : `openpile.construct.Model` object
        Model where structure and boundary conditions are defined.
    solver: str, by default 'MNR'
        solver. literally 'NR': "Newton-Raphson" or 'MNR': "Modified Newton-Raphson"
    max_iter: int, by defaut 100
        maximum number of iterations for convergence

    Returns
    -------
    results : `openpile.analyses.Result` object
        Results of the analysis 
    """
    
    if model.soil is None:
        UserWarning('SoilProfile must be provided when creating the Model.')
        
    else:
        # initialise global force and displacement vectors 
        F = kernel.mesh_to_global_force_dof_vector(model.global_forces)
        U = kernel.mesh_to_global_disp_dof_vector(model.global_disp)
        # initialise global stiffness matrix
        K = kernel.build_stiffness_matrix(model, u=U, kind="initial")
        # initialise global supports vector 
        supports = kernel.mesh_to_global_restrained_dof_vector(model.global_restrained)
        
        # validate boundary conditions
        # validation.check_boundary_conditions(model)

        # Initialise residual forces 
        Rg = F
        control = np.linalg.norm(F[~supports])
        
        # incremental calculations to convergence 
        iter_no = 0
        while iter_no <= 100:
            iter_no += 1
            
            # solve system
            u_inc, Q = kernel.solve_equations(K, Rg, U, restraints=supports)

            # add up increment displacements
            U += u_inc
                      
            #internal forces calculations
            q_int = kernel.struct_internal_force(model, U)
                        
            # calculate residual forces
            Rg = F - q_int - Q
            
            # check if converged
            if np.linalg.norm(Rg[~supports]) < 1e-4*control:
                print(f"Converged at iteration no. {iter_no}")
                break

            if iter_no == 100:
                print("Not converged after 100 iterations.")    


        NVM = structural_forces_to_df(model,q_int)
        df_u = disp_to_df(model, U)
        
        results = Result(displacements=df_u, forces=NVM)
  
        return results



