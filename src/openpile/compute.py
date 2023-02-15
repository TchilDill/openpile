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
    _coordinates: pd.DataFrame
    _displacements: pd.DataFrame
    _forces: pd.DataFrame
    
    @property
    def x_coordinates(self):
        return self._coordinates['x [m]']

    @property
    def y_coordinates(self):    
        
        return self._coordinates['y [m]']
    
    @property
    def deflection(self):
        return pd.Series(data = {'Deflection [m]':self._displacements[1::3]})
    
    @property
    def rotation(self):
        return pd.Series(data = {'Deflection [m]':self._displacements[2::3]})
    
    @property
    def deflection(self):
        return self._displacements[1::3]

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
    
    if mesh.soil is None:
        F = kernel.mesh_to_global_force_dof_vector(mesh.global_forces)
        K = kernel.build_stiffness_matrix(mesh, soil_flag = False)
        U = kernel.mesh_to_global_disp_dof_vector(mesh.global_disp)
        supports = kernel.mesh_to_global_restrained_dof_vector(mesh.global_restrained)
        
        
        # validate boundary conditions
        validate.check_boundary_conditions(mesh)
        
        u, _ = kernel.solve_equations(K, F, U, restraints=supports)
        
        return u

def simple_winkler_analysis(mesh):
    pass


