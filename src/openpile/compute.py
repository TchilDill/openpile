"""
`Compute` module
==================

The `compute` module is used to run various simulations. 

Every function from this module returns an `openpile.compute.Result` object. 

"""

import openpile.utils.kernel as kernel 
import openpile.utils.validation as validate
import numpy as np

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
        K = kernel.build_stiffness_matrix(mesh)
        U = kernel.mesh_to_global_disp_dof_vector(mesh.global_disp)
        supports = kernel.mesh_to_global_restrained_dof_vector(mesh.global_restrained)
        
        # validate boundary conditions
        validate.check_boundary_conditions(mesh)
        
        u, _ = kernel.solve_equations(K, F, U, restraints=supports)
        
        return u

def simple_winkler_analysis(mesh):
    pass


