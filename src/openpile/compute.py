"""
`Compute` module
==================

The `compute` module is used to run various simulations. 


The :

- the pile
- the soil profile
- the mesh

"""

import openpile.utils.kernel as kernel 
import openpile.utils.validation as validate
import numpy as np

def lateral_loading(mesh):
    
    if mesh.soil is None:
        F = kernel.mesh_to_global_force_dof_vector(mesh.global_forces)
        K = kernel.build_stiffness_matrix(mesh)
        U = kernel.mesh_to_global_disp_dof_vector(mesh.global_disp)
        supports = kernel.mesh_to_global_restrained_dof_vector(mesh.global_restrained)
        
        # validate boundary conditions
        validate.check_boundary_conditions(mesh)
        
        u, _ = kernel.solve_equations(K, F, U, restraints=supports)
        
        return u

def lateral_displacement(mesh):
    pass


