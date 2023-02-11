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
import pandas as pd
import matplotlib.pyplot as plt

def lateral_loading(mesh):
    
    def repeat_inner(arr):
        
        arr = arr.reshape(-1,1)
        
        arr_inner = arr[1:-1]
        arr_inner = np.tile(arr_inner,(2)).reshape(-1)
        
        return np.hstack([arr[0],arr_inner,arr[-1]])
    
    def structural_forces_to_df(mesh,q):
            z = mesh.nodes_coordinates['z [m]'].values
            z = repeat_inner(z)
            L = kernel.mesh_to_element_length(mesh).reshape(-1)

            N = np.vstack( (q_int[0::6], -q_int[3::6]) ).reshape(-1,order='F')
            V = np.vstack( (q_int[1::6], -q_int[4::6]) ).reshape(-1,order='F')
            M = np.vstack( (q_int[2::6], q_int[2::6]-L*q_int[1::6]) ).reshape(-1,order='F')
            
            structural_forces_to_DataFrame = pd.DataFrame(data={
                'Elevation [m]': z,
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
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
        ax1.set_ylabel('Elevation [m VREF]')
        ax1.set_xlabel('Deflection [m]')
        ax1.grid(which='both')
        ax1.plot(u[1::3],mesh.nodes_coordinates['z [m]'].values)

        ax2.set_xlabel('Normal [kN]')
        ax2.grid(which='both')
        ax2.fill_betweenx(NVM['Elevation [m]'].values,NVM['N [kN]'].values)
        ax2.set_xlim([NVM['N [kN]'].values.min()-0.1, NVM['N [kN]'].values.max()+0.1])
        ax2.set_yticklabels('')

        ax3.set_xlabel('Shear [kN]')
        ax3.grid(which='both')
        ax3.fill_betweenx(NVM['Elevation [m]'].values,NVM['V [kN]'].values)
        ax3.set_xlim([NVM['V [kN]'].values.min()-0.1, NVM['V [kN]'].values.max()+0.1])
        ax3.set_yticklabels('')

        ax4.set_xlabel('Moment [kNm]')
        ax4.grid(which='both')
        ax4.fill_betweenx(NVM['Elevation [m]'].values,NVM['M [kNm]'].values)
        ax4.set_xlim([NVM['M [kNm]'].values.min()-0.1, NVM['M [kNm]'].values.max()+0.1])
        ax4.set_yticklabels('')
        
        return u, NVM, fig