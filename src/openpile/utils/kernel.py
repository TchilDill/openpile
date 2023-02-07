"""
`Kernel` module
===============

#TODO

"""
# general utilities for openfile

import numpy as np
import pandas as pd
from numba import njit, prange
from openpile.construct import Mesh, Pile

@njit(cache=True)
def jit_solve(A, b):
    return np.linalg.solve(A, b)

@njit(cache=True)
def jit_dot(a, b):
    return a.dot(b)

@njit(cache=True)
def jit_eigh(A):
    return np.linalg.eigh(A)

@njit(cache=True)
def reverse_indices(A, B):
    return np.array([x for x in A if x not in B])

@njit(cache=True)
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))

def solve_equations(K, F, U, restraints=None):
    r"""function that solves the system of equations

    The function uses numba to speed up the computational time.

    The system of equations to solve is the following :eq:`solve_eq_3dof_example` assuming a trivial 3 degree of freedom system:
    
    .. math::
        :label: solve_eq_3dof_example
    
        \begin{pmatrix}
        U_1\\ 
        U_2\\ 
        U_3
        \end{pmatrix} = 
        \begin{bmatrix}
        K_{11} & K_{12} & K_{13} \\ 
        K_{21} & K_{22} & K_{23}\\ 
        K_{31} & K_{23} & K_{33}
        \end{bmatrix} \cdot \begin{pmatrix}
        F_1 \\ 
        F_2 \\ 
        F_3
        \end{pmatrix}
        \\
        \\

    Parameters
    ----------
    K : float numpy array of dim (ndof, ndof)
        Global stiffness matrix with all dofs in unit:1/kPa.
    F : float numpy array of dim (ndof, 1)
        External force vetor (can also be denoted as load vector) in unit:kN.
    restraints: boolean numpy array of dim (ndof, 1)
        External vector of restrained dof.
        
    Returns
    -------
    U : float numpy array of dim (ndof, 1)
        Global displacement vector in unit:m.
    Q : float numpy array of dim (ndof, 1)
        Global reaction force vector in unit:kN
    
    Example
    -------
    
    >>> from openpile.utils.kernel import solve_equations
    >>> import numpy as np
    
    >>> K = np.random.rand(3,3) + np.identity(3) # ensure the matrix can be inverted
    >>> F = np.random.rand(3,1)
    >>> U, _ = solve_equations(K,F)
    """    
    
    if restraints is not None:
        
        prescribed_dof_true = np.where(restraints==True)[0]
        prescribed_dof_false =  np.where(restraints==False)[0]
        
        prescribed_disp = U
            
        Fred = numba_ix(F,prescribed_dof_false,np.array([0])) - jit_dot(numba_ix(K,prescribed_dof_false,prescribed_dof_true), numba_ix(prescribed_disp,prescribed_dof_true,np.array([0])))
        Kred = numba_ix(K, prescribed_dof_false,prescribed_dof_false)

        U[prescribed_dof_false] = jit_solve(Kred,Fred)
    else:
        U = jit_solve(K,F)
    
    Q = jit_dot(K, U) - F
    
    return U, Q

def solve_equations_2(K, F, BC, _test=False):
    r"""function that solves the system of equations

    The function uses numba to speed up the computational time.

    The system of equations to solve is the following :eq:`solve_eq_3dof_example` assuming a trivial 3 degree of freedom system:
    
    .. math::
        :label: solve_eq_3dof_example
    
        \begin{pmatrix}
        U_1\\ 
        U_2\\ 
        U_3
        \end{pmatrix} = 
        \begin{bmatrix}
        K_{11} & K_{12} & K_{13} \\ 
        K_{21} & K_{22} & K_{23}\\ 
        K_{31} & K_{23} & K_{33}
        \end{bmatrix} \cdot \begin{pmatrix}
        F_1 \\ 
        F_2 \\ 
        F_3
        \end{pmatrix}
        \\
        \\

    Parameters
    ----------
    K : numpy array of dim (ndof, ndof)
        Global stiffness matrix with all dofs in unit:1/kPa.
    F : numpy array of dim (ndof, 1)
        External force vetor (can also be denoted as load vector) in unit:kN.
    BC : list (of tuples)
        tuples where first item is the prescribed dof, and second item is the precribed displacement in unit:m.
        
    Returns
    -------
    U : numpy array of dim (ndof, 1)
        Global displacement vector in unit:m.
    Q : numpy array of dim (ndof, 1)
        Global reaction force vector in unit:kN
    
    Example
    -------
    
    >>> from openpile.utils.kernel import solve_equations
    >>> import numpy as np
    
    >>> K = np.random.rand(3,3) + np.identity(3) # ensure the matrix can be inverted
    >>> F = np.random.rand(3,1)
    >>> BC = [(0,0),] # restrain first dof
    >>> U, Q = solve_equations(K,F,BC)
    """    
    
    if BC or _test:
        prescribed_dof_all = np.arange(len(F))
        prescribed_dof_true =  np.arange(len(BC))
        prescribed_disp = np.zeros([len(BC),1],dtype=float)
        for idx, val in enumerate(BC):
            prescribed_dof_true[idx] = int(val[0])
            prescribed_disp[idx] = val[1]
            
        prescribed_dof_false =  reverse_indices(prescribed_dof_all, prescribed_dof_true)

        U = np.zeros(F.shape,dtype=float)
        U[prescribed_dof_true] = prescribed_disp

        Fred = numba_ix(F,prescribed_dof_false,np.array([0])) - jit_dot(numba_ix(K,prescribed_dof_false,prescribed_dof_true), prescribed_disp) 
        Kred = numba_ix(K, prescribed_dof_false,prescribed_dof_false)

        U[prescribed_dof_false] = jit_solve(Kred,Fred)        
    else:
        U = jit_solve(K,F)
    
    Q = jit_dot(K, U) - F
    
    return U, Q
 
def elem_mechanical_stiffness_matrix(mesh):
    """creates pile element stiffness matrix based on mesh info and element number

    Parameters
    ----------
    mesh : openpile class object `openpile.construct.Mesh`
        includes information on soil/structure, elements, nodes and other mesh-related data. 

    Returns
    -------
    k: numpy array (3d)
        element stiffness matrix of all elememts related to the mechanical properties of the structure

    Raises
    ------
    ValueError
        openpile_mesh.element.type only accepts 'EB' type (for Euler-Bernoulli) or 'T' type (for Timoshenko)
    ValueError
        Timoshenko beams cannot be used yet for non-circular pile types
    ValueError
        ndof per node can be either 2 or 3

    """      
       
    # elememt coordinates along z and y-axes 
    ez = np.array([mesh.element_properties['z_top [m]'].to_numpy(dtype=float) , mesh.element_properties['z_bottom [m]'].to_numpy(dtype=float)])
    ey = np.array([mesh.element_properties['y_top [m]'].to_numpy(dtype=float) , mesh.element_properties['y_bottom [m]'].to_numpy(dtype=float)])
    # length calcuated via pythagorus theorem
    L = np.sqrt(( (ez[0] - ez[1])**2 + (ey[0] - ey[1])**2 )).reshape((-1,1,1))
    # elastic properties
    nu = mesh.pile._nu
    E = mesh.element_properties['E [kPa]'].to_numpy(dtype=float).reshape((-1,1,1))
    G = E/(2+2*nu)
    #cross-section properties
    I = mesh.element_properties['I [m4]'].to_numpy(dtype=float).reshape((-1,1,1))
    A = mesh.element_properties['Area [m2]'].to_numpy(dtype=float).reshape((-1,1,1))
    d = mesh.element_properties['Diameter [m]'].to_numpy(dtype=float).reshape((-1,1,1))
    wt = mesh.element_properties['Wall thickness [m]'].to_numpy(dtype=float).reshape((-1,1,1))

    # calculate shear component in stiffness matrix (if Timorshenko)
    if mesh.element_type == 'EulerBernoulli':
        kappa = 0
    elif mesh.element_type == 'Timoshenko':
        if mesh.pile.type == 'Circular':
            a = 0.5 * d
            b = 0.5 * ( d - 2*wt )
            nom = 6 * (a**2 + b**2)**2 * (1 + nu)**2
            denom = 7 * a**4 + 34 * a**2 * b**2 + 7 * b**4 + nu * ( 12 * a**4 + 48 * a**2 * b**2 + 12 * b**4) + nu**2 * (4 * a**4 + 16 * a**2 * b**2 + 4 * b**4)
            kappa = nom/denom
        else:
            raise ValueError("Timoshenko beams cannot be used yet for non-circular pile types")
    else:
        raise ValueError("openpile_mesh.element.type only accepts 'EB' type (for Euler-Bernoulli) of 'T' type (for Timoshenko)")
            
    phi = 12*E*I*kappa/(A*G*L**2)
    X = A*E/L
    Y1 = (12*E*I) / ((1+phi)*L**3)
    Y2 = ( 6*E*I) / ((1+phi)*L**2)
    Y3 = ((4+phi)*E*I) / ((1+phi)*L)
    Y4 = ((2-phi)*E*I) / ((1+phi)*L)
    N = np.zeros(shape=X.shape)
    
    k = np.block(
        [
            [ X,   N,   N, -X,   N,   N],
            [ N,  Y1,  Y2,  N, -Y1,  Y2],
            [ N,  Y2,  Y3,  N, -Y2,  Y4],
            [-X,   N,   N,  X,   N,   N],
            [ N, -Y1, -Y2,  N,  Y1, -Y2],
            [ N,  Y2,  Y4,  N, -Y2,  Y3],
        ]
    )
             
    return k

@njit(parallel=True, cache=True)
def jit_build(k, ndim, n_elem, node_per_element, ndof_per_node):
    
    #pre-allocate stiffness matrix
    K = np.zeros((ndim, ndim), dtype=float)
    
    for i in prange(n_elem):
        #dummy stiffness matrix that updates at each iteration
        K_temp = np.zeros((ndim, ndim), dtype=float)
        #select relevant rows/columns of dummy stiffness matrix
        start = ndof_per_node*i
        end = ndof_per_node*i+ndof_per_node*node_per_element
        K_temp[start:end, start:end] = k[i]
        
        #update global stiffness matrix
        K += K_temp
    
    return K


def build_stiffness_matrix(openpile_mesh):
    """Builds the stiffness matrix based on the mesh(element and node) properties 

    Element stiffness matrices are first computed for each element and then loaded in the global stiffness matrix through summation.
    Different element stiffness matrices are computed depending on the specifications found in the openpile.mesh object.

    Parameters
    ----------
    openpile_mesh : obj from openpile library
        stores all information about nodes elements, and other mesh-related items, see #TODO:link to mesh class 

    Returns
    -------
    K : numpy array of dim (ndof, ndof)
        Global stiffness matrix with all dofs in unit:1/kPa.
    """

    
    #number of dof per node
    ndof_per_node = 3
    #number of nodes per element
    node_per_element = 2
    #number of elements in mesh
    n_elem = openpile_mesh.element_number
    #global dimension of the global stiffness matrix can be inferred 
    ndim_global = ndof_per_node * n_elem + ndof_per_node
    
    #mechanical stiffness properties
    k_elem = elem_mechanical_stiffness_matrix(openpile_mesh)
    
    K = jit_build(k_elem, ndim_global, n_elem, 2, 3)
        
    return K

def mesh_to_global_force_dof_vector(df:pd.DataFrame):

    # extract each column (one line per node)
    force_dof_vector = df[['Pz [kN]', 'Py [kN]', 'Mx [kNm]']].values.reshape(-1,1).astype(np.float64)
    
    return force_dof_vector

def mesh_to_global_disp_dof_vector(df:pd.DataFrame):

    # extract each column (one line per node)
    disp_dof_vector = df[['Tz [m]', 'Ty [m]', 'Rx [rad]']].values.reshape(-1,1).astype(np.float64)
    
    return disp_dof_vector

def mesh_to_global_restrained_dof_vector(df:pd.DataFrame):

    # extract each column (one line per node)
    restrained_dof_vector = df[['Tz', 'Ty', 'Rx']].values.reshape(-1,1)
    
    return restrained_dof_vector

def computer():
    """This function is the solver of openpile.

    The solver reads the boundary conditions, the global Force and Displacement vector and iterate to find convergence.
    """
    
