# general utilities for openfile

import numpy as np
from numba import njit

@njit
def jit_solve(A, b):
    return np.linalg.solve(A, b)

@njit
def jit_dot(a, b):
    return a.dot(b)

@njit
def jit_eigh(A):
    return np.linalg.eigh(A)

@njit
def reverse_indices(A, B):
    return np.array([x for x in A if x not in B])

@njit  
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

def solve_equations(K, F, BC, _test=False):
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
    
    Q = K @ U - F
    
    return U, Q
 
def elem_mechanical_stiffness_matrix(openpile_mesh, elem_no):
    """creates element stiffness matrix based on mesh info and element number

    Parameters
    ----------
    openpile_mesh : openpile class object
        includes information on elements, nodes and other mesh-related data. 
    elem_no : int
        element number for which matrix needs to be computed. element number 0 is at the top of the mesh.

    Returns
    -------
    k: numpy array (2d)
        element stiffness matrix related to the mechanical properties of the structure

    Raises
    ------
    ValueError
        mesh can currently be only of type "Linear 2-dof", meaning 2-dof per node (shear and moment)
    """
    
    if openpile_mesh.element.type == 'lin2':
        
        EI = openpile_mesh.element.E[elem_no] * openpile_mesh.element.I[elem_no]
        L = openpile_mesh.element.L[elem_no]
        
        k = np.array([
            [ 12,    6*L,   -12,   -6*L],
            [6*L,  4*L*L,  -6*L,  2*L*L],
            [-12,   -6*L,    12,    6*L],
            [6*L,  2*L*L,  -6*L,  4*L*L],
        ]) * EI / L**3
    else:
        raise ValueError('openpile_mesh.element.type only accepts 2-dof linear element')
        
    return k
   
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
    ndof_per_node = openpile_mesh.node.ndof
    #number of nodes per element
    node_per_element = openpile_mesh.element.node
    #number of elements in mesh
    n_elem = openpile_mesh.element.number
    #global dimension of the global stiffness matrix can be inferred 
    ndim_global = ndof_per_node * n_elem + ndof_per_node
    
    #pre-allocate stiffness matrix
    K = np.zeros((ndim_global, ndim_global), dtype=float)
    
    for i in range(n_elem):
        
        #mechanical stiffness properties
        k_elem = elem_mechanical_stiffness_matrix(openpile_mesh, i)
        
        #dummy stiffness matrix that updates at each iteration
        K_temp = np.zeros((ndim_global, ndim_global), dtype=float)
        #select relevant rows/columns of dummy stiffness matrix
        start = ndof_per_node*i
        end = ndof_per_node*i+ndof_per_node*node_per_element
        K_temp[start:end, start:end] = k_elem
        
        #update global stiffness matrix
        K += K_temp
        
    return K