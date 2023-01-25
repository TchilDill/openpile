"""
`Kernel` module
===============

#TODO

"""
# general utilities for openfile

import numpy as np
from numba import njit, prange

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
    
    Q = jit_dot(K, U) - F
    
    return U, Q
 
def elem_mechanical_stiffness_matrix(openpile_mesh, pile):
    """creates pile element stiffness matrix based on mesh info and element number

    Parameters
    ----------
    openpile_mesh : openpile class object
        includes information on elements, nodes and other mesh-related data. 
    pile: openpile class obect
        includes information on pile

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
    
    #number of dof per node
    ndof_per_node = openpile_mesh.node_dof_count
    #number of nodes per element
    node_per_element = openpile_mesh.element_node_count
    #number of element
    elem_no = openpile_mesh.element_count
    #element stiffnedd matrix dimension
    dim = ndof_per_node*node_per_element
    
    k = np.zeros(shape=(elem_no,dim,dim), dtype=float)
    
    for i in range(elem_no):
            
        E = pile.E
        I = openpile_mesh.element.I[i]
        L = openpile_mesh.element.L[i]
        G = openpile_mesh.element.E[i]/(2+2*pile.nu)
        A = openpile_mesh.element.A[i]
    
        if openpile_mesh.element.type == 'EB':
            kappa = 0
        elif openpile_mesh.element.type == 'T':
            if pile.type == 'Circular':
                a = 0.5 * openpile_mesh.element.spread[i]
                b = 0.5 * ( openpile_mesh.element.spread[i] - 2*openpile_mesh.element.thickness[i] )
                nom = 6 * (a**2 + b**2)**2 * (1 + pile.nu)**2
                denom = 7 * a**4 + 34 * a**2 * b**2 + 7 * b**4 + pile.nu * ( 12 * a**4 + 48 * a**2 * b**2 + 12 * b**4) + pile.nu**2 * (4 * a**4 + 16 * a**2 * b**2 + 4 * b**4)
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
        
        k[i] = np.array(
            [
                [ X,   0,   0, -X,   0,   0],
                [ 0,  Y1,  Y2,  0, -Y1,  Y2],
                [ 0,  Y2,  Y3,  0, -Y2,  Y4],
                [-X,   0,   0,  X,   0,   0],
                [ 0, -Y1, -Y2,  0,  Y1, -Y2],
                [ 0,  Y2,  Y4,  0, -Y2,  Y3],
            ]
        )
             
    return k

@njit(parallel=True, cache=True)
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
    
    #mechanical stiffness properties
    k_elem = elem_mechanical_stiffness_matrix(openpile_mesh)
    
    if openpile_mesh.settings.soilprofile:
        k_soil = elem_soil_stiffness_matrix(openpile_mesh)
        k_elem += k_soil 
    
    for i in prange(n_elem):
        #dummy stiffness matrix that updates at each iteration
        K_temp = np.zeros((ndim_global, ndim_global), dtype=float)
        #select relevant rows/columns of dummy stiffness matrix
        start = ndof_per_node*i
        end = ndof_per_node*i+ndof_per_node*node_per_element
        K_temp[start:end, start:end] = k_elem[i]
        
        #update global stiffness matrix
        K += K_temp
        
    return K