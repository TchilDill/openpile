# general utilities for openfile

import numpy as np
from numba import njit

@njit
def jit_solve(A, b):
    return np.linalg.solve(A, b)

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

def solve_equations(K , F, BC):
    """function that solves the system of equations

    _extended_summary_

    Parameters
    ----------
    K : numpy array
        Global stiffness matrix with all dofs in kPa.
    F : numpy array 
        External force vetor (can also be denoted as load vector) in kN.
    BC : list (of tuples)
        tuples where first item is the prescribed dof, and second item is the precribed displacement in metres.
    """    
    
    if BC:
        prescribed_dof_all = np.arange(len(F))
        prescribed_dof_True =  np.arange(len(BC))
        prescribed_disp = np.zeros([len(BC),1],dtype=float)
        for idx, val in enumerate(BC):
            prescribed_dof_True[idx] = int(val[0])
            prescribed_disp[idx] = val[1]
            
        prescribed_dof_False =  reverse_indices(prescribed_dof_all, prescribed_dof_True)

        u = np.zeros(F.shape,dtype=float)
        u[prescribed_dof_True] = prescribed_disp

        Fred = numba_ix(F,prescribed_dof_False,np.array([0])) - numba_ix(K,prescribed_dof_False,prescribed_dof_True) @ prescribed_disp 
        Kred = numba_ix(K, prescribed_dof_False,prescribed_dof_False)

        u[prescribed_dof_False] = jit_solve(Kred,Fred)        
    else:
        u = jit_solve(K,F)
    
    Q = K @ u - F
    
    return u, Q
