"""
---------------
`Kernel` module
---------------

#TODO

"""
# general utilities for openfile

import numpy as np
import pandas as pd
from numba import njit, prange, f4, f8, b1, char
import numba as nb
from typing_extensions import Literal

from openpile.construct import Model, Pile

from openpile.core import misc


@njit(f8[:](f8[:, :], f8[:]), cache=True)
def jit_solve(A, b):
    return np.linalg.solve(A, b)


@njit(parallel=True, cache=True)
def jit_eigh(A):
    return np.linalg.eigh(A)


@njit(parallel=True, cache=True)
def reverse_indices(A, B):
    return np.array([x for x in A if x not in B])


@njit(parallel=True, cache=True)
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
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


def global_dof_vector_to_consistent_stacked_array(dof_vector, dim):
    arr = dof_vector.reshape((-1, 3))
    arr_inner = np.tile(arr[1:-1], 2)
    arr = np.vstack([arr[0, :], arr_inner.reshape((-1, 3)), arr[-1, :]])
    arr = arr.reshape(-1, dim, 1)

    return arr


def solve_equations(K, F, U, restraints):
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

    if restraints.any():
        prescribed_dof_true = np.where(restraints)[0]
        prescribed_dof_false = np.where(~restraints)[0]

        prescribed_disp = U

        Fred = F[prescribed_dof_false] - numba_ix(
            K, prescribed_dof_false, prescribed_dof_true
        ).dot(prescribed_disp[prescribed_dof_true])
        Kred = numba_ix(K, prescribed_dof_false, prescribed_dof_false)

        U[prescribed_dof_false] = jit_solve(Kred, Fred)
    else:
        U = jit_solve(K, F)

    Q = K.dot(U) - F

    return U, Q


def mesh_to_element_length(model) -> np.ndarray:
    # elememt coordinates along z and y-axes
    ez = np.array(
        [
            model.element_properties["x_top [m]"].to_numpy(dtype=float),
            model.element_properties["x_bottom [m]"].to_numpy(dtype=float),
        ]
    )
    ey = np.array(
        [
            model.element_properties["y_top [m]"].to_numpy(dtype=float),
            model.element_properties["y_bottom [m]"].to_numpy(dtype=float),
        ]
    )
    # length calcuated via pythagorus theorem
    L = np.sqrt(((ez[0] - ez[1]) ** 2 + (ey[0] - ey[1]) ** 2)).reshape(-1, 1, 1)

    return L


def elem_mechanical_stiffness_matrix(model):
    """creates pile element stiffness matrix based on model info and element number

    Parameters
    ----------
    model : openpile class object `openpile.construct.Model`
        includes information on soil/structure, elements, nodes and other model-related data.

    Returns
    -------
    k: numpy array (3d)
        element stiffness matrix of all elememts related to the mechanical properties of the structure

    Raises
    ------
    ValueError
        Model.element.type only accepts 'EB' type (for Euler-Bernoulli) or 'T' type (for Timoshenko)
    ValueError
        Timoshenko beams cannot be used yet for non-circular pile types
    ValueError
        ndof per node can be either 2 or 3

    """

    # calculate length vector
    L = mesh_to_element_length(model)
    # elastic properties
    nu = model.pile._nu
    E = model.element_properties["E [kPa]"].to_numpy(dtype=float).reshape((-1, 1, 1))
    G = E / (2 + 2 * nu)
    # cross-section properties
    I = model.element_properties["I [m4]"].to_numpy(dtype=float).reshape((-1, 1, 1))
    A = model.element_properties["Area [m2]"].to_numpy(dtype=float).reshape((-1, 1, 1))
    d = (
        model.element_properties["Diameter [m]"]
        .to_numpy(dtype=float)
        .reshape((-1, 1, 1))
    )
    wt = (
        model.element_properties["Wall thickness [m]"]
        .to_numpy(dtype=float)
        .reshape((-1, 1, 1))
    )

    # calculate shear component in stiffness matrix (if Timorshenko)
    if model.element_type == "EulerBernoulli":
        kappa = 0
    elif model.element_type == "Timoshenko":
        if model.pile.kind == "Circular":
            a = 0.5 * d
            b = 0.5 * (d - 2 * wt)
            nom = 6 * (a**2 + b**2) ** 2 * (1 + nu) ** 2
            denom = (
                7 * a**4
                + 34 * a**2 * b**2
                + 7 * b**4
                + nu * (12 * a**4 + 48 * a**2 * b**2 + 12 * b**4)
                + nu**2 * (4 * a**4 + 16 * a**2 * b**2 + 4 * b**4)
            )
            kappa = nom / denom
        else:
            raise ValueError(
                "Timoshenko beams cannot be used yet for non-circular pile types"
            )
    else:
        raise ValueError(
            "Model.element.type only accepts 'EB' type (for Euler-Bernoulli) of 'T' type (for Timoshenko)"
        )

    phi = 12 * E * I * kappa / (A * G * L**2)
    X = A * E / L
    Y1 = (12 * E * I) / ((1 + phi) * L**3)
    Y2 = (6 * E * I) / ((1 + phi) * L**2)
    Y3 = ((4 + phi) * E * I) / ((1 + phi) * L)
    Y4 = ((2 - phi) * E * I) / ((1 + phi) * L)
    N = np.zeros(shape=X.shape)

    k = np.block(
        [
            [X, N, N, -X, N, N],
            [N, Y1, Y2, N, -Y1, Y2],
            [N, Y2, Y3, N, -Y2, Y4],
            [-X, N, N, X, N, N],
            [N, -Y1, -Y2, N, Y1, -Y2],
            [N, Y2, Y4, N, -Y2, Y3],
        ]
    )

    return k


def elem_py_stiffness_matrix(model, u, kind):
    """creates soil element stiffness matrix based on model info and element number.

    The soil stiffness matrix assumes the soil stiffness to vary lineraly along the elements.

    #TODO: proof here

    Parameters
    ----------
    model : openpile class object `openpile.construct.Model`
        includes information on soil/structure, elements, nodes and other model-related data.
    u: np.ndarray
        Global displacement vector
    kind: str
        "initial", "secant" or "tangent"

    Returns
    -------
    k: numpy array (3d)
        soil consistent stiffness matrix of all elememts related to the p-y soil springs' stiffness

    Raises
    ------
    ValueError
        ndof per node can be either 2 or 3

    """

    # calculate length vector
    L = mesh_to_element_length(model)

    # calculate the spring stiffness
    ksoil = calculate_springs_stiffness(u=u[1::3], springs=model.py_springs, kind=kind)

    N = 0 * L
    A = 2 * L / 7
    B = L**2 / 28
    C = L**3 / 168
    D = 9 * L / 140
    E = L**2 / 70
    F = 3 * L / 35
    G = -(L**2) / 60
    H = -(L**3) / 280
    I = L**3 / 280

    ktop = (
        np.block(
            [
                [N, N, N, N, N, N],
                [N, A, B, N, D, G],
                [N, B, C, N, E, H],
                [N, N, N, N, N, N],
                [N, D, E, N, F, G],
                [N, G, H, N, G, I],
            ]
        )
        * ksoil[:, 0]
    )

    kbottom = (
        np.block(
            [
                [N, N, N, N, N, N],
                [N, F, -G, N, D, -E],
                [N, -G, -H, N, -G, -I],
                [N, N, N, N, N, N],
                [N, D, -G, N, A, -B],
                [N, -E, -I, N, -B, C],
            ]
        )
        * ksoil[:, 1]
    )

    return ktop + kbottom


@njit(parallel=True, cache=True)
def jit_build(k, ndim, n_elem, node_per_element, ndof_per_node):
    # pre-allocate stiffness matrix
    K = np.zeros((ndim, ndim), dtype=np.float64)

    for i in prange(n_elem):
        # dummy stiffness matrix that updates at each iteration
        K_temp = np.zeros((ndim, ndim), dtype=np.float64)
        # select relevant rows/columns of dummy stiffness matrix
        start = ndof_per_node * i
        end = ndof_per_node * i + ndof_per_node * node_per_element
        K_temp[start:end, start:end] = k[i]

        # update global stiffness matrix
        K += K_temp

    return K


def build_stiffness_matrix(model, u=None, kind=None):
    """Builds the stiffness matrix based on the model(element and node) properties

    Element stiffness matrices are first computed for each element and then loaded in the global stiffness matrix through summation.
    Different element stiffness matrices are computed depending on the specifications found in the Model object.

    Parameters
    ----------
    model : obj from openpile library
        stores all information about nodes elements, and other model-related items, see #TODO:link to model class
    u: np.ndarray, Optional
        Global displacement vector. Must be given if soil_flag is not None
    kind: str, Optional
        "initial", "secant" or "tangent". Must be given if soil_flag is not None

    Returns
    -------
    K : numpy array of dim (ndof, ndof)
        Global stiffness matrix with all dofs in unit:1/kPa.
    """

    # number of dof per node
    ndof_per_node = 3
    # number of nodes per element
    node_per_element = 2
    # number of elements in mesh
    n_elem = model.element_number
    # global dimension of the global stiffness matrix can be inferred
    ndim_global = ndof_per_node * n_elem + ndof_per_node

    # mechanical stiffness properties
    k = elem_mechanical_stiffness_matrix(model)
    # add soil contribution
    if model.soil is not None:
        # gives warning if soil is given without displacements or type of stiffness
        if u is None or kind is None:
            UserWarning("'u' and 'kind' must be stipulated.")
        else:
            if model.distributed_lateral:
                k += elem_py_stiffness_matrix(model, u, kind)
            elif model.distributed_moment:
                k += 0
            elif model.base_shear:
                k += 0
            elif model.base_moment:
                k += 0

    K = jit_build(k, ndim_global, n_elem, node_per_element, ndof_per_node)

    return K


def mesh_to_global_force_dof_vector(df: pd.DataFrame) -> np.ndarray:
    # extract each column (one line per node)
    force_dof_vector = (
        df[["Px [kN]", "Py [kN]", "Mz [kNm]"]].values.reshape(-1).astype(np.float64)
    )

    return force_dof_vector


def mesh_to_global_disp_dof_vector(df: pd.DataFrame) -> np.ndarray:
    # extract each column (one line per node)
    disp_dof_vector = (
        df[["Tx [m]", "Ty [m]", "Rz [rad]"]].values.reshape(-1).astype(np.float64)
    )

    return disp_dof_vector


def mesh_to_global_restrained_dof_vector(df: pd.DataFrame) -> np.ndarray:
    # extract each column (one line per node)
    restrained_dof_vector = df[["Tx", "Ty", "Rz"]].values.reshape(-1)

    return restrained_dof_vector


def struct_internal_force(model, u) -> np.ndarray:
    # number of dof per node
    ndof_per_node = 3
    # number of nodes per element
    node_per_element = 2

    # create mech consistent stiffness matrix
    k = elem_mechanical_stiffness_matrix(model)

    # add soil contribution
    if model.soil is not None:
        kind = "secant"
        if model.distributed_lateral:
            k += elem_py_stiffness_matrix(model, u, kind)
        elif model.distributed_moment:
            k += 0
        elif model.base_shear:
            k += 0
        elif model.base_moment:
            k += 0

    # create array u of shape [n_elem x 6 x 1]
    u = global_dof_vector_to_consistent_stacked_array(
        u, ndof_per_node * node_per_element
    )
    # compute internal forces and reshape into global dof vector
    F_int = (-1) * np.matmul(k, u).reshape((-1))

    return F_int


@njit(cache=True)
def calculate_springs_stiffness(
    u: np.ndarray, springs: np.ndarray, kind: Literal["initial", "secant", "tangent"]
):
    """Calculate springs stiffness

    Parameters
    ----------
    u : np.ndarray
        displacements to calculate stiffness.
        For dofs related to t-z curves, u = U[::3] where U is the global displacement vector.
        For dofs related to p-y curves, u = U[1::3] where U is the global displacement vector.
        For dofs related to m-t curves, u = U[2::3] where U is the global displacement vector.
    springs : np.ndarray
        soil-structure interaction springs array of shape (n_elem, 2, 2, spring_dim)
    kind : str
        defines whether it is initial, secant of tangent stiffness to define

    Returns
    -------
    k: np.ndarray
        secant or tangent stiffness for all elements. Array of shape(n_elem,2,1,1)
    """

    # double inner values
    # d = misc.repeat_inner(u)
    d = np.zeros(((len(u) - 1) * 2), dtype=np.float64)
    i = 0
    while i < len(d) + 1:
        if i == 0:
            d[i] = u[0]
            i += 1
        elif i == len(d):
            d[i] = u[-1]
            i += 1
        elif i == len(d) - 1:
            d[i] = u[-2]
            i += 1
        elif i % 2 != 0:
            x = int((i + 1) / 2)
            d[i] = u[x]
            d[i + 1] = u[x]
            i += 2
        elif i % 2 == 0:
            x = int(i / 2)
            d[i] = u[x]
            d[i + 1] = u[x]
            i += 2

    # displacemet with same dimension as spring
    d = np.abs(d).reshape((-1, 2, 1, 1))

    k = np.zeros(d.shape, dtype=np.float64)

    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if np.sum(springs[i, j, 1]) == 0:
                pass
            else:
                if kind == "initial" or d[i, j, 0, 0] == 0.0:
                    dx = springs[i, j, 1, 1] - springs[i, j, 1, 0]
                    p0 = springs[i, j, 0, 0]
                    p1 = springs[i, j, 0, 1]
                elif kind == "secant":
                    dx = d[i, j, 0, 0]
                    p0 = springs[i, j, 0, 0]
                    if d[i, j, 0, 0] > np.max(springs[i, j, 1]):
                        p1 = springs[i, j, 0, -1]
                    else:
                        p1 = np.interp(dx, springs[i, j, 1], springs[i, j, 0])
                elif kind == "tangent":
                    dx = min(0.0005, d[i, j, 0, 0])
                    if (d[i, j, 0, 0] - dx) > np.max(springs[i, j, 1]):
                        p0 = springs[i, j, 0, -1]
                    else:
                        p0 = np.interp(
                            d[i, j, 0, 0] - dx, springs[i, j, 1], springs[i, j, 0]
                        )
                    if d[i, j, 0, 0] > np.max(springs[i, j, 1]):
                        p1 = springs[i, j, 0, -1]
                    else:
                        p1 = np.interp(
                            d[i, j, 0, 0], springs[i, j, 1], springs[i, j, 0]
                        )

            k[i, j, 0, 0] = abs((p1 - p0) / dx)

    return k


def computer():
    """This function is the solver of openpile.

    The solver reads the boundary conditions, the global Force and Displacement vector and iterate to find convergence.
    """
