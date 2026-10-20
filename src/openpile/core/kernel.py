"""
---------------
`Kernel` module
---------------

#TODO

"""
# general utilities for openfile

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing_extensions import Literal

from openpile.construct import CircularPileSection
from openpile.core.validation import UserInputError
from openpile.core import misc
from openpile.core._model_build import parameter2elements


def jit_solve(A, b):
    if sparse.issparse(A):
        return spsolve(A, b)
    return np.linalg.solve(A, b)


def jit_eigh(A):
    return np.linalg.eigh(A)


def reverse_indices(A, B):
    return np.array([x for x in A if x not in B])


def double_inner_njit(in_arr):
    # equivalent to misc.repeat_inner(u)
    out_arr = np.zeros(((len(in_arr) - 1) * 2), dtype=np.float64)
    if len(in_arr) >= 2:
        out_arr[0] = in_arr[0]
        out_arr[-1] = in_arr[-1]
        out_arr[1:-1] = np.repeat(in_arr[1:-1], 2)

    return out_arr


def global_dof_vector_to_consistent_stacked_array(dof_vector, dim):
    arr = dof_vector.reshape((-1, 3))
    arr_inner = np.tile(arr[1:-1], 2)
    arr = np.vstack([arr[0, :], arr_inner.reshape((-1, 3)), arr[-1, :]])
    arr = arr.reshape(-1, dim, 1)

    return arr


def solve_equations(K, F, U, restraints):
    r"""function that solves the system of equations

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
    
    """

    if restraints.any():
        prescribed_dof_true = np.where(restraints)[0]
        prescribed_dof_false = np.where(~restraints)[0]

        prescribed_disp = U

        # row/column slicing works for both dense arrays and sparse CSR matrices
        Kft = K[prescribed_dof_false][:, prescribed_dof_true]
        Kred = K[prescribed_dof_false][:, prescribed_dof_false]

        Fred = F[prescribed_dof_false] - Kft.dot(prescribed_disp[prescribed_dof_true])

        U[prescribed_dof_false] = jit_solve(Kred, Fred)
    else:
        U = jit_solve(K, F)

    Q = K.dot(U) - F

    return U, Q


def mesh_to_element_length(model) -> np.ndarray:

    # initialize element properties
    elem_prop = model.element_properties

    # elememt coordinates along z and y-axes
    ez = np.array(
        [
            elem_prop["z_top [m]"].to_numpy(dtype=float),
            elem_prop["z_bottom [m]"].to_numpy(dtype=float),
        ]
    )
    ey = np.array(
        [
            elem_prop["y_top [m]"].to_numpy(dtype=float),
            elem_prop["y_bottom [m]"].to_numpy(dtype=float),
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


    # initialize element properties
    elem_prop = model.element_properties

    # elastic properties / cross-section properties
    E = parameter2elements(
        model.pile.sections,
        lambda x: x.material.young_modulus,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    I = parameter2elements(
        model.pile.sections,
        lambda x: x.second_moment_of_area,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    nu = parameter2elements(
        model.pile.sections,
        lambda x: x.material.poisson,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    A = parameter2elements(
        model.pile.sections,
        lambda x: x.area,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    G = E / (2 + 2 * nu)

    # calculate shear component in stiffness matrix (if Timorshenko)
    if model.element_type == "EulerBernoulli":
        kappa = 0
    elif model.element_type == "Timoshenko":
        if all([isinstance(section, CircularPileSection) for section in model.pile.sections]):
            # pile diameter (called width)
            d = parameter2elements(
                model.pile.sections,
                lambda x: x.width,
                elem_prop["z_top [m]"].values,
                elem_prop["z_bottom [m]"].values,
            ).reshape((-1, 1, 1))
            # wall thickness
            wt = parameter2elements(
                model.pile.sections,
                lambda x: x.thickness,
                elem_prop["z_top [m]"].values,
                elem_prop["z_bottom [m]"].values,
            ).reshape((-1, 1, 1))

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
            raise ValueError("Timoshenko beams cannot be used yet for non-circular pile types")

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
    ksoil = calculate_py_springs_stiffness(u=u[1::3], springs=model._py_springs, kind=kind)

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


def elem_tz_stiffness_matrix(model, u, kind):
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
        soil consistent stiffness matrix of all elememts related to the t-z soil springs' stiffness

    Raises
    ------
    ValueError
        ndof per node can be either 2 or 3

    """

    # calculate length vector
    L = mesh_to_element_length(model)

    # calculate the spring stiffness
    ksoil = calculate_tz_springs_stiffness(u=u[::3], springs=model._tz_springs, kind=kind)

    N = 0 * L
    A = L / 4
    B = L / 12

    ktop = (
        np.block(
            [
                [A, N, N, B, N, N],
                [N, N, N, N, N, N],
                [N, N, N, N, N, N],
                [B, N, N, B, N, N],
                [N, N, N, N, N, N],
                [N, N, N, N, N, N],
            ]
        )
        * ksoil[:, 0]
    )

    kbottom = (
        np.block(
            [
                [B, N, N, B, N, N],
                [N, N, N, N, N, N],
                [N, N, N, N, N, N],
                [B, N, N, A, N, N],
                [N, N, N, N, N, N],
                [N, N, N, N, N, N],
            ]
        )
        * ksoil[:, 1]
    )

    return ktop + kbottom


def elem_mt_stiffness_matrix(model, u, kind):
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

    # calculate the py spring stiffness
    kspy = calculate_py_springs_stiffness(u=u[1::3], springs=model._py_springs, kind="secant")
    upy = double_inner_njit(u[1::3]).reshape(-1, 2, 1, 1)
    p_mobilised = kspy * upy

    # calculate the spring stiffness
    ksoil = calculate_mt_springs_stiffness(
        u=u[2::3],
        mt_springs=model._mt_springs,
        py_springs=model._py_springs,
        p_mobilised=p_mobilised,
        kind=kind,
    )

    # initialize element properties
    elem_prop = model.element_properties

    # elastic properties / cross-section properties
    E = parameter2elements(
        model.pile.sections,
        lambda x: x.material.young_modulus,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    I = parameter2elements(
        model.pile.sections,
        lambda x: x.second_moment_of_area,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    nu = parameter2elements(
        model.pile.sections,
        lambda x: x.material.poisson,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    A = parameter2elements(
        model.pile.sections,
        lambda x: x.area,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    G = E / (2 + 2 * nu)

    # calculate shear component in stiffness matrix (if Timorshenko)
    if model.element_type == "EulerBernoulli":
        N = 0 * L
        A = 6 / (5 * L)
        B = N + 1 / 10
        C = 2 * L / 15
        D = -L / 30
    elif model.element_type == "Timoshenko":
        if all([isinstance(section, CircularPileSection) for section in model.pile.sections]):
            # pile diameter (called width)
            d = parameter2elements(
                model.pile.sections,
                lambda x: x.width,
                elem_prop["z_top [m]"].values,
                elem_prop["z_bottom [m]"].values,
            ).reshape((-1, 1, 1))
            # wall thickness
            wt = parameter2elements(
                model.pile.sections,
                lambda x: x.thickness,
                elem_prop["z_top [m]"].values,
                elem_prop["z_bottom [m]"].values,
            ).reshape((-1, 1, 1))

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

            omega = E * I * kappa / (A * G * L**2)

        else:
            raise ValueError("Timoshenko beams cannot be used yet for non-circular pile types")

        phi = (12 * omega + 1) ** 2

        N = 0 * L
        A = 6 * (120 * omega**2 + 20 * omega + 1) / ((5 * L) * phi) + 12 * I / (A * L**3 * phi)
        B = 1 / (10 * phi) + 6 * I / (A * L**2 * phi)
        C = (2 * L * (90 * omega**2 + 15 * omega + 1)) / (15 * phi) + 4 * I * (
            36 * omega**2 + 6 * omega + 1
        ) / (A * L * phi)
        D = -L * (360 * omega**2 + 60 * omega + 1) / (30 * phi) - 2 * I * (
            72 * omega**2 + 12 * omega - 1
        ) / (A * L * phi)

    else:
        raise ValueError(
            "Model.element.type only accepts 'EB' type (for Euler-Bernoulli) of 'T' type (for Timoshenko)"
        )

    km = np.block(
        [
            [N, N, N, N, N, N],
            [N, A, B, N, -A, B],
            [N, B, C, N, -B, D],
            [N, N, N, N, N, N],
            [N, -A, -B, N, A, -B],
            [N, B, D, N, -B, C],
        ]
    ) * (0.5 * ksoil[:, 0] + 0.5 * ksoil[:, 1])

    return km


def elem_p_delta_stiffness_matrix(model, u):
    """creates stress stiffness matrix based on axial stress in pile.

    #TODO: proof here

    Parameters
    ----------
    model : openpile class object `openpile.construct.Model`
        includes information on soil/structure, elements, nodes and other model-related data.
    u: np.ndarray
        Global displacement vector

    Returns
    -------
    k: numpy array (3d)
        stress stiffness matrix of all elememts related to axial stress

    Raises
    ------
    ValueError
        ndof per node can be either 2 or 3

    """

    # calculate length vector
    L = mesh_to_element_length(model)

    # internal forces
    f = pile_internal_forces(model, u)
    P = f[::6].reshape(-1, 1, 1)

    # initialize element properties
    elem_prop = model.element_properties

    # elastic properties
    nu = model.pile.material.poisson
    E = model.pile.material.young_modulus * np.ones(L.shape).reshape((-1, 1, 1))
    G = E / (2 + 2 * nu)
    # cross-section properties
    I = parameter2elements(
        model.pile.sections,
        lambda x: x.second_moment_of_area,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))
    A = parameter2elements(
        model.pile.sections,
        lambda x: x.area,
        elem_prop["z_top [m]"].values,
        elem_prop["z_bottom [m]"].values,
    ).reshape((-1, 1, 1))

    # calculate shear component in stiffness matrix (if Timorshenko)
    if model.element_type == "EulerBernoulli":
        N = 0 * L
        N1 = 0 * L
        N2 = 0 * L
        A = 6 / (5 * L)
        B = N + 1 / 10
        C = 2 * L / 15
        D = -L / 30
    elif model.element_type == "Timoshenko":
        if all([isinstance(section, CircularPileSection) for section in model.pile.sections]):
            # pile diameter (called width)
            d = parameter2elements(
                model.pile.sections,
                lambda x: x.width,
                elem_prop["z_top [m]"].values,
                elem_prop["z_bottom [m]"].values,
            ).reshape((-1, 1, 1))
            # wall thickness
            wt = parameter2elements(
                model.pile.sections,
                lambda x: x.thickness,
                elem_prop["z_top [m]"].values,
                elem_prop["z_bottom [m]"].values,
            ).reshape((-1, 1, 1))

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

            omega = E * I * kappa / (A * G * L**2)

        else:
            raise ValueError("Timoshenko beams cannot be used yet for non-circular pile types")

        phi = (12 * omega + 1) ** 2

        N = 0 * L
        N1 = 0 / L
        N2 = 0 / L
        A = 6 * (120 * omega**2 + 20 * omega + 1) / ((5 * L) * phi) + 12 * I / (A * L**3 * phi)
        B = 1 / (10 * phi) + 6 * I / (A * L**2 * phi)
        C = (2 * L * (90 * omega**2 + 15 * omega + 1)) / (15 * phi) + 4 * I * (
            36 * omega**2 + 6 * omega + 1
        ) / (A * L * phi)
        D = -L * (360 * omega**2 + 60 * omega + 1) / (30 * phi) - 2 * I * (
            72 * omega**2 + 12 * omega - 1
        ) / (A * L * phi)

    k = (
        np.block(
            [
                [N, N, N1, N, N, N2],
                [N, A, B, N, -A, B],
                [N1, B, C, -N1, -B, D],
                [N, N, -N1, N, N, -N2],
                [N, -A, -B, N, A, -B],
                [N2, B, D, -N2, -B, C],
            ]
        )
        * -P
    )

    return k


@dataclass(frozen=True)
class StiffnessPattern:
    """Constant CSR structure of the global stiffness matrix.

    It depends only on the mesh connectivity, not on stiffness values, so it can
    be computed once and reused at every iteration of the solver.
    """

    order: np.ndarray    # permutation of the element-matrix triplets matching (row, col) sort order
    group: np.ndarray    # CSR data slot each element-matrix entry contributes to
    indices: np.ndarray  # CSR column indices
    indptr: np.ndarray   # CSR row pointer


def stiffness_pattern(ndim, n_elem, node_per_element, ndof_per_node) -> StiffnessPattern:
    n_loc = ndof_per_node * node_per_element
    # global dof index of each local dof, shape (n_elem, n_loc)
    dofs = np.arange(n_elem)[:, None] * ndof_per_node + np.arange(n_loc)[None, :]
    rows = np.repeat(dofs, n_loc, axis=1).ravel()
    cols = np.tile(dofs, (1, n_loc)).ravel()

    # sort triplets into CSR layout and fold duplicate entries (shared nodes) into unique slots
    order = np.lexsort((cols, rows))
    rows_s, cols_s = rows[order], cols[order]
    new_slot = np.empty(rows_s.shape, dtype=bool)
    new_slot[0] = True
    new_slot[1:] = (rows_s[1:] != rows_s[:-1]) | (cols_s[1:] != cols_s[:-1])

    return StiffnessPattern(
        order=order,
        group=np.cumsum(new_slot) - 1,
        indices=cols_s[new_slot],
        indptr=np.concatenate(([0], np.cumsum(np.bincount(rows_s[new_slot], minlength=ndim)))),
    )


def jit_build(k, ndim, n_elem, node_per_element, ndof_per_node, pattern: StiffnessPattern = None):
    # assemble all element matrices into a CSR matrix in one vectorized scatter-add
    if pattern is None:
        pattern = stiffness_pattern(ndim, n_elem, node_per_element, ndof_per_node)

    data = np.bincount(
        pattern.group,
        weights=k.reshape(n_elem, -1).ravel()[pattern.order],
        minlength=pattern.indices.size,
    )
    return sparse.csr_matrix((data, pattern.indices, pattern.indptr), shape=(ndim, ndim))


def build_stiffness_matrix(model, n_elem, u=None, kind=None):
    """Builds the stiffness matrix based on the model(element and node) properties

    Element stiffness matrices are first computed for each element and then loaded in the global stiffness matrix through summation.
    Different element stiffness matrices are computed depending on the specifications found in the Model object.

    Parameters
    ----------
    model : obj from openpile library
        stores all information about nodes elements, and other model-related items, see #TODO:link to model class
    n_elem: int
        Number of elements in the mesh
    u: np.ndarray, Optional
        Global displacement vector. Must be given if soil_flag is not None
    p_mobilised: np.ndarray, Optional
        Mobilised p-value. Must be given if soil_flag is not None
    kind: str, Optional
        "initial", "secant" or "tangent". Must be given if soil_flag is not None

    Returns
    -------
    K : scipy.sparse.csr_matrix of dim (ndof, ndof)
        Global stiffness matrix with all dofs in unit:1/kPa.
    """

    # number of dof per node
    ndof_per_node = 3
    # number of nodes per element
    node_per_element = 2
    # global dimension of the global stiffness matrix can be inferred
    ndim_global = ndof_per_node * n_elem + ndof_per_node

    # mechanical stiffness properties
    k = elem_mechanical_stiffness_matrix(model)
    # k += elem_p_delta_stiffness_matrix(model, u)

    # add soil contribution
    if model.soil is not None:
        # gives warning if soil is given without displacements or type of stiffness
        if u is None or kind is None:
            UserWarning("'u' and 'kind' must be stipulated.")
        else:
            if model.distributed_lateral:
                k += elem_py_stiffness_matrix(model, u, kind)

            if model.distributed_axial:
                k += elem_tz_stiffness_matrix(model, u, kind)

            if model.distributed_moment:
                if not model.distributed_lateral:
                    raise UserInputError(
                        "Error: Distributed moment cannot be calculated without distributed lateral springs."
                    )
                k += elem_mt_stiffness_matrix(model, u, kind)

    # the CSR pattern only depends on the mesh connectivity, cache it on the model
    # so that repeated calls (Newton-Raphson iterations) skip the setup work
    pattern = getattr(model, "_stiffness_pattern", None)
    if pattern is None or pattern.indptr.size - 1 != ndim_global:
        pattern = stiffness_pattern(ndim_global, n_elem, node_per_element, ndof_per_node)
        model._stiffness_pattern = pattern

    K = jit_build(k, ndim_global, n_elem, node_per_element, ndof_per_node, pattern)

    # add base springs contribution
    if model.soil is not None:
        if model.base_shear:
            K[-2, -2] += calculate_base_spring_stiffness(u[-2], model._Hb_spring, kind)

        if model.base_moment:
            K[-1, -1] += calculate_base_spring_stiffness(u[-1], model._Mb_spring, kind)

        if model.base_axial:
            K[-3, -3] += calculate_base_axial_spring_stiffness(u[-3], model._qz_spring, kind)

    return K


def mesh_to_global_force_dof_vector(df: pd.DataFrame) -> np.ndarray:
    # extract each column (one line per node)
    force_dof_vector = df[["Pz [kN]", "Py [kN]", "Mx [kNm]"]].values.reshape(-1).astype(np.float64)

    return force_dof_vector


def mesh_to_global_disp_dof_vector(df: pd.DataFrame) -> np.ndarray:
    # extract each column (one line per node)
    disp_dof_vector = df[["Tz [m]", "Ty [m]", "Rx [rad]"]].values.reshape(-1).astype(np.float64)

    return disp_dof_vector


def mesh_to_global_restrained_dof_vector(df: pd.DataFrame) -> np.ndarray:
    # extract each column (one line per node)
    restrained_dof_vector = df[["Tz", "Ty", "Rx"]].values.reshape(-1)

    return restrained_dof_vector


def pile_internal_forces(model, u):
    # number of dof per node
    ndof_per_node = 3
    # number of nodes per element
    node_per_element = 2

    # create mech consistent stiffness matrix
    # (soil m-t springs act as a stress stiffness matrix and are therefore
    # not included in the internal force recovery)
    k = elem_mechanical_stiffness_matrix(model)

    # create array u of shape [n_elem x 6 x 1]
    u = global_dof_vector_to_consistent_stacked_array(u, ndof_per_node * node_per_element)
    # compute internal forces and reshape into global dof vector
    F_int = (-1) * np.matmul(k, u).reshape((-1))

    return F_int


def calculate_base_spring_stiffness(
    u: np.ndarray, spring: np.ndarray, kind: Literal["initial", "secant", "tangent"]
):
    """Calculate springs stiffness for py or t-z springs.

    Parameters
    ----------
    u : float
        base displacement or rotation to calculate stiffness
    spring : np.ndarray
        soil-structure interaction base springs array of shape (1, 1, 1, spring_dim)
    kind : str
        defines whether it is initial, secant of tangent stiffness to define

    Returns
    -------
    k: float
        secant or tangent stiffness for all elements.
    """

    # displacemet with same dimension as spring
    d = abs(u)

    if np.sum(spring[0, 0, 0]) == 0:
        k = 0.0
    else:
        if kind == "initial" or d == 0.0:
            dx = spring[0, 0, 1, 1] - spring[0, 0, 1, 0]
            p0 = spring[0, 0, 0, 0]
            p1 = spring[0, 0, 0, 1]
        elif kind == "secant":
            dx = d
            p0 = spring[0, 0, 0, 0]
            if d > np.max(spring[0, 0, 1]):
                p1 = spring[0, 0, 0, -1]
            else:
                p1 = np.interp(dx, spring[0, 0, 1], spring[0, 0, 0])
        elif kind == "tangent":
            dx = min(0.0005, d)
            if (d - dx) > np.max(spring[0, 0, 1]):
                p0 = spring[0, 0, 0, -1]
            else:
                p0 = np.interp(d - dx, spring[0, 0, 1], spring[0, 0, 0])
            if d > np.max(spring[0, 0, 1]):
                p1 = spring[0, 0, 0, -1]
            else:
                p1 = np.interp(d, spring[0, 0, 1], spring[0, 0, 0])

        k = abs((p1 - p0) / dx)

    return k


def calculate_base_axial_spring_stiffness(
    u: np.ndarray, spring: np.ndarray, kind: Literal["initial", "secant", "tangent"]
):
    """Calculate springs stiffness for base axial spring.

    Parameters
    ----------
    u : float
        base displacement or rotation to calculate stiffness
    spring : np.ndarray
        soil-structure interaction base springs array of shape (1, 1, 1, spring_dim)
    kind : str
        defines whether it is initial, secant of tangent stiffness to define

    Returns
    -------
    k: float
        secant or tangent stiffness for all elements.
    """

    # displacement with same dimension as spring
    d = u

    # determine place where 0 value of qz-spring is located
    # TODO: make this dynamic
    spring_0_index = round(spring.shape[-1] / 2) - 1

    if np.sum(np.abs(spring[0, 0, 0])) == 0:
        k = 0.0
    else:
        if kind == "initial" or d == 0.0:
            dx = spring[0, 0, 1, spring_0_index] - spring[0, 0, 1, spring_0_index + 1]
            p0 = spring[0, 0, 0, spring_0_index]
            p1 = spring[0, 0, 0, spring_0_index + 1]
        elif kind == "secant":
            dx = d
            p0 = spring[0, 0, 0, spring_0_index]
            p1 = np.interp(dx, spring[0, 0, 1][::-1], spring[0, 0, 0][::-1])
        elif kind == "tangent":
            dx = min(0.0005, abs(d))
            if d > 0:
                p0 = np.interp(d - dx, spring[0, 0, 1][::-1], spring[0, 0, 0][::-1])
            else:
                p0 = np.interp(d + dx, spring[0, 0, 1][::-1], spring[0, 0, 0][::-1])

            p1 = np.interp(d, spring[0, 0, 1][::-1], spring[0, 0, 0][::-1])

        k = abs((p1 - p0) / dx)

    return k


def calculate_py_springs_stiffness(
    u: np.ndarray, springs: np.ndarray, kind: Literal["initial", "secant", "tangent"]
):
    """Calculate springs stiffness for py springs.

    Parameters
    ----------
    u : np.ndarray
        displacements to calculate stiffness.
        For dofs related to t-z curves, u = U[::3] where U is the global displacement vector.
        For dofs related to p-y curves, u = U[1::3] where U is the global displacement vector.
    springs : np.ndarray
        soil-structure interaction py springs array of shape (n_elem, 2, 2, spring_dim)
    kind : str
        defines whether it is initial, secant of tangent stiffness to define

    Returns
    -------
    k: np.ndarray
        secant or tangent stiffness for all elements. Array of shape(n_elem,2,1,1)
    """

    # double inner values for u
    d = double_inner_njit(u)

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
                        p0 = np.interp(d[i, j, 0, 0] - dx, springs[i, j, 1], springs[i, j, 0])
                    if d[i, j, 0, 0] > np.max(springs[i, j, 1]):
                        p1 = springs[i, j, 0, -1]
                    else:
                        p1 = np.interp(d[i, j, 0, 0], springs[i, j, 1], springs[i, j, 0])

                k[i, j, 0, 0] = abs((p1 - p0) / dx)

    return k


def calculate_tz_springs_stiffness(
    u: np.ndarray, springs: np.ndarray, kind: Literal["initial", "secant", "tangent"]
):
    """Calculate springs stiffness for t-z springs.

    Parameters
    ----------
    u : np.ndarray
        displacements to calculate stiffness.
        For dofs related to t-z curves, u = U[::3] where U is the global displacement vector.
        For dofs related to p-y curves, u = U[1::3] where U is the global displacement vector.
    springs : np.ndarray
        soil-structure interaction py springs array of shape (n_elem, 2, 2, spring_dim)
    kind : str
        defines whether it is initial, secant of tangent stiffness to define

    Returns
    -------
    k: np.ndarray
        secant or tangent stiffness for all elements. Array of shape(n_elem,2,1,1)
    """

    # double inner values for u
    d = double_inner_njit(u)

    # displacement with same dimension as spring
    d = np.array(d).reshape((-1, 2, 1, 1))

    k = np.zeros(d.shape, dtype=np.float64)

    # determine place where 0 value of tz-spring is located
    spring_0_index = round(springs.shape[-1] / 2) - 1

    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if np.sum(np.abs(springs[i, j, 1])) == 0:
                pass
            else:
                if kind == "initial" or d[i, j, 0, 0] == 0.0:
                    dx = springs[i, j, 1, spring_0_index] - springs[i, j, 1, spring_0_index + 1]
                    p0 = springs[i, j, 0, spring_0_index]
                    p1 = springs[i, j, 0, spring_0_index + 1]
                elif kind == "secant":
                    dx = d[i, j, 0, 0]
                    p0 = springs[i, j, 0, spring_0_index]
                    p1 = np.interp(dx, springs[i, j, 1][::-1], springs[i, j, 0][::-1])
                elif kind == "tangent":
                    dx = min(0.0005, abs(d[i, j, 0, 0]))
                    if d[i, j, 0, 0] > 0:
                        p0 = np.interp(
                            d[i, j, 0, 0] - dx, springs[i, j, 1][::-1], springs[i, j, 0][::-1]
                        )
                    else:
                        p0 = np.interp(
                            d[i, j, 0, 0] + dx, springs[i, j, 1][::-1], springs[i, j, 0][::-1]
                        )

                    p1 = np.interp(d[i, j, 0, 0], springs[i, j, 1][::-1], springs[i, j, 0][::-1])

                k[i, j, 0, 0] = abs((p1 - p0) / dx)

    return k


def calculate_mt_springs_stiffness(
    u: np.ndarray,
    mt_springs: np.ndarray,
    py_springs: np.ndarray,
    p_mobilised: np.ndarray,
    kind: Literal["initial", "secant", "tangent"],
):
    """Calculate springs stiffness for rotational springs

    .. note::
        The difference with the py function is that rotational springs can depend on lateral springs

    Parameters
    ----------
    u : np.ndarray
        displacements to calculate stiffness.
        For dofs related to m-t curves, u = U[2::3] where U is the global displacement vector.
    mt_springs : np.ndarray
        soil-structure interaction m-t springs array of shape (n_elem, 2, 2, py_spring_dim, mt_spring_dim)
    py_springs : np.ndarray
        soil-structure interaction p-y springs array of shape (n_elem, 2, 2, py_spring_dim)
    p_mobilised : np.ndarray
        current p value of p-y springs of shape (nelem, 2, 1, 1)
    kind : str
        defines whether it is initial, secant of tangent stiffness to define

    Returns
    -------
    k: np.ndarray
        secant or tangent stiffness for all elements. Array of shape(n_elem,2,1,1)
    """

    # double inner values for u
    d = double_inner_njit(u)

    # displacemet and p_mobilised with same dimension as spring
    d = np.abs(d).reshape((-1, 2, 1, 1))
    p = p_mobilised.reshape((-1, 2, 1, 1))

    k = np.zeros(d.shape, dtype=np.float64)

    # get the proper m-t spring
    m = np.zeros(mt_springs.shape[4])
    t = np.zeros(mt_springs.shape[4])

    # i for each element
    for i in range(k.shape[0]):
        # j for top and bottom values of spring
        for j in range(k.shape[1]):
            if np.sum(mt_springs[i, j, 1, 0]) == 0:
                # check if first m-vector is not a defined spring
                # if that is the case, we do not calculate any stiffness
                pass
            else:

                for ii in range((mt_springs.shape[4])):
                    if ii == 0:
                        pass
                    else:
                        m[ii] = np.interp(
                            p[i, j, 0, 0], py_springs[i, j, 0, :], mt_springs[i, j, 0, :, ii]
                        )
                        t[ii] = np.interp(
                            p[i, j, 0, 0], py_springs[i, j, 0, :], mt_springs[i, j, 1, :, ii]
                        )

                if kind == "initial" or d[i, j, 0, 0] == 0.0:
                    dt = t[1] - t[0]
                    m0 = m[0]
                    m1 = m[1]
                elif kind == "secant":
                    dt = d[i, j, 0, 0]
                    m0 = m[0]
                    if d[i, j, 0, 0] > t[-1]:
                        m1 = m[-1]
                    else:
                        m1 = np.interp(dt, t, m)
                elif kind == "tangent":
                    dt = min(0.01 * t[1], d[i, j, 0, 0])
                    if (d[i, j, 0, 0] - dt) > t[-1]:
                        m0 = m[-1]
                    else:
                        m0 = np.interp(d[i, j, 0, 0] - dt, t, m)
                    if (d[i, j, 0, 0]) > t[-1]:
                        m1 = m[-1]
                    else:
                        m1 = np.interp(d[i, j, 0, 0], t, m)

                k[i, j, 0, 0] = abs(m1 - m0) / (dt)

    return k


def computer():
    """This function is the solver of openpile.

    The solver reads the boundary conditions, the global Force and Displacement vector and iterate to find convergence.
    """
