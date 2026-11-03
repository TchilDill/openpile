import math as m
from operator import le
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from openpile.soilmodels import LateralModel, AxialModel
from openpile.construct import Pile

from typing import Callable, List, Dict, Optional, Union
from typing_extensions import Literal, Annotated, Optional
from pydantic import (
    BaseModel,
    Field,
)

from scipy.sparse import coo_matrix

#to be later pushed into factory.py
class Element(BaseModel):
    """Base class for all elements in the model. They always have two nodes.

    Typically we are dealine with 1D element, so this class is the parent class of the BarElement and BeamElement classes. The Element class is not meant to be instantiated directly, but rather to be subclassed by specific element types.
    """
    z: list[float] = Field(..., description="Z coordinate of the element's first node.")
    E: float = Field(..., description="Young's modulus of the element material.")
    x: list[float] = Field(default_factory=lambda: [0, 0], description="X coordinate of the element's first node.")
    y: list[float] = Field(default_factory=lambda: [0, 0], description="Y coordinate of the element's first node.")
    rho: float = Field(default_factory=lambda: 0, description="Density of the element material.")

    @property
    def length(self) -> float:
        """Calculate the length of the element based on its node coordinates."""
        return m.sqrt((self.x[1] - self.x[0]) ** 2 + (self.y[1] - self.y[0]) ** 2 + (self.z[1] - self.z[0]) ** 2)


class BarElement(Element):
    """Class representing a bar element in the model.

    Inherits from the Element class and adds properties specific to bar elements, such as cross-sectional area.
    """
    A: float = Field(..., description="Cross-sectional area of the bar element.")

    @property
    def consistent_stiffness_matrix(self) -> np.ndarray:
        """Calculate the consistent stiffness matrix for the bar element.

        Returns
        -------
        np.ndarray
            The consistent stiffness matrix of the bar element.
        """
        L = self.length
        k = (self.E * self.A) / L
        return k * np.array([[1, -1], [-1, 1]])

    @property
    def consistent_mass_matrix(self) -> np.ndarray:
        """Calculate the consistent mass matrix for the bar element.

        Returns
        -------
        np.ndarray
            The consistent mass matrix of the bar element.
        """
        L = self.length
        m = (self.rho * self.A * L) / 6
        return m * np.array([[2, 1], [1, 2]])

    @property
    def lumped_mass_matrix(self) -> np.ndarray:
        """Calculate the lumped mass matrix for the bar element.

        Returns
        -------
        np.ndarray
            The lumped mass matrix of the bar element.
        """
        L = self.length
        m = (self.rho * self.A * L) / 2
        return m * np.array([[1, 0], [0, 1]])

    def get_soil_spring_matrix(self, ksoil) -> np.ndarray:
        """Calculate the soil spring matrix for the bar element.

        Parameters
        ----------
        ksoil : float
            Soil spring stiffness per unit length.

        Returns
        -------
        np.ndarray
            The soil spring matrix of the bar element.
        """
        L = self.length
        A = L/3
        B = L/6
        return ksoil * np.array([[A, B], [B, A]])


    def get_soil_damping_matrix(self, csoil) -> np.ndarray:
        """Calculate the soil damping matrix for the bar element.

        Parameters
        ----------
        csoil : float
            Soil damping coefficient per unit length.

        Returns
        -------
        np.ndarray
            The soil damping matrix of the bar element.
        """
        L = self.length
        A = L/3
        B = L/6
        return csoil * np.array([[A, B], [B, A]])


def _section_lookup(pile: Pile, elevations: np.ndarray, key: Callable) -> np.ndarray:
    """Look up a piecewise-constant pile-section property at given elevations.

    Parameters
    ----------
    pile : Pile
        The pile whose sections define the property steps.
    elevations : np.ndarray
        Elevations at which to evaluate the property.
    key : callable
        Function mapping a pile section to the property value, e.g. ``lambda s: s.area``.

    Returns
    -------
    np.ndarray
        Property values at the given elevations, same shape as ``elevations``.
    """
    sections = sorted(pile.sections, key=lambda s: s.bottom)
    bottoms = np.array([s.bottom for s in sections])
    values = np.array([key(s) for s in sections])
    idx = np.clip(np.searchsorted(bottoms, elevations, side="right") - 1, 0, len(sections) - 1)
    return values[idx]


class WeapMesh(BaseModel):
    """Class representing a mesh of elements in the model.

    Contains a list of elements and provides methods to manipulate and analyze the mesh.
    """
    elements: List[Element] = Field(..., description="List of elements in the mesh.")

    @property
    def num_elements(self) -> int:
        """Get the number of elements in the mesh.

        Returns
        -------
        int
            The number of elements in the mesh.
        """
        return len(self.elements)

    def add_element(self, element: Element) -> None:
        """Add an element to the mesh.

        Parameters
        ----------
        element : Element
            The element to be added to the mesh.
        """
        self.elements.append(element)

    def get_total_length(self) -> float:
        """Calculate the total length of all elements in the mesh.

        Returns
        -------
        float
            The total length of all elements in the mesh.
        """
        return sum(element.length for element in self.elements)

    def get_critical_time_step(self, safety_factor: float = 1.6) -> float:
        """Calculate the critical time step for the mesh based on the smallest element length and material properties.

        Parameters
        ----------
        safety_factor : float, optional
            A safety factor to apply to the critical time step calculation (default is 1.6 like in GRLWEAP).  
        
        Returns
        -------
        float
            The critical time step for the mesh.
        """
        return 1/safety_factor * min(element.length/np.sqrt(element.E*1e3 / element.rho) for element in self.elements)

    def get_stiffness_array_of_elements(self) -> np.ndarray:
        """Collect the consistent stiffness matrix of each element in the mesh.

        Every element contributes a 2x2 stiffness matrix relating the axial
        displacements of its two nodes. The matrices are stacked along a new
        leading axis, one per element, in the order of ``self.elements``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_elements, 2, 2)`` containing the
            consistent stiffness matrix of each element, in mesh order.
            With ``E`` in kPa, ``A`` in m^2 and length in m, the entries
            are in kN/m.
        """
        nelem = self.num_elements

        # assemble 2*2*nelem pile stiffness array
        Kp = np.zeros((nelem, 2, 2))
        for i, elem in enumerate(self.elements):
            Kp[i] = elem.consistent_stiffness_matrix

        return Kp
    

    def get_stiffness_matrix(self) -> np.ndarray:
        """Assemble the global stiffness matrix of the mesh.

        The 2x2 consistent stiffness matrix of each element is scattered
        into a global sparse matrix: element ``i`` connects global nodes
        ``i`` and ``i + 1``, so the matrix is symmetric and tridiagonal.
        Entries are scaled by 1e3 to convert from kN/m to N/m 
        as it is used in the equation of motion.

        Returns
        -------
        np.ndarray
            Global stiffness matrix of shape ``(num_nodes, num_nodes)``
            with ``num_nodes = num_elements + 1``, in N/m. Although the
            annotation is ``np.ndarray``, a ``scipy.sparse.csr_matrix``
            is actually returned.

        See Also
        --------
        get_stiffness_array_of_elements : Per-element stiffness matrices
            used to build this global matrix.
        """
        nnodes = self.num_elements + 1
        nelem = self.num_elements

        Kp = self.get_stiffness_array_of_elements()

        # assemble sparse Kp COO matrix
        row = np.zeros(4*nelem, dtype=int)
        col = np.zeros(4*nelem, dtype=int)
        for i in range(nelem):
            row[4*i:4*i+4] = [i, i, i+1, i+1]
            col[4*i:4*i+4] = [i, i+1, i, i+1]
        data = Kp.reshape(-1) * 1e3

        return coo_matrix((data, (row, col)), shape=(nnodes, nnodes)).tocsr()

    def get_lumped_mass_diagonal_vector(self) -> np.ndarray:
        """Assemble the diagonal of the lumped mass matrix of the mesh.

        Each element's 2x2 lumped mass matrix contributes half of the
        element mass to each of its two nodes. Since the lumped mass
        matrix is diagonal, only the nodal masses are assembled: element
        ``i`` adds its mass to global nodes ``i`` and ``i + 1``.

        Returns
        -------
        np.ndarray
            Nodal lumped masses of shape ``(num_nodes,)`` with
            ``num_nodes = num_elements + 1``, in kg (with ``rho`` in
            kg/m^3, ``A`` in m^2 and length in m).

        See Also
        --------
        BarElement.lumped_mass_matrix : Per-element lumped mass matrix
            whose diagonal entries are accumulated here.
        """
        nnodes = self.num_elements + 1

        # assemble the nodal masses directly, without building the full matrix
        m_diag = np.zeros(nnodes)
        for i, elem in enumerate(self.elements):
            m_diag[i:i+2] += elem.lumped_mass_matrix.diagonal()

        return m_diag


    @classmethod
    def from_pile(
        cls, 
        pile: Pile, 
        max_elem_length: float,
        water_elevation: Optional[float] = None,
    ) -> "WeapMesh":
        """Create a WeapMesh instance from a Pile instance.

        Parameters
        ----------
        pile : Pile
            The Pile instance from which to create the WeapMesh.
        max_elem_length : float
            The maximum desired length of each element in the mesh.
        water_elevation : float, optional
            The elevation of the water table, if applicable. 
            This parameter is used to determine the effective unit weight of the pile material in submerged conditions.

        Returns
        -------
        WeapMesh
            A mesh of bar elements discretizing the pile.
        """

        # node elevations: pile top/bottom and every section boundary are nodes,
        # and each segment between consecutive boundaries is subdivided into
        # equally sized elements, so no element crosses a section boundary
        z_bounds = np.unique(
            np.concatenate(
                [[pile.top_elevation, pile.bottom_elevation]]
                + [[s.top, s.bottom] for s in pile.sections]
            )
        )[::-1]  # descending, from pile top to pile bottom
        z_nodes = [z_bounds[0]]
        for zt, zb in zip(z_bounds[:-1], z_bounds[1:]):
            n_seg = m.ceil((zt - zb) / max_elem_length)
            z_nodes.extend(zt - (zt - zb) / n_seg * np.arange(1, n_seg + 1))
        z_nodes = np.array(z_nodes)
        z_top, z_bot = z_nodes[:-1], z_nodes[1:]

        # look up E, A and density at the element midpoints: since section
        # boundaries are nodes, each element lies fully inside one section
        # and gets that section's exact properties (no smearing at boundaries)
        z_mid = 0.5 * (z_top + z_bot)
        E = _section_lookup(pile, z_mid, lambda s: s.material.young_modulus)
        A = _section_lookup(pile, z_mid, lambda s: s.area)
        rho = _section_lookup(pile, z_mid, lambda s: s.material.unitmass)
        if water_elevation is not None:
            # adjust density for submerged elements
            rho = np.where(z_mid < water_elevation, rho - 1000, rho)

        elements = [
            BarElement(z=[zt, zb], E=e, A=ar, rho=r)
            for zt, zb, e, ar, r in zip(z_top, z_bot, E, A, rho)
        ]
        return cls(elements=elements)


def run_explicit_weap_analysis_with_top_excitation(
    mesh: WeapMesh,
    acc: np.ndarray,
    t_end: float,
):
    """Run an explicit WEAP analysis with a given excitation (acceleration time series).

    Parameters
    ----------
    mesh : WeapMesh
        The mesh to be used for the analysis.
    acc : np.ndarray
        An array containing the acceleration time in first column and acceleration in 2nd column for the analysis.
    t_end : float
        The end time of the analysis.

    """

    # gravitational acceleration in m/s²
    g = -9.81

    dt = mesh.get_critical_time_step()
    # number of time steps, 
    # +1 for t=0s and t=t_end, 
    ntimesteps = int(t_end / dt) + 1
    # create the time vector
    time = np.linspace(0, t_end, ntimesteps)

    # transform acceleration time series so that it has the same timestep and t_end
    acc_t = acc[:,0]
    acc_a = acc[:,1]
    input_acc = np.interp(time, acc_t, acc_a, left=0, right=0 )

    # number of nodes
    nnodes = mesh.num_elements + 1
    # number of elements
    nelem = mesh.num_elements

    # get the stiffness matrix of the mesh
    K = mesh.get_stiffness_matrix()

    # get the lumped mass diagonal vector and its inverse for practical use in the solver
    Mv = mesh.get_lumped_mass_diagonal_vector()
    Mv_inv = 1/Mv 

    # initialize the initial vectors for the central difference method
    u_current = np.zeros(nnodes)  
    u_next = np.zeros(nnodes)  # initialize the current displacement vector 
    v_next = np.zeros(nnodes)  # initialize the current velocity vector


    top_disp = []
    top_vel = []
    top_acc = []

    for i, _ in enumerate(time):

        # update vectors for the current time step
        u_prev = u_current.copy()  # store the current displacement vector as the previous one for the next time step
        u_current = u_next.copy()  # store the current velocity vector as the previous one for the next time step
        v_current = v_next.copy()  # store the current velocity vector as the previous one for the next time step
        
        # we know the acceleration input at that time step already
        R_ext = np.zeros(nnodes)
        #TODO include the effect of gravity with R_ext += Mv * g  # apply the gravity at all nodes

        # include the effect of the input acceleration at the top node
        R_ext[0] = input_acc[i] * Mv[0]  # apply the acceleration at the top node
        # we also know the current displacment vector u, which is the solution from the previous time step
        R_int = K @ u_current

        # compute the acceleration at the current time step
        a_current = Mv_inv * (R_ext - R_int)  

        # compute the displacement at the next time step using the central difference method
        u_next = dt**2 * a_current + 2*u_current - u_prev
        v_next = (u_next - u_prev) / (2*dt)  

        # record top node disp, vel, acc for post-processing
        top_disp.append(u_next[0])
        top_vel.append(v_current[0])
        top_acc.append(a_current[0])

    return time, np.array(top_disp), np.array(top_vel), np.array(top_acc)