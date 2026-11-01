import math as m
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics

from openpile.materials import PileMaterial
import openpile.materials as materials
from openpile.core import misc, _model_build
from openpile.soilmodels import LateralModel, AxialModel
from openpile.core.misc import generate_color_string
from openpile.core._model_build import (
    check_springs,
    get_soil_properties,
    apply_bc,
    validate_bc,
    parameter2elements,
)

from openpile.construct import Pile

from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Optional, Union
from typing_extensions import Literal, Annotated, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    InstanceOf,
    Field,
    model_validator,
    computed_field,
    AfterValidator
)
from functools import cached_property

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

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


class BeamElement(Element):
    """Class representing a beam element in the model.

    Inherits from the Element class and adds properties specific to beam elements, such as moment of inertia.
    """
    I: float = Field(..., description="Moment of inertia of the beam element.")


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

    @classmethod
    def from_pile(cls, pile: Pile, max_elem_length: float) -> "WeapMesh":
        """Create a WeapMesh instance from a Pile instance.

        Parameters
        ----------
        pile : Pile
            The Pile instance from which to create the WeapMesh.
        max_elem_length : float
            The maximum desired length of each element in the mesh.

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

        elements = [
            BarElement(z=[zt, zb], E=e, A=ar, rho=r)
            for zt, zb, e, ar, r in zip(z_top, z_bot, E, A, rho)
        ]
        return cls(elements=elements)