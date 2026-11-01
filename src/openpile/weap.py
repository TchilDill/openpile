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
from typing import List, Dict, Optional, Union
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
            A mesh with one bar element per pile section.
        """

        #check the pile length and get the number of elements needed
        pile_length = pile.length
        num_elements = m.ceil(pile_length / max_elem_length)
        element_length = pile_length / num_elements

        # interpolate E and A along the pile for specific elevation for the number of elements needed
        # we want to consider both values at the top and bottom of elements for that
        

        elements = [
            BarElement(
                z=[section.top, section.bottom],
                E=section.material.young_modulus,
                A=section.cross_sectional_area,
            )
            for section in pile.sections
        ]
        return cls(elements=elements)