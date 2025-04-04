"""
`Materials` module
===================

This module can be used to create new materials for structure components, e.g. a Pile object.

Example
-------

.. doctest::

    >>> from openpile.construct import Pile, CircularPileSection
    >>> from openpile.materials import PileMaterial

    >>> # Create a Pile
    >>> pile = Pile(
    ...     name = "",
    ...     material=PileMaterial.custom(
    ...         name="concrete",unitweight=25, young_modulus=30e6, poisson_ratio=0.15
    ...         ),
    ...     sections=[
    ...         CircularPileSection(
    ...             top=0, 
    ...             bottom=-10, 
    ...             diameter=1.0, 
    ...             thickness=0.05
    ...         ),
    ...     ]
    ... )

    >>> pile.weight
    37.30641276137878



"""


# Import libraries

from pydantic import (
    BaseModel,
    Field,
)

from abc import ABC
from pydantic import BaseModel, ConfigDict, Field


class AbstractPileMaterial(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")


class PileMaterial(AbstractPileMaterial):
    """A class to define the material of a pile.
    This class is used to define the material properties of a pile, including
    unit weight, Young's modulus, and Poisson's ratio. The class also provides
    methods to calculate the shear modulus and to create custom materials.

    Parameters
    ----------
    name : str
        The name of the material.
    uw : float
        The unit weight of the material in kN/m³.
    E : float
        The Young's modulus of the material in kN/m².
    nu : float
        The Poisson's ratio of the material. Must be between -1 and 0.5.
    """

    #: name of the material
    name: str = Field(min_length=1, max_length=20)
    #: unit weight [kN/m³]
    uw: float = Field(gt=0.0)
    #: Young's modulus [kN/m²]
    E: float = Field(gt=0.0)
    #: Poisson's ratio [-]
    nu: float = Field(gt=-1.0, le=0.5)

    @property
    def unitweight(self):
        """The unit weight of the material in kN/m³."""
        return self.uw

    @property
    def young_modulus(self):
        """The Young's modulus of the material in kN/m²."""
        return self.E

    @property
    def poisson(self):
        """The Poisson's ratio of the material. Must be between -1 and 0.5."""
        return self.nu

    @property
    def shear_modulus(self):
        """The shear modulus of the material in kN/m². Calculated from Young's modulus and Poisson's ratio."""
        return self.young_modulus / (2 + 2 * self.poisson)

    @classmethod
    def custom(
        cls, unitweight: float, young_modulus: float, poisson_ratio: float, name: str = "Custom"
    ):
        """a redundant constructor to create a custom material with the given parameters provided.

        Parameters
        ----------
        unitweight : float
            The unit weight of the material in kN/m³.
        young_modulus : float
            The Young's modulus of the material in kN/m².
        poisson_ratio : float
            The Poisson's ratio of the material. Must be between -1 and 0.5.
        name : str, optional
            the name of the material, by default "Custom"

        Returns
        -------
        openpile.materials.PileMaterial
        """
        return cls(name=name, uw=unitweight, E=young_modulus, nu=poisson_ratio)


steel = PileMaterial(name="Steel", uw=78.0, E=210e6, nu=0.3)
concrete = PileMaterial(name="Concrete", uw=24.0, E=30e6, nu=0.2)
