"""
`Materials` module
===================

This module can be used to create new materials for structure components, e.g. a Pile object.

Example
-------

.. doctest:

    >>> from openpile.construct import Pile, CircularPileSection
    >>> from openpile.materials import PileMaterial
    >>> 
    >>> # Create a Pile
    ... pile = Pile(
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
    name: str = Field(min_length=1, max_length=20)
    uw: float = Field(gt=0.0)
    E: float = Field(gt=0.0)
    nu: float = Field(gt=-1.0, le=0.5)

    @property
    def unitweight(self):
        return self.uw

    @property
    def young_modulus(self):
        return self.E

    @property
    def poisson(self):
        return self.nu

    @property
    def shear_modulus(self):
        return self.young_modulus / (2 + 2 * self.poisson)

    @classmethod
    def custom(
        cls, unitweight: float, young_modulus: float, poisson_ratio: float, name: str = "Custom"
    ):
        return cls(name=name, uw=unitweight, E=young_modulus, nu=poisson_ratio)
    

steel = PileMaterial(name="Steel", uw=78.0, E=210e6, nu=0.3)
concrete = PileMaterial(name="Concrete", uw=24.0, E=30e6, nu=0.2)