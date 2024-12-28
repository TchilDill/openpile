"""
`Materials` module
===================

"""


# Import libraries

from typing import List, Dict, Optional, Union, Callable, Tuple, ClassVar
from typing_extensions import Literal
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    confloat,
    conlist,
)
from pydantic.dataclasses import dataclass


from abc import ABC
from typing import List, Dict, Optional, Union
from typing_extensions import Literal, Annotated, Optional
from pydantic import BaseModel, AfterValidator, ConfigDict, Field, model_validator


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

    @classmethod
    def steel(cls):
        return cls(name="Steel", uw=78.0, E=210e6, nu=0.3)

    @classmethod
    def concrete(cls):
        return cls(name="Concrete", uw=24.0, E=30e6, nu=0.2)
