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


from abc import ABC, abstractproperty
from typing import List, Dict, Optional, Union
from typing_extensions import Literal, Annotated, Optional
from pydantic import BaseModel, AfterValidator, ConfigDict, Field, model_validator

class AbstractPileMaterial(BaseModel, ABC):
    model_config = ConfigDict(extra='forbid')

    @abstractproperty
    def unitweight(self):
        pass

    @abstractproperty
    def young(self):
        pass

    @abstractproperty
    def poisson(self):
        pass

    @abstractproperty
    def shear_modulus(self):
        return self.young / (2 + 2 * self.poisson)

class PileMaterial(AbstractPileMaterial):
    uw: float = Field(gt=0.0)
    E: float = Field(gt=0.0)
    nu: float = Field(gt=-1.0, le=0.5)

    @property
    def unitweight(self):
        return self.uw
    
    @property
    def young(self):
        return self.E
    
    @property
    def poisson(self):
        return self.nu
    
    @classmethod
    def custom(cls, unitweight:float,  young_modulus:float, poisson_ratio:float):
        return cls(uw=unitweight,  E=young_modulus, nu=poisson_ratio)

    @classmethod
    def steel(cls):
        return cls(uw=78.0,  E=210e6, nu=0.3)

    @classmethod
    def concrete(cls):
        return cls(uw=24.0,  E=30e6, nu=0.2)