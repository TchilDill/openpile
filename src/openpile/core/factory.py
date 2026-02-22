

# Import libraries
import math as m
import numpy as np
import pandas as pd

from typing import List, Dict, Optional, Union, Callable, Tuple, ClassVar
from typing_extensions import Literal
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    confloat,
    conlist,
)

from openpile.core.misc import from_list2x_parse_top_bottom, var_to_str, get_value_at_current_depth
from openpile.utils import py_curves, Hb_curves, mt_curves, Mb_curves, tz_curves, qz_curves
from openpile.utils.misc import _fmax_api_sand, _fmax_api_clay, _Qmax_api_clay, _Qmax_api_sand
from openpile.utils.hooks import InitialSubgradeReaction

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from typing_extensions import Literal, Annotated, Optional
from pydantic import BaseModel, BeforeValidator, PlainSerializer, ConfigDict, Field, model_validator


# ND ARRAY TYPE DEFINITION ------------------------------------
def nd_array_custom_before_validator(x):
    # custom before validation logic
    return x

def nd_array_custom_serializer(x):
    # custom serialization logic
    return str(x)

NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]


# SPRING CLASSES ----------------------------------------------

class FancySpring(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    X: Union[NdArray, List[float]]         # Return the x-axis values for the spring
    Y: Union[NdArray, List[float]]         # Return the y-axis values for the spring

    signature = "NonLinearElasticSpring"                         # info on spring

class ConstantSpring(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        
    )
    k: Union[NdArray, List[float]]         # stiffness


class Node(BaseModel, ABC):

    @property
    @abstractmethod
    def coordinates(self) -> tuple[float, float, float]:
        # coordinates are x, y, z 
        # z is the elevation upward positive, 
        # y is lateral leftward positive, 
        # x is out of plane towards us being positive
        pass

    @property
    @abstractmethod
    def number(self) -> str:
        pass
