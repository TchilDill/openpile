

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


class Node(BaseModel):
    number: int
    x: float
    y: float
    z: float

class BeamElement(BaseModel):
    number: int
    nodes: Annotated[List[Node],conlist(min_length=2, max_length=2)]

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @property
    def length(self) -> float:
        # calculate length of beam element from the coordinates of the nodes
        return m.sqrt(
            (self.nodes[1].x - self.nodes[0].x)**2 + 
            (self.nodes[1].y - self.nodes[0].y)**2 + 
            (self.nodes[1].z - self.nodes[0].z)**2
            )


