"""

"""

# Import libraries
import math as m
import numpy as np
import pandas as pd
from numba import njit, prange

from typing import List, Dict, Optional, Union
from typing_extensions import Literal
from pydantic import (
    BaseModel,
    Field,
    root_validator,
    PositiveFloat,
    confloat,
    conlist,
    Extra,
)
from pydantic.dataclasses import dataclass

from openpile.core.misc import from_list2x_parse_top_bottom, var_to_str
from openpile.utils import py_curves


# CONSTITUTIVE MODELS CLASSES ---------------------------------


class PydanticConfigFrozen:
    arbitrary_types_allowed = True
    allow_mutation = False
    extra = Extra.forbid


class ConstitutiveModel:
    pass


class LateralModel(ConstitutiveModel):
    pass


class AxialModel(ConstitutiveModel):
    pass


@dataclass(config=PydanticConfigFrozen)
class API_sand(LateralModel):
    """A class to establish the API sand model.

    Parameters
    ----------
    phi: float
        internal angle of friction in degrees
    Neq: float
        number of equivalent cycles, (Neq=1 means Static, Neq>1 means Cyclic)
    p_multiplier: float
        multiplier for p-values
    y_multiplier: float
        multiplier for y-values

    """

    #: soil friction angle [deg], if a variation in values, two values can be given.
    phi: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: Number of equivalent cycles of the curve. 1 = static curve, >1 = cyclic curve.
    Neq: confloat(ge=1.0, le=100.0)
    #: p-multiplier
    p_multiplier: confloat(ge=0.0) = 1.0
    #: y-multiplier
    y_multiplier: confloat(gt=0.0) = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature = np.array([True, False, False, False], dtype=bool)

    def __str__(self):
        if self.Neq == 1:
            Neq = "Static, N < 1 cycle"
        else:
            Neq = "Cyclic, N = 100 cycles"

        return f"\tAPI sand\n\tphi = {var_to_str(self.phi)}°\n\t{Neq}"

    def py_spring_fct(
        self,
        sig: float,
        X: float,
        layer_height: float,
        depth_from_top_of_layer: float,
        D: float,
        L: float = None,
        below_water_table: bool = True,
        ymax: float = 0.0,
        output_length: int = 15,
    ):
        # validation
        if depth_from_top_of_layer > layer_height:
            raise ValueError("Spring elevation outside layer")

        # define phi
        phi_t, phi_b = from_list2x_parse_top_bottom(self.phi)
        phi = phi_t + (phi_b - phi_t) * depth_from_top_of_layer / layer_height

        p, y = py_curves.api_sand(
            sig=sig,
            X=X,
            phi=phi,
            D=D,
            Neq=self.Neq,
            below_water_table=below_water_table,
            ymax=ymax,
            output_length=output_length,
        )

        return p * self.p_multiplier, y * self.y_multiplier


@dataclass(config=PydanticConfigFrozen)
class API_clay(LateralModel):
    """A class to establish the API clay model.

    Parameters
    ----------
    Su: float
        Undrained shear strength in kPa
    eps50: float
        strain at 50% failure load [-]
    J: float
        empirical factor varying depending on clay stiffness, varies between 0.25 and 0.50
    stiff_clay_threshold: float
        undrained shear strength [kPa] at which stiff clay curve is computed
    Neq: float
        number of equivalent cycles, (Neq=1 means Static, Neq>1 means Cyclic)
    p_multiplier: float
        multiplier for p-values
    y_multiplier: float
        multiplier for y-values

    """

    #: undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: strain at 50% failure load [-], if a variation in values, two values can be given.
    eps50: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: Number of equivalent cycles of the curve. 1 = static curve, >1 = cyclic curve.
    Neq: confloat(ge=1.0, le=100.0)
    #: empirical factor varying depending on clay stiffness
    J: confloat(ge=0.25, le=0.5) = 0.5
    #: undrained shear strength [kPa] at which stiff clay curve is computed
    stiff_clay_threshold: PositiveFloat = 96
    #: p-multiplier
    p_multiplier: confloat(ge=0.0) = 1.0
    #: y-multiplier
    y_multiplier: confloat(gt=0.0) = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature = np.array([True, False, False, False], dtype=bool)

    def __str__(self):
        if self.Neq == 1:
            Neq = "Static, N < 1 cycle"
        else:
            Neq = "Cyclic, N = 100 cycles"

        return f"\tAPI clay\n\tSu = {var_to_str(self.Su)} kPa\n\teps50 = {var_to_str(self.eps50)}\n\t{Neq}"

    def py_spring_fct(
        self,
        sig: float,
        X: float,
        layer_height: float,
        depth_from_top_of_layer: float,
        D: float,
        L: float = None,
        below_water_table: bool = True,
        ymax: float = 0.0,
        output_length: int = 15,
    ):
        # validation
        if depth_from_top_of_layer > layer_height:
            raise ValueError("Spring elevation outside layer")

        # define Su
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height

        # define eps50
        eps50_t, eps50_b = from_list2x_parse_top_bottom(self.eps50)
        eps50 = eps50_t + (eps50_b - eps50_t) * depth_from_top_of_layer / layer_height

        p, y = py_curves.api_clay(
            sig=sig,
            X=X,
            Su=Su,
            eps50=eps50,
            D=D,
            J=self.J,
            stiff_clay_threshold=self.stiff_clay_threshold,
            Neq=self.Neq,
            ymax=ymax,
            output_length=output_length,
        )

        return p * self.p_multiplier, y * self.y_multiplier
