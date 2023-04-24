"""
`SoilModels` module
===================
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
from openpile.utils import py_curves, Hb_curves, mt_curves, Mb_curves


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
class API_clay(AxialModel):
    #: undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: t-multiplier
    t_multiplier: confloat(ge=0.0) = 1.0
    #: z-multiplier
    z_multiplier: confloat(gt=0.0) = 1.0
    #: Q-multiplier
    Q_multiplier: confloat(ge=0.0) = 1.0
    #: w-multiplier
    w_multiplier: confloat(gt=0.0) = 1.0

    def __str__(self):
        return f"\tAPI clay\n\tSu = {var_to_str(self.Su)} kPa"


@dataclass(config=PydanticConfigFrozen)
class Cowden_clay(LateralModel):
    """A class to establish the PISA Cowden clay model.

    Parameters
    ----------
    Su: float or list[top_value, bottom_value]
        Undrained shear strength. Value to range from 0 to 100 [unit: kPa]
    G0: float or list[top_value, bottom_value]
        Small-strain shear modulus [unit: kPa]
    p_multiplier: float
        multiplier for p-values
    y_multiplier: float
        multiplier for y-values
    m_multiplier: float
        multiplier for m-values
    t_multiplier: float
        multiplier for t-values
    """

    #: Undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: p-multiplier
    p_multiplier: confloat(ge=0.0) = 1.0
    #: y-multiplier
    y_multiplier: confloat(gt=0.0) = 1.0
    #: m-multiplier
    m_multiplier: confloat(ge=0.0) = 1.0
    #: t-multiplier
    t_multiplier: confloat(gt=0.0) = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature = np.array([True, True, True, True], dtype=bool)

    def __str__(self):
        return f"\Cowden clay (PISA)\n\tSu = {var_to_str(self.Su)} kPa.\n\tG0 = {round(self.G0/1000,1)} MPa"

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

        # define Dr
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        p, y = py_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            output_length=output_length,
        )

        return p * self.p_multiplier, y * self.y_multiplier

    def Hb_spring_fct(
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

        # define Dr
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        Hb, y = Hb_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return Hb, y

    def mt_spring_fct(
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

        # define Dr
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        p_array, _ = py_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            output_length=output_length,
        )

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        for count, _ in enumerate(p_array):
            m[count, :], t[count, :] = mt_curves.cowden_clay(
                X=X,
                Su=Su,
                G0=Gmax,
                D=D,
                output_length=output_length,
            )

        return m * self.m_multiplier, t * self.t_multiplier

    def Mb_spring_fct(
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

        # define Dr
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        Mb, y = Mb_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return Mb, y


@dataclass(config=PydanticConfigFrozen)
class Dunkirk_sand(LateralModel):
    """A class to establish the PISA Dunkirk sand model.

    Parameters
    ----------
    Dr: float or list[top_value, bottom_value]
        relative density of sand. Value to range from 0 to 100. [unit: -]
    G0: float or list[top_value, bottom_value]
        Small-strain shear modulus [unit: kPa]
    p_multiplier: float
        multiplier for p-values
    y_multiplier: float
        multiplier for y-values
    m_multiplier: float
        multiplier for m-values
    t_multiplier: float
        multiplier for t-values
    """

    #: soil friction angle [deg], if a variation in values, two values can be given.
    Dr: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: p-multiplier
    p_multiplier: confloat(ge=0.0) = 1.0
    #: y-multiplier
    y_multiplier: confloat(gt=0.0) = 1.0
    #: m-multiplier
    m_multiplier: confloat(ge=0.0) = 1.0
    #: t-multiplier
    t_multiplier: confloat(gt=0.0) = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature = np.array([True, True, True, True], dtype=bool)

    def __str__(self):
        return f"\tDunkirk sand (PISA)\n\tDr = {var_to_str(self.Dr)}%. \n\tG0 = {round(self.G0/1000,1)} MPa"

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

        # define Dr
        Dr_t, Dr_b = from_list2x_parse_top_bottom(self.Dr)
        Dr = Dr_t + (Dr_b - Dr_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        p, y = py_curves.dunkirk_sand(
            sig=sig,
            X=X,
            Dr=Dr,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return p * self.p_multiplier, y * self.y_multiplier

    def Hb_spring_fct(
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

        # define Dr
        Dr_t, Dr_b = from_list2x_parse_top_bottom(self.Dr)
        Dr = Dr_t + (Dr_b - Dr_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        Hb, y = Hb_curves.dunkirk_sand(
            sig=sig,
            X=X,
            Dr=Dr,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return Hb, y

    def mt_spring_fct(
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

        # define Dr
        Dr_t, Dr_b = from_list2x_parse_top_bottom(self.Dr)
        Dr = Dr_t + (Dr_b - Dr_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        p_array, _ = py_curves.dunkirk_sand(
            sig=sig,
            X=X,
            Dr=Dr,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        for count, p_iter in enumerate(p_array):
            m[count], t[count, :] = mt_curves.dunkirk_sand(
                sig=sig,
                X=X,
                Dr=Dr,
                G0=Gmax,
                p=p_iter,
                D=D,
                L=L,
                output_length=output_length,
            )

        return m * self.m_multiplier, t * self.t_multiplier

    def Mb_spring_fct(
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

        # define Dr
        Dr_t, Dr_b = from_list2x_parse_top_bottom(self.Dr)
        Dr = Dr_t + (Dr_b - Dr_t) * depth_from_top_of_layer / layer_height
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        Mb, y = Mb_curves.dunkirk_sand(
            sig=sig,
            X=X,
            Dr=Dr,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return Mb, y


@dataclass(config=PydanticConfigFrozen)
class API_sand(LateralModel):
    """A class to establish the API sand model.

    Parameters
    ----------
    phi: float or list[top_value, bottom_value]
        internal angle of friction in degrees
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    p_multiplier: float
        multiplier for p-values
    y_multiplier: float
        multiplier for y-values

    """

    #: soil friction angle [deg], if a variation in values, two values can be given.
    phi: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: types of curves, can be of ("static","cyclic")
    kind: Literal["static", "cyclic"]
    #: p-multiplier
    p_multiplier: confloat(ge=0.0) = 1.0
    #: y-multiplier
    y_multiplier: confloat(gt=0.0) = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature = np.array([True, False, False, False], dtype=bool)

    def __str__(self):
        return f"\tAPI sand\n\tphi = {var_to_str(self.phi)}°\n\t{self.kind} curves"

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
            kind=self.kind,
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
    Su: float or list[top_value, bottom_value]
        Undrained shear strength in kPa
    eps50: float or list[top_value, bottom_value]
        strain at 50% failure load [-]
    J: float
        empirical factor varying depending on clay stiffness, varies between 0.25 and 0.50
    stiff_clay_threshold: float
        undrained shear strength [kPa] at which stiff clay curve is computed
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    p_multiplier: float
        multiplier for p-values
    y_multiplier: float
        multiplier for y-values

    """

    #: undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: strain at 50% failure load [-], if a variation in values, two values can be given.
    eps50: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: types of curves, can be of ("static","cyclic")
    kind: Literal["static", "cyclic"]
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
        return f"\tAPI clay\n\tSu = {var_to_str(self.Su)} kPa\n\teps50 = {var_to_str(self.eps50)}\n\t{self.kind} curves"

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
            kind=self.kind,
            ymax=ymax,
            output_length=output_length,
        )

        return p * self.p_multiplier, y * self.y_multiplier
