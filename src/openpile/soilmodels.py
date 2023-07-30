"""
`SoilModels` module
===================

"""

# Guide to create soil models
# ---------------------------
#
#


# Import libraries
import math as m
import numpy as np
import pandas as pd
from numba import njit, prange

from typing import List, Dict, Optional, Union, Callable
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
from openpile.utils import py_curves, Hb_curves, mt_curves, Mb_curves, tz_curves


# CONSTITUTIVE MODELS CLASSES ---------------------------------


class PydanticConfigFrozen:
    arbitrary_types_allowed = True
    allow_mutation = False


class ConstitutiveModel:
    pass


class LateralModel(ConstitutiveModel):
    pass


class AxialModel(ConstitutiveModel):
    pass


@dataclass(config=PydanticConfigFrozen)
class API_clay_axial(AxialModel):

    #: undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], confloat(ge=0.0)] = 1.0
    #: z-multiplier
    z_multiplier: Union[Callable[[float], float], confloat(gt=0.0)] = 1.0
    #: Q-multiplier
    Q_multiplier: confloat(ge=0.0) = 1.0
    #: w-multiplier
    w_multiplier: confloat(gt=0.0) = 1.0

    def __str__(self):
        return f"\tAPI clay\n\tSu = {var_to_str(self.Su)} kPa"


@dataclass(config=PydanticConfigFrozen)
class Cowden_clay(LateralModel):
    """A class to establish the PISA Cowden clay model as per Byrne et al 2020 (see [BHBG20]_).

    Parameters
    ----------
    Su: float or list[top_value, bottom_value]
        Undrained shear strength. Value to range from 0 to 100 [unit: kPa]
    G0: float or list[top_value, bottom_value]
        Small-strain shear modulus [unit: kPa]
    p_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for p-values
    y_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for y-values
    m_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for m-values
    t_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for t-values

    See also
    --------
    :py:func:`openpile.utils.py_curves.cowden_clay`, :py:func:`openpile.utils.mt_curves.cowden_clay`,
    :py:func:`openpile.utils.Hb_curves.cowden_clay`, :py:func:`openpile.utils.Mb_curves.cowden_clay`


    """

    #: Undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], confloat(ge=0.0)] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], confloat(gt=0.0)] = 1.0
    #: m-multiplier
    m_multiplier: Union[Callable[[float], float], confloat(ge=0.0)] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], confloat(gt=0.0)] = 1.0

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

        y, p = py_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            output_length=output_length,
        )

        # parse multipliers and apply results
        y_mult = self.y_multiplier if isinstance(self.y_multiplier, float) else self.y_multiplier(X)
        p_mult = self.p_multiplier if isinstance(self.p_multiplier, float) else self.p_multiplier(X)

        return y * y_mult, p * p_mult

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

        y, Hb = Hb_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return y, Hb

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

        _, p_array = py_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            output_length=output_length,
        )

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        for count, _ in enumerate(p_array):
            t[count, :], m[count, :] = mt_curves.cowden_clay(
                X=X,
                Su=Su,
                G0=Gmax,
                D=D,
                output_length=output_length,
            )

        # parse multipliers and apply results
        t_mult = self.t_multiplier if isinstance(self.t_multiplier, float) else self.t_multiplier(X)
        m_mult = self.m_multiplier if isinstance(self.m_multiplier, float) else self.m_multiplier(X)

        return t * t_mult, m * m_mult

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

        y, Mb = Mb_curves.cowden_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return y, Mb


@dataclass(config=PydanticConfigFrozen)
class Dunkirk_sand(LateralModel):
    """A class to establish the PISA Dunkirk sand model as per  Burd et al (2020) (see [BTZA20]_)..

    Parameters
    ----------
    Dr: float or list[top_value, bottom_value]
        relative density of sand. Value to range from 0 to 100. [unit: -]
    G0: float or list[top_value, bottom_value]
        Small-strain shear modulus [unit: kPa]
    p_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for p-values
    y_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for y-values
    m_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for m-values
    t_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for t-values

    See also
    --------
    :py:func:`openpile.utils.py_curves.dunkirk_sand`, :py:func:`openpile.utils.mt_curves.dunkirk_sand`,
    :py:func:`openpile.utils.Hb_curves.dunkirk_sand`, :py:func:`openpile.utils.Mb_curves.dunkirk_sand`

    """

    #: soil friction angle [deg], if a variation in values, two values can be given.
    Dr: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], confloat(ge=0.0)] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], confloat(gt=0.0)] = 1.0
    #: m-multiplier
    m_multiplier: Union[Callable[[float], float], confloat(ge=0.0)] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], confloat(gt=0.0)] = 1.0

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

        y, p = py_curves.dunkirk_sand(
            sig=sig,
            X=X,
            Dr=Dr,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        # parse multipliers and apply results
        y_mult = self.y_multiplier if isinstance(self.y_multiplier, float) else self.y_multiplier(X)
        p_mult = self.p_multiplier if isinstance(self.p_multiplier, float) else self.p_multiplier(X)

        return y * y_mult, p * p_mult

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

        y, Hb = Hb_curves.dunkirk_sand(
            sig=sig,
            X=X,
            Dr=Dr,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return y, Hb

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

        _, p_array = py_curves.dunkirk_sand(
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
            t[count, :], m[count] = mt_curves.dunkirk_sand(
                sig=sig,
                X=X,
                Dr=Dr,
                G0=Gmax,
                p=p_iter,
                D=D,
                L=L,
                output_length=output_length,
            )

        # parse multipliers and apply results
        t_mult = self.t_multiplier if isinstance(self.t_multiplier, float) else self.t_multiplier(X)
        m_mult = self.m_multiplier if isinstance(self.m_multiplier, float) else self.m_multiplier(X)

        return t * t_mult, m * m_mult

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

        y, Mb = Mb_curves.dunkirk_sand(
            sig=sig,
            X=X,
            Dr=Dr,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return y, Mb


@dataclass(config=PydanticConfigFrozen)
class API_sand(LateralModel):
    """A class to establish the API sand model.

    Parameters
    ----------
    phi: float or list[top_value, bottom_value]
        internal angle of friction in degrees
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    G0: float or list[top_value, bottom_value] or None
        Small-strain shear modulus [unit: kPa], by default None
    p_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for p-values
    y_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for y-values
    extension: str, by default None
        turn on extensions by calling them in this variable
        for API_sand, rotational springs can be added to the model with the extension "mt_curves"

    See also
    --------
    :py:func:`openpile.utils.py_curves.api_sand`

    """

    #: soil friction angle [deg], if a variation in values, two values can be given.
    phi: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: types of curves, can be of ("static","cyclic")
    kind: Literal["static", "cyclic"] = "static"
    #: small-strain stiffness [kPa], if a variation in values, two values can be given.
    G0: Optional[Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]] = None
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], confloat(ge=0.0)] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], confloat(gt=0.0)] = 1.0
    #: extensions available for soil model
    extension: Optional[Literal["mt_curves"]] = None

    # define class variables needed for all soil models
    m_multiplier = 1.0
    t_multiplier = 1.0

    def __post_init__(self):
        # spring signature which tells that API sand only has p-y curves in normal conditions
        # signature if e.g. of the form [p-y:True, Hb:False, m-t:False, Mb:False]
        if self.extension == "mt_curves":
            self.spring_signature = np.array([True, False, True, False], dtype=bool)
        else:
            self.spring_signature = np.array([True, False, False, False], dtype=bool)

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

        y, p = py_curves.api_sand(
            sig=sig,
            X=X,
            phi=phi,
            D=D,
            kind=self.kind,
            below_water_table=below_water_table,
            ymax=ymax,
            output_length=output_length,
        )

        # parse multipliers and apply results
        y_mult = self.y_multiplier if isinstance(self.y_multiplier, float) else self.y_multiplier(X)
        p_mult = self.p_multiplier if isinstance(self.p_multiplier, float) else self.p_multiplier(X)

        return y * y_mult, p * p_mult

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

        # define phi
        phi_t, phi_b = from_list2x_parse_top_bottom(self.phi)
        phi = phi_t + (phi_b - phi_t) * depth_from_top_of_layer / layer_height

        # define p vector
        _, p = py_curves.api_sand(
            sig=sig,
            X=X,
            phi=phi,
            D=D,
            kind=self.kind,
            below_water_table=below_water_table,
            ymax=ymax,
            output_length=output_length,
        )

        if p.max() > 0:
            p_norm = p / p.max()
        else:
            p_norm = p

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        z, tz = tz_curves.api_sand(
            sig=sig,
            delta=phi - 5,
            K=0.8,
            tensile_factor=1.0,
            output_length=output_length,
        )

        # trasnform tz vector
        tz_pos = tz[tz >= 0]
        z_pos = z[z >= 0]
        diff_length_t = output_length - len(tz_pos)
        diff_length_z = output_length - len(z_pos)
        tz_pos = np.append(tz_pos, [tz_pos[-1]] * diff_length_t)
        z_pos = np.append(z_pos, [z_pos[-1] + i * 0.1 for i in range(diff_length_z)])

        t = np.arctan(z_pos.reshape((1, -1)) / (0.5 * D)) * np.ones((output_length, 1))
        m = 1 / 4 * np.pi * D**2 * tz_pos.reshape((1, -1)) * p_norm.reshape((-1, 1))

        return t, m


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
    G0: float or list[top_value, bottom_value] or None
        Small-strain shear modulus [unit: kPa], by default None
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    p_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for p-values
    y_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for y-values
    extension: str, by default None
        turn on extensions by calling them in this variable
        for API_clay, rotational springs can be added to the model with the extension "mt_curves"


    See also
    --------
    :py:func:`openpile.utils.py_curves.api_clay`

    """

    #: undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: strain at 50% failure load [-], if a variation in values, two values can be given.
    eps50: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    #: types of curves, can be of ("static","cyclic")
    kind: Literal["static", "cyclic"] = "static"
    #: small-strain stiffness [kPa], if a variation in values, two values can be given.
    G0: Optional[Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]] = None
    #: empirical factor varying depending on clay stiffness
    J: confloat(ge=0.25, le=0.5) = 0.5
    #: undrained shear strength [kPa] at which stiff clay curve is computed
    stiff_clay_threshold: PositiveFloat = 96
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], confloat(ge=0.0)] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], confloat(gt=0.0)] = 1.0
    #: extensions available for soil model
    extension: Optional[Literal["mt_curves"]] = None

    # define class variables needed for all soil models
    m_multiplier = 1.0
    t_multiplier = 1.0

    def __post_init__(self):
        # spring signature which tells that API clay only has p-y curves in normal conditions
        # signature if e.g. of the form [p-y:True, Hb:False, m-t:False, Mb:False]
        if self.extension == "mt_curves":
            self.spring_signature = np.array([True, False, True, False], dtype=bool)
        else:
            self.spring_signature = np.array([True, False, False, False], dtype=bool)

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

        y, p = py_curves.api_clay(
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

        # parse multipliers and apply results
        y_mult = self.y_multiplier if isinstance(self.y_multiplier, float) else self.y_multiplier(X)
        p_mult = self.p_multiplier if isinstance(self.p_multiplier, float) else self.p_multiplier(X)

        return y * y_mult, p * p_mult

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

        # define Su
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        z, tz = tz_curves.api_clay(
            sig=sig,
            Su=Su,
            D=D,
            residual=1.0,
            tensile_factor=1.0,
            output_length=output_length,
        )

        # trasnform tz vector so that we only get the positive side of the vectors
        tz_pos = tz[tz >= 0]
        z_pos = z[z >= 0]
        # check how many elements we got rid off
        diff_length_t = output_length - len(tz_pos)
        diff_length_z = output_length - len(z_pos)
        # add new elements at the end of the positive only vectors
        tz_pos = np.append(tz_pos, [tz_pos[-1]] * diff_length_t)
        z_pos = np.append(z_pos, [z_pos[-1] + i * 0.1 for i in range(diff_length_z)])

        # calculate m and t vectors (they are all the same for clay)
        t = np.arctan(z_pos.reshape((1, -1)) / (0.5 * D)) * np.ones((output_length, 1))
        m = 1 / 4 * np.pi * D**2 * tz_pos.reshape((1, -1)) * np.ones((output_length, 1))

        return t, m
