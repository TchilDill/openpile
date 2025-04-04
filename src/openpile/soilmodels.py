r"""
`SoilModels` module
===================

This module comprises of the Soil Models available in OpenPile. 
`openpile.soilmodels.LateralModel` and `openpile.soilmodels.AxialModel` are defined  here in this module and can be called to
a :class:`openpile.construct.Layer` with the `lateral_model` or `axial_model` argument.


.. rubric:: References 

.. [MuOn83] Murchison, J.M., and O'Neill, M.,W., 1983. *An Evaluation of p-y Relationships 
    in Sands.* Rserach Report No. GT.DF02-83, Department of Civil Engineering, 
    University of Houston, Houston, Texas, May, 1983.
.. [MuOn84] Murchison, J.M., and O'Neill, M.,W., 1984. *Evaluation of p-y relationships 
    in cohesionless soils.* In Proceedings of Analysis and Design of Pile Foundations, 
    San Francisco, October 1-5, pp. 174-191.
.. [DNV-RP-C212] DNVGL RP-C212. *Recommended Practice, Geotechnical design*.
    Edition 2019-09 - Amended 2021-09.
.. [API2000] API, December 2000. *Recommended Practice for Planning, Designing, and Constructing 
    Fixed Offshore Platforms - Working Stress Design (RP 2A-WSD)*, Twenty-First Edition.
.. [API2014] API, October 2014. *Recommended Practice 2GEO/ISO 19901-4, Geotechnical and 
   Foundation Design Considerations*, 1st Edition
.. [Matl70] Matlock, H. (1970). *Correlations for Design of Laterally Loaded Piles in Soft Clay*. 
    Offshore Technology Conference Proceedings, Paper No. OTC 1204, Houston, Texas. 
.. [BaCA06] Battacharya,  S.,  Carrington,  T.  M.  and  Aldridge,  T.  R.  (2006),  
    *Design  of  FPSO  Piles  against  Storm  Loading*. Proceedings Annual Offshore Technology 
    Conference, OTC17861, Houston, Texas, May, 2006.
.. [KrRK81] Kraft, L.M., Ray, R.P., and Kagawa, T. (1981). *Theoretical t-z curves*. 
    Journal of the Geotechnical Engineering Division, ASCE, Vol. 107, No. GT11, pp. 1543-1561.
.. [BHBG20] Byrne, B. W., Houlsby, G. T., Burd, H. J., Gavin, K. G., Igoe, D. J. P., 
    Jardine, R. J., Martin, C. M., McAdam, R. A., Potts, D. M., Taborda, D. M. G. & Zdravkovic ́, L. (2020). 
    PISA design model for monopiles for offshore wind turbines: application 
    to a stiff glacial clay till. Géotechnique, https://doi.org/10.1680/ jgeot.18.P.255.
.. [BTZA20] Burd, H. J., Taborda, D. M. G., Zdravkovic ́, L., Abadie, C. N., Byrne, B. W., 
    Houlsby, G. T., Gavin, K. G., Igoe, D. J. P., Jardine, R. J., Martin, C. M., McAdam, R. A., 
    Pedro, A. M. G. & Potts, D. M. (2020). PISA design model for monopiles for offshore wind 
    turbines: application to a marine sand. Géotechnique, https://doi.org/10.1680/jgeot.18.P.277.
.. [BABH20] Burd, H. J., Abadie, C. N., Byrne, B. W., Houlsby, G. T., Martin, C. M., McAdam, R. A., 
    Jardine, R.J., Pedro, A.M., Potts, D.M., Taborda, D.M., Zdravković, L., and Andrade, M.P. 
    (2020). Application of the PISA Design Model to Monopiles Embedded in Layered Soils. 
    Géotechnique 70(11): 1-55. https://doi.org/10.1680/jgeot.20.PISA.009
.. [Rees97] Reese, L.C. (1997), Analysis of Laterally Loaded Piles in Weak Rock, Journal of Geotechnical
    and Geoenvironmental Engineering, ASCE, vol. 123 (11) Nov., ASCE, pp. 1010-1017.
.. [SøIA10] Sorensen, S.P.H. & Ibsen, L.B. & Augustesen, A.H. (2010), Effects of diameter on 
    initial stiffness of p-y curves for large-diameter piles in sand, Numerical Methods in 
    Geotechnical Engineering, CRC Press, pp. 907-912.
.. .. [Søre12] Sorensen, S.P.H. (2012), Soil-Structure Interaction For Nonslender, Large-Diameter 
..     Offshore Monopiles. PhD Thesis, Department of Civil Engineering, Aalborg University, Denmark.


Introduction - lateral soil models
----------------------------------

Lateral models are capable of creating lateral and rotational springs. 

The following lateral models are included in openpile. 

* :class:`openpile.soilmodels.API_sand`
* :class:`openpile.soilmodels.API_clay`
* :class:`openpile.soilmodels.Modified_Matlock_clay`
* :class:`openpile.soilmodels.Reese_weakrock`
* :class:`openpile.soilmodels.Dunkirk_sand`
* :class:`openpile.soilmodels.Cowden_clay`
* :class:`openpile.soilmodels.Custom_pisa_sand`
* :class:`openpile.soilmodels.Custom_pisa_clay`

Typically, each model relates to soil spring definitions stored in either:

* :mod:`openpile.utils.py_curves`
* :mod:`openpile.utils.mt_curves`
* :mod:`openpile.utils.Hb_curves`
* :mod:`openpile.utils.Mb_curves`
* :mod:`openpile.utils.tz_curves`
* :mod:`openpile.utils.qz_curves`

Introduction - axial soil models
--------------------------------

The axial model are capable of calculating skin friction along the pile and end-bearing at pile tip.

The following axial models are included in openpile. 

* :py:class:`openpile.soilmodels.API_sand_axial`
* :py:class:`openpile.soilmodels.API_clay_axial`

This soil model then provides soil springs as given by the function(s) below and depending on the type of material:

* :py:func:`openpile.utils.tz_curves.api_sand`
* :py:func:`openpile.utils.tz_curves.api_sand_kraft`
* :py:func:`openpile.utils.qz_curves.api_sand`
* :py:func:`openpile.utils.tz_curves.api_clay`
* :py:func:`openpile.utils.tz_curves.api_clay_kraft`
* :py:func:`openpile.utils.qz_curves.api_clay`


"""


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
from pydantic.dataclasses import dataclass

from openpile.core.misc import from_list2x_parse_top_bottom, var_to_str, get_value_at_current_depth
from openpile.utils import py_curves, Hb_curves, mt_curves, Mb_curves, tz_curves, qz_curves
from openpile.utils.misc import _fmax_api_sand, _fmax_api_clay, _Qmax_api_clay, _Qmax_api_sand
from openpile.utils.hooks import InitialSubgradeReaction

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from typing_extensions import Literal, Annotated, Optional
from pydantic import BaseModel, AfterValidator, ConfigDict, Field, model_validator


# CONSTITUTIVE MODELS CLASSES ---------------------------------


class LateralModel(BaseModel, ABC):
    model_config = ConfigDict(
        extra="allow",
    )

    @abstractmethod
    def py_spring_fct(
        self,
        sig: float,
        X: float,
        layer_height: float,
        depth_from_top_of_layer: float,
        D: float,
        L: float,
        below_water_table: bool,
        ymax: float,
        output_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def Hb_spring_fct(
        self, ymax: float = 1, output_length: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        return np.linspace(0, ymax, output_length), np.zeros(output_length)

    def mt_spring_fct(
        self, ymax: float = 0.2, output_length: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        return np.linspace(0, ymax, output_length), np.zeros(output_length)

    def Mb_spring_fct(
        self, ymax: float = 0.2, output_length: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        return np.linspace(0, ymax, output_length), np.zeros(output_length)


class AxialModel(BaseModel, ABC):
    model_config = ConfigDict(
        extra="allow",
    )

    @property
    @abstractmethod
    def method(self) -> str:
        pass

    @abstractmethod
    def unit_shaft_friction(self, *args):
        pass

    @abstractmethod
    def unit_tip_resistance(self, *args):
        pass

    @abstractmethod
    def tz_spring_fct(self, *args):
        pass

    @abstractmethod
    def Qz_spring_fct(self, *args):
        pass


class API_clay_axial(AxialModel):
    """A class to assign API clay for t-z and q-z curves in a Layer.

    Parameters
    ----------
    Su: float or function taking the depth as input argument and returning a value
        Undrained shear strength of soil. [unit: kPa]
    plugging: str
        defines whether pile behave plugged or unplugged,
        can be one of ('none', 'compression', 'tension', 'both'), by default 'none'.

        The plugging criterion is only relevant if the pile section has an open geometry.
    alpha_limit: float
        Limit of unit shaft friction, normalized with undrained shear strength, by default it is 1.0
    t_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for t-values, by default it is 1.0
    z_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for z-values, by default it is 1.0
    Q_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for Q-values, by default it is 1.0
    w_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for w-values, by default it is 1.0
    t_residual: float
        Residual value of t-z curves, by default it is 0.9 and can range from 0.7 to 0.9.
    tension_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for tensile strength, by default it is 1.0.

    Returns
    -------
    AxialModel
        AxialModel object with API clay.

    Example
    -------

    >>> from openpile.construct import Layer
    >>> from openpile.soilmodels import API_clay_axial, API_clay
    >>> clay_layer = Layer(
    ...                 name="clay",
    ...                 top=-20,
    ...                 bottom=-40,
    ...                 weight=18,
    ...                 lateral_model=API_clay(Su=[50, 70], eps50=0.015, kind="cyclic"),
    ...                 axial_model=API_clay_axial(Su=[50, 70])
    ...             )
    >>> print(clay_layer) # doctest: +NORMALIZE_WHITESPACE
    Name: clay
    Elevation: (-20.0) - (-40.0) m
    Weight: 18.0 kN/m3
    Lateral model: 	API clay
        Su = 50.0-70.0 kPa
        eps50 = 0.015
        cyclic curves
        ext: None
    Axial model: 	API clay (Unplugged)
        Su = 50.0-70.0 kPa
        alpha_limit = 1.0


    See also
    --------
    :py:func:`openpile.utils.tz_curves.api_sand`, :py:func:`openpile.utils.qz_curves.api_sand`,

    """

    #: undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    # pile behaviour upon axial loading
    plugging: Literal["none", "compression", "tension", "both"] = "none"
    #: limiting value of unit shaft friction normalized to undrained shear strength
    alpha_limit: Annotated[float, Field(gt=0.0)] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: z-multiplier
    z_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: Q-multiplier
    Q_multiplier: Annotated[float, Field(ge=0.0)] = 1.0
    #: w-multiplier
    w_multiplier: Annotated[float, Field(gt=0.0)] = 1.0
    #: t-residual
    t_residual: Annotated[float, Field(ge=0.7, le=0.9)] = 0.9
    #: tension factor
    tension_multiplier: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    def __str__(self):
        if self.plugging == "both":
            out = f"\tAPI clay (Plugged)\n\tSu = {var_to_str(self.Su)} kPa\n\talpha_limit = {var_to_str(self.alpha_limit)}"
        elif self.plugging == "none":
            out = f"\tAPI clay (Unplugged)\n\tSu = {var_to_str(self.Su)} kPa\n\talpha_limit = {var_to_str(self.alpha_limit)}"
        elif self.plugging == "compression":
            out = f"\tAPI clay (Plugged in compression, unplugged in tension)\n\tSu = {var_to_str(self.Su)} kPa\n\talpha_limit = {var_to_str(self.alpha_limit)}"
        elif self.plugging == "tension":
            out = f"\tAPI clay (Unplugged in compression, plugged in tension)\n\tSu = {var_to_str(self.Su)} kPa\n\talpha_limit = {var_to_str(self.alpha_limit)}"
        if self.t_multiplier != 1.0:
            out += f"\n\tt-multiplier = {var_to_str(self.t_multiplier)}"
        if self.z_multiplier != 1.0:
            out += f"\n\tz-multiplier = {var_to_str(self.z_multiplier)}"
        if self.Q_multiplier != 1.0:
            out += f"\n\tQ-multiplier = {var_to_str(self.Q_multiplier)}"
        if self.w_multiplier != 1.0:
            out += f"\n\tw-multiplier = {var_to_str(self.w_multiplier)}"
        if self.t_residual != 0.9:
            out += f"\n\tt-residual = {var_to_str(self.t_residual)}"
        if self.tension_multiplier != 1.0:
            out += f"\n\ttension_multiplier = {var_to_str(self.tension_multiplier)}"
        return out

    def unit_shaft_friction(self, sig, depth_from_top_of_layer, layer_height):
        # define Su
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height

        return _fmax_api_clay(sig, Su, self.alpha_limit)

    def unit_tip_resistance(self, sig, depth_from_top_of_layer, layer_height):
        # define Su
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height

        return _Qmax_api_clay(Su=Su)

    def unit_shaft_signature(self, *args, **kwargs):
        "This function determines how the unit shaft friction should be applied on outer an inner side of the pile"
        return {"out": 1.0, "in": 1.0 * self.inside_friction}
        # for CPT based methods, it should be: return {'out': out_perimeter/(out_perimeter+in_perimeter), 'in':in_perimeter/(out_perimeter+in_perimeter)}

    def tz_spring_fct(
        self,
        circumference_in: float,
        circumference_out: float,
        sig: float,
        layer_height: float,
        depth_from_top_of_layer: float,
        D: float,
        output_length: int = 15,
        **kwargs,
    ):

        # define Su
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height

        if self.plugging == "none":
            circumference_total = circumference_out + circumference_in
            tens_fac = self.tension_multiplier
        elif self.plugging == "tension":
            circumference_total = circumference_out + circumference_in
            tens_fac = self.tension_multiplier * circumference_in / (circumference_total)
        elif self.plugging == "compression":
            circumference_total = circumference_out
            tens_fac = (
                self.tension_multiplier
                * (circumference_in + circumference_out)
                / (circumference_total)
            )
        elif self.plugging == "both":
            circumference_total = circumference_out
            tens_fac = self.tension_multiplier * circumference_in / (circumference_total)

        z, t = tz_curves.api_clay(
            sig=sig,
            Su=Su,
            D=D,
            residual=self.t_residual,
            tensile_factor=tens_fac,
            output_length=output_length,
        )

        return z * self.z_multiplier, t * self.t_multiplier * circumference_total

    def Qz_spring_fct(
        self,
        tip_area: float,
        footprint: float,
        layer_height: float,
        depth_from_top_of_layer: float,
        D: float,
        output_length: int = 15,
        **kwargs,
    ):

        # define Su
        Su_t, Su_b = from_list2x_parse_top_bottom(self.Su)
        Su = Su_t + (Su_b - Su_t) * depth_from_top_of_layer / layer_height

        w, Q = qz_curves.api_clay(Su=Su, D=D, output_length=output_length)

        if self.plugging == "both" or self.plugging == "compression":
            area = footprint
        else:
            area = tip_area

        return w * self.w_multiplier, Q * self.Q_multiplier * area

    def method(self) -> str:
        return "API"


class API_sand_axial(AxialModel):
    """A class to assign API sand for t-z and q-z curves in a Layer.

    Parameters
    ----------
    delta: float or list[top_value, bottom_value]
        interface friction angle of sand/pile. Typical value ranges between 70% and 100% of internal friction angle of soil. [unit: degrees]
    K: float
        Coefficient of earth pressure against pile, it should be 0.8 for open-ended piles and 1.0 for closed-ended piles.
    plugging: str
        defines whether pile behave plugged or unplugged, can be one of ('none', 'compression', 'tension', 'both'), by default 'none'
    t_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for t-values, by default it is 1.0
    z_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for z-values, by default it is 1.0
    Q_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for Q-values, by default it is 1.0
    w_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for w-values, by default it is 1.0
    tension_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for tensile strength, by default it is 1.0

    Returns
    -------
    AxialModel
        AxialModel object with API sand.

    Example
    -------

    >>> from openpile.construct import Layer
    >>> from openpile.soilmodels import API_sand_axial, API_sand
    >>> sand_layer = Layer(
    ...                 name="sand",
    ...                 top=-20,
    ...                 bottom=-40,
    ...                 weight=18,
    ...                 lateral_model=API_sand(phi=30, kind='cyclic'),
    ...                 axial_model=API_sand_axial(
    ...                     delta=25,
    ...                 ),
    ...             )
    >>> print(sand_layer) # doctest: +NORMALIZE_WHITESPACE
    Name: sand
    Elevation: (-20.0) - (-40.0) m
    Weight: 18.0 kN/m3
    Lateral model: 	API sand
        phi = 30.0°
        cyclic curves
        ext: None
    Axial model: 	API sand (Unplugged)
        delta = 25.0 deg
        K = 0.8

    See also
    --------
    :py:func:`openpile.utils.tz_curves.api_sand`, :py:func:`openpile.utils.qz_curves.api_sand`,

    """

    #: interface friction angle [deg], if a variation in values, two values can be given.
    delta: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[Annotated[float, Field(ge=0, le=45)]], Field(min_length=1, max_length=2)],
    ]
    #: coefficient of lateral earth pressure, for open-ended piles, a value of 0.8 should be considered while 1.0 for close-ended piles
    K: Annotated[float, Field(ge=0.8, le=1.0)] = 0.8
    # pile behaviour upon axial loading
    plugging: Literal["none", "compression", "tension", "both"] = "none"
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: z-multiplier
    z_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: Q-multiplier
    Q_multiplier: Annotated[float, Field(ge=0.0)] = 1.0
    #: w-multiplier
    w_multiplier: Annotated[float, Field(gt=0.0)] = 1.0
    #: inner_shaft_friction
    inside_friction: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    #: tension factor
    tension_multiplier: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    def __str__(self):
        if self.plugging == "both":
            out = f"\tAPI sand (Plugged)\n\tdelta = {var_to_str(self.delta)} deg\n\tK = {var_to_str(self.K)}"
        elif self.plugging == "none":
            out = f"\tAPI sand (Unplugged)\n\tdelta = {var_to_str(self.delta)} deg\n\tK = {var_to_str(self.K)}"
        elif self.plugging == "compression":
            out = f"\tAPI sand (Plugged in compression, unplugged in tension)\n\tdelta = {var_to_str(self.delta)} deg\n\tK = {var_to_str(self.K)}"
        elif self.plugging == "tension":
            out = f"\tAPI sand (Unplugged in compression, plugged in tension)\n\tdelta = {var_to_str(self.delta)} deg\n\tK = {var_to_str(self.K)}"
        if self.t_multiplier != 1.0:
            out += f"\n\tt-multiplier = {var_to_str(self.t_multiplier)}"
        if self.z_multiplier != 1.0:
            out += f"\n\tz-multiplier = {var_to_str(self.z_multiplier)}"
        if self.Q_multiplier != 1.0:
            out += f"\n\tQ-multiplier = {var_to_str(self.Q_multiplier)}"
        if self.w_multiplier != 1.0:
            out += f"\n\tw-multiplier = {var_to_str(self.w_multiplier)}"
        if self.tension_multiplier != 1.0:
            out += f"\n\ttension_multiplier = {var_to_str(self.tension_multiplier)}"
        return out

    def unit_shaft_friction(self, sig, depth_from_top_of_layer, layer_height):
        # define interface friction angle
        delta_t, delta_b = from_list2x_parse_top_bottom(self.delta)
        delta = delta_t + (delta_b - delta_t) * depth_from_top_of_layer / layer_height

        return _fmax_api_sand(sig, delta, self.K)

    def unit_tip_resistance(self, sig, depth_from_top_of_layer, layer_height):
        # define interface friction angle
        delta_t, delta_b = from_list2x_parse_top_bottom(self.delta)
        delta = delta_t + (delta_b - delta_t) * depth_from_top_of_layer / layer_height

        return _Qmax_api_sand(sig=sig, delta=delta)

    def unit_shaft_signature(self, *args, **kwargs):
        "This function determines how the unit shaft friction should be applied on outer an inner side of the pile"
        return {"out": 1.0, "in": 1.0 * self.inside_friction}
        # for CPT based methods, it should be: return {'out': out_perimeter/(out_perimeter+in_perimeter), 'in':in_perimeter/(out_perimeter+in_perimeter)}

    def tz_spring_fct(
        self,
        circumference_out: float,
        circumference_in: float,
        sig: float,
        layer_height: float,
        depth_from_top_of_layer: float,
        D: float,
        output_length: int = 15,
        **kwargs,
    ):

        # define interface friction angle
        delta_t, delta_b = from_list2x_parse_top_bottom(self.delta)
        delta = delta_t + (delta_b - delta_t) * depth_from_top_of_layer / layer_height

        if self.plugging == "none":
            tens_fac = self.tension_multiplier
            circumference_total = circumference_out + circumference_in
        elif self.plugging == "tension":
            circumference_total = circumference_out + circumference_in
            tens_fac = self.tension_multiplier * circumference_in / (circumference_total)
        elif self.plugging == "compression":
            circumference_total = circumference_out
            tens_fac = (
                self.tension_multiplier
                * (circumference_in + circumference_out)
                / (circumference_total)
            )
        elif self.plugging == "both":
            circumference_total = circumference_out
            tens_fac = self.tension_multiplier * circumference_in / (circumference_total)

        z, t = tz_curves.api_sand(
            sig=sig,
            delta=delta,
            K=self.K,
            tensile_factor=tens_fac,
            output_length=output_length,
        )

        return z * self.z_multiplier, t * self.t_multiplier * circumference_total

    def Qz_spring_fct(
        self,
        tip_area: float,
        footprint: float,
        sig: float,
        layer_height: float,
        depth_from_top_of_layer: float,
        D: float,
        output_length: int = 15,
        **kwargs,
    ):

        # define interface friction angle
        delta_t, delta_b = from_list2x_parse_top_bottom(self.delta)
        delta = delta_t + (delta_b - delta_t) * depth_from_top_of_layer / layer_height

        w, Q = qz_curves.api_sand(sig=sig, delta=delta, D=D, output_length=output_length)

        if self.plugging == "both" or self.plugging == "compression":
            area = footprint
        else:
            area = tip_area

        return w * self.w_multiplier, Q * self.Q_multiplier * area

    def method(self) -> str:
        return "API"


class Bothkennar_clay(LateralModel):
    """A class to establish the PISA Bothkennar clay model as per Burd et al 2020 (see [BABH20]_).

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
    :py:func:`openpile.utils.py_curves.bothkennar_clay`, :py:func:`openpile.utils.mt_curves.bothkennar_clay`,
    :py:func:`openpile.utils.Hb_curves.bothkennar_clay`, :py:func:`openpile.utils.Mb_curves.bothkennar_clay`


    """

    #: Undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: m-multiplier
    m_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature: ClassVar[np.array] = np.array([True, True, True, True], dtype=bool)

    def __str__(self):
        return f"\tCowden clay (PISA)\n\tSu = {var_to_str(self.Su)} kPa.\n\tG0 = {round(self.G0/1000,1)} MPa"

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
        # define G0
        G0_t, G0_b = from_list2x_parse_top_bottom(self.G0)
        Gmax = G0_t + (G0_b - G0_t) * depth_from_top_of_layer / layer_height

        y, p = py_curves.bothkennar_clay(
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

        y, Hb = Hb_curves.bothkennar_clay(
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

        _, p_array = py_curves.bothkennar_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            output_length=output_length,
        )

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        for count, _ in enumerate(p_array):
            t[count, :], m[count, :] = mt_curves.bothkennar_clay(
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

        y, Mb = Mb_curves.bothkennar_clay(
            X=X,
            Su=Su,
            G0=Gmax,
            D=D,
            L=L,
            output_length=output_length,
        )

        return y, Mb


class Cowden_clay(LateralModel):
    r"""A class to establish the PISA Cowden clay model as per Byrne et al 2020 (see [BHBG20]_).

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

    Notes
    -----
    This soil model was formulated as part of the Joint Industry Project PISA,
    that focused on formulating soil springs for large diameter monopiles as found
    in the offshore wind industry. This resulted in soil springs formulated in a normalized
    space based on a conic function backbone curve and the few following soil parameters,
    (i) undrained shear strength and (ii) small-strain shear stiffness.

    This soil model provides soil springs as given by the function(s):

    * :py:func:`openpile.utils.py_curves.cowden_clay`
    * :py:func:`openpile.utils.mt_curves.cowden_clay`
    * :py:func:`openpile.utils.Hb_curves.cowden_clay`
    * :py:func:`openpile.utils.Mb_curves.cowden_clay`

    .. note::
        This standard model only account for monotonic reaction curves and as usual,
        it reflects the site conditions of the site the curves were calibrated from,
        a site in Cowden, England where overconsolidated glacial till is found.
        More details can be found in [BHBG20]_.

    The model is validated in the below figure by performing a benchmark of OpenPile
    against the source material, [BHBG20]_.

    .. figure:: _static/validation/CowdenClay_D1_D2.png
        :width: 80%

        Validation against piles D1 and D2 documented in [BHBG20]_.

    """

    #: Undrained shear strength [kPa], if a variation in values, two values can be given.
    Su: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: m-multiplier
    m_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature: ClassVar[np.array] = np.array([True, True, True, True], dtype=bool)

    def __str__(self):
        return f"\tCowden clay (PISA)\n\tSu = {var_to_str(self.Su)} kPa.\n\tG0 = {round(self.G0/1000,1)} MPa"

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


class Dunkirk_sand(LateralModel):
    r"""A class to establish the PISA Dunkirk sand model as per  Burd et al (2020) (see [BTZA20]_)..

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

    Notes
    -----
    This soil model was formulated as part of the Joint Industry Project PISA,
    that focused on formulating soil springs for large diameter monopiles as found
    in the offshore wind industry. This resulted in soil springs formulated in a normalized
    space based on a conic function backbone curve and the few following soil parameters,
    (i) relative density and (ii) small-strain shear stiffness.

    This soil model provides soil springs as given by the function(s):

    * :py:func:`openpile.utils.py_curves.dunkirk_sand`
    * :py:func:`openpile.utils.mt_curves.dunkirk_sand`
    * :py:func:`openpile.utils.Hb_curves.dunkirk_sand`
    * :py:func:`openpile.utils.Mb_curves.dunkirk_sand`

    .. note::
        This standard model only account for monotonic reaction curves and as usual, it reflects the site
        conditions of the site the curves were calibrated from, a site in Dunkirk, France where dense sand is found.
        More details can be found in [BTZA20]_.

    This soil model was formulated as part of the Joint Industry Project PISA, that focused on formulating soil springs for large diameter monopiles as found in the offshore wind industry.
    This resulted in soil springs formulated in a normalized space based on a conic function backbone curve and the few following soil parameters,
    (i) relative density and (ii) small-strain shear stiffness.

    Validation is shown in the below figure by performing a benchmark of OpenPile
    against the source material, [BTZA20]_. OpenPile shows some differences in result for high lateral load.
    This is due to the slight difference in input given in OpenPile compares to the source material.

    .. figure:: _static/validation/GDSM_D2t.png
        :width: 80%

        Validation against pile D2t documented in [BTZA20]_.

    """

    #: Relative density [%], if a variation in values, two values can be given.
    Dr: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: m-multiplier
    m_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature: ClassVar[np.array] = np.array([True, True, True, True], dtype=bool)

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


class API_sand(LateralModel):
    """A class to establish the API sand model.


    The API sand soil model is based on the publication by
    O'neill and Murchison, preceded by work from Reese, L.C. and others (
    see [MuOn83]_ and [MuOn84]_).

    This soil model provides soil springs as given by the function(s):

    * :py:func:`openpile.utils.py_curves.api_sand`

    Parameters
    ----------
    phi: float or list[top_value, bottom_value]
        internal angle of friction in degrees
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    G0: float or list[top_value, bottom_value] or None
        Small-strain shear modulus [unit: kPa], by default None
    initial_subgrade_modulus: float or list[top_value, bottom_value] or None
        User-defined initial subgrade modulus  [unit: kN/m^3], by default None which default to API definition based on friction angle
    Modification: str or None, by default None
        Application of well-known modification to API sand. Modifications available are:

        - "Kallehave" - which calls the p-y springs :py:func:`openpile.utils.hooks.InitialSubgradeReaction.kallehave_sand()`.
        - "Sørensen" - which calls the p-y springs with the initial subgrade modulus :py:func:`openpile.utils.hooks.InitialSubgradeReaction.sørensen2010_sand()`.

    p_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for p-values
    y_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for y-values
    extension: str, by default None
        turn on extensions by calling them in this variable.
        Rotational springs can be added to the model with the extension "mt_curves"

    See also
    --------
    :py:func:`openpile.utils.py_curves.api_sand`, :py:func:`openpile.utils.hooks.durkhop`

    """

    #: soil friction angle [deg], if a variation in values, two values can be given.
    phi: Union[
        Annotated[float, Field(ge=15, le=45)],
        Annotated[List[confloat(ge=15, le=45)], Field(min_length=1, max_length=2)],
    ]
    #: types of curves, can be of ("static","cyclic")
    kind: Literal["static", "cyclic"] = "static"
    #: small-strain stiffness [kPa], if a variation in values, two values can be given.
    G0: Optional[
        Union[
            Annotated[float, Field(gt=0.0)],
            Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        ]
    ] = None
    #: user-defined initial subgrade modulus [kN/m^3], if a variation in values, two values can be given.
    initial_subgrade_modulus: Optional[
        Union[
            Annotated[float, Field(gt=0.0)],
            Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        ]
    ] = None
    #: Application of well-known modification to API sand
    Modification: Optional[Literal["Kallehave", "Sørensen"]] = None
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: extensions available for soil model
    extension: Optional[Literal["mt_curves"]] = None

    # common class variables
    m_multiplier: ClassVar[float] = 1.0
    t_multiplier: ClassVar[float] = 1.0

    def model_post_init(self, *args, **kwargs):
        # spring signature which tells that API sand only has p-y curves in normal conditions
        # signature if e.g. of the form [p-y:True, Hb:False, m-t:False, Mb:False]
        if self.extension == "mt_curves":
            self.spring_signature = np.array([True, False, True, False], dtype=bool)
        else:
            self.spring_signature = np.array([True, False, False, False], dtype=bool)
        return self

    def __str__(self):
        return f"\tAPI sand\n\tphi = {var_to_str(self.phi)}°\n\t{self.kind} curves\n\text: {self.extension}"

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

        # initial k
        if self.initial_subgrade_modulus is None:
            subgrade_modulus = 0.0
        else:
            subgrade_modulus_t, subgrade_modulus_b = from_list2x_parse_top_bottom(
                self.initial_subgrade_modulus
            )
            subgrade_modulus = (
                subgrade_modulus_t
                + (subgrade_modulus_b - subgrade_modulus_t) * depth_from_top_of_layer / layer_height
            )

        if not self.Modification:
            y, p = py_curves.api_sand(
                sig=sig,
                X=X,
                phi=phi,
                D=D,
                kind=self.kind,
                below_water_table=below_water_table,
                k=subgrade_modulus,
                ymax=ymax,
                output_length=output_length,
            )
        elif self.Modification == "Kallehave":
            y, p = py_curves.api_sand(
                sig=sig,
                X=X,
                phi=phi,
                D=D,
                kind=self.kind,
                below_water_table=below_water_table,
                k=InitialSubgradeReaction.kallehave_sand(phi),
                ymax=ymax,
                output_length=output_length,
            )
        elif self.Modification == "Sørensen":
            y, p = py_curves.api_sand(
                sig=sig,
                X=X,
                phi=phi,
                D=D,
                kind=self.kind,
                below_water_table=below_water_table,
                k=InitialSubgradeReaction.sørensen2010_sand(phi),
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
        _, p = self.py_spring_fct(
            sig=sig,
            X=X,
            layer_height=layer_height,
            depth_from_top_of_layer=depth_from_top_of_layer,
            D=D,
            L=L,
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

        # transform tz vector
        tz_pos = tz[tz >= 0]
        z_pos = z[z >= 0]
        diff_length_t = output_length - len(tz_pos)
        diff_length_z = output_length - len(z_pos)
        tz_pos = np.append(tz_pos, [tz_pos[-1]] * diff_length_t)
        z_pos = np.append(z_pos, [z_pos[-1] + i * 0.1 for i in range(diff_length_z)])

        t = np.arctan(z_pos.reshape((1, -1)) / (0.5 * D)) * np.ones((output_length, 1))
        m = 1 / 4 * np.pi * D**2 * tz_pos.reshape((1, -1)) * p_norm.reshape((-1, 1))

        return t, m


class API_clay(LateralModel):
    """A class to establish the API clay model as per [API2014]_.

    Parameters
    ----------
    Su: float or list[top_value, bottom_value]
        Undrained shear strength in kPa
    eps50: float or list[top_value, bottom_value]
        strain at 50% failure load [-]
    J: float
        empirical factor varying depending on clay stiffness, varies between 0.25 and 0.50
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
    Su: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: strain at 50% failure load [-], if a variation in values, two values can be given.
    eps50: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: types of curves, can be of ("static","cyclic")
    kind: Literal["static", "cyclic"] = "static"
    #: small-strain stiffness [kPa], if a variation in values, two values can be given.
    G0: Optional[
        Union[
            Annotated[float, Field(gt=0.0)],
            Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        ]
    ] = None
    #: empirical factor varying depending on clay stiffness
    J: Annotated[float, Field(ge=0.25, le=0.5)] = 0.5
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: extensions available for soil model
    extension: Optional[Literal["mt_curves"]] = None

    # common class variables
    m_multiplier: ClassVar[float] = 1.0
    t_multiplier: ClassVar[float] = 1.0

    def model_post_init(self, *args, **kwargs):
        # spring signature which tells that API clay only has p-y curves in normal conditions
        # signature if e.g. of the form [p-y:True, Hb:False, m-t:False, Mb:False]
        if self.extension == "mt_curves":
            self.spring_signature = np.array([True, False, True, False], dtype=bool)
        else:
            self.spring_signature = np.array([True, False, False, False], dtype=bool)
        return self

    def __str__(self):
        return f"\tAPI clay\n\tSu = {var_to_str(self.Su)} kPa\n\teps50 = {var_to_str(self.eps50)}\n\t{self.kind} curves\n\text: {self.extension}"

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


class Modified_Matlock_clay(LateralModel):
    """A class to establish the Modified Matlock clay model, see [BaCA06]_.

    Parameters
    ----------
    Su: float or list[top_value, bottom_value]
        Undrained shear strength in kPa
    eps50: float or list[top_value, bottom_value]
        strain at 50% failure load [-]
    J: float
        empirical factor varying depending on clay stiffness, varies between 0.25 and 0.50
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
    Su: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: strain at 50% failure load [-], if a variation in values, two values can be given.
    eps50: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: types of curves, can be of ("static","cyclic")
    kind: Literal["static", "cyclic"] = "static"
    #: small-strain stiffness [kPa], if a variation in values, two values can be given.
    G0: Optional[
        Union[
            Annotated[float, Field(gt=0.0)],
            Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        ]
    ] = None
    #: empirical factor varying depending on clay stiffness
    J: Annotated[float, Field(ge=0.25, le=0.5)] = 0.5
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: extensions available for soil model
    extension: Optional[Literal["mt_curves"]] = None

    # common class variables
    m_multiplier: ClassVar[float] = 1.0
    t_multiplier: ClassVar[float] = 1.0

    def model_post_init(self, *args, **kwargs):
        # spring signature which tells that API clay only has p-y curves in normal conditions
        # signature if e.g. of the form [p-y:True, Hb:False, m-t:False, Mb:False]
        if self.extension == "mt_curves":
            self.spring_signature = np.array([True, False, True, False], dtype=bool)
        else:
            self.spring_signature = np.array([True, False, False, False], dtype=bool)
        return self

    def __str__(self):
        return f"\tModified Matlock clay\n\tSu = {var_to_str(self.Su)} kPa\n\teps50 = {var_to_str(self.eps50)}\n\t{self.kind} curves\n\text: {self.extension}"

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

        y, p = py_curves.modified_Matlock(
            sig=sig,
            X=X,
            Su=Su,
            eps50=eps50,
            D=D,
            J=self.J,
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


class Reese_weakrock(LateralModel):
    """A class to establish the Reese weakrock model.

    Parameters
    ----------
    Ei: float or list[top_value, bottom_value]
        Initial modulus of rock [unit: kPa]
    qu: float or list[top_value, bottom_value]
        compressive strength of rock [unit: kPa]
    RQD: float or list[top_value, bottom_value]
        Rock Quality Designation [unit: %]
    k: float
        dimensional constant randing from 0.0005 to 0.00005, by default 0.0005
    ztop: float
        absolute depth of top layer elevation with respect to rock surface [m]
    p_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for p-values
    y_multiplier: float or function taking the depth as argument and returns the multiplier
        multiplier for y-values

    """

    #: initial modulus of rock [kPa], if a variation in values, two values can be given.
    Ei: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: compressive strength of rock [kPa], if a variation in values, two values can be given.
    qu: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
    ]
    #: Rock Quality Designation
    RQD: Annotated[float, Field(ge=0.0, le=100.0)]
    #: dimnesional constant
    k: Annotated[float, Field(ge=0.00005, le=0.0005)]
    #: absolute depth of top layer elevation with respect to rock surface [m]
    ztop: Annotated[float, Field(ge=0.0)]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0

    # common class variables
    m_multiplier: ClassVar[float] = 1.0
    t_multiplier: ClassVar[float] = 1.0
    spring_signature: ClassVar[np.array] = np.array([True, False, False, False], dtype=bool)

    def __str__(self):
        return f"\tReese weakrock\n\tEi = {var_to_str(self.Ei)}kPa\n\tqu = {var_to_str(self.qu)}kPa\n\tRQD = {var_to_str(self.RQD)}%"

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

        # define Ei
        Ei_t, Ei_b = from_list2x_parse_top_bottom(self.Ei)
        Ei = Ei_t + (Ei_b - Ei_t) * depth_from_top_of_layer / layer_height

        # define qu
        qu_t, qu_b = from_list2x_parse_top_bottom(self.qu)
        qu = qu_t + (qu_b - qu_t) * depth_from_top_of_layer / layer_height

        y, p = py_curves.reese_weakrock(
            Ei=Ei,
            xr=(X + self.ztop),
            RQD=self.RQD,
            qu=qu,
            D=D,
            k=self.k,
            output_length=output_length,
        )

        # parse multipliers and apply results
        y_mult = self.y_multiplier if isinstance(self.y_multiplier, float) else self.y_multiplier(X)
        p_mult = self.p_multiplier if isinstance(self.p_multiplier, float) else self.p_multiplier(X)

        return y * y_mult, p * p_mult


class Custom_pisa_sand(LateralModel):
    """A class to establish a sand model as per PISA framework with custom normalized parameters.

    Parameters
    ----------
    G0: float or list[top_value, bottom_value]
        Small-strain shear modulus [unit: kPa]
    py_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the distributed lateral springs
    py_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the distributed lateral springs, must be greater than or equal to 0 and less than or equal to 1.
    py_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the curve of the distributed lateral springs
    py_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the curve of the distributed lateral springs
    mt_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the distributed rotational springs
    mt_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the distributed rotational springs, must be greater than or equal to 0 and less than or equal to 1.
    mt_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the curve of the distributed rotational springs
    mt_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the curve of the distributed rotational springs
    Hb_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the base shear spring
    Hb_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the base shear spring, must be greater than or equal to 0 and less than or equal to 1.
    Hb_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the base shear spring
    Hb_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the curve of the base shear spring
    Mb_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the base rotational spring
    Mb_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the base rotational spring, must be greater than or equal to 0 and less than or equal to 1.
    Mb_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the base rotational spring
    Mb_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the base rotational spring
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
    :py:func:`openpile.utils.py_curves.custom_pisa_sand`, :py:func:`openpile.utils.mt_curves.custom_pisa_sand`,
    :py:func:`openpile.utils.Hb_curves.custom_pisa_sand`, :py:func:`openpile.utils.Mb_curves.custom_pisa_sand`

    """

    #: small-strain shear stiffness modulus [kPa]
    G0: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of p-y curve
    py_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of p-y curve, must be greater than or equal to 0 and less than or equal to 1.
    py_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of p-y curve
    py_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of p-y curve
    py_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of m-t curve
    mt_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of m-t curve, must be greater than or equal to 0 and less than 1.
    mt_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of m-t curve
    mt_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of m-t curve
    mt_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of Hb-y curve
    Hb_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of Hb-y curve, must be greater than or equal to 0 and less than 1.
    Hb_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of Hb-y curve
    Hb_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of Hb-y curve
    Hb_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of Mb-y curve
    Mb_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of Mb-y curve, must be greater than or equal to 0 and less than 1.
    Mb_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of Mb-y curve
    Mb_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of Mb-y curve
    Mb_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: m-multiplier
    m_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature: ClassVar[np.array] = np.array([True, True, True, True], dtype=bool)

    def __str__(self):
        return f"\tCustom PISA sand\n\tG0 = {round(self.G0/1000,1)} MPa"

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

        y, p = py_curves.custom_pisa_sand(
            sig=sig,
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.py_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.py_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.py_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.py_Y, depth_from_top_of_layer, layer_height, X),
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

        y, Hb = Hb_curves.custom_pisa_sand(
            sig=sig,
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.Hb_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.Hb_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.Hb_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.Hb_Y, depth_from_top_of_layer, layer_height, X),
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

        _, p_array = py_curves.custom_pisa_sand(
            sig=sig,
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.py_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.py_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.py_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.py_Y, depth_from_top_of_layer, layer_height, X),
            output_length=output_length,
        )

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        for count, p_iter in enumerate(p_array):
            t[count, :], m[count] = mt_curves.custom_pisa_sand(
                sig=sig,
                G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
                D=D,
                X_ult=get_value_at_current_depth(
                    self.mt_X, depth_from_top_of_layer, layer_height, X
                ),
                n=get_value_at_current_depth(self.mt_n, depth_from_top_of_layer, layer_height, X),
                k=get_value_at_current_depth(self.mt_k, depth_from_top_of_layer, layer_height, X),
                Y_ult=get_value_at_current_depth(
                    self.mt_Y, depth_from_top_of_layer, layer_height, X
                ),
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

        y, Mb = Mb_curves.custom_pisa_sand(
            sig=sig,
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.Mb_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.Mb_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.Mb_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.Mb_Y, depth_from_top_of_layer, layer_height, X),
            output_length=output_length,
        )

        return y, Mb


class Custom_pisa_clay(LateralModel):
    """A class to establish a clay model as per PISA framework with custom normalized parameters.

    Parameters
    ----------
    Su: float or list[top_value, bottom_value]
        Undrained shear strength [unit: kPa]
    G0: float or list[top_value, bottom_value]
        Small-strain shear modulus [unit: kPa]
    py_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the distributed lateral springs
    py_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the distributed lateral springs, must be greater than or equal to 0 and less than or equal to 1.
    py_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the curve of the distributed lateral springs
    py_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the curve of the distributed lateral springs
    mt_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the distributed rotational springs
    mt_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the distributed rotational springs, must be greater than or equal to 0 and less than or equal to 1.
    mt_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the curve of the distributed rotational springs
    mt_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the curve of the distributed rotational springs
    Hb_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the base shear spring
    Hb_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the base shear spring, must be greater than or equal to 0 and less than or equal to 1.
    Hb_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the base shear spring
    Hb_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the curve of the base shear spring
    Mb_X: float, or list with top and bottom values, or function taking the depth as argument
        normalized displacement at ultimate resistance of the base rotational spring
    Mb_n: float, or list with top and bottom values, or function taking the depth as argument
        normalized curvature of the conic function of the base rotational spring, must be greater than or equal to 0 and less than or equal to 1.
    Mb_k: float, or list with top and bottom values, or function taking the depth as argument
        normalized initial stiffness of the base rotational spring
    Mb_Y: float, or list with top and bottom values, or function taking the depth as argument
        normalized maximum resistance of the base rotational spring
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
    :py:func:`openpile.utils.py_curves.custom_pisa_sand`, :py:func:`openpile.utils.mt_curves.custom_pisa_sand`,
    :py:func:`openpile.utils.Hb_curves.custom_pisa_sand`, :py:func:`openpile.utils.Mb_curves.custom_pisa_sand`

    """

    #: undrained shear strength [kPa]
    Su: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: small-strain shear stiffness modulus [kPa]
    G0: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of p-y curve
    py_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of p-y curve, must be greater than or equal to 0 and less than or equal to 1.
    py_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of p-y curve
    py_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of p-y curve
    py_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of m-t curve
    mt_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of m-t curve, must be greater than or equal to 0 and less than 1.
    mt_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of m-t curve
    mt_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of m-t curve
    mt_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of Hb-y curve
    Hb_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of Hb-y curve, must be greater than or equal to 0 and less than 1.
    Hb_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of Hb-y curve
    Hb_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of Hb-y curve
    Hb_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized displacement at ultimate resistance of Mb-y curve
    Mb_X: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized curvature of the conic function of Mb-y curve, must be greater than or equal to 0 and less than 1.
    Mb_n: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Annotated[List[confloat(ge=0.0, lt=1.0)], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized initial stiffness of the curve  of Mb-y curve
    Mb_k: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: normalized maximum resistance of the curve of Mb-y curve
    Mb_Y: Union[
        Annotated[float, Field(gt=0.0)],
        Annotated[List[PositiveFloat], Field(min_length=1, max_length=2)],
        Callable[[float], float],
    ]
    #: p-multiplier
    p_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: y-multiplier
    y_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0
    #: m-multiplier
    m_multiplier: Union[Callable[[float], float], Annotated[float, Field(ge=0.0)]] = 1.0
    #: t-multiplier
    t_multiplier: Union[Callable[[float], float], Annotated[float, Field(gt=0.0)]] = 1.0

    # spring signature which tells that API sand only has p-y curves
    # signature if of the form [p-y:True, Hb:False, m-t:False, Mb:False]
    spring_signature: ClassVar[np.array] = np.array([True, True, True, True], dtype=bool)

    def __str__(self):
        return f"\tCustom PISA sand\n\tG0 = {round(self.G0/1000,1)} MPa"

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

        y, p = py_curves.custom_pisa_clay(
            Su=get_value_at_current_depth(self.Su, depth_from_top_of_layer, layer_height, X),
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.py_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.py_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.py_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.py_Y, depth_from_top_of_layer, layer_height, X),
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

        y, Hb = Hb_curves.custom_pisa_clay(
            Su=get_value_at_current_depth(self.Su, depth_from_top_of_layer, layer_height, X),
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.Hb_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.Hb_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.Hb_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.Hb_Y, depth_from_top_of_layer, layer_height, X),
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

        _, p_array = py_curves.custom_pisa_clay(
            Su=get_value_at_current_depth(self.Su, depth_from_top_of_layer, layer_height, X),
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.py_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.py_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.py_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.py_Y, depth_from_top_of_layer, layer_height, X),
            output_length=output_length,
        )

        m = np.zeros((output_length, output_length), dtype=np.float32)
        t = np.zeros((output_length, output_length), dtype=np.float32)

        for count, _ in enumerate(p_array):
            t[count, :], m[count] = mt_curves.custom_pisa_clay(
                Su=get_value_at_current_depth(self.Su, depth_from_top_of_layer, layer_height, X),
                G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
                D=D,
                X_ult=get_value_at_current_depth(
                    self.mt_X, depth_from_top_of_layer, layer_height, X
                ),
                n=get_value_at_current_depth(self.mt_n, depth_from_top_of_layer, layer_height, X),
                k=get_value_at_current_depth(self.mt_k, depth_from_top_of_layer, layer_height, X),
                Y_ult=get_value_at_current_depth(
                    self.mt_Y, depth_from_top_of_layer, layer_height, X
                ),
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

        y, Mb = Mb_curves.custom_pisa_clay(
            Su=get_value_at_current_depth(self.Su, depth_from_top_of_layer, layer_height, X),
            G0=get_value_at_current_depth(self.G0, depth_from_top_of_layer, layer_height, X),
            D=D,
            X_ult=get_value_at_current_depth(self.Mb_X, depth_from_top_of_layer, layer_height, X),
            n=get_value_at_current_depth(self.Mb_n, depth_from_top_of_layer, layer_height, X),
            k=get_value_at_current_depth(self.Mb_k, depth_from_top_of_layer, layer_height, X),
            Y_ult=get_value_at_current_depth(self.Mb_Y, depth_from_top_of_layer, layer_height, X),
            output_length=output_length,
        )

        return y, Mb
