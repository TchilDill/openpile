"""
`tz_curves` module
------------------


"""

# Import libraries
from openpile.utils.misc import _fmax_api_clay, _fmax_api_sand

import math as m
import numpy as np
from numba import njit, prange
from random import random

# Kraft et al (1989) formulation aid to ease up implementation
def kraft_modification(
    fmax: float,
    D: float,
    G0: float,
    residual: float = 1.0,
    tensile_factor: float = 1.0,
    RF: float = 0.9,
    zif: float = 10.0,
    output_length: int = 15,
):
    """
    Creates the t-z curve from relevant input with the Kraft et al (1981) formulation.

    Parameters
    ----------
    fmax: float
        unit skin friction [unit: kPa]
    D: float
        Pile diameter [unit: m]
    G0: float
        small-strain stiffness [unit: kPa]
    residual: float
        residual strength after peak strength
    tensile_factor: float
        strength factor for negative values of the curve
    RF: float
        curve fitting factor as per Kraft et al. (1981), by default 0.9
    zif: float
        dimensionless zone of influence as per Kraft et al (1981) that corresponds to the radius of the zone of influence divided by the pile radius, by default 10.0
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        t vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]

    """

    if output_length < 15:
        output_length = 15
    if output_length % 2 == 0:
        output_length += 1

    half_length = round(output_length / 2) - 2

    # define t till tmax
    tpos = np.linspace(0, fmax, half_length)
    # define z till zmax
    zpos = tpos * 0.5 * D / G0 * np.log((zif - RF * tpos / fmax) / (1 - RF * tpos / fmax))
    # define z where t = tmax, a.k.a zmax here
    zmax = fmax * D / (2 * G0) * m.log((zif - RF) / (1 - RF))
    # define z where z=tres, which is zmax + 5mm
    zres = zmax + 0.005

    z = np.append(zpos, [zres, zres + 0.005])
    t = np.append(tpos, [residual * fmax, residual * fmax])

    return np.append(-z[-1:0:-1], z), np.append(-t[-1:0:-1] * tensile_factor, t)


# API clay backbone fct
def backbone_api_clay(
    residual: float = 0.9,
    tensile_factor: float = 1.0,
    output_length: int = 15,
):
    """
    Creates the API clay t-z curve backbone (i.e. normalized with strength and diameter) from relevant input as per [API2000]_.

    Parameters
    ----------
    D: float
        Pile width [unit: m]
    residual: float
        residual strength after peak strength, according to API-RP-2A,
        this value is between 0.7 and 0.9, default to 0.9
    tensile_factor: float
        strength factor for negative values of the curve
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        t vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]


    """
    # cannot have less than 15
    if output_length < 15:
        output_length = 15

    # piecewise function
    zlist = [0.0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, 0.0300]
    tlist = [0.0, 0.3, 0.5, 0.75, 0.90, 1.00, residual, residual]

    # determine z vector
    z = np.array(zlist, dtype=np.float32)
    z = np.concatenate((-z[-1:0:-1], z))
    # define t vector
    t = np.array(tlist, dtype=np.float32)
    t = np.concatenate((-tensile_factor * t[-1:0:-1], t))

    add_values = output_length - len(z)
    add_z_values = np.zeros((add_values), dtype=np.float32)
    add_t_values = np.zeros((add_values), dtype=np.float32)

    for i in range(add_values):
        add_z_values[i] = 0.02 + random() * 0.01
        add_t_values[i] = residual

    z = np.append(z, add_z_values)
    t = np.append(t, add_t_values)

    z = np.sort(z)
    z_id_sorted = np.argsort(z)

    t = t[z_id_sorted]

    return z, t


# API clay function
def api_clay(
    sig: float,
    Su: float,
    D: float,
    alpha_limit: float = 1.0,
    residual: float = 0.9,
    tensile_factor: float = 1.0,
    output_length: int = 15,
):
    r"""
    Creates the API clay t-z curve from relevant input as per [API2000]_.

    Parameters
    ----------
    sig: float
        Vertical effective stress [unit: kPa]
    Su : float
        Undrained shear strength [unit: kPa]
    D: float
        Pile width [unit: m]
    alpha_limit: float
        maximum value of the alpha parameter, default to 1.0
    residual: float
        residual strength after peak strength, according to API-RP-2A,
        this value is between 0.7 and 0.9, default to 0.9
    tensile_factor: float
        strength factor for negative values of the curve
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        t vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]

    Notes
    -----

    The backbone curve is computed via the piecewise formulation by [API2000]_ or through
    Kraft's formulation that captures small-strain characteristics of the soil [KrRK81]_ in the backbone curve.

    .. note::
        Kraft's formulation is accessed by the user by stipulating the small-strain shear
        stiffness of the soil, :math:`G_0`


    The API guidelines describe the axial soil springs in a manner that accounts for the undrained shear strength of the clay.
    The springs are characterized as follows:

    1. **Unit Skin Friction** :math:`f_s`: For clay, this is based on the undrained shear strength :math:`S_u` of the soil and a factor that accounts for the adhesion between the clay and the pile.

        .. math::

            f_s = \alpha \cdot S_u < f_{s,\text{max}}

        where:

        - :math:`\alpha` is the adhesion factor, which depends on the type of clay and the pile material.
          It typically ranges from 0.5 to 1.0 for soft clays and 0.3 to 0.6 for stiff clays.
          As per the API guidelines, this adhesion factor can be calculated as:
        - :math:`S_u` is the undrained shear strength of the clay.
        - Limit Skin Friction :math:`f_{s,\text{max}}` is the maximum unit skin friction for clay,
          which can be directly related to the undrained shear strength and the adhesion factor.
          In general, the limit skin friction is set to the undrained shear strength.

    2. A backbone curve computed via the piecewise formulation seen in [API2000]_.

    """
    # cannot have less than 15
    if output_length < 15:
        output_length = 15

    # unit skin friction
    f = _fmax_api_clay(sig, Su, alpha_limit)

    # call backbone curve
    z, t = backbone_api_clay(residual, tensile_factor, output_length)

    return z * D, t * f


@njit(cache=True)
def api_clay_kraft(
    sig: float,
    Su: float,
    D: float,
    G0: float,
    alpha_limit: float = 1.0,
    residual: float = 1.0,
    tensile_factor: float = 1.0,
    RF: float = 0.9,
    zif: float = 10.0,
    output_length: int = 15,
):
    r"""
    Creates the API clay t-z curve (see [API2000]_) with the Kraft et al (1981) formulation (see [KrRK81]_).

    Parameters
    ----------
    sig: float
        Vertical effective stress [unit: kPa]
    Su : float
        Undrained shear strength [unit: kPa]
    D: float
        Pile diameter [unit: m]
    G0: float
        small-strain stiffness [unit: kPa]
    alpha_limit: float
        limit of the alpha parameter, by default 1.0
    residual: float
        residual strength after peak strength, by default 1.0
    tensile_factor: float
        strength factor for negative values of the curve, by default 1.0
    RF: float
        curve fitting factor as per Kraft et al. (1981), by default 0.9
    zif: float
        dimensionless zone of influence as per Kraft et al (1981) that corresponds to the radius of the zone of influence divided by the pile radius, by default 10.0
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        t vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]

    See also
    --------
    :py:func:`openpile.utils.tz_curves.api_clay`

    Notes
    -----

    The backbone curve is computed via the piecewise formulation
    by [API2000]_ or through
    Kraft's formulation that captures small-strain
    characteristics of the soil [KrRK81]_ in the backbone curve.

    .. note::
        Kraft's formulation is accessed by the user by stipulating the small-strain shear
        stiffness of the soil, :math:`G_0`


    For clay, the API guidelines describe the axial soil springs in a manner that accounts for the undrained shear strength of the clay.
    The springs are characterized as follows:

    1. **Unit Skin Friction** :math:`f_s`: For clay, this is based on the undrained shear strength :math:`S_u` of the soil and a factor that accounts for the adhesion between the clay and the pile.

        .. math::

            f_s = \alpha \cdot S_u < f_{s,\text{max}}

        where:

        - :math:`\alpha` is the adhesion factor, which depends on the type of clay and the pile material.
          It typically ranges from 0.5 to 1.0 for soft clays and 0.3 to 0.6 for stiff clays.
          As per the API guidelines, this adhesion factor can be calculated as:
        - :math:`S_u` is the undrained shear strength of the clay.
        - Limit Skin Friction :math:`f_{s,\text{max}}` is the maximum unit skin friction for clay,
          which can be directly related to the undrained shear strength and the adhesion factor.
          In general, the limit skin friction is set to the undrained shear strength.

    2. A backbone curve computed via the piecewise formulation seen in [API2000]_.

    """

    return kraft_modification(
        _fmax_api_clay(sig, Su, alpha_limit),
        D,
        G0,
        residual,
        tensile_factor,
        RF,
        zif,
        output_length,
    )


# API sand function
def backbone_api_sand(
    tensile_factor: float = 1.0,
    output_length: int = 7,
):
    """
    Creates the API sand t-z curve (see [API2000]_).

    Parameters
    ----------
    tensile_factor: float
        strength factor for negative values of the curve
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 7, by default 7

    Returns
    -------
    numpy 1darray
        t vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]


    """
    # cannot have less than 7
    if output_length < 15:
        output_length = 15

    # piecewise function
    zlist = [0.0, 0.0254, 0.03, 0.032, 0.034, 0.036, 0.038, 0.04]
    tlist = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # determine z vector
    z = np.array(zlist, dtype=np.float32)
    z = np.concatenate((-z[-1:0:-1], z))
    # define t vector
    t = np.array(tlist, dtype=np.float32)
    t = np.concatenate((-tensile_factor * t[-1:0:-1], t))

    add_values = output_length - len(z)
    add_z_values = np.zeros((add_values), dtype=np.float32)
    add_t_values = np.zeros((add_values), dtype=np.float32)

    for i in range(add_values):
        add_z_values[i] = 0.038 + random() * 0.01
        add_t_values[i] = 1.0

    z = np.append(z, add_z_values)
    t = np.append(t, add_t_values)

    z = np.sort(z)
    z_id_sorted = np.argsort(z)

    t = t[z_id_sorted]

    return z, t


# API sand function
def api_sand(
    sig: float,
    delta: float,
    K: float = 0.8,
    tensile_factor: float = 1.0,
    output_length: int = 7,
):
    r"""
    Creates the API sand t-z curve (see [API2000]_).

    Parameters
    ----------
    sig: float
        Vertical effective stress [unit: kPa]
    delta: float
        interface friction angle [unit: degrees]
    K: float
        coefficient of lateral pressure (0.8 for open-ended piles and 1.0 for cloased-ended)
    tensile_factor: float
        strength factor for negative values of the curve
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 7, by default 7

    Returns
    -------
    numpy 1darray
        t vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]

    Notes
    -----

    The backbone curve is computed via the piecewise formulation
    by [API2000]_ or through
    Kraft's formulation that captures small-strain
    characteristics of the soil [KrRK81]_ in the backbone curve.

    .. note::
        Kraft's formulation is accessed by the user by stipulating the small-strain shear
        stiffness of the soil, :math:`G_0`

    The API guidelines provide methods to estimate the resistance offered by sandy soils along the pile.
    These springs are defined based on the following considerations:

    1. **Unit Skin Friction** :math:`f_s`: This is the frictional resistance per unit area along the pile shaft. It depends on the effective overburden pressure and the soil-pile interface properties.

        .. math::

        f_s = \sigma_v^\prime \cdot K \cdot tan(\delta) < f_{s,\text{max}}

        where:

        - \sigma_v^\prime is the effective vertical stress at the depth considered.
        - K is the coefficient of horizontal earth pressure (typically ranges from 0.4 to 1.0 for sands).
        - \delta is the angle of friction between the pile and the sand, often taken as a fraction of the soil's internal friction angle :math:`\varphi` (usually :math:`\delta = 0.7 \varphi` to :math:`\varphi`).
        - Limit Skin Friction :math:`f_{s,\text{max}}` is the maximum unit skin friction that can be mobilized. It is typically given by empirical correlations or laboratory tests. The following is assumed in OpenPile:

        .. list-table:: Correlation between interface friction angle and shaft friction
            :header-rows: 0

            * - :math:`\delta` [degrees]
              - 15
              - 20
              - 25
              - 30
              - 35
            * - :math:`f_{s,\texttt{max}}` [kPa]
              - 47.8
              - 67
              - 81.3
              - 95.7
              - 114.8

    2. A backbone curve computed via the piecewise formulation seen in [API2000]_.

    """

    # unit skin friction
    f = _fmax_api_sand(sig, delta, K)

    # call backbone curve
    z, t = backbone_api_sand(tensile_factor, output_length)

    return z, t * f


@njit(cache=True)
def api_sand_kraft(
    sig: float,
    delta: float,
    D: float,
    G0: float,
    K: float = 0.8,
    residual: float = 1.0,
    tensile_factor: float = 1.0,
    RF: float = 0.9,
    zif: float = 10.0,
    output_length: int = 15,
):
    r"""
    Creates the API sand t-z curve (see [API2000]_) with the Kraft et al (1981) formulation (see [KrRK81]_).

    Parameters
    ----------
    sig: float
        Vertical effective stress [unit: kPa]
    delta: float
        interface friction angle [unit: degrees]
    D: float
        Pile width [unit: m]
    G0: float
        small-strain stiffness [unit: kPa]
    K: float
        coefficient of lateral pressure (0.8 for open-ended piles and 1.0 for cloased-ended), by default 0.8
    residual: float
        residual strength after peak strength, by default 1.0
    tensile_factor: float
        strength factor for negative values of the curve, by default 1.0
    RF: float
        curve fitting factor as per Kraft et al. (1981), by default 0.9
    zif: float
        dimensionless zone of influence as per Kraft et al (1981) that corresponds to the radius of the zone of influence divided by the pile radius, by default 10.0
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        t vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]

    See also
    --------
    :py:func:`openpile.utils.tz_curves.api_sand`

    Notes
    -----

    The backbone curve is computed via the piecewise formulation
    by [API2000]_ or through
    Kraft's formulation that captures small-strain
    characteristics of the soil [KrRK81]_ in the backbone curve.

    .. note::
        Kraft's formulation is accessed by the user by stipulating the small-strain shear
        stiffness of the soil, :math:`G_0`

    The API guidelines provide methods to estimate the resistance offered by sandy soils along the pile.
    These springs are defined based on the following considerations:

    1. **Unit Skin Friction** :math:`f_s`: This is the frictional resistance per unit area along the pile shaft. It depends on the effective overburden pressure and the soil-pile interface properties.

        .. math::

        f_s = \sigma_v^\prime \cdot K \cdot tan(\delta) < f_{s,\text{max}}

        where:

        - \sigma_v^\prime is the effective vertical stress at the depth considered.
        - K is the coefficient of horizontal earth pressure (typically ranges from 0.4 to 1.0 for sands).
        - \delta is the angle of friction between the pile and the sand, often taken as a fraction of the soil's internal friction angle :math:`\varphi` (usually :math:`\delta = 0.7 \varphi` to :math:`\varphi`).
        - Limit Skin Friction :math:`f_{s,\text{max}}` is the maximum unit skin friction that can be mobilized. It is typically given by empirical correlations or laboratory tests. The following is assumed in OpenPile:

        .. list-table:: Correlation between interface friction angle and shaft friction
            :header-rows: 0

            * - :math:`\delta` [degrees]
              - 15
              - 20
              - 25
              - 30
              - 35
            * - :math:`f_{s,\texttt{max}}` [kPa]
              - 47.8
              - 67
              - 81.3
              - 95.7
              - 114.8

    2. A backbone curve computed via the piecewise formulation seen in [API2000]_.

    """
    return kraft_modification(
        _fmax_api_sand(sig, delta, K), D, G0, residual, tensile_factor, RF, zif, output_length
    )
