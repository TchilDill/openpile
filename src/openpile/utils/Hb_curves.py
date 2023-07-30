"""
`Hb_curves` module
==================

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

from openpile.core.misc import conic


@njit(cache=True)
def dunkirk_sand(
    sig: float,
    X: float,
    Dr: float,
    G0: float,
    D: float,
    L: float,
    output_length: int = 20,
):
    """
    Creates the base shear spring from the PISA sand formulation
    published by Burd et al (2020) (see [BTZA20]_).
    Also called the General Dunkirk Sand Model (GDSM).

    Parameters
    ----------
    sig : float
        vertical/overburden effective stress [unit: kPa]
    X : float
        Depth below ground level [unit: m]
    Dr : float
        Sand relative density Value must be between 0 and 100 [unit: -]
    G0 : float
        Small-strain shear modulus [unit: kPa]
    D : float
        Pile diameter [unit: m]
    L : float
        Embedded pile length [unit: m]
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        Hb vector [unit: kN]
    1darray
        y vector [unit: m]
    """
    # correct relative density for decimal value
    Dr = Dr / 100

    # Generalised Dunkirk Sand Model parameters
    v_hu1 = 0.5150 + 2.883 * Dr
    v_hu2 = 0.1695 - 0.7018 * Dr
    k_h1 = 6.505 - 2.985 * Dr
    k_h2 = -0.007969 - 0.4299 * Dr
    n_h1 = 0.09978 + 0.7974 * Dr
    n_h2 = 0.004994 - 0.07005 * Dr
    p_u1 = 0.09952 + 0.7996 * Dr
    p_u2 = 0.03988 - 0.1606 * Dr

    # Depth variation parameters
    v_max = v_hu1 + v_hu2 * L / D
    k = k_h1 + k_h2 * L / D
    n = n_h1 + n_h2 * L / D
    p_max = p_u1 + p_u2 * L / D

    # calculate normsalised conic function
    y, p = conic(v_max, n, k, p_max, output_length)

    # return non-normalised curve
    return y * (sig * D / G0), p * (sig * D**2)


@njit(cache=True)
def cowden_clay(
    X: float,
    Su: float,
    G0: float,
    D: float,
    L: float,
    output_length: int = 20,
):
    """
    Creates the base shear spring from the PISA clay formulation
    published by Byrne et al 2020 (see [BHBG20]_) and calibrated based pile
    load tests at Cowden (north east coast of England).

    Parameters
    ----------
    X : float
        Depth below ground level [unit: m]
    Su : float
        Undrained shear strength [unit: kPa]
    G0 : float
        Small-strain shear modulus [unit: kPa]
    D : float
        Pile diameter [unit: m]
    L : float
        Embedded pile length [unit: m]
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        Hb vector [unit: kN]
    1darray
        y vector [unit: m]

    References
    ----------
    .. [1] Byrne, B. W., Houlsby, G. T., Burd, H. J., Gavin, K. G., Igoe, D. J. P., Jardine,
           R. J., Martin, C. M., McAdam, R. A., Potts, D. M., Taborda, D. M. G. & Zdravkovic ́,
           L. (2020). PISA design model for monopiles for offshore wind turbines: application
           to a stiff glacial clay till. Géotechnique, https://doi.org/10.1680/ jgeot.18.P.255.

    """

    # Generalised Dunkirk Sand Model parameters
    v_hu1 = 235.7
    v_hu2 = 0.00
    k_h1 = 2.717
    k_h2 = -0.3575
    n_h1 = 0.8793
    n_h2 = -0.03150
    p_u1 = 0.4038
    p_u2 = 0.04812

    # Depth variation parameters
    v_max = v_hu1 + v_hu2 * L / D
    k = k_h1 + k_h2 * L / D
    n = n_h1 + n_h2 * L / D
    p_max = p_u1 + p_u2 * L / D

    # calculate normsalised conic function
    y, p = conic(v_max, n, k, p_max, output_length)

    # return non-normalised curve
    return y * (Su * D / G0), p * (Su * D**2)
