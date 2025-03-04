"""
`Hb_curves` module
------------------

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

from openpile.core.misc import conic


@njit(cache=True)
def bothkennar_clay(
    X: float,
    Su: float,
    G0: float,
    D: float,
    L: float,
    output_length: int = 20,
):
    """
    Creates the base shear spring from the PISA clay formulation
    published by Burd et al 2020 (see [BABH20]_) and calibrated based on Bothkennar clay
    response (a normally consolidated soft clay).

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
        y vector [unit: m]
    1darray
        Hb vector [unit: kN]

    """

    # Generalised Bothkennar clay Model parameters
    v_hu1 = 291.5
    v_hu2 = 0.00
    k_h1 = 3.008
    k_h2 = -0.2701
    n_h1 = 0.3113
    n_h2 = 0.04263
    p_u1 = 0.5279
    p_u2 = 0.06864

    # Depth variation parameters
    v_max = v_hu1 + v_hu2 * L / D
    k = k_h1 + k_h2 * L / D
    n = n_h1 + n_h2 * L / D
    p_max = p_u1 + p_u2 * L / D

    # calculate normsalised conic function
    y, p = conic(v_max, n, k, p_max, output_length)

    # return non-normalised curve
    return y * (Su * D / G0), p * (Su * D**2)


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
        y vector [unit: m]
    1darray
        Hb vector [unit: kN]
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
        y vector [unit: m]
    1darray
        Hb vector [unit: kN]

    """

    # Generalised Cowden clay Model parameters
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


@njit(cache=True)
def custom_pisa_sand(
    sig: float,
    G0: float,
    D: float,
    X_ult: float,
    n: float,
    k: float,
    Y_ult: float,
    output_length: int = 20,
):
    """
    Creates a base shear spring with the PISA sand formulation and custom user inputs.

    Parameters
    ----------
    sig : float
        vertical/overburden effective stress [unit: kPa]
    Dr : float
        Sand relative density Value must be between 0 and 100 [unit: -]
    G0 : float
        Small-strain shear modulus [unit: kPa]
    D : float
        Pile diameter [unit: m]
    X_ult : float
        Normalized displacement at maximum strength
    k : float
        Normalized stiffness parameter
    n : float
        Normalized curvature parameter, must be between 0 and 1
    Y_ult : float
        Normalized maximum strength parameter
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        Hb vector [unit: kN]
    """

    # calculate normsalised conic function
    x, y = conic(X_ult, n, k, Y_ult, output_length)

    # return non-normalised curve
    return x * (sig * D / G0), y * (sig * D**2)


@njit(cache=True)
def custom_pisa_clay(
    Su: float,
    G0: float,
    D: float,
    X_ult: float,
    n: float,
    k: float,
    Y_ult: float,
    output_length: int = 20,
):
    """
    Creates a base shear spring with the PISA clay formulation and custom user inputs.

    Parameters
    ----------
    Su : float
        Undrained shear strength [unit: kPa]
    G0 : float
        Small-strain shear modulus [unit: kPa]
    D : float
        Pile diameter [unit: m]
    X_ult : float
        Normalized displacement at maximum strength
    k : float
        Normalized stiffness parameter
    n : float
        Normalized curvature parameter, must be between 0 and 1
    Y_ult : float
        Normalized maximum strength parameter
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        Hb vector [unit: kN]

    """
    # calculate normsalised conic function
    x, y = conic(X_ult, n, k, Y_ult, output_length)

    # return non-normalised curve
    return x * (Su * D / G0), y * (Su * D**2)
