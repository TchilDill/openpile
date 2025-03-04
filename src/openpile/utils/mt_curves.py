"""
`mt_curves` module
------------------

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

from openpile.core.misc import conic

# SPRING FUNCTIONS --------------------------------------------
@njit(cache=True)
def bothkennar_clay(
    X: float,
    Su: float,
    G0: float,
    D: float,
    output_length: int = 20,
):
    """
    Create the rotational springs from the PISA clay formulation
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
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        t vector of length [unit: rad]
    1darray
        m vector [unit: kN]

    """

    # Bothkennar clay parameters
    k_m1 = 1.698
    k_m2 = -0.1576
    n_m1 = 0.00
    n_m2 = 0.00
    m_m1 = 0.4862
    m_m2 = -0.05674

    # Depth variation parameters
    k = k_m1 + k_m2 * X / D
    n = n_m1 + n_m2 * X / D
    m_max = m_m1 + m_m2 * X / D
    psi_max = m_max / k

    # calculate normsalised conic function
    t, m = conic(psi_max, n, k, m_max, output_length)

    # return non-normalised curve
    return t * (Su / G0), m * (Su * D**2)


@njit(cache=True)
def cowden_clay(
    X: float,
    Su: float,
    G0: float,
    D: float,
    output_length: int = 20,
):
    """
    Create the rotational springs from the PISA clay formulation
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
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        t vector of length [unit: rad]
    1darray
        m vector [unit: kN]

    """

    # Cowden clay parameters
    k_m1 = 1.420
    k_m2 = -0.09643
    n_m1 = 0.00
    n_m2 = 0.00
    m_m1 = 0.2899
    m_m2 = -0.04775

    # Depth variation parameters
    k = k_m1 + k_m2 * X / D
    n = n_m1 + n_m2 * X / D
    m_max = m_m1 + m_m2 * X / D
    psi_max = m_max / k

    # calculate normsalised conic function
    t, m = conic(psi_max, n, k, m_max, output_length)

    # return non-normalised curve
    return t * (Su / G0), m * (Su * D**2)


@njit(cache=True)
def dunkirk_sand(
    sig: float,
    X: float,
    Dr: float,
    G0: float,
    p: float,
    D: float,
    L: float,
    output_length: int = 20,
):
    """
    Create the rotational springs from the PISA sand
    formulation published by Burd et al (2020) (see [BTZA20]_).
    Also called the General Dunkirk Sand Model (GDSM)

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
    p : float
        radial stress computed via p-y curves [unit: kN/m]
    D : float
        Pile diameter [unit: m]
    L : float
        Embedded pile length [unit: m]
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        t vector of length [unit: rad]
    1darray
        m vector [unit: kN]
    """
    # correct relative density for decimal value
    Dr = Dr / 100

    # correct p value
    p = abs(p)

    # Generalised Dunkirk Sand Model parameters
    k_m1 = 17.00
    k_m2 = 0.00
    n_m = 0.0
    m_u1 = 0.2605
    m_u2 = -0.1989 + 0.2019 * Dr

    # Depth variation parameters
    k = k_m1 + k_m2 * X / D
    n = n_m
    m_max = m_u1 + m_u2 * X / L
    psi_max = m_max / k

    # calculate normsalised conic function
    t, m = conic(psi_max, n, k, m_max, output_length)

    # return non-normalised curve
    return t * (sig / G0), m * (p * D)


@njit(cache=True)
def custom_pisa_sand(
    sig: float,
    G0: float,
    p: float,
    D: float,
    X_ult: float,
    n: float,
    k: float,
    Y_ult: float,
    output_length: int = 20,
):
    """Creates a rotational spring with the PISA sand formulation and custom user inputs.

    Parameters
    ----------
    sig : float
        vertical/overburden effective stress [unit: kPa]
    G0 : float
        Small-strain shear modulus [unit: kPa]
    p : float
        Radial stress computed via p-y curves [unit: kN/m]
    D : float
        Pile diameter [m]
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
        p vector [unit: kN/m]
    """

    # calculate normsalised conic function
    x, y = conic(X_ult, n, k, Y_ult, output_length)

    # return non-normalised curve
    return x * (sig / G0), y * (p * D)


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
    Creates a rotational spring with the PISA clay formulation and custom user inputs.


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
        p vector [unit: kN/m]

    """
    # calculate normsalised conic function
    x, y = conic(X_ult, n, k, Y_ult, output_length)

    # return non-normalised curve
    return x * (Su / G0), y * (Su * D**2)
