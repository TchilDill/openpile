"""
`Mb_curves` module
==================

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

from openpile.core.misc import conic

# SPRING FUNCTIONS --------------------------------------------
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
    Create the base moment springs from the PISA clay formulation
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
        Pile length [unit: m]
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        Mb vector [unit: kN]
    1darray
        t vector of length [unit: rad]

    """

    # Cowden clay parameters
    k_m1 = 0.2146
    k_m2 = -0.002132
    n_m1 = 1.079
    n_m2 = -0.1087
    m_m1 = 0.8192
    m_m2 = -0.08588
    psi_u = 173.1

    # Depth variation parameters
    k = k_m1 + k_m2 * L / D
    n = n_m1 + n_m2 * L / D
    m_max = m_m1 + m_m2 * L / D
    psi_max = psi_u

    # calculate normsalised conic function
    t, m = conic(psi_max, n, k, m_max, output_length)

    # return non-normalised curve
    return t * (Su / G0), m * (Su * D**3)


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
    Create the base moment spring from the PISA sand
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
    D : float
        Pile diameter [unit: m]
    L : float
        Embedded pile length [unit: m]
    output_length : int, optional
        Number of datapoints in the curve, by default 20

    Returns
    -------
    1darray
        Mb vector [unit: kN]
    1darray
        t vector of length [unit: rad]
    """
    # correct relative density for decimal value
    Dr = Dr / 100

    # Generalised Dunkirk Sand Model parameters
    k_m1 = 0.3515
    k_m2 = 0.00
    n_m = 0.3 + 0.4986 * Dr
    m_u1 = 0.09981 + 0.3710 * Dr
    m_u2 = 0.01998 - 0.09041 * Dr
    psi_u = 44.89

    # Depth variation parameters
    k = k_m1 + k_m2 * L / D
    n = n_m
    m_max = m_u1 + m_u2 * L / D
    psi_max = psi_u

    # calculate normsalised conic function
    t, m = conic(psi_max, n, k, m_max, output_length)

    # return non-normalised curve
    return t * (sig / G0), m * (sig * D**3)
