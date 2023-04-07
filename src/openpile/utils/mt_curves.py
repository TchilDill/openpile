

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
    output_length: int = 20,  
):
    """
    Create the rotational springs from the PISA clay formulation 
    published by Byrne et al (2020) and calibrated based pile load tests 
    at Cowden (north east coast of England).

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
    ndarray 
        p vector of length [<output length>]
    ndarray 
        y vector of length [<output length>]
    """

    # Cowden clay parameters
    k_m1 = 1.420
    k_m2 = -0.09643
    n_m1 = 0.00
    n_m2 = 0.00
    m_m1 = 0.2899
    m_m2 = -0.04775

    # Depth variation parameters
    k = k_m1 + k_m2 * X/D
    n = n_m1 + n_m2 * X/D
    m_max = m_m1 + m_m2 * X/D
    psi_max = m_max / k

    # calculate normsalised conic function
    t, m = conic(psi_max, n, k, m_max, output_length)

    # return non-normalised curve
    return m*(Su*D**2), t*(Su/G0)


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
    formulation published by Burd et al (2020).
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
    ndarray 
        p vector of length [<output length>]
    ndarray 
        y vector of length [<output length>]
    """
    #correct relative density for decimal value
    Dr = Dr/100

    # correct p value
    p = abs(p)


    # Generalised Dunkirk Sand Model parameters
    k_m1 = 17.00
    k_m2 = 0.00
    n_m = 0.0
    m_u1 = 0.2605
    m_u2 = -0.1989 + 0.2019 * Dr

    # Depth variation parameters
    k = k_m1 + k_m2 * X/D
    n = n_m
    m_max = m_u1 + m_u2 * X/L
    psi_max = m_max / k

    # calculate normsalised conic function
    t, m = conic(psi_max, n, k, m_max, output_length)

    # return non-normalised curve
    return m*(p*D), t*(sig/G0)