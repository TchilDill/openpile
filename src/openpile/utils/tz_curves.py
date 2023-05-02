"""
`tz_curves` module
==================

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

# SPRING FUNCTIONS --------------------------------------------

# API clay function
@njit(cache=True)
def api_clay(
    sig: float,
    Su: float,
    D: float,
    residual: float = 0.9,
    tensile_factor: float = 1.0,
    output_length: int = 15,
):
    """
    Creates the API clay t-z curve from relevant input.

    Parameters
    ----------
    sig: float
        Vertical effective stress [unit: kPa]
    Su : float
        Undrained shear strength [unit: kPa]
    D: float
        Pile diameter [unit: m]
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

    # important variables
    if sig == 0.0:
        psi = Su / 0.001
    else:
        psi = Su / sig

    if psi > 1.0:
        alpha = min(0.5 * psi ** (-0.25), 1.0)
    else:
        alpha = min(0.5 * psi ** (-0.5), 1.0)

    # Unit skin friction [kPa]
    f = alpha * Su

    # piecewise function
    zlist = [0.0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, 0.0300]
    tlist = [0.0, 0.3, 0.5, 0.75, 0.90, 1.00, residual, residual]

    # determine z vector
    z = np.array(zlist, dtype=np.float32) * D
    z = np.concatenate((-z[-1:0:-1], z))
    # define t vector
    t = np.array(tlist, dtype=np.float32) * f
    t = np.concatenate((-tensile_factor * t[-1:0:-1], t))

    add_values = output_length - len(z)
    add_z_values = np.zeros((add_values), dtype=np.float32)
    add_t_values = np.zeros((add_values), dtype=np.float32)

    for i in range(add_values):
        add_z_values[i] = (0.02 + random() * 0.01) * D
        add_t_values[i] = residual * f

    z = np.append(z, add_z_values)
    t = np.append(t, add_t_values)

    z = np.sort(z)
    z_id_sorted = np.argsort(z)

    t = t[z_id_sorted]

    return t, z


# API sand function
@njit(cache=True)
def api_sand(
    sig: float,
    delta: float,
    K: float = 0.8,
    tensile_factor: float = 1.0,
    output_length: int = 7,
):
    """
    Creates the API sand t-z curve from relevant input.

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
    """
    # cannot have less than 7
    if output_length < 7:
        output_length = 7

    # important variables
    delta_table = np.array([0, 15, 20, 25, 30, 35, 100], dtype=np.float32)
    fs_max_table = np.array([47.8, 47.8, 67, 81.3, 95.7, 114.8, 114.8], dtype=np.float32)

    # limit unit skin friction according to API ref page 59
    fs_max = np.interp(delta, delta_table, fs_max_table)

    # Unit skin friction [kPa]
    f = min(fs_max, K * sig * m.tan(delta * m.pi / 180.0))

    # piecewise function
    zlist = [0.0, 0.0254, 0.03, 0.04]
    tlist = [0.0, 1.0, 1.0, 1.0]

    # determine z vector
    z = np.array(zlist, dtype=np.float32)
    z = np.concatenate((-z[-1:0:-1], z))
    # define t vector
    t = np.array(tlist, dtype=np.float32) * f
    t = np.concatenate((-tensile_factor * t[-1:0:-1], t))

    add_values = output_length - len(z)
    add_z_values = np.zeros((add_values), dtype=np.float32)
    add_t_values = np.zeros((add_values), dtype=np.float32)

    for i in range(add_values):
        add_z_values[i] = 0.03 + random() * 0.01
        add_t_values[i] = f

    z = np.append(z, add_z_values)
    t = np.append(t, add_t_values)

    z = np.sort(z)
    z_id_sorted = np.argsort(z)

    t = t[z_id_sorted]

    return t, z
