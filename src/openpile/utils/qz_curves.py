"""
`qz_curves` module
==================

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

# SPRING FUNCTIONS --------------------------------------------

# API sand function
@njit(cache=True)
def api_sand(
    sig: float,
    delta: float,
    D: float,
    output_length: int = 7,
):
    """
    Creates the API sand Q.z curve from relevant input.

    Parameters
    ----------
    sig : float
        Vertical effective stress [unit: kPa]
    delta : float
        interface friction angle [unit: degrees]
    D : float
        Pile diameter [unit: m]
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 8, by default 8

    Returns
    -------
    numpy 1darray
        Q vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]
    """
    # cannot have less than 8
    if output_length < 8:
        output_length = 8

    # important variables
    delta_table = np.array([0, 15, 20, 25, 30, 35, 100], dtype=np.float32)
    Nq_table = np.array([8, 8, 12, 20, 40, 50, 50], dtype=np.float32)
    Qmax_table = np.array([1900, 1900, 2900, 4800, 9600, 12000, 12000], dtype=np.float32)

    Nq = np.interp(delta, delta_table, Nq_table)
    Qmax = np.interp(delta, delta_table, Qmax_table)

    # Unit end-bearing [kPa]
    f = min(Qmax, sig * Nq)

    # piecewise function
    zlist = [-0.002, 0.0, 0.002, 0.013, 0.042, 0.073, 0.100, 0.200]
    Qlist = [0.0, 0.0, 0.25, 0.50, 0.75, 0.90, 1.00, 1.00]

    # determine z vector
    z = np.array(zlist, dtype=np.float32) * D

    # define t vector
    Q = np.array(Qlist, dtype=np.float32) * f

    add_values = output_length - len(z)
    add_z_values = np.zeros((add_values), dtype=np.float32)
    add_Q_values = np.zeros((add_values), dtype=np.float32)

    for i in range(add_values):
        add_z_values[i] = (0.1 + random() * 0.1) * D
        add_Q_values[i] = f

    z = np.append(z, add_z_values)
    Q = np.append(Q, add_Q_values)

    z = np.sort(z)
    z_id_sorted = np.argsort(z)

    Q = Q[z_id_sorted]

    return Q, z


# API clay Q-z function
@njit(cache=True)
def api_clay(
    Su: float,
    D: float,
    output_length: int = 7,
):
    """
    Creates the API clay Q.z curve from relevant input.

    Parameters
    ----------
    Su : float
        Undrained shear strength [unit: kPa]
    D: float
        Pile diameter [unit: m]
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 8, by default 8

    Returns
    -------
    numpy 1darray
        Q vector [unit: kPa]
    numpy 1darray
        z vector [unit: m]
    """
    # cannot have less than 8
    if output_length < 8:
        output_length = 8

    # Unit end-bearing [kPa]
    f = 9 * Su

    # piecewise function
    zlist = [-0.002, 0.0, 0.002, 0.013, 0.042, 0.073, 0.100, 0.200]
    Qlist = [0.0, 0.0, 0.25, 0.50, 0.75, 0.90, 1.00, 1.00]

    # determine z vector
    z = np.array(zlist, dtype=np.float32) * D

    # define t vector
    Q = np.array(Qlist, dtype=np.float32) * f

    add_values = output_length - len(z)
    add_z_values = np.zeros((add_values), dtype=np.float32)
    add_Q_values = np.zeros((add_values), dtype=np.float32)

    for i in range(add_values):
        add_z_values[i] = (0.1 + random() * 0.1) * D
        add_Q_values[i] = f

    z = np.append(z, add_z_values)
    Q = np.append(Q, add_Q_values)

    z = np.sort(z)
    z_id_sorted = np.argsort(z)

    Q = Q[z_id_sorted]

    return Q, z
