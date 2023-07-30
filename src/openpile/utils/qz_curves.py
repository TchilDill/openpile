"""
`qz_curves` module
==================

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

import openpile.utils.misc as misc

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

    # unit toe reistance [kPa]
    f = misc._Qmax_api_sand(sig, delta)

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

    return z, Q


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

    # unit toe reistance [kPa]
    f = misc._Qmax_api_clay

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

    return z, Q
