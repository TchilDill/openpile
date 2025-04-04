"""
`qz_curves` module
------------------

"""

# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

from openpile.utils.misc import _Qmax_api_clay, _Qmax_api_sand

# SPRING FUNCTIONS --------------------------------------------


# API clay Q-z function
def _backbone_api(
    output_length: int = 15,
):
    """
    Creates the API Q.z curve backbone from relevant input.

    Parameters
    ----------
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        z vector [unit: m]
    numpy 1darray
        Q vector [unit: kPa]
    """
    # cannot have less than 8
    if output_length < 15:
        output_length = 15

    # piecewise function
    zlist = [
        -0.2,
        -0.15,
        -0.1,
        -0.073,
        -0.042,
        -0.013,
        -0.002,
        0.0,
        0.002,
        0.013,
        0.042,
        0.073,
        0.100,
        0.15,
        0.200,
    ]
    Qlist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.50, 0.75, 0.90, 1.00, 1.00, 1.00]

    # determine z vector
    z = np.array(zlist, dtype=np.float32)

    # define t vector
    Q = np.array(Qlist, dtype=np.float32)

    add_values = output_length - len(z)
    add_z_values = np.zeros((add_values), dtype=np.float32)
    add_Q_values = np.zeros((add_values), dtype=np.float32)

    for i in range(add_values):
        add_z_values[i] = 0.1 + random() * 0.1
        add_Q_values[i] = 1.0

    z = np.append(z, add_z_values)
    Q = np.append(Q, add_Q_values)

    z = np.sort(z)
    z_id_sorted = np.argsort(z)

    Q = Q[z_id_sorted]

    return z, Q


# API clay Q-z function
def api_clay(
    Su: float,
    D: float,
    output_length: int = 15,
):
    r"""
    Creates the API clay Q.z curve from relevant input.

    Parameters
    ----------
    Su : float
        Undrained shear strength [unit: kPa]
    D: float
        Pile diameter [unit: m]
    output_length : int, optional
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        z vector [unit: m]
    numpy 1darray
        Q vector [unit: kPa]

    Notes
    -----

    The maximum resistance is calculated as follows:

    * API clay: :math:`Q_{max} = 9 S_u`

    where :math:`S_u` is the clay undrained shear strength.


    The backbone curve is computed via the piecewise formulation presented in [API2000]_.
    """

    # unit toe reistance [kPa]
    f = _Qmax_api_clay(Su)

    # call backbone curve
    z, Q = _backbone_api(output_length)

    return z * D, Q * f


# API sand function
def api_sand(
    sig: float,
    delta: float,
    D: float,
    output_length: int = 15,
):
    r"""
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
        Number of discrete point along the springs, cannot be lower than 15, by default 15

    Returns
    -------
    numpy 1darray
        z vector [unit: m]
    numpy 1darray
        Q vector [unit: kPa]

    Notes
    -----

    The maximum resistance is calculated as follows:

    * API sand: :math:`Q_{max} = N_q \sigma^\prime_v`

    where :math:`\sigma_v^\prime` is the overburden effective stress and :math:`N_q` is
    the end-bearing factor depending on the interface friction angle :math:`\varphi`, see below table.

    +---------------------------+------+------+------+------+-------+
    | :math:`\varphi` [degrees] | 15.0 | 20.0 | 25.0 | 30.0 | 35.0  |
    +---------------------------+------+------+------+------+-------+
    | :math:`N_q` [kPa]         | 8.0  | 12.0 | 20.0 | 40.0 | 50.0  |
    +---------------------------+------+------+------+------+-------+
    | :math:`Q_{max}` [kPa]     | 1900 | 2900 | 4800 | 9600 | 12000 |
    +---------------------------+------+------+------+------+-------+


    The backbone curve is computed via the piecewise formulation presented in [API2000]_.
    """

    # max tip resistance
    f = _Qmax_api_sand(sig, delta)

    # call backbone curve
    z, Q = _backbone_api(output_length)

    return z * D, Q * f
