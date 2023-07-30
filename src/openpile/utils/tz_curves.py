"""
`tz_curves` module
==================


"""

# Import libraries
import openpile.utils.misc as misc

import math as m
import numpy as np
from numba import njit, prange
from random import random

# Kraft et al (1989) formulation aid to ease up implementation
@njit(cache=True)
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

    # define t till tmax
    tpos = np.linspace(0, fmax, output_length - 2)
    # define z till zmax
    zpos = tpos * 0.5 * D / G0 * np.log((zif - RF * tpos / fmax) / (1 - RF * tpos / fmax))
    # define z where t = tmax, a.k.a zmax here
    zmax = fmax * D / (2 * G0) * m.log((zif - RF) / (1 - RF))
    # define z where z=tres, which is zmax + 5mm
    zres = zmax + 0.005

    z = np.append(zpos, [zres, zres + 0.005])
    t = np.append(tpos, [residual * fmax, residual * fmax])

    return np.append(-z[-1::-1], np.append([0.0], z[1:])), np.append(
        -t[-1::-1] * tensile_factor, np.append([0.0], t[1:])
    )


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
    Creates the API clay t-z curve from relevant input as per [API2000]_.

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

    See also
    --------
    `API_clay`_

    """
    # cannot have less than 15
    if output_length < 15:
        output_length = 15

    # unit skin friction
    f = misc._fmax_api_clay(sig, Su)

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

    return z, t


@njit(cache=True)
def api_clay_kraft(
    sig: float,
    Su: float,
    D: float,
    G0: float,
    residual: float = 1.0,
    tensile_factor: float = 1.0,
    RF: float = 0.9,
    zif: float = 10.0,
    output_length: int = 15,
):
    """
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
    `API_clay`_, :py:func:`openpile.utils.tz_curves.api_clay`

    """
    return kraft_modification(
        misc._fmax_api_clay(sig, Su), D, G0, residual, tensile_factor, RF, zif, output_length
    )


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

    See also
    --------
    `API_sand`_

    """
    # cannot have less than 7
    if output_length < 7:
        output_length = 7

    # unit skin friction
    f = misc._fmax_api_sand(sig, delta, K)

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

    return z, t


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
    """
    Creates the API sand t-z curve (see [API2000]_) with the Kraft et al (1981) formulation (see [KrRK81]_).

    Parameters
    ----------
    sig: float
        Vertical effective stress [unit: kPa]
    delta: float
        interface friction angle [unit: degrees]
    D: float
        Pile diameter [unit: m]
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
    `API_sand`_, :py:func:`openpile.utils.tz_curves.api_sand`

    """
    return kraft_modification(
        misc._fmax_api_sand(sig, delta, K), D, G0, residual, tensile_factor, RF, zif, output_length
    )
