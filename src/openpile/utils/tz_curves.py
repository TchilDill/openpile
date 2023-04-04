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
    X: float,
    Su: float,
    D: float,
    residual: float,
    output_length: int = 15,
):
    """
    Creates the API clay t.z curve from relevant input.

    ---------
    input:
        sig: float
            Vertical effective stress [unit: kPa]
        X: float
            Depth of the curve w.r.t. mudline [unit: m]
        Su : float
            Undrained shear strength [unit: kPa]
        D: float
            Pile diameter [unit: m]
        residual: float
            residual strength after peak strength, according to API-RP-2A, this value is between 0.7 and 0.9 
        output_length: int, by default 15
            Number of discrete point along the springs
    ---------
    Returns curve with 2 vectors:
        t: numpy 1darray
            t vector [unit: kPa]
        z: numpy 1darray
            z vector [unit: m]
    ---------
    """
    # important variables
    if sig == 0.0:
        psi = Su / 0.001
    else:
        psi = Su / sig

    if psi > 1.0:
        alpha = min( 0.5 * psi**(-0.25), 1.0 )
    else:
        alpha = min( 0.5 * psi**(-0.5), 1.0 )

    # Unit skin friction [kPa]
    f = alpha * Su

    # piecewise function
    zlist = [0.0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, 0.03]
    tlist = [0.0 , 0.3, 0.5, 0.75, 0.90, 1.00, residual, residual]

    # determine z vector
    z = np.array(zlist, dtype=np.float32) * D

    # define t vector
    t = np.array(tlist, dtype=np.float32) * f

    add_values = output_length - len(z)
    add_z_values = []
    for _ in range(add_values):
        zval = 0.0017*D + random() * D * (0.0199 - 0.0017)
        tval = np.interp(zval, z, t)

        z = np.append(z, zval)
        t = np.append(t, tval)

    z = np.sort(z)
    t = np.sort(t)

    return t, z
