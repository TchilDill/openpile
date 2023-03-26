# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

# SPRING FUNCTIONS --------------------------------------------


# API sand function
@njit(parallel=True, cache=True)
def api_sand(
    sig: float,
    X: float,
    phi: float,
    D: float,
    Neq: float = 1.0,
    below_water_table: bool = True,
    ymax: float = 0.0,
    output_length: int = 20,
):
    """
    Creates the API sand p-y curve from relevant input.

    ---------
    input:
        sig: float
            Vertical effective stress [unit: kPa]
        X: float
            Depth of the curve w.r.t. mudline [unit: m]
        phi: float
            internal angle of friction of the sand layer [unit: degrees]
        D: float
            Pile width [unit: m]
        Neq: float, by default 1.0
            Number of equivalent cycles [unit: -]
        below_water_table: bool, by default False
            switch to calculate initial subgrade modulus below/above water table
        ymax: float, by default 0.0
            maximum value of y, default goes to 99.9% of ultimate resistance
        output_length: int, by default 10
            Number of discrete point along the springs
    ---------
    Returns curve with 2 vectors:
        p: numpy 1darray
            p vector [unit: kPa/metre of pile length]
        y: numpy 1darray
            y vector [unit: m]
    ---------

    """
    # A value - only thing that changes between cyclic or static
    if Neq == 1:
        A = max(0.9, 3 - 0.8 * X / D)
    else:
        A = 0.9

    # initial subgrade modulus in kpa from fit with API (2014)
    if below_water_table:
        k_phi = max((0.1978 * phi**2 - 10.232 * phi + 136.82) * 1000, 5400)
    else:
        k_phi = max((0.2153 * phi**2 - 8.232 * phi + 63.657) * 1000, 5400)

    # Calculate Pmax (regular API)
    ## Factors according to Mosher and Dawkins 2008.(regular API)
    b = 0.4
    Beta = 45 + phi / 2
    rad = m.pi / 180
    C1 = (
        (b * m.tan(phi * rad) * m.sin(Beta * rad))
        / (m.tan((Beta - phi) * rad) * m.cos((phi / 2) * rad))
        + ((m.tan(Beta * rad)) ** 2 * m.tan((phi / 2) * rad))
        / (m.tan((Beta - phi) * rad))
        + b
        * m.tan(Beta * rad)
        * (m.tan(phi * rad) * m.sin(Beta * rad) - m.tan((phi / 2) * rad))
    )
    C2 = (
        m.tan(Beta * rad) / m.tan((Beta - phi) * rad)
        - (m.tan((45 - phi / 2) * rad)) ** 2
    )
    C3 = b * m.tan(phi * rad) * (m.tan(Beta * rad)) ** 4 + (
        m.tan((45 - phi / 2) * rad)
    ) ** 2 * ((m.tan(Beta * rad)) ** 8 - 1)

    ## Pmax for shallow and deep zones (regular API)
    Pmax = min(C3 * sig * D, C1 * sig * X + C2 * sig * D)

    # creation of 'y' array
    if ymax == 0.0:
        # ensure X cannot be zero
        z = max(X, 0.1)
        if Pmax == 0:
            f = (C3 * D + C2 * D) / 2
        else:
            f = Pmax
        ymax = 4 * A * f / (k_phi * z)

    # determine y vector from 0 to ymax
    # y_ini = np.array([0,],dtype=np.float64)
    # y = np.append(y_ini,np.logspace(np.log10(0.0001*D),np.log10(ymax),output_length-1))
    y = np.linspace(0, ymax, output_length)

    ## calculate p vector
    p = np.zeros(shape=len(y), dtype=np.float32)
    for i in prange(len(y)):
        if Pmax == 0:
            p[i] = 0
        else:
            p[i] = A * Pmax * m.tanh((k_phi * X * y[i]) / (A * Pmax))

    return p, y


# API sand function
@njit(parallel=True, cache=True)
def api_clay(
    sig: float,
    X: float,
    Su: float,
    eps50: float,
    D: float,
    J: float = 0.5,
    stiff_clay_threshold=96,
    Neq: float = 1.0,
    ymax: float = 0.0,
    output_length: int = 15,
):
    """
    Creates the API clay p-y curve from relevant input.


    ---------
    input:
        sig: float
            Vertical effective stress [unit: kPa]
        X: float
            Depth of the curve w.r.t. mudline [unit: m]
        Su : float
            Undrained shear strength [unit: kPa]
        eps50: float
            strain at 50% ultimate resistance [-]
        D: float
            Pile width [unit: m]
        J: float, by default 0.5
            empirical factor varying depending on clay stiffness
        stiff_clay_threshold: float, by default 96.0
            undrained shear strength at which stiff clay curve is computed [unit: kPa]
        Neq: float, by default 1.0
            Number of equivalent cycles [unit: -]
        ymax: float, by default 0.0
            maximum value of y, if null the maximum is calculated such that the whole curve is computed
        output_length: int, by default 20
            Number of discrete point along the springs
    ---------
    Returns curve with 2 vectors:
        p: numpy 1darray
            p vector [unit: kPa/metre of pile length]
        y: numpy 1darray
            y vector [unit: m]
    ---------
    """
    # important variables
    y50 = 2.5 * eps50 * D
    if X == 0.0 or Su == 0:
        Xr = 2.5 * D
    else:
        Xr = max((6 * D) / (sig / X * D / Su + J), 2.5 * D)

    # creation of 'y' array
    if ymax == 0.0:
        ymax = 16 * y50

    # Calculate Pmax (regular API)
    ## Pmax for shallow and deep zones (regular API)
    Pmax_shallow = (3 * Su + sig) * D + J * Su * X
    Pmax_deep = 9 * Su * X
    Pmax = min(Pmax_deep, Pmax_shallow)

    ylist_in = [0.0, 0.1 * y50, 0.21 * y50, 1 * y50, 3 * y50, 8 * y50, 15 * y50, ymax]
    ylist_out = []
    for i in range(len(ylist_in)):
        if ylist_in[i] <= ymax:
            ylist_out.append(ylist_in[i])

    # determine y vector from 0 to ymax
    y = np.array(ylist_out, dtype=np.float32)
    add_values = output_length - len(y)
    add_y_values = []
    for _ in range(add_values):
        add_y_values.append(0.1 * y50 + random() * (ymax - 0.1 * y50))
    y = np.append(y, add_y_values)
    y = np.sort(y)

    # define p vector
    p = np.zeros(shape=len(y), dtype=np.float32)

    for i in prange(len(y)):
        if Neq == 1:
            # derive static curve
            if y[i] > 8 * y50:
                p[i] = Pmax
            else:
                p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33
        else:
            # derive cyclic curve
            if X <= Xr:
                if Su <= stiff_clay_threshold:
                    if y[i] > 15 * y50:
                        p[i] = 0.7185 * Pmax * X / Xr
                    elif y[i] > 3 * y50:
                        p[i] = (
                            0.7185
                            * Pmax
                            * (1 - (1 - X / Xr) * (y[i] - 3 * y50) / (12 * y50))
                        )
                    else:
                        p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33
                else:
                    if y[i] > 15 * y50:
                        p[i] = 0.5 * Pmax * X / Xr
                    elif y[i] > 1 * y50:
                        p[i] = (
                            0.5
                            * Pmax
                            * (1 - (1 - X / Xr) * (y[i] - 1 * y50) / (14 * y50))
                        )
                    else:
                        p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

            elif X > Xr:
                if Su <= stiff_clay_threshold:
                    if y[i] > 3 * y50:
                        p[i] = 0.7185 * Pmax
                    else:
                        p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33
                else:
                    if y[i] > 1 * y50:
                        p[i] = 0.5 * Pmax
                    else:
                        p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

        # modification of initial slope of the curve (DNVGL RP-C203 B.2.2.4)
        if y[i] == 0.1 * y50:
            p[i] = 0.23 * Pmax

    return p, y
