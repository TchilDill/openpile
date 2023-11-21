"""
`py_curves` module
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
def bothkennar_clay(
    X: float,
    Su: float,
    G0: float,
    D: float,
    output_length: int = 20,
):
    """
    Creates a spring from the PISA clay formulation
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
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]

    """

    # # Bothkennar clay parameters
    v_pu = 173.8
    k_p1 = 12.05
    k_p2 = -1.547
    n_p1 = 0.7204
    n_p2 = -0.002679
    p_u1 = 7.743
    p_u2 = -3.945

    # Depth variation parameters
    v_max = v_pu
    k = k_p1 + k_p2 * X / D
    n = n_p1 + n_p2 * X / D
    p_max = p_u1 + p_u2 * m.exp(-0.8456 * X / D)

    # calculate normsalised conic function
    y, p = conic(v_max, n, k, p_max, output_length)

    # return non-normalised curve
    return y * (Su * D / G0), p * (Su * D)


@njit(cache=True)
def cowden_clay(
    X: float,
    Su: float,
    G0: float,
    D: float,
    output_length: int = 20,
):
    """
    Creates a spring from the PISA clay formulation
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
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]

    """

    # # Cowden clay parameters
    v_pu = 241.4
    k_p1 = 10.6
    k_p2 = -1.650
    n_p1 = 0.9390
    n_p2 = -0.03345
    p_u1 = 10.7
    p_u2 = -7.101

    # Depth variation parameters
    v_max = v_pu
    k = k_p1 + k_p2 * X / D
    n = n_p1 + n_p2 * X / D
    p_max = p_u1 + p_u2 * m.exp(-0.3085 * X / D)

    # calculate normsalised conic function
    y, p = conic(v_max, n, k, p_max, output_length)

    # return non-normalised curve
    return y * (Su * D / G0), p * (Su * D)


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
    Creates a lateral spring from the PISA sand formulation
    published  by Burd et al (2020) (see [BTZA20]_).
    Also called the General Dunkirk Sand Model (GDSM).

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
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]
    """
    # correct relative density for decimal value
    Dr = Dr / 100

    # Generalised Dunkirk Sand Model parameters
    v_pu = 146.1 - 92.11 * Dr
    k_p1 = 8.731 - 0.6982 * Dr
    k_p2 = -0.9178
    n_p = 0.917 + 0.06193 * Dr
    p_u1 = 0.3667 + 25.89 * Dr
    p_u2 = 0.3375 - 8.9 * Dr

    # Depth variation parameters
    v_max = v_pu
    k = k_p1 + k_p2 * X / D
    n = n_p
    p_max = p_u1 + p_u2 * X / L

    # calculate normsalised conic function
    y, p = conic(v_max, n, k, p_max, output_length)

    # return non-normalised curve
    return y * (sig * D / G0), p * (sig * D)


# API sand function
@njit(parallel=True, cache=True)
def api_sand(
    sig: float,
    X: float,
    phi: float,
    D: float,
    kind: str = "static",
    below_water_table: bool = True,
    ymax: float = 0.0,
    output_length: int = 20,
):
    """
    Creates the API sand p-y curve from relevant input.

    Parameters
    ----------
    sig: float
        Vertical effective stress [unit: kPa]
    X: float
        Depth of the curve w.r.t. mudline [unit: m]
    phi: float
        internal angle of friction of the sand layer [unit: degrees]
    D: float
        Pile width [unit: m]
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    below_water_table: bool, by default False
        switch to calculate initial subgrade modulus below/above water table
    ymax: float, by default 0.0
        maximum value of y, default goes to 99.9% of ultimate resistance
    output_length: int, by default 20
        Number of discrete point along the springs

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]
    """
    # A value - only thing that changes between cyclic or static
    if kind == "static":
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
        + ((m.tan(Beta * rad)) ** 2 * m.tan((phi / 2) * rad)) / (m.tan((Beta - phi) * rad))
        + b * m.tan(Beta * rad) * (m.tan(phi * rad) * m.sin(Beta * rad) - m.tan((phi / 2) * rad))
    )
    C2 = m.tan(Beta * rad) / m.tan((Beta - phi) * rad) - (m.tan((45 - phi / 2) * rad)) ** 2
    C3 = b * m.tan(phi * rad) * (m.tan(Beta * rad)) ** 4 + (m.tan((45 - phi / 2) * rad)) ** 2 * (
        (m.tan(Beta * rad)) ** 8 - 1
    )

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

    return y, p


# API clay function
@njit(parallel=True, cache=True)
def api_clay(
    sig: float,
    X: float,
    Su: float,
    eps50: float,
    D: float,
    J: float = 0.5,
    stiff_clay_threshold=96,
    kind: str = "static",
    ymax: float = 0.0,
    output_length: int = 20,
):
    """
    Creates the API clay p-y curve from relevant input.

    Parameters
    ----------
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
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    ymax: float, by default 0.0
        maximum value of y, if null the maximum is calculated such that the whole curve is computed
    output_length: int, by default 20
        Number of discrete point along the springs

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]
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
    Pmax_deep = 9 * Su * D
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
        if kind == "static":
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
                        p[i] = 0.7185 * Pmax * (1 - (1 - X / Xr) * (y[i] - 3 * y50) / (12 * y50))
                    else:
                        p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33
                else:
                    if y[i] > 15 * y50:
                        p[i] = 0.5 * Pmax * X / Xr
                    elif y[i] > 1 * y50:
                        p[i] = 0.5 * Pmax * (1 - (1 - X / Xr) * (y[i] - 1 * y50) / (14 * y50))
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

    return y, p


@njit(parallel=True, cache=True)
def reese_weakrock(
    Ei: float,
    qu: float,
    RQD: float,
    xr: float,
    D: float,
    k: float = 0.0005,
    output_length=20,
):
    """creates the Reese weakrock p-y curve based on the work of Reese (1997), see reference [Rees97]_.


    Parameters
    ----------
    Ei : float
        initial modulus of rock [kPa]
    qu : float
        compressive strength of rock [kPa]
    RQD : float
        Rock Quality Designation [%]
    xr : float
        depth from rock surface [m]
    D : float
        pile width [m]
    k : float, optional
        dimensional constant, randing from 0.0005 to 0.00005, by default 0.0005
    output_length : int, optional
        length of output arrays, by default 20

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]

    """

    output_length = max(10, output_length)

    # Rqd forced to be within 0 and 100
    rqd = max(min(100, RQD), 0)

    # determine alpha
    alpha = 1.0 - 2 / 3 * rqd / 100

    # determine ultimate resistance of rock
    Pmax = min(5.2 * alpha * qu * D, alpha * qu * D * (1 + 1.4 * xr / D))

    # initial portion of p-y curve
    Epy_i = Ei * min(500, 100 + 400 * xr / (3 * D))

    # yA & yrm
    yrm = k * D
    yA = (Pmax / (2 * (yrm) ** 0.25 * Epy_i)) ** 1.333

    # define y
    ymax = max(1.05 * yA, (2 * yrm ** (0.25)) ** 4)
    y1 = np.linspace(yA, max(0.15 * ymax, 1.01 * yA), int(output_length / 2))
    y2 = np.linspace(max(yA * 1.02, 0.25 * ymax), ymax, output_length - len(y1) - 2)
    y = np.concatenate((np.array([0.0]), y1, y2, np.array([1.2 * ymax])))

    # define p
    p = np.zeros(y.size)
    for i in range(len(p)):
        if y[i] <= yA:
            p[i] = min(Pmax, Epy_i * y[i])
        else:
            p[i] = min(Pmax, Pmax / 2 * (y[i] / yrm) ** 0.25)

    return y, p


@njit(cache=True)
def custom_pisa_sand(
    sig: float,
    G0: float,
    D: float,
    X_ult: float,
    n: float,
    k: float,
    Y_ult: float,
    output_length: int = 20,
):
    """Creates a lateral spring with the PISA sand formulation and custom user inputs.

    Parameters
    ----------
    sig : float
        vertical/overburden effective stress [unit: kPa]
    G0 : float
        Small-strain shear modulus [unit: kPa]
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
    y, p = conic(X_ult, n, k, Y_ult, output_length)

    # return non-normalised curve
    return y * (sig * D / G0), p * (sig * D)


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
    Creates a spring from the PISA clay formulation
    published by Byrne et al 2020 (see [BHBG20]_) and calibrated based pile
    load tests at Cowden (north east coast of England).

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
    y, p = conic(X_ult, n, k, Y_ult, output_length)

    # return non-normalised curve
    return y * (Su * D / G0), p * (Su * D)
