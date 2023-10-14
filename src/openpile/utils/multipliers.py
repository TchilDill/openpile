"""
`multipliers` module
====================



"""

import math as m


def durkhop(D: float, ra: float = 0.3) -> object:
    """This function generates p-multipliers as per Durkhop et al (see [Duhr09]_) for application
    to cyclic curves of the `openpile.soilmodels.API_sand` model.

    The default behaviour, i.e. when the parameter `ra` = 0.3 is that the API sand cyclic curves are unchanged
    with a equivalent number of cycles equal to 100.
    When the parameter `ra` is equal to 1.0, the cyclic curves are returned back to monotonic curves.
    In between 0.3 and 1.0, the multipliers may be thought as cyclic curves with a lower number of equivalent cycles.

    Parameters
    ----------
    D : float
        Pile diameter [m]
    ra : float, optional
        empirical factor dependent on the number of load cycles, 1.0 for monotonic loading and
        0.3 for cyclic loading at 100 cycles, by default 0.3

    Returns
    -------
    object
        callable for use as p_multipler in `openpile.soilmodels.API_sand` in cyclic mode.

    Example
    -------

    >>> from openpile.construct import Layer
    >>> from openpile.soilmodels import API_sand
    >>> from openpile.utils.multipliers import durkhop
    >>> # Create a Layer with API_sand and monotonic curves with Durkhop approach
    >>> a_layer = Layer(
    ...     name="medium dense sand",
    ...     top=0,
    ...     bottom=-40,
    ...     weight=18,
    ...     lateral_model=API_sand(
    ...         phi=33,
    ...         kind="cyclic",
    ...         p_multiplier=durkhop(D=7.0, ra=1.0)
    ...     ),
    ... )


    Reference
    ---------

    .. [Duhr09] Dührkop, J. (2009). *Zum Einfluss von Aufweitungen und zyklischen Lasten auf
        das Verformungsverhalten lateral 385 beanspruchter Pfähle in Sand*. Ph.D. Thesis,
        Institut für Geotechnik und Baubetrieb, Technische Universität Hamburg-Harburg, Vol. 20 (in German).

    """

    func = lambda x: 1 / 0.9 * max(0.9, ra * (3 - 1.143 * x / D) + 0.343 * x / D)

    return func

def durkhop_normalized(D: float, ra: float = 0.3) -> object:
    """This function generates multipliers that represent ratios the cyclic and monotonic curves of
    the traditional API sand model.


    Parameters
    ----------
    D : float
        Pile diameter [m]
    ra : float, optional
        empirical factor dependent on the number of load cycles, 1.0 for monotonic loading and
        0.3 for cyclic loading at 100 cycles, by default 0.3

    Returns
    -------
    object
        callable for use as p_multipler in `openpile.soilmodels.LateralModel` or `openpile.soilmodels.AxialModel`
        The function input is the depth and the function output is the multiplier applied
        for the spring at the said depth.

    See also
    --------
    `durkhop`
    """

    func = lambda x: 0.9 / max(0.9, ra * (3 - 1.143 * x / D) + 0.343 * x / D)

    return func

def cowden_clay_py_k(D:float):
    """function that returns the depth variation function 
    of the normalized initial stiffness of the cowden_clay p-y curve as per [BHBG20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    k_p1 = 10.6
    k_p2 = -1.650
    return lambda x: max(0.001,k_p1 + k_p2 * x / D)

def cowden_clay_py_n(D:float):
    """function that returns the depth variation function 
    of the normalized curvature of the cowden_clay p-y curve as per [BHBG20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    n_p1 = 0.9390
    n_p2 = -0.03345
    return lambda x : min(0.999,max(0,n_p1 + n_p2 * x / D))

def cowden_clay_py_X_ult():
    """function that returns the depth variation function 
    of the normalized maximum displacement of the cowden_clay p-y curve as per [BHBG20]_.

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    return lambda x : 241.4

def cowden_clay_py_Y_ult(D:float):
    """function that returns the depth variation function 
    of the normalized curvature of the cowden_clay p-y curve as per [BHBG20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    p_u1 = 10.7
    p_u2 = -7.101
    return lambda x : max(0.001,p_u1 + p_u2 * m.exp(-0.3085 * x / D))

def dunkirk_sand_py_k(D:float, Dr: float):
    """function that returns the depth variation function 
    of the normalized initial stiffness of the dunkirk_sand p-y curve as per [BTZA20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]
    Dr : float
        sand relative density [%]

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    k_p1 = 8.731 - 0.6982 * Dr
    k_p2 = -0.9178
    return lambda x : max(0.001,k_p1 + k_p2 * x / D)

def dunkirk_sand_py_n(Dr: float):
    """function that returns the depth variation function 
    of the normalized curvature of the dunkirk_sand p-y curve as per [BTZA20]_.

    Parameters
    ----------
    Dr : float
        sand relative density [%]

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    return lambda x : min(0.999,max(0,0.917 + 0.06193 * Dr))

def dunkirk_sand_py_X_ult(Dr:float):
    """function that returns the depth variation function 
    of the normalized ultimate displacement of the dunkirk_sand p-y curve as per [BTZA20]_.

    Parameters
    ----------
    Dr : float
        sand relative density [%]

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    return lambda x : 146.1 - 92.11 * Dr

def dunkirk_sand_py_Y_ult(L:float, Dr: float):
    """function that returns the depth variation function 
    of the normalized ultimate resistance of the dunkirk_sand p-y curve as per [BTZA20]_.

    Parameters
    ----------
    L : float
        pile embedment length [m]
    Dr : float
        sand relative density [%]

    Returns
    -------
    callable
        depth variation function accepting one input argument, the depth from ground (strictly positive) [m]
    """
    p_u1 = 0.3667 + 25.89 * Dr
    p_u2 = 0.3375 - 8.9 * Dr
    return lambda x : max(0.001,p_u1 + p_u2 * x / L)
