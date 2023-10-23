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
    When the parameter `ra` is equal to 1.0, the cyclic curves are approach the static curves.
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

    Example
    -------


    See also
    --------
    `openpile.utils.multipliers.durkhop`
    """

    func = lambda x: 0.9 / max(0.9, ra * (3 - 1.143 * x / D) + 0.343 * x / D)

    return func


def get_cowden_clay_py_norm_param(D: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the cowden_clay p-y conic formulation as per [BHBG20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['py_k','py_n','py_X', 'py_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    # normalized initial stiffness
    k_p1 = 10.6
    k_p2 = -1.650
    k_func = lambda x: max(0.001, k_p1 + k_p2 * x / D)

    # normalized curvature
    n_p1 = 0.9390
    n_p2 = -0.03345
    n_func = lambda x: min(0.999, max(0, n_p1 + n_p2 * x / D))

    # normalized displacement at peak resistance
    X_func = lambda x: 241.4

    # normalized peak resistance
    p_u1 = 10.7
    p_u2 = -7.101
    Y_func = lambda x: max(0.001, p_u1 + p_u2 * m.exp(-0.3085 * x / D))

    return {"py_k": k_func, "py_n": n_func, "py_X": X_func, "py_Y": Y_func}


def get_cowden_clay_mt_norm_param(D: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the cowden_clay rotational spring conic formulation as per [BHBG20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['mt_k','mt_n','mt_X', 'mt_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    # normalized initial stiffness
    k_m1 = 1.420
    k_m2 = -0.09643
    k_func = lambda x: max(0.001, k_m1 + k_m2 * x / D)

    # normalized curvature
    n_func = lambda x: 0.0

    # normalized peak resistance
    m_m1 = 0.2899
    m_m2 = -0.04775
    Y_func = lambda x: max(0.001, m_m1 + m_m2 * x / D)

    # normalized displacement at peak resistance
    X_func = lambda x: max(0.001, m_m1 + m_m2 * x / D) / max(0.001, k_m1 + k_m2 * x / D)

    return {"mt_k": k_func, "mt_n": n_func, "mt_X": X_func, "mt_Y": Y_func}


def get_cowden_clay_Hb_norm_param(D: float, L: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the cowden_clay base shear spring conic formulation as per [BHBG20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]
    L : float
        pile embedment length [m]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['Hb_k','Hb_n','Hb_X', 'Hb_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    # normalized initial stiffness
    k_h1 = 2.717
    k_h2 = -0.3575
    k_func = lambda x: max(0.001, k_h1 + k_h2 * L / D)

    # normalized curvature
    n_h1 = 0.8793
    n_h2 = -0.03150
    n_func = lambda x: min(0.999, max(0, n_h1 + n_h2 * L / D))

    # normalized peak resistance
    p_u1 = 0.4038
    p_u2 = 0.04812
    Y_func = lambda x: max(0.001, p_u1 + p_u2 * L / D)

    # normalized displacement at peak resistance
    X_func = lambda x: 235.7

    return {"Hb_k": k_func, "Hb_n": n_func, "Hb_X": X_func, "Hb_Y": Y_func}


def get_cowden_clay_Mb_norm_param(D: float, L: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the cowden_clay base moment spring conic formulation as per [BHBG20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]
    L : float
        pile embedment length [m]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['Mb_k','Mb_n','Mb_X', 'Mb_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    # normalized initial stiffness
    k_m1 = 0.2146
    k_m2 = -0.002132
    k_func = lambda x: max(0.001, k_m1 + k_m2 * L / D)

    # normalized curvature
    n_m1 = 1.079
    n_m2 = -0.1087
    n_func = lambda x: min(0.999, max(0, n_m1 + n_m2 * L / D))

    # normalized peak resistance
    m_m1 = 0.8192
    m_m2 = -0.08588
    Y_func = lambda x: max(0.001, m_m1 + m_m2 * L / D)

    # normalized displacement at peak resistance
    X_func = lambda x: 173.1

    return {"Mb_k": k_func, "Mb_n": n_func, "Mb_X": X_func, "Mb_Y": Y_func}


def get_dunkirk_sand_py_norm_param(D: float, L: float, Dr: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the dunkirk_sand p-y conic formulation as per [BTZA20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]
    L : float
        pile embedment length [m]
    Dr : float
        sand relative density [%]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['py_k','py_n','py_X', 'py_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    Dr = Dr / 100

    # normalized initial stiffness
    k_p1 = 8.731 - 0.6982 * Dr
    k_p2 = -0.9178
    k_func = lambda x: max(0.001, k_p1 + k_p2 * x / D)

    # normalized curvature
    n_func = lambda x: min(0.999, max(0, 0.917 + 0.06193 * Dr))

    # normalized displacement at peak resistance
    X_func = lambda x: 146.1 - 92.11 * Dr

    # normalized peak resistance
    p_u1 = 0.3667 + 25.89 * Dr
    p_u2 = 0.3375 - 8.9 * Dr
    Y_func = lambda x: max(0.001, p_u1 + p_u2 * x / L)

    return {"py_k": k_func, "py_n": n_func, "py_X": X_func, "py_Y": Y_func}


def get_dunkirk_sand_mt_norm_param(L: float, Dr: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the dunkirk_sand rotational spring conic formulation as per [BTZA20]_.

    Parameters
    ----------
    L : float
        pile embedment length [m]
    Dr : float
        sand relative density [%]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['mt_k','mt_n','mt_X', 'mt_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    Dr = Dr / 100

    # normalized initial stiffness
    k_func = lambda x: 17.0

    # normalized curvature
    n_func = lambda x: 0.0

    # normalized peak resistance
    m_u1 = 0.2605
    m_u2 = -0.1989 + 0.2019 * Dr
    Y_func = lambda x: max(0.001, m_u1 + m_u2 * x / L)

    # normalized displacement at peak resistance
    X_func = lambda x: max(0.001, m_u1 + m_u2 * x / L) / 17.0

    return {"mt_k": k_func, "mt_n": n_func, "mt_X": X_func, "mt_Y": Y_func}


def get_dunkirk_sand_Hb_norm_param(D: float, L: float, Dr: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the dunkirk_sand base shear spring conic formulation as per [BTZA20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]
    L : float
        pile embedment length [m]
    Dr : float
        sand relative density [%]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['Hb_k','Hb_n','Hb_X', 'Hb_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    Dr = Dr / 100

    # normalized initial stiffness
    k_h1 = 6.505 - 2.985 * Dr
    k_h2 = -0.007969 - 0.4299 * Dr
    k_func = lambda x: max(0.001, k_h1 + k_h2 * L / D)

    # normalized curvature
    n_h1 = 0.8793
    n_h2 = -0.03150
    n_func = lambda x: min(0.999, max(0.0, n_h1 + n_h2 * L / D))

    # normalized peak resistance
    p_u1 = 0.4038
    p_u2 = 0.04812
    Y_func = lambda x: max(0.001, p_u1 + p_u2 * L / D)

    # normalized displacement at peak resistance
    X_func = lambda x: 235.7

    return {"Hb_k": k_func, "Hb_n": n_func, "Hb_X": X_func, "Hb_Y": Y_func}


def get_dunkirk_sand_Mb_norm_param(D: float, L: float, Dr: float):
    """function that returns the depth variation functions for the 4 normalized parameters
    of the dunkirk_sand base moment spring conic formulation as per [BTZA20]_.

    Parameters
    ----------
    D : float
        pile diameter [m]
    L : float
        pile embedment length [m]
    Dr : float
        sand relative density [%]

    Returns
    -------
    dict
        dictionary of depth variation functions with keys['Mb_k','Mb_n','Mb_X', 'Mb_Y']
        accepting one input argument, the depth from ground (strictly positive) [m]
    """
    Dr = Dr / 100

    # normalized initial stiffness
    k_func = lambda x: 0.3515

    # normalized curvature
    n_h1 = 0.8793
    n_h2 = -0.03150
    n_func = lambda x: min(0.999, max(0.0, 0.3 + 0.4986 * Dr))

    # normalized peak resistance
    m_u1 = 0.09981 + 0.3710 * Dr
    m_u2 = 0.01998 - 0.09041 * Dr
    Y_func = lambda x: max(0.001, m_u1 + m_u2 * L / D)

    # normalized displacement at peak resistance
    X_func = lambda x: 44.89

    return {"Mb_k": k_func, "Mb_n": n_func, "Mb_X": X_func, "Mb_Y": Y_func}
