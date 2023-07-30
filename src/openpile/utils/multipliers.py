"""
`multipliers` module
====================



"""

import numpy as np


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
    >>> Layer(
    >>>     name="medium dense sand",
    >>>     top=0,
    >>>     bottom=-40,
    >>>     weight=18,
    >>>     lateral_model=API_sand(
    >>>         phi=33,
    >>>         kind="cyclic",
    >>>         p_multiplier=durkhop(D=7.0, ra=1.0)
    >>>     ),
    >>> )


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
