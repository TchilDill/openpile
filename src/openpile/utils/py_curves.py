"""
`py_curves` module
------------------

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

    See also
    --------
    :py:func:`openpile.utils.py_curves.cowden_clay`, :py:func:`openpile.utils.py_curves.custom_clay`
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

    See also
    --------
    :py:func:`openpile.utils.py_curves.bothkennar_clay`, :py:func:`openpile.utils.py_curves.custom_clay`
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
    r"""
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

    Notes
    -----

    The curve backbone is defined as a conic function, see below.

    .. figure:: _static/PISA_conic_function.png
        :width: 80%

        PISA Conic function: (a) conic form; (b) bilinear form, after [BHBG20]_.

    """
    output_length = max(8, output_length)

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
    k: float = 0.0,
    ymax: float = 0.0,
    output_length: int = 20,
):
    r"""Creates the API sand p-y curve from relevant input.

    
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
    k: float, by default 0.0
        user-defined initial subgrade modulus [kN/m^3], if kept as zero, it is calculated as per API guidelines, see Notes below
    ymax: float, by default 0.0
        maximum value of y, default goes to 99.9% of ultimate resistance
    output_length: int, by default 20
        Number of discrete point along the springs, cannot be less than 8

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]

        
    Notes
    -----
    
    p-y formulation
        The API sand **p-y formulation** is presented in both the API and DNVGL standards,
        see, [DNV-RP-C212]_, [API2000]_ or [API2014]_.

        Granular soils are modelled by the sand p-y model as described 
        with the following backbone formula:
        
        .. math::

            p = A \cdot P_{max} \cdot \tanh \left( \frac{k \cdot X}{A \cdot P_{max} }  y \right) 

        where:

        * :math:`A` is a factor to account for static of cyclic loading 
        * :math:`P_{max}` is the ultimate resistance of the p-y curve 
        * :math:`k` is the initial modulus of subgrade reaction
        * :math:`X` is the depth below mudline of the p-y curve.

    Factor A
        The factor A takes into account whether the curve represent 
        static(also called monotonic) or cycling loading and is equal to:

        .. math::

            A = 
            \begin{cases} 
            \begin{split}
            0.9 & \text{  for cyclic loading} \\ 
            \\
            3 - 0.8 \frac{X}{D} \ge 0.9 & \text{  for static loading}
                \end{split}
            \end{cases}

        where:

        * :math:`D` is the pile diameter. 
    
    Initial subgrade reaction modulus
        The initial subgrade reaction, represented by the factor k is
        approximated by the following equation in which the output is given in kN/m³ 
        and where :math:`\phi` is inserted in degrees: 

        .. math::

            k = 
            \begin{cases} 
            \begin{split}
            197.8 \cdot \phi^2 - 10232 \cdot \phi + 136820 \ge 5400 & \text{ ,  below water table} \\ 
            \\
            215.3 \cdot \phi^2 - 8232 \cdot \phi + 63657 \ge 5400  & \text{ ,  above water table}
            \end{split}
            \end{cases}

        The equation is a fit to the recommended values in [API2000]_.  The correspondence 
        of this fit is illustrated in below figure:

        .. figure:: /_static/py_API_sand/k_vs_phi.jpg
            :width: 80%

            Subgrade reaction moduli fits calculated by openpile.

        .. note::

            The initial subgrade modulus can be user-defined by setting the parameter 'k' to a value greater than zero.
            Furthermore, many researchers have proposed different values for the initial subgrade modulus, see :py:class:`openpile.soilmodels.hooks.InitialSubgradeReaction`.

    Ultimate resistance
        The ultimate resistance :math:`P_{max}` is calculated via the coefficients C1, C2 and C3 found 
        in the below figure. 

        .. figure:: _static/py_API_sand/C_coeffs_graph.jpg
            :width: 80%

            Coefficients to calculate the maximum resistance. (as given in [MuOn84]_) 

        The ultimate resistance is found via the below equation:

        .. math::

            P_{max} = \left( 
                C1 \cdot \sigma^{\prime} \cdot X + C2 \cdot \sigma^{\prime} \cdot D \right) \lt
                C3 \cdot \sigma^{\prime} \cdot D 

        where:

        * :math:`\sigma^{\prime}` is the vertical effective stress

    """
    output_length = max(8, output_length)

    # A value - only thing that changes between cyclic or static
    if kind == "static":
        A = max(0.9, 3 - 0.8 * X / D)
    else:
        A = 0.9

    # initial subgrade modulus in kpa from fit with API (2014)
    if k != 0:
        if k < 0:
            raise ValueError("'initial_subgrade_modulus' must be stricly positive.")
        k_phi = k
    else:
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
    kind: str = "static",
    ymax: float = 0.0,
    output_length: int = 20,
):
    r"""
    Creates the API clay p-y curve from API RP2GEO (2014) from relevant input.

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
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    ymax: float, by default 0.0
        maximum value of y, if null the maximum is calculated such that the whole curve is computed
    output_length: int, by default 20
        Number of discrete point along the springs, cannot be less than 8

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]

    See also
    --------

    :py:class:`openpile.soilmodels.API_clay`, :py:func:`openpile.utils.py_curves.matlock_1970`

    Notes
    -----

    The p-y clay formulation is presented in both the API and DNVGL standards,
    see [DNV-RP-C212]_ and [API2014]_.

    The p-y curve is defined as per Matlock original equations (see notes in :py:func:`openpile.utils.py_curves.matlock_1970`) but defined at specific points, in a piece-wise manner.

    Static piece-wise points are defined as follows:
        +------------------+-------------------+
        | :math:`p/P_{max}`|  :math:`y/y_{50}` |
        +==================+===================+
        | 0.00             |  0.0              |
        +------------------+-------------------+
        | 0.23             |  0.1              |
        +------------------+-------------------+
        | 0.33             |  0.3              |
        +------------------+-------------------+
        | 0.50             |  1.0              |
        +------------------+-------------------+
        | 0.72             |  3                |
        +------------------+-------------------+
        | 1.00             |  8                |
        +------------------+-------------------+
        | 1.00             |  :math:`\infty`   |
        +------------------+-------------------+

    Cyclic piece-wise points are defined as follows:
        +----------------------------------------------------+------------------+
        |                 :math:`p/P_{max}`                  | :math:`y/y_{50}` |
        +====================================================+==================+
        | 0.00                                               | 0.0              |
        +----------------------------------------------------+------------------+
        | 0.23                                               | 0.1              |
        +----------------------------------------------------+------------------+
        | 0.33                                               | 0.3              |
        +----------------------------------------------------+------------------+
        | 0.50                                               | 1.0              |
        +----------------------------------------------------+------------------+
        | 0.72                                               | 3.0              |
        +----------------------------------------------------+------------------+
        | :math:`\min\left(0.72; 0.72 \dfrac{z}{X_R}\right)` | 15.0             |
        +----------------------------------------------------+------------------+
        | :math:`\min\left(0.72; 0.72 \dfrac{z}{X_R}\right)` | :math:`\infty`   |
        +----------------------------------------------------+------------------+

        where:

        * :math:`z` is the depth below mudline
        * :math:`X_R` is the transition zone depth below mudline
        * :math:`y_{50}` is the equivalent displacement for deformation at 50% ultimate resistance
        * :math:`P_{max}` is the ultimate resistance of the p-y curve
    """

    output_length = max(8, output_length)

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

    ylist_in = [0.0, 0.1 * y50, 0.3 * y50, 1 * y50, 3 * y50, 8 * y50, 15 * y50, ymax]
    ylist_out = []
    for i in range(len(ylist_in)):
        if ylist_in[i] <= ymax:
            ylist_out.append(ylist_in[i])

    # determine y vector from 0 to ymax
    y = np.array(ylist_out, dtype=np.float32)
    add_values = output_length - len(y)
    add_y_values = []
    for _ in range(add_values):
        add_y_values.append(15 * y50 + random() * (ymax - 15 * y50))
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
                if y[i] > 15 * y50:
                    p[i] = 0.7185 * Pmax * X / Xr
                elif y[i] > 3 * y50:
                    p[i] = 0.7185 * Pmax * (1 - (1 - X / Xr) * (y[i] - 3 * y50) / (12 * y50))
                else:
                    p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

            elif X > Xr:
                if y[i] > 3 * y50:
                    p[i] = 0.7185 * Pmax
                else:
                    p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

    return y, p


# API clay function
@njit(parallel=True, cache=True)
def matlock_1970(
    sig: float,
    X: float,
    Su: float,
    eps50: float,
    D: float,
    J: float = 0.5,
    kind: str = "static",
    ymax: float = 0.0,
    output_length: int = 20,
):
    r"""
    Creates the original clay p-y curve from the work performed by [Matl70]_.

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
    kind: str, by default "static"
        types of curves, can be of ("static","cyclic")
    ymax: float, by default 0.0
        maximum value of y, if null the maximum is calculated such that the whole curve is computed
    output_length: int, by default 20
        Number of discrete point along the springs, cannot be less than 9

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]

    See also
    --------
    :py:func:`openpile.utils.py_curves.api_clay`, :py:class:`openpile.soilmodels.API_clay`

    Notes
    -----

    Ultimate resistance
        The **ultimate resistance** is calculated via the capacity of two failure mechanisms,
        one that is shallow (wedge-type failure) and another that is deep (flow-around failure).

        .. math::

            P_{max} &= min(P_{shallow}, P_{deep})
            \\\\
            P_{shallow} &= D (3 S_u + \sigma^{\prime}) + J \cdot S_u \cdot X
            \\\\
            P_{deep} &=  9 \cdot S_u \cdot D

        where: 

        * :math:`S_u` is the undrained shear strßength in Unconfined and 
          unconsolidated (UU) Trixial tests.
        * :math:`\sigma^{\prime}` is the vertical effective stress.
        * :math:`J` is an empirical factor determined by Matlock to fit results 
          to pile load tests. This value can vary from 0.25 to 0.50 depending on 
          the clay characteristics
        * :math:`X` is the depth below ground level

        
    Strain normalization
        Strain normalization is performed with a parameter :math:`y_{50}` that is used to scale the curve with respect
        to the structure's scale and soil type.

        .. math::

            y_{50} = 2.5 \cdot \varepsilon_{50} \cdot D

        where: 

        * :math:`D` is the pile width or diameter
        * :math:`\varepsilon_{50}` is the strain at 50% ultimate resistance
          in Unconfined and unconsolidated (UU) Triaxial tests.

          
    Transition zone
        A transition zone is defined which corresponds to the depth at which the failure 
        around the pile is not governed by the free-field boundary, i.e. the ground level.
        Below the transition zone, a flow-around type of failure.

        The transition zone is defined by the following formula:

        .. math::

            X_R = \left( \frac{6 \cdot D}{\gamma^{\prime} \cdot \frac{D}{S_u} + J} \right) \ge  2.5 \cdot D

    
    Initial stiffness of p-y curve
        The initial stiffness :math:`k_{ini}` is infinite and can be capped from the Matlock original as in :py:func:`openpile.utils.py_curves.api_clay`:  


    p-y formulation (static loading, Neq = 1) 

        .. math::

            p = 
            \begin{cases} 
            \begin{split}
            0.5 \cdot P_{max} \left( \frac{y}{y_{50}} \right)^{0.33} & \text{  for } y \le 8 y_{50} \\ 
            \\
            P_{max} & \text{  for } y \gt 8 y_{50}
            \end{split}
            \end{cases}  

    p-y formulation (cyclic loading, Neq > 1)

        .. math::

            p = 
            \begin{cases} 
            \begin{split}
            0.5 \cdot P_{max} \left( \frac{y}{y_{50}} \right)^{0.33} & \text{  for } y \le 3 y_{50} \\ 
            \\
            0.72 \cdot P_{max} & \text{  for } y \gt 3 y_{50}
            \end{split}
            \end{cases}  

        For cyclic loading and curves above the transition zone ( i.e. :math:`X \le Xr`), 
        the p-y curve can be generated according to: 

        .. math::

            p = 
            \begin{cases} 
            \begin{split}
            0.5 \cdot P_{max} \left( \frac{y}{y_{50}} \right) & \text{  for } y \le 3 y_{50} \\ 
            \\
            0.72 \cdot P_{max} \left[ 1 - \left( 1 - \frac{X}{X_R} \right) \left( \frac{y - 3 y_{50}}{12 y_{50}} \right)  \right] & \text{  for } 3 y_{50} \lt y \le 15 y_{50} \\
            \\
            0.72 \cdot P_{max} \left( \frac{X}{X_R} \right) & \text{  for } y \gt 15 y_{50} \\
            \end{split}
            \end{cases}  
    """
    output_length = max(9, output_length)

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

    ylist_in = [0.0, 0.001 * y50, 0.01 * y50, 0.10 * y50, 1 * y50, 3 * y50, 8 * y50, 15 * y50, ymax]
    ylist_out = []
    for i in range(len(ylist_in)):
        if ylist_in[i] <= ymax:
            ylist_out.append(ylist_in[i])

    # determine y vector from 0 to ymax
    y = np.array(ylist_out, dtype=np.float32)
    add_values = output_length - len(y)
    add_y_values = []
    for _ in range(add_values):
        add_y_values.append(0.01 * y50 + random() * (8 - 0.01) * y50)
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
                if y[i] > 15 * y50:
                    p[i] = 0.7185 * Pmax * X / Xr
                elif y[i] > 3 * y50:
                    p[i] = 0.7185 * Pmax * (1 - (1 - X / Xr) * (y[i] - 3 * y50) / (12 * y50))
                else:
                    p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

            elif X > Xr:
                if y[i] > 3 * y50:
                    p[i] = 0.7185 * Pmax
                else:
                    p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

    return y, p


# API clay function
@njit(parallel=True, cache=True)
def modified_Matlock(
    sig: float,
    X: float,
    Su: float,
    eps50: float,
    D: float,
    J: float = 0.5,
    kind: str = "cyclic",
    ymax: float = 0.0,
    output_length: int = 20,
):
    """
    Creates the Modified Matlock for stiff clay p-y curve as defined in Bathacharya et al 2006 (see [BaCA06]).

    The modification takes places in the cyclic version of the curves. Static curves are kept the same as the original curves (see [Matl70]_), see :func:`Openpile.utils.py_curves.matlock_1970`.

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
    kind: str, by default "cyclic"
        types of curves, can be of ("static","cyclic")
    ymax: float, by default 0.0
        maximum value of y, if null the maximum is calculated such that the whole curve is computed
    output_length: int, by default 20
        Number of discrete point along the springs, cannot be less than 8

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]


    See also
    --------
    :py:func:`openpile.utils.py_curves.api_clay`, :py:func:`openpile.utils.py_curves.matlock_1970`

    Notes
    -----

    Differences with standard Matlock (1970) formula
        For an undrained shear strength of 96 kPa (assumed as the threshold at which a clay is considered stiff),
        this formulation may be deemed more relevant to account for a more brittle fracture and degradation
        of the soil, see [BaCA06]_.

        .. figure:: _static/schematic_curves.png
            :width: 80%

            Schematic of original (soft clay response) and modified (stiff clay response), after [BaCA06]_.
    """

    output_length = max(8, output_length)

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

    ylist_in = [0.0, 0.01 * y50, 0.10 * y50, 1 * y50, 3 * y50, 8 * y50, 15 * y50, ymax]
    ylist_out = []
    for i in range(len(ylist_in)):
        if ylist_in[i] <= ymax:
            ylist_out.append(ylist_in[i])

    # determine y vector from 0 to ymax
    y = np.array(ylist_out, dtype=np.float32)
    add_values = output_length - len(y)
    add_y_values = []
    for _ in range(add_values):
        add_y_values.append(0.01 * y50 + random() * (8 - 0.01) * y50)
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
                if y[i] > 15 * y50:
                    p[i] = 0.5 * Pmax * X / Xr
                elif y[i] > 1 * y50:
                    p[i] = 0.5 * Pmax * (1 - (1 - X / Xr) * (y[i] - 1 * y50) / (14 * y50))
                else:
                    p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

            elif X > Xr:
                if y[i] > 1 * y50:
                    p[i] = 0.5 * Pmax
                else:
                    p[i] = 0.5 * Pmax * (y[i] / y50) ** 0.33

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
    r"""creates the Reese weakrock p-y curve based on the work of Reese (1997), see reference [Rees97]_.

    Parameters
    ----------
    Ei : float
        initial rock mass modulus of rock [kPa]
    qu : float
        unconfined compressive strength of rock [kPa]
    RQD : float
        Rock Quality Designation [%]
    xr : float
        depth from rock surface [m]
    D : float
        pile width [m]
    k : float, optional
        dimensional constant, randing from 0.0005 to 0.00005, by default 0.0005
    output_length : int, optional
        length of output arrays, by default 20, cannot be less than 8

    Returns
    -------
    1darray
        y vector [unit: m]
    1darray
        p vector [unit: kN/m]

        
    Notes
    -----
    This formulation was derived for weak rocks and based on [Rees97]_.
    This empirical model is mostly based on experimental data of pile load tests near San Francisco 
    where the rock unconfined compressive strength varies from 1.86 MPa near the surface to 16.0 MPa.
    Pressuremeter tests results were used by Reese in this formulation as the initial modulus of the rock.     

    The curve is characterized by a linear initial portion, a non-linear response for the remaining 
    part of the curve, and a maximum resistance value that can be mobilized

    Ultimate resistance of rock
        .. math::
            P_{max} = \alpha \cdot q_u \cdot D \left(1 + 1.4 \dfrac{x_r}{D}\right) \le 5.2 \alpha \cdot q_u \cdot D

        where:

        * :math:`\alpha = 1 - \dfrac{2}{3} \dfrac{\text{RQD}}{100}`
        * :math:`\text{RQD}` is the rock quality designation in percentage
        * :math:`q_u` is the unconfined compressive strength of rock
        * :math:`D` is the pile diameter
        * :math:`x_r` is the depth from rock surface

    Initial portion of p-y curve
        The initial part of the curve is defined for :math:`y \le yA`, with a linear p-y curve stiffness of :math:`E_{py_i}`.

        .. math::
            y_A &= \left(\dfrac{P_{max}}{2 \cdot y_{rm}^{0.25} \cdot E_{py_i}}\right)^{4/3} \\
            \\
            E_{py_i} &= \left(100 + 400 \dfrac{x_r}{3D} \right) E_i \le 500 E_i 


        where:

        * :math:`E_rm` is the rock mass modulus of rock
        * :math:`y_{rm} = k \cdot D`
        * :math:`k` is a dimensional constant ranging from 0.0005 to 0.00005

    Remaining non-linear response
        The remaining portion of the curve is defined with the following equation:

        .. math::
            p = \dfrac{P_{max}}{2} \left(\dfrac{y}{y_{rm}}\right)^{1/4} \le P_{max}
    """

    output_length = max(8, output_length)

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

    See also
    --------
    :py:func:`openpile.utils.py_curves.dunkirk_sand`, :py:func:`openpile.utils.hooks.dunkirk_sand_pisa_norm_param`

    Example
    -------

    .. plot::

        import matplotlib.pyplot as plt
        from openpile.utils.py_curves import custom_pisa_sand, dunkirk_sand


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

    See also
    --------
    :py:func:`openpile.utils.py_curves.cowden_clay`, :py:func:`openpile.utils.py_curves.bothkennar_clay`,
    :py:func:`openpile.utils.hooks.cowden_clay_pisa_norm_param`
    """
    # calculate normsalised conic function
    y, p = conic(X_ult, n, k, Y_ult, output_length)

    # return non-normalised curve
    return y * (Su * D / G0), p * (Su * D)
