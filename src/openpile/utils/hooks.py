"""
`hooks` module
--------------



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

    .. plot::

        import matplotlib.pyplot as plt
        from openpile.construct import Layer
        from openpile.utils import hooks
        from openpile.soilmodels import API_sand
        # settings for py curve generation
        kw = {'sig':50, 'X':5, 'layer_height':10, 'depth_from_top_of_layer':5, 'D':5, 'L':30}
        # a, b soil models are traditional cyclic and static API sand models
        a = API_sand(
            phi=33,
            kind="static",
        )
        b = API_sand(
            phi=33,
            kind="cyclic",
        )
        plt.plot(*a.py_spring_fct(**kw), label='static')
        plt.plot(*b.py_spring_fct(**kw), label='cyclic')
        # soil model c is the API sand model with `durkhop` multipliers
        for ra in [0.3,0.45,0.6,0.75,0.9]:
            c = API_sand(
                phi=33,
                kind="cyclic",
                p_multiplier=hooks.durkhop(D=7.0, ra=ra)
            )
            plt.plot(*c.py_spring_fct(**kw), ':',label=f'Durkhop multipliers, ra={ra}')
        plt.legend()

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
    `openpile.utils.hooks.durkhop`
    """

    func = lambda x: 0.9 / max(0.9, ra * (3 - 1.143 * x / D) + 0.343 * x / D)

    return func


class PISA_depth_variation:
    r"""stores functions that returns the depth varying normalized parameters
    of the p-y, m-t, H-b and M-b conic formulations as per [BTZA20]_ and [BHBG20]_, and [BABH20]_.

    These functions can be then be used to model the original models, e.g. Cowden Clay,
    or make modifications to the original models.

    Example
    -------

    .. plot::

        import matplotlib.pyplot as plt
        from openpile.utils.py_curves import custom_pisa_sand, dunkirk_sand
        from openpile.utils.hooks import PISA_depth_variation

        # plot original dunkirk sand curve
        params = {'sig':50, 'X':5, 'Dr':75, 'G0':50e3, 'D':6, 'L':20}
        plt.plot(*dunkirk_sand(**params), label='Original Dunkirk Sand model', linestyle='-')

        # load PISA dunkirk sand depth variation functions
        funcs = PISA_depth_variation.dunkirk_sand_py_pisa_norm_param(D=6, L=20, Dr=75)

        # plot same curve with custom pisa sand curve
        params = {'sig':50, 'G0':50e3, 'D':6, 'X_ult':funcs['py_X'](5), 'Y_ult':funcs['py_Y'](5), 'n':funcs['py_n'](5), 'k':funcs['py_k'](5)}
        plt.plot(*custom_pisa_sand(**params), label='Custom Sand equivalent to Dunkirk sand model', linestyle='--')

        # plot an alrtered curve with custom pisa sand curve
        params['Y_ult'] /= 2
        plt.plot(*custom_pisa_sand(**params), label='Custom Sand with altered ultimate strength', linestyle='--')

        plt.title('Dunkirk Sand model vs. Custom PISA sand model')
        plt.legend()


    """

    @staticmethod
    def dunkirk_sand_pisa_norm_param(D: float, L: float):
        """returns the depth variation functions for all normalized parameters
        of the dunkirk sand conic formulations as per [BTZA20]_.

        Parameters
        ----------
        D : float
            pile diameter [m]
        L : float
            pile embedment [m]
        """

        py = PISA_depth_variation.dunkirk_sand_py_pisa_norm_param(D=D)
        mt = PISA_depth_variation.dunkirk_sand_mt_pisa_norm_param(D=D)
        Hb = PISA_depth_variation.dunkirk_sand_Hb_pisa_norm_param(D=D, L=L)
        Mb = PISA_depth_variation.dunkirk_sand_Mb_pisa_norm_param(D=D, L=L)

        return {**py, **mt, **Hb, **Mb}

    @staticmethod
    def cowden_clay_pisa_norm_param(D: float, L: float):
        """returns the depth variation functions for all normalized parameters
        of the cowden_clay conic formulations as per [BHBG20]_.

        Parameters
        ----------
        D : float
            pile diameter [m]
        L : float
            pile embedment [m]
        """

        py = PISA_depth_variation.cowden_clay_py_pisa_norm_param(D=D)
        mt = PISA_depth_variation.cowden_clay_mt_pisa_norm_param(D=D)
        Hb = PISA_depth_variation.cowden_clay_Hb_pisa_norm_param(D=D, L=L)
        Mb = PISA_depth_variation.cowden_clay_Mb_pisa_norm_param(D=D, L=L)

        return {**py, **mt, **Hb, **Mb}

    @staticmethod
    def cowden_clay_py_pisa_norm_param(D: float):
        """returns the depth variation functions for the 4 normalized parameters
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
        k_func = lambda x: max(0.0001, k_p1 + k_p2 * x / D)

        # normalized curvature
        n_p1 = 0.9390
        n_p2 = -0.03345
        n_func = lambda x: min(0.999, max(0, n_p1 + n_p2 * x / D))

        # normalized displacement at peak resistance
        X_func = lambda x: 241.4

        # normalized peak resistance
        p_u1 = 10.7
        p_u2 = -7.101
        Y_func = lambda x: max(0.0001, p_u1 + p_u2 * m.exp(-0.3085 * x / D))

        return {"py_k": k_func, "py_n": n_func, "py_X": X_func, "py_Y": Y_func}

    @staticmethod
    def cowden_clay_mt_pisa_norm_param(D: float):
        """returns the depth variation functions for the 4 normalized parameters
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
        k_func = lambda x: max(0.0001, k_m1 + k_m2 * x / D)

        # normalized curvature
        n_func = lambda x: 0.0

        # normalized peak resistance
        m_m1 = 0.2899
        m_m2 = -0.04775
        Y_func = lambda x: max(0.0001, m_m1 + m_m2 * x / D)

        # normalized displacement at peak resistance
        X_func = lambda x: max(0.0001, m_m1 + m_m2 * x / D) / max(0.0001, k_m1 + k_m2 * x / D)

        return {"mt_k": k_func, "mt_n": n_func, "mt_X": X_func, "mt_Y": Y_func}

    @staticmethod
    def cowden_clay_Hb_pisa_norm_param(D: float, L: float):
        """returns the depth variation functions for the 4 normalized parameters
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
        k_func = lambda x: max(0.0001, k_h1 + k_h2 * L / D)

        # normalized curvature
        n_h1 = 0.8793
        n_h2 = -0.03150
        n_func = lambda x: min(0.999, max(0, n_h1 + n_h2 * L / D))

        # normalized peak resistance
        p_u1 = 0.4038
        p_u2 = 0.04812
        Y_func = lambda x: max(0.0001, p_u1 + p_u2 * L / D)

        # normalized displacement at peak resistance
        X_func = lambda x: 235.7

        return {"Hb_k": k_func, "Hb_n": n_func, "Hb_X": X_func, "Hb_Y": Y_func}

    @staticmethod
    def cowden_clay_Mb_pisa_norm_param(D: float, L: float):
        """returns the depth variation functions for the 4 normalized parameters
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
        k_func = lambda x: max(0.0001, k_m1 + k_m2 * L / D)

        # normalized curvature
        n_m1 = 1.079
        n_m2 = -0.1087
        n_func = lambda x: min(0.999, max(0, n_m1 + n_m2 * L / D))

        # normalized peak resistance
        m_m1 = 0.8192
        m_m2 = -0.08588
        Y_func = lambda x: max(0.0001, m_m1 + m_m2 * L / D)

        # normalized displacement at peak resistance
        X_func = lambda x: 173.1

        return {"Mb_k": k_func, "Mb_n": n_func, "Mb_X": X_func, "Mb_Y": Y_func}

    @staticmethod
    def dunkirk_sand_py_pisa_norm_param(D: float, L: float, Dr: float):
        """returns the depth variation functions for the 4 normalized parameters
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

    @staticmethod
    def dunkirk_sand_mt_pisa_norm_param(L: float, Dr: float):
        """returns the depth variation functions for the 4 normalized parameters
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

    @staticmethod
    def dunkirk_sand_Hb_pisa_norm_param(D: float, L: float, Dr: float):
        """returns the depth variation functions for the 4 normalized parameters
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

    @staticmethod
    def dunkirk_sand_Mb_pisa_norm_param(D: float, L: float, Dr: float):
        """returns the depth variation functions for the 4 normalized parameters
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


class InitialSubgradeReaction:
    """This class stores functions that calculate the initial
    subgrade modulus of soil reaction curves.

    This class includes functions to calculate the initial subgrade reaction of py curves for sand.

    Example
    -------

    .. plot::

        # imports
        from matplotlib import pyplot as plt
        from openpile.utils.py_curves import api_sand
        from openpile.utils.hooks import InitialSubgradeReaction

        # sand parameters
        settings = {'phi':35,'X':5,'sig':50,'D':4, 'below_water_table':False}

        # create a plot
        _, ax = plt.subplots()
        ax.set_title("Varying the initial subgrade reaction modulus")
        # plot API sand
        ax.plot(*api_sand(**settings,), label='API sand')
        # plot API sand with Kallehave modification
        ax.plot(*api_sand(**settings, k=InitialSubgradeReaction.kallehave_sand(**settings)), label="Kallehave sand")
        ax.plot(*api_sand(**settings, k=InitialSubgradeReaction.sørensen2010_sand(**settings)), label="Sørensen sand")
        ax.legend()
    """

    @staticmethod
    def api_sand(phi: float, below_water_table: bool, *args, **kwargs):
        """Calculates the initial subgrade modulus 'k' in the API sand p-y curve.
        The value calculated here is based on a visual fit.

        Parameters
        ----------
        phi : float
            internal angle of friction in degrees
        below_water_table : bool
            whether the curve is below or above the water table

        Returns
        -------
        float
            initial subgrade modulus [kN/m^3]
        """
        if below_water_table:
            return max((0.1978 * phi**2 - 10.232 * phi + 136.82) * 1000, 5400)
        else:
            return max((0.2153 * phi**2 - 8.232 * phi + 63.657) * 1000, 5400)

    @staticmethod
    def kallehave_sand(phi: float, below_water_table: bool, X: float, D: float, *args, **kwargs):
        """Calculates the initial subgrade modulus based on modification of the API sand p-y curve, presented in #REF.

        Parameters
        ----------
        phi : float
            internal angle of friction in degrees
        below_water_table : bool
            whether the curve is below or above the water table
        X : float
            depth from the ground surface [m]
        D : float
            pile diameter [m]

        Returns
        -------
        float
            initial subgrade modulus [kN/m^3]
        """
        return (
            InitialSubgradeReaction.api_sand(phi, below_water_table)
            * (X / 2.5) ** 0.6
            * (D / 0.61) ** 0.5
        )

    @staticmethod
    def sørensen2010_sand(phi: float, X: float, D: float, *args, **kwargs):
        """Calculates the initial subgrade modulus based on modification of the API sand p-y curve,
        presented in [SøIA10]_.

        Parameters
        ----------
        phi : float
            internal angle of friction in degrees
        below_water_table : bool
            whether the curve is below or above the water table
        X : float
            depth from the ground surface [m]
        D : float
            pile diameter [m]

        Returns
        -------
        float
            initial subgrade modulus [kN/m^3]
        """
        Dref = 1.0
        Xref = 1.0
        return 1 / X * 50e3 * (X / Xref) ** 0.6 * (D / Dref) ** 0.5 * m.radians(phi) ** 3.6
