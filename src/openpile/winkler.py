r"""
`winkler` module
==================

The `winkler` module is used to run 1D Finite Element analyses. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from dataclasses import dataclass
from copy import deepcopy

from openpile.core import kernel
import openpile.core.validation as validation
import openpile.utils.graphics as graphics
import openpile.core.misc as misc


class PydanticConfig:
    arbitrary_types_allowed = True
    frozen = True
    underscore_attrs_are_private = True


def springs_mob_to_df(model, d):

    # elevations
    x = model.nodes_coordinates["z [m]"].values
    x = kernel.double_inner_njit(x)

    # PY springs
    # py secant stiffness
    py_ks = kernel.calculate_py_springs_stiffness(
        u=d[1::3], springs=model._py_springs, kind="secant"
    ).flatten()
    py_mob = kernel.double_inner_njit(d[1::3]) * py_ks
    # calculate max spring values
    py_max = model._py_springs[:, :, 0].max(axis=2).flatten()

    # mt springs
    # mt secant stiffness
    mt_ks = kernel.calculate_mt_springs_stiffness(
        d[2::3], model._mt_springs, model._py_springs, py_mob.reshape((-1, 2, 1, 1)), kind="secant"
    ).flatten()
    mt_mob = kernel.double_inner_njit(d[2::3]) * mt_ks
    # calculate max spring values
    mt_max = model._mt_springs[:, :, 0, -1].max(axis=2).flatten()

    # create DataFrame
    df = pd.DataFrame(
        data={
            "Elevation [m]": x,
            "p_mobilized [kN/m]": np.abs(py_mob),
            "p_max [kN/m]": py_max,
            "m_mobilized [kNm/m]": np.abs(mt_mob),
            "m_max [kNm/m]": mt_max,
        }
    )

    return df


def reaction_forces_to_df(model, Q):
    z = model.nodes_coordinates["z [m]"].values
    Q = Q.reshape(-1, 3)

    df = pd.DataFrame(
        data={
            "Elevation [m]": z,
            "Nr [kN]": Q[:, 0],
            "Vr [kN]": Q[:, 1],
            "Mr [kNm]": Q[:, 2],
        }
    )
    df[["Nr [kN]"]] = df[["Nr [kN]"]].mask(df[["Nr [kN]"]].abs() < 1e-3, 0.0)
    df[["Vr [kN]"]] = df[["Vr [kN]"]].mask(df[["Vr [kN]"]].abs() < 1e-3, 0.0)
    df[["Mr [kNm]"]] = df[["Mr [kNm]"]].mask(df[["Mr [kNm]"]].abs() < 1e-3, 0.0)

    return df[np.any(df[["Nr [kN]", "Vr [kN]", "Mr [kNm]"]].abs() > 1e-3, axis=1)].reset_index(
        drop=True
    )


def structural_forces_to_df(model, q):
    z = model.nodes_coordinates["z [m]"].values
    z = misc.repeat_inner(z)
    L = kernel.mesh_to_element_length(model).reshape(-1)

    N = np.vstack((-q[0::6], q[3::6])).reshape(-1, order="F")
    V = np.vstack((-q[1::6], q[4::6])).reshape(-1, order="F")
    M = np.vstack((-q[2::6], -q[2::6] + L * q[1::6])).reshape(-1, order="F")

    structural_forces_to_DataFrame = pd.DataFrame(
        data={
            "Elevation [m]": z,
            "N [kN]": N,
            "V [kN]": V,
            "M [kNm]": M,
        }
    )

    return structural_forces_to_DataFrame


def disp_to_df(model, u):
    z = model.nodes_coordinates["z [m]"].values

    Tz = u[::3].reshape(-1)
    Ty = u[1::3].reshape(-1)
    Rx = u[2::3].reshape(-1)

    disp_to_DataFrame = pd.DataFrame(
        data={
            "Elevation [m]": z,
            "Settlement [m]": Tz,
            "Deflection [m]": Ty,
            "Rotation [rad]": Rx,
        }
    )

    return disp_to_DataFrame


@dataclass
class WinklerResult:
    """The `WinklerResult` class is created by any analyses from the :py:mod:`openpile.winkler` module.

    As such the user can use the following properties and/or methods for any return values of an analysis.

    """

    _name: str
    _d: pd.DataFrame
    _f: pd.DataFrame
    _Q: pd.DataFrame
    _dist_mob: pd.DataFrame = None
    _hb_mob: tuple = None
    _mb_mob: tuple = None
    _details: dict = None

    @property
    def displacements(self):
        """Retrieves displacements along each dimensions

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their displacements
        """
        return self._d

    @property
    def forces(self):
        """Retrieves forces along the pile (Normal force, shear force and bending moment)

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their forces
        """
        return self._f

    @property
    def reactions(self):
        """Retrieves reaction forces (where supports are given)

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their forces
        """
        return self._Q

    @property
    def settlement(self):
        """Retrieves degrees of freedom for settlement

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their normal displacements
        """
        return self._d[["Elevation [m]", "Settlement [m]"]]

    @property
    def deflection(self):
        """Retrieves degrees of freedom for deflection

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their transversal displacements
        """
        return self._d[["Elevation [m]", "Deflection [m]"]]

    @property
    def rotation(self):
        """Retrieves rotational degrees of freedom

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their rotations
        """
        return self._d[["Elevation [m]", "Rotation [rad]"]]

    @property
    def py_mobilization(self):
        """Retrieves mobilized resistance of districuted lateral p-y curves.

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and the mobilized resistance in kN/m.
        """

        if self._dist_mob is None:
            return None
        else:
            return self._dist_mob[["Elevation [m]", "p_mobilized [kN/m]", "p_max [kN/m]"]]

    @property
    def mt_mobilization(self):
        """Retrieves mobilized resistance of distributed moment rotational curves.

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and the mobilized resistance in kNm/m.
        """
        if self._dist_mob is None:
            return None
        else:
            return self._dist_mob[["Elevation [m]", "m_mobilized [kNm/m]", "m_max [kNm/m]"]]

    @property
    def Hb_mobilization(self):
        """Retrieves mobilized resistance of base shear.

        Returns
        -------
        tuple
            the mobilised value and the maximum resistance in kN
        """
        return self._hb_mob if self._hb_mob is not None else None

    @property
    def Mb_mobilization(self):
        """Retrieves mobilized resistance of base moment.

        Returns
        -------
        tuple
            the mobilised value and the maximum resistance in kNm
        """
        return self._mb_mob if self._mb_mob is not None else None

    def plot_deflection(self, assign=False):
        r"""
        Plots the deflection of the pile.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned

        Example
        -------
        The plot looks like:

        .. plot::
            :context: reset
            :include-source: False

            from openpile.construct import Pile, SoilProfile, Layer, Model
            from openpile.soilmodels import API_clay, API_sand
            p = Pile.create_tubular(
                name="<pile name>", top_elevation=0, bottom_elevation=-40, diameter=7.5, wt=0.075
            )
            # Create a 40m deep offshore Soil Profile with a 15m water column
            sp = SoilProfile(
                name="Offshore Soil Profile",
                top_elevation=0,
                water_line=15,
                layers=[
                    Layer(
                        name="medium dense sand",
                        top=0,
                        bottom=-20,
                        weight=18,
                        lateral_model=API_sand(phi=33, kind="cyclic"),
                    ),
                    Layer(
                        name="firm clay",
                        top=-20,
                        bottom=-40,
                        weight=18,
                        lateral_model=API_clay(Su=[50, 70], eps50=0.015, kind="cyclic"),
                    ),
                ],
            )
            # Create Model
            M = Model(name="<model name>", pile=p, soil=sp)
            # Apply bottom fixity along z-axis
            M.set_support(elevation=-40, Tz=True)
            # Apply axial and lateral loads
            M.set_pointload(elevation=0, Pz=-20e3, Py=5e3)
            # Run analysis
            result = M.solve()
            # plot the results
            result.plot_deflection()

        """
        fig = graphics.plot_deflection(self)
        return fig if assign else None

    def plot_forces(self, assign=False):
        r"""Plots the pile sectional forces.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned

        Example
        -------
        The plot looks like:

        .. plot::
            :context: close-figs
            :include-source: False

            result.plot_forces()
        """
        fig = graphics.plot_forces(self)
        return fig if assign else None

    def plot_lateral_results(self, assign=False):
        r"""Plots the pile deflection and sectional forces.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned


        Example
        -------
        The plot looks like:

        .. plot::
            :context: close-figs
            :include-source: False

            result.plot_lateral_results()
        """
        fig = graphics.plot_results(self)
        return fig if assign else None

    def plot_axial_results(self, assign=False):
        r"""Plots the pile settlements and normal forces.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned

        """
        fig = graphics.plot_settlement(self)
        return fig if assign else None

    def plot(self, assign=False):
        r"""Same behaviour as :py:meth:`openpile.analyze.plot_lateral_results`.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned

        Example
        -------

        The plot looks like:

        .. plot::
            :context: close-figs
            :include-source: False

            result.plot()
        """
        return self.plot_lateral_results(assign)

    def details(self) -> dict:
        """Provide a summary of the results.

        Returns
        -------
        dict
            info on the results

        """

        return {
            **self._details,
            "Max. normal force [kN]": round(self._f["N [kN]"].max(), 2),
            "Min. normal force [kN]": round(self._f["N [kN]"].min(), 2),
            "Max. shear force [kN]": round(self._f["V [kN]"].max(), 2),
            "Min. shear force [kN]": round(self._f["V [kN]"].min(), 2),
            "Max. moment [kNm]": round(self._f["M [kNm]"].max(), 2),
            "Min. moment [kNm]": round(self._f["M [kNm]"].min(), 2),
            "Max. settlement [m]": round(self._d["Settlement [m]"].max(), 3),
            "Min. settlement [m]": round(self._d["Settlement [m]"].min(), 3),
            "Max. deflection [m]": round(self._d["Deflection [m]"].max(), 3),
            "Min. deflection [m]": round(self._d["Deflection [m]"].min(), 3),
            "Max. rotation [rad]": round(self._d["Rotation [rad]"].max(), 3),
            "Min. rotation [rad]": round(self._d["Rotation [rad]"].min(), 3),
        }


def beam(model):
    """
    Function where loading or displacement defined in the model boundary conditions
    are used to solve the system of equations, this is a linear problem and is solved with one iteration.

    Parameters
    ----------
    model : `openpile.construct.Model` object
        Model where structure and boundary conditions are defined.

    Returns
    -------
    results : `openpile.compute.Result` object
        Results of the analysis
    """

    # validate boundary conditions
    # validation.check_boundary_conditions(model)

    # initialise global force
    F = kernel.mesh_to_global_force_dof_vector(model.global_forces)
    # initiliase prescribed displacement vector
    U = kernel.mesh_to_global_disp_dof_vector(model.global_disp)
    # initialise global supports vector
    supports = (kernel.mesh_to_global_restrained_dof_vector(model.global_restrained)) | (U != 0.0)

    # initialise displacement vectors
    d = np.zeros(U.shape)

    # initialise global stiffness matrix
    K = kernel.build_stiffness_matrix(model, d)
    # first run with no stress stiffness matrix
    d, Q = kernel.solve_equations(K, F, U, restraints=supports)

    # internal forces
    q_int = kernel.pile_internal_forces(model, d)

    # Final results
    results = WinklerResult(
        _name=f"{model.name} ({model.pile.name})",
        _d=disp_to_df(model, d),
        _f=structural_forces_to_df(model, q_int),
        _Q=reaction_forces_to_df(model, Q),
        _details={
            "converged @ iter no.": 1,
            "error": 0.0,
            "tolerance": None,
        },
    )

    return results


def winkler(model, max_iter: int = 100):
    """
    Function where loading or displacement defined in the model boundary conditions
    are used to solve the system of equations via the iterative Newton-Raphson scheme.

    Parameters
    ----------
    model : `openpile.construct.Model` object
        Model where structure and boundary conditions are defined.
    max_iter: int, by defaut 100
        maximum number of iterations for convergence

    Returns
    -------
    results : `openpile.analyses.Result` object
        Results of the analysis
    """

    if model.soil is None:
        UserWarning("A SoilProfile must be provided to the model before running this model.")

    else:
        # initialise global force
        F = kernel.mesh_to_global_force_dof_vector(model.global_forces)
        # initiliase prescribed displacement vector
        U = kernel.mesh_to_global_disp_dof_vector(model.global_disp)
        # initialise displacement vectors
        d = np.zeros(U.shape)
        # initialise global stiffness matrix
        K = kernel.build_stiffness_matrix(model, u=d, kind="initial")
        # initialise global supports vector
        supports = (kernel.mesh_to_global_restrained_dof_vector(model.global_restrained)) | (
            U != 0.0
        )

        # validate boundary conditions
        # validation.check_boundary_conditions(model)

        # Initialise residual forces
        Rg = deepcopy(F)

        # incremental calculations to convergence
        iter_no = 0
        while iter_no <= max_iter:

            # solve system
            try:
                u_inc, Q = kernel.solve_equations(K, Rg, U, restraints=supports)
                # bump iteration number
                iter_no += 1
            except np.linalg.LinAlgError:
                print(
                    """Cannot converge. Failure of the pile-soil system.\n
                      Boundary conditions may not be realistic or values may be too large."""
                )
                # dummy output vars
                Q = np.full(F.shape, np.nan)
                d = np.full(U.shape, np.nan)
                nr_tol = np.nan
                break

            # External forces
            F_ext = F - Q
            control = np.linalg.norm(F_ext)
            nr_tol = 1e-4 * control

            # add up increment displacements
            d += u_inc

            # calculate internal forces
            K_secant = kernel.build_stiffness_matrix(model, u=d, kind="secant")
            F_int = -K_secant.dot(d)

            # calculate residual forces
            Rg = F_ext + F_int

            # check if converged
            if np.linalg.norm(Rg[~supports]) < nr_tol and iter_no > 1:
                # do not accept convergence without iteration (without a second call to solve equations)
                print(f"Converged at iteration no. {iter_no}")

                # final stiffness matrix
                K_final = kernel.build_stiffness_matrix(model, u=d, kind="secant")

                # calculate final reaction forces
                _, Q = kernel.solve_equations(
                    K_final,
                    kernel.mesh_to_global_force_dof_vector(model.global_forces),
                    kernel.mesh_to_global_disp_dof_vector(model.global_disp),
                    restraints=supports,
                )

                break

            if iter_no == 100:
                print("Not converged after 100 iterations.")

                # dummy output vars
                Q = np.full(F.shape, np.nan)
                d = np.full(U.shape, np.nan)

            # re-calculate global stiffness matrix for next iterations
            K = kernel.build_stiffness_matrix(model, u=d, kind="tangent")

            # reset prescribed displacements to converge properly in case
            # of displacement-driven analysis
            U[:] = 0.0

        # Internal forces calculations with dim(nelem,6,6)
        q_int = kernel.pile_internal_forces(model, d)

        # Final results
        results = WinklerResult(
            _name=f"{model.name} ({model.pile.name}/{model.soil.name})",
            _d=disp_to_df(model, d),
            _f=structural_forces_to_df(model, q_int),
            _Q=reaction_forces_to_df(model, Q),
            _dist_mob=springs_mob_to_df(model, d),
            _hb_mob=(
                abs(d[-2])
                * kernel.calculate_base_spring_stiffness(d[-2], model._Hb_spring, kind="secant"),
                model._Hb_spring.flatten().max(),
            ),
            _mb_mob=(
                abs(d[-1])
                * kernel.calculate_base_spring_stiffness(d[-1], model._Mb_spring, kind="secant"),
                model._Mb_spring.flatten().max(),
            ),
            _details={
                "converged @ iter no.": iter_no,
                "error [kN]": round(np.linalg.norm(Rg[~supports]), 3),
                "tolerance [kN]": round(nr_tol, 3),
            },
        )

        return results


def simple_winkler_analysis(*args, **kwargs):
    """
    .. versionremoved: 1.0.0
        Use :func:`winkler` instead that keeps the same functional behaviour.
    """


def simple_beam_analysis(*args, **kwargs):
    """
    .. versionremoved: 1.0.0
        Use :func:`beam` instead that keeps the same functional behaviour.

    """
