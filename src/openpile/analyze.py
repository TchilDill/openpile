"""
`analyze` module
==================

The `analyze` module is used to run 1D Finite Element analyses. 

Every function from this module returns an `openpile.analyze.AnalyzeResult` object. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from dataclasses import dataclass

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
    x = model.nodes_coordinates["x [m]"].values
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
    x = model.nodes_coordinates["x [m]"].values
    Q = Q.reshape(-1, 3)

    df = pd.DataFrame(
        data={
            "Elevation [m]": x,
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
    x = model.nodes_coordinates["x [m]"].values
    x = misc.repeat_inner(x)
    L = kernel.mesh_to_element_length(model).reshape(-1)

    N = np.vstack((-q[0::6], q[3::6])).reshape(-1, order="F")
    V = np.vstack((-q[1::6], q[4::6])).reshape(-1, order="F")
    M = np.vstack((-q[2::6], -q[2::6] + L * q[1::6])).reshape(-1, order="F")

    structural_forces_to_DataFrame = pd.DataFrame(
        data={
            "Elevation [m]": x,
            "N [kN]": N,
            "V [kN]": V,
            "M [kNm]": M,
        }
    )

    return structural_forces_to_DataFrame


def disp_to_df(model, u):
    x = model.nodes_coordinates["x [m]"].values

    Tx = u[::3].reshape(-1)
    Ty = u[1::3].reshape(-1)
    Rx = u[2::3].reshape(-1)

    disp_to_DataFrame = pd.DataFrame(
        data={
            "Elevation [m]": x,
            "Settlement [m]": Tx,
            "Deflection [m]": Ty,
            "Rotation [rad]": Rx,
        }
    )

    return disp_to_DataFrame


@dataclass
class AnalyzeResult:
    """The `AnalyzeResult` class is created by any analyses from the :py:mod:`openpile.analyze` module.

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
        """
        Plots the deflection of the pile.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned


        The plot looks like:

        .. image:: ../../docs/source/_static/usage/analyses_plots/deflection_results_plot.png
            :width: 60%
        """
        fig = graphics.plot_deflection(self)
        return fig if assign else None

    def plot_forces(self, assign=False):
        """Plots the pile sectional forces.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned


        The plot looks like:

        .. image:: ../../docs/source/_static/usage/analyses_plots/forces_results_plot.png
            :width: 60%
        """
        fig = graphics.plot_forces(self)
        return fig if assign else None

    def plot_lateral_results(self, assign=False):
        """Plots the pile deflection and sectional forces.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned


        The plot looks like:

        .. image:: ../../docs/source/_static/usage/analyses_plots/main_results_plot.png
            :width: 60%
        """
        fig = graphics.plot_results(self)
        return fig if assign else None

    def plot(self, assign=False):
        """Same behaviour as :py:meth:`openpile.analyze.plot_lateral_results`.

        Parameters
        ----------
        assign : bool, optional
            by default False

        Returns
        -------
        None or matplotlib.pyplot.figure
            if assign is True, a matplotlib figure is returned

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
    are used to solve the system of equations, .

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
    validation.check_boundary_conditions(model)

    # initialise global force
    F = kernel.mesh_to_global_force_dof_vector(model.global_forces)
    # initiliase prescribed displacement vector
    U = kernel.mesh_to_global_disp_dof_vector(model.global_disp)
    # initialise global supports vector
    supports = kernel.mesh_to_global_restrained_dof_vector(model.global_restrained)

    # initialise displacement vectors
    d = np.zeros(U.shape)

    # initialise global stiffness matrix
    K = kernel.build_stiffness_matrix(model, d)
    # first run with no stress stiffness matrix
    d, Q = kernel.solve_equations(K, F, U, restraints=supports)
    # rerun global stiffness matrix
    K = kernel.build_stiffness_matrix(model, d)
    # second run with stress stiffness matrix
    d, Q = kernel.solve_equations(K, F, U, restraints=supports)

    # internal forces
    q_int = kernel.pile_internal_forces(model, d)

    # Final results
    results = AnalyzeResult(
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
        UserWarning("SoilProfile must be provided when creating the Model.")

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
        supports = kernel.mesh_to_global_restrained_dof_vector(model.global_restrained)

        # validate boundary conditions
        # validation.check_boundary_conditions(model)

        # Initialise residual forces
        Rg = F

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
            if np.linalg.norm(Rg[~supports]) < nr_tol:
                # do not accept convergence without iteration (without a second call to solve equations)
                print(f"Converged at iteration no. {iter_no}")
                break

            if iter_no == 100:
                print("Not converged after 100 iterations.")

            # re-calculate global stiffness matrix for next iterations
            K = kernel.build_stiffness_matrix(model, u=d, kind="tangent")

            # reset prescribed displacements to converge properly in case
            # of displacement-driven analysis
            U[:] = 0.0

        # Internal forces calculations with dim(nelem,6,6)
        q_int = kernel.pile_internal_forces(model, d)

        # Final results
        results = AnalyzeResult(
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


def simple_winkler_analysis(model, max_iter: int = 100):

    # deprecation warning
    warnings.warn(
        "\nThe method Analyze.simple_winkler_analysis() will be removed in version 1.0.0."
        "\nPlease use the Analyze.winkler() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return winkler(model, max_iter)


def simple_beam_analysis(model):

    # deprecation warning
    warnings.warn(
        "\nThe method Analyze.simple_beam_analysis() will be removed in version 1.0.0."
        "\nPlease use the Analyze.beam() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return beam(model)
