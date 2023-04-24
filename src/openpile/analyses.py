"""
`Analyses` module
==================

The `analyses` module is used to run various simulations. 

Every function from this module returns an `openpile.compute.Result` object. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from pydantic import BaseModel, Field, root_validator
from pydantic.dataclasses import dataclass

from openpile.core import kernel
import openpile.core.validation as validation
import openpile.utils.graphics as graphics
import openpile.core.misc as misc


class PydanticConfig:
    arbitrary_types_allowed = True


def springs_mob_to_df(model, d):

    # elevations
    x = model.nodes_coordinates["x [m]"].values
    x = kernel.double_inner_njit(x)

    # PY springs
    # py secant stiffness
    py_ks = kernel.calculate_py_springs_stiffness(
        u=d[1::3], springs=model._py_springs, kind="secant"
    ).flatten()
    py_u = kernel.double_inner_njit(d[1::3])
    py_mob = py_u * py_ks
    # calculate max spring values
    max_springs_values = model._py_springs[:, :, 0].max(axis=2).flatten()

    # create DataFrame
    df = pd.DataFrame(
        data={
            "Elevation [m]": x,
            "p_mobilized [kN/m]": np.abs(py_mob),
            "p_max [kN/m]": max_springs_values,
        }
    )

    return df


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


@dataclass(config=PydanticConfig)
class Result:
    name: str
    displacements: pd.DataFrame
    forces: pd.DataFrame
    springs_mobilization: pd.DataFrame

    class Config:
        frozen = True

    @property
    def settlement(self):
        """Retrieves degrees of freedom for settlement

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their normal displacements
        """
        return self.displacements[["Elevation [m]", "Settlement [m]"]]

    @property
    def deflection(self):
        """Retrieves degrees of freedom for deflection

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their transversal displacements
        """
        return self.displacements[["Elevation [m]", "Deflection [m]"]]

    @property
    def rotation(self):
        """Retrieves rotational degrees of freedom

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and their rotations
        """
        return self.displacements[["Elevation [m]", "Rotation [rad]"]]

    @property
    def py_mobilization(self):
        """Retrieves mobilized resistance of p-y curves.

        Returns
        -------
        pandas.DataFrame
            Table with the nodes elevations along the pile and the mobilized resistance in kN/m.
        """
        return self.springs_mobilization[["Elevation [m]", "p_mobilized [kN/m]", "p_max [kN/m]"]]

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

    def plot(self, assign=False):
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


def simple_beam_analysis(model):
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

    if model.soil is None:
        # initialise global force
        F = kernel.mesh_to_global_force_dof_vector(model.global_forces)
        # initiliase prescribed displacement vector
        U = kernel.mesh_to_global_disp_dof_vector(model.global_disp)
        # initialise global stiffness matrix
        K = kernel.build_stiffness_matrix(model)
        # initialise global supports vector
        supports = kernel.mesh_to_global_restrained_dof_vector(model.global_restrained)

        # validate boundary conditions
        validation.check_boundary_conditions(model)

        u, _ = kernel.solve_equations(K, F, U, restraints=supports)

        # internal forces
        q_int = kernel.struct_internal_force(model, u)

        # Final results
        results = Result(
            name=f"{model.name} ({model.pile.name}/{model.soil.name})",
            displacements=disp_to_df(model, u),
            forces=structural_forces_to_df(model, q_int),
        )

        return results


def simple_winkler_analysis(model, max_iter: int = 100):
    """
    Function where loading or displacement defined in the model boundary conditions
    are used to solve the system of equations, .

    #TODO

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
        K = kernel.build_stiffness_matrix(model, f=None, u=d, kind="initial")
        # initialise internal forces
        q_int = np.zeros((model.element_number * 6))
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
            except np.linalg.LinAlgError:
                print(
                    """Cannot converge. Failure of the pile-soil system.\n
                      Boundary conditions may not be realistic or values may be too large."""
                )
                break

            # External forces
            F_ext = F - Q
            control = np.linalg.norm(F_ext)

            # add up increment displacements
            d += u_inc

            # calculate internal forces
            K_secant = kernel.build_stiffness_matrix(model, f=q_int, u=d, kind="secant")
            F_int = -K_secant.dot(d)

            # calculate residual forces
            Rg = F_ext + F_int

            # check if converged
            if np.linalg.norm(Rg[~supports]) < 1e-4 * control:
                # do not accept convergence without iteration (without a second call to solve equations)
                if iter_no > 0:
                    print(f"Converged at iteration no. {iter_no}")
                    break

            if iter_no == 100:
                print("Not converged after 100 iterations.")

            # Internal forces calculations with dim(nelem,6,6)
            q_int = kernel.struct_internal_force(model, q_int, d)

            # re-calculate global stiffness matrix for next iterations
            K = kernel.build_stiffness_matrix(model, f=q_int, u=d, kind="tangent")

            # reset prescribed displacements to converge properly in case
            # of displacement-driven analysis
            U[:] = 0.0

            # nump iteration number
            iter_no += 1

        # Internal forces calculations with dim(nelem,6,6)
        q_int = kernel.struct_internal_force(model, q_int, d)

        # Final results
        results = Result(
            name=f"{model.name} ({model.pile.name}/{model.soil.name})",
            displacements=disp_to_df(model, d),
            forces=structural_forces_to_df(model, q_int),
            springs_mobilization=springs_mob_to_df(model, d),
        )

        return results
