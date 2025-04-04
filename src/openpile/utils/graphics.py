# import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from openpile.core.misc import generate_color_string

mpl.rcParams["figure.subplot.wspace"] = 0.4


def plot_deflection(result):
    fig, ax = plt.subplots()

    fig.suptitle(f"{result._name} - Pile Deflection")

    ax = U_plot(ax, result)

    return fig


def plot_settlement(result):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(f"{result._name} - Pile settlements")

    ax1 = settlement_plot(ax1, result)
    ax2 = F_plot(ax2, result, "N [kN]")

    ax2.set_yticklabels("")
    ax2.set_ylabel("")

    return fig


def plot_forces(result):
    # create 4 subplots with (deflectiom, normal force, shear force, bending moment)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    fig.suptitle(f"{result._name} - Sectional forces")

    ax1 = F_plot(ax1, result, "N [kN]")
    ax2 = F_plot(ax2, result, "V [kN]")
    ax3 = F_plot(ax3, result, "M [kNm]")

    for axis in [ax2, ax3]:
        axis.set_yticklabels("")
        axis.set_ylabel("")

    return fig


def plot_results(result):
    # create 4 subplots with (deflectiom, normal force, shear force, bending moment)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    fig.suptitle(f"{result._name} - Analysis results")

    ax1 = U_plot(ax1, result)
    ax2 = F_plot(ax2, result, "N [kN]")
    ax3 = F_plot(ax3, result, "V [kN]")
    ax4 = F_plot(ax4, result, "M [kNm]")

    for axis in [ax2, ax3, ax4]:
        axis.set_yticklabels("")
        axis.set_ylabel("")

    return fig


def soil_plot(SoilProfile, ax=None):
    def add_soil_profile(SoilProfile, ax, pile=None):

        ax.set_title(label=f"Soil Profile overview - {SoilProfile.name}")
        ax.set_xlim(left=0, right=1)

        # make data
        offset = 5
        yBot = SoilProfile.bottom_elevation
        if pile is None:
            yTop = max(SoilProfile.top_elevation + offset, SoilProfile.water_line + offset)
        else:
            yTop = max(
                SoilProfile.top_elevation + offset,
                SoilProfile.water_line + offset,
                pile.top_elevation + offset,
            )

        # axes
        ax.set_ylim(bottom=yBot, top=yTop)
        ax.set_ylabel("Elevation [m VREF]")
        ax.set_xticks([])

        for layer in SoilProfile.layers:
            ax.add_patch(
                Rectangle(
                    xy=(-100, layer.bottom),
                    width=200,
                    height=layer.top - layer.bottom,
                    facecolor=layer.color,
                )
            )

            ax.text(
                0.02,
                0.5 * (layer.top + layer.bottom),
                layer.name,
                bbox={"facecolor": [0.98, 0.96, 0.85], "alpha": 1, "edgecolor": "none", "pad": 1},
            )

        ax.plot(
            np.array([-100, 0.1, 100]),
            SoilProfile.water_line + np.zeros((3)),
            mfc="dodgerblue",
            marker=7,
            linewidth=1,
            color="dodgerblue",
        )

        # grid
        ax.minorticks_on()
        ax.grid()
        ax.grid(axis="y", which="minor", color=[0.75, 0.75, 0.75], linestyle="-", linewidth=0.5)

        ax.plot(
            np.array([-100, 0.1, 100]),
            SoilProfile.water_line + np.zeros((3)),
            mfc="dodgerblue",
            marker=7,
            linewidth=1,
            color="dodgerblue",
        )

        return ax

    if ax is None:
        _, ax = plt.subplots()
    ax = add_soil_profile(SoilProfile, ax, pile=None)

    return ax


def connectivity_plot(model, ax=None):
    # TODO docstring

    support_color = "b"
    # create 4 subplots with (deflectiom, normal force, shear force, bending moment)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel("z [m]")
    ax.set_xlabel("y [m]")
    ax.set_title(f"{model.name} - Connectivity plot")
    ax.axis("equal")
    ax.grid(which="both")

    # plot mesh with + scatter points to see nodes.
    z = model.nodes_coordinates["z [m]"]
    y = model.nodes_coordinates["y [m]"]
    ax.plot(y, z, "-k", marker="+")

    total_length = (
        (model.nodes_coordinates["z [m]"].max() - model.nodes_coordinates["z [m]"].min()) ** 2
        + (model.nodes_coordinates["y [m]"].max() - model.nodes_coordinates["y [m]"].min()) ** 2
    ) ** (0.5)

    ylim = ax.get_ylim()

    # plots SUPPORTS
    # Plot supports along z
    support_along_z = model.global_restrained["Tz"].values
    support_along_z_down = np.copy(support_along_z)
    support_along_z_down[-1] = False
    support_along_z_up = np.copy(support_along_z)
    support_along_z_up[:-1] = False
    ax.scatter(
        y[support_along_z_down],
        z[support_along_z_down],
        color=support_color,
        marker=7,
        s=100,
    )
    ax.scatter(
        y[support_along_z_up],
        z[support_along_z_up],
        color=support_color,
        marker=6,
        s=100,
    )

    # Plot supports along y
    support_along_y = model.global_restrained["Ty"].values
    ax.scatter(y[support_along_y], z[support_along_y], color=support_color, marker=5, s=100)

    # Plot supports along z
    support_along_z = model.global_restrained["Rx"].values
    ax.scatter(y[support_along_z], z[support_along_z], color=support_color, marker="s", s=35)

    # plot LOADS
    arrows = []

    normalized_arrow_size = (
        0.10 * total_length
    )  # max arrow length will be 20% of the total structure length

    load_max = model.global_forces["Py [kN]"].abs().max()
    for yval, zval, load in zip(z, y, model.global_forces["Py [kN]"]):
        if load == 0:
            pass
        else:
            style = "Simple, tail_width=1, head_width=5, head_length=3"
            kw = dict(arrowstyle=style, color="r")
            arrow_length = normalized_arrow_size * abs(load / load_max)
            if load > 0:
                arrows.append(FancyArrowPatch((-arrow_length, yval), (zval, yval), **kw))
            elif load < 0:
                arrows.append(FancyArrowPatch((arrow_length, yval), (zval, yval), **kw))

    load_max = model.global_forces["Pz [kN]"].abs().max()
    for idx, (yval, zval, load) in enumerate(zip(z, y, model.global_forces["Pz [kN]"])):
        if load == 0:
            pass
        else:
            style = "Simple, tail_width=1, head_width=5, head_length=3"
            kw = dict(arrowstyle=style, color="r")
            arrow_length = normalized_arrow_size * abs(load / load_max)
            if load > 0:
                if idx == len(z) - 1:
                    arrows.append(FancyArrowPatch((zval, yval), (zval, yval + arrow_length), **kw))
                else:
                    arrows.append(FancyArrowPatch((zval, yval - arrow_length), (zval, yval), **kw))
            elif load < 0:
                if idx == len(z) - 1:
                    arrows.append(FancyArrowPatch((zval, yval), (zval, yval - arrow_length), **kw))
                else:
                    arrows.append(FancyArrowPatch((zval, yval + arrow_length), (zval, yval), **kw))

    load_max = model.global_forces["Mx [kNm]"].abs().max()
    for idx, (yval, zval, load) in enumerate(zip(z, y, model.global_forces["Mx [kNm]"])):
        if load == 0:
            pass
        else:
            kw = dict(arrowstyle=style, color="r")
            arrow_length = normalized_arrow_size * abs(load / load_max)
            style = "Simple, tail_width=1, head_width=5, head_length=3"
            if load > 0:
                if idx == len(z) - 1:
                    arrows.append(
                        FancyArrowPatch(
                            (arrow_length / 1.5, yval),
                            (-arrow_length / 1.5, yval),
                            connectionstyle="arc3,rad=0.5",
                            **kw,
                        )
                    )
                else:
                    arrows.append(
                        FancyArrowPatch(
                            (-arrow_length / 1.5, yval),
                            (arrow_length / 1.5, yval),
                            connectionstyle="arc3,rad=0.5",
                            **kw,
                        )
                    )
            elif load < 0:
                if idx == len(z) - 1:
                    arrows.append(
                        FancyArrowPatch(
                            (arrow_length / 1.5, yval),
                            (-arrow_length / 1.5, yval),
                            connectionstyle="arc3,rad=-0.5",
                            **kw,
                        )
                    )
                else:
                    arrows.append(
                        FancyArrowPatch(
                            (-arrow_length / 1.5, yval),
                            (arrow_length / 1.5, yval),
                            connectionstyle="arc3,rad=-0.5",
                            **kw,
                        )
                    )

    ax.set_ylim(ylim[0] - 0.11 * total_length, ylim[1] + 0.11 * total_length)

    if model.soil is not None:

        ax.set_ylim(
            min(model.bottom, ylim[0]) - 0.11 * total_length,
            max(model.top, ylim[1]) + 0.11 * total_length,
        )

        for layer in model.soil.layers:
            ax.add_patch(
                Rectangle(
                    xy=(-2 * total_length, layer.bottom),
                    width=4 * total_length,
                    height=layer.top - layer.bottom,
                    facecolor=layer.color,
                    alpha=0.4,
                )
            )

            ax.text(
                -1 * total_length,
                0.5 * (layer.top + layer.bottom) + 0.1 * (layer.top - layer.bottom),
                layer.name,
                bbox={"facecolor": [0.98, 0.96, 0.85], "alpha": 1, "edgecolor": "none", "pad": 1},
            )

            ax.plot(
                np.array([-2 * total_length, -0.6 * total_length, 2 * total_length]),
                model.soil.water_line + np.zeros((3)),
                mfc="dodgerblue",
                marker=7,
                linewidth=1,
                color="dodgerblue",
            )
        ax.set_xlim((-0.7 * total_length, 0.3 * total_length))

    else:
        ax.set_xlim((-0.5 * total_length, 0.5 * total_length))

    for arrow in arrows:
        ax.add_patch(arrow)

    return ax


def U_plot(axis: plt.axis, result):
    # TODO docstring

    axis.set_ylabel("Elevation [m VREF]", fontsize=8)
    axis.set_xlabel("Deflection [mm]", fontsize=8)
    axis.tick_params(axis="both", labelsize=8)
    axis.grid(which="both")

    y = result.displacements["Elevation [m]"].values
    x = np.zeros(shape=y.shape)
    deflection = result.displacements["Deflection [m]"] * 1e3

    axis.plot(x, y, color="0.4")
    axis.plot(deflection, y, color="0.0", lw=2)

    return axis


def settlement_plot(axis: plt.axis, result):
    # TODO docstring

    axis.set_ylabel("Elevation [m VREF]", fontsize=8)
    axis.set_xlabel("Settlement [mm]", fontsize=8)
    axis.tick_params(axis="both", labelsize=8)
    axis.grid(which="both")

    y = result.displacements["Elevation [m]"].values
    x = np.zeros(shape=y.shape)
    settlement = result.settlement["Settlement [m]"] * 1e3

    axis.plot(x, y, color="0.4")
    axis.plot(settlement, y, color="0.0", lw=2)

    return axis


def F_plot(axis: plt.axis, result, force: str):
    # TODO docstring

    # Define plot colors
    force_facecolor = "#E6DAA6"  # beige
    force_edgecolor = "#AAA662"  # khaki

    axis.set_ylabel("Elevation [m VREF]", fontsize=8)
    axis.set_xlabel(force, fontsize=8)
    axis.tick_params(axis="both", labelsize=8)
    axis.grid(which="both")

    f = result.forces[force]
    y = result.forces["Elevation [m]"]

    axis.fill_betweenx(y, f, edgecolor=force_edgecolor, facecolor=force_facecolor)
    axis.plot(np.zeros(shape=y.shape), y, color="0.4")

    axis.set_xlim(
        [
            min(0, f.min() - 0.1 * abs(f.min() + 1.0)),
            max(0, f.max() + 0.1 * abs(f.max() + 1.0)),
        ]
    )

    return axis
