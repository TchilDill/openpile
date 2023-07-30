# misc functions


import random
import numpy as np
import pandas as pd
import math as m

from numba import njit
import matplotlib.colors as mcolors


def from_list2x_parse_top_bottom(var):
    """provide top and bottom values of layer based on float or list inputs"""
    if isinstance(var, float) or isinstance(var, int):
        top = var
        bottom = var
    elif isinstance(var, list) and len(var) == 1:
        top = var[0]
        bottom = var[0]
    elif isinstance(var, list) and len(var) == 2:
        top = var[0]
        bottom = var[1]
    else:
        print("Soil Layer variable is not a float nor a list")
        raise TypeError

    return top, bottom


def var_to_str(var):
    if isinstance(var, float) or isinstance(var, int):
        var_print = var
    elif isinstance(var, list):
        var_print = "-".join(str(v) for v in var)
    else:
        raise ValueError("not a float nor list")
    return var_print


def generate_color_string(kind=None):
    if kind is None:
        colors = list(mcolors.CSS4_COLORS.values())
    elif kind == "earth":
        colors = [
            "#B4A390",
            "#927E75",
            "#796363",
            "#99A885",
            "#D4DBBA",
            "#FDEDCF",
            "#EEEAC3",
            "#F5D498",
            "#ECB992",
            "#DDA175",
            "#AB7B5E",
            "#8F6854",
        ]
    return colors[random.randint(0, len(colors) - 1)]


def repeat_inner(arr):
    arr = arr.reshape(-1, 1)

    arr_inner = arr[1:-1]
    arr_inner = np.tile(arr_inner, (2)).reshape(-1)

    return np.hstack([arr[0], arr_inner, arr[-1]])


def get_reduced_springs(springs: np.ndarray, elevations: np.ndarray, kind: str) -> pd.DataFrame:
    """
    Returns soil springs created for the given model in one DataFrame.

    Parameters
    ----------
    springs : ndarray dim[nelem,2,2,spring_dim]
        Springs at top and bottom of element
    elevations : ndarray
        self.nodes_coordinates["x [m]"].values
    kind : str
        type of spring to extract. one of ["p-y", "m-t", "Hb-y", "Mb-t", "t-z"]

    Returns
    -------
    pd.DataFrame
        Soil springs
    """

    # get rid of the dimension dependent on the p-value, we keep max m-t spring
    if kind == "m-t":
        springs = springs.max(axis=3)

    spring_dim = springs.shape[-1]
    nelem = springs.shape[0]
    nnode = len(elevations)

    column_values_spring = [f"VAL {i}" for i in range(spring_dim)]

    id = np.repeat(np.arange(nelem + 1), 2)
    x = np.repeat(elevations, 2)

    influence = np.abs(np.gradient(elevations))
    influence[0] = influence[0] / 2
    influence[-1] = influence[-1] / 2

    springs[:, 0, 0, :] = springs[:, 0, 0, :] * influence[:-1].reshape(-1, 1)
    springs[:, 1, 0, :] = springs[:, 1, 0, :] * influence[1:].reshape(-1, 1)

    reduced_springs = np.zeros((nnode * 2, spring_dim))

    # first spring resistance and disp values
    reduced_springs[0, :] = springs[0, 0, 0, :]
    reduced_springs[1, :] = springs[0, 0, 1, :]
    # last spring resistance and disp values
    reduced_springs[-2, :] = springs[-1, 1, 0, :]
    reduced_springs[-1, :] = springs[-1, 1, 1, :]
    # calculation of weighted springs when node based
    j = 0
    for i in range(2, nelem * 2 - 1, 2):
        j += 1
        reduced_springs[i, :] = (
            springs[j - 1, 1, 0, :] * influence[j - 1] + springs[j, 0, 0, :] * influence[j]
        ) / (influence[j - 1] + influence[j])
        reduced_springs[i + 1, :] = (
            springs[j - 1, 1, 1, :] * influence[j - 1] + springs[j, 0, 1, :] * influence[j]
        ) / (influence[j - 1] + influence[j])

    df = pd.DataFrame(
        data={
            "Node no.": id,
            "Elevation [m]": x,
        }
    )

    df["type"] = kind.split("-") * len(elevations)
    df[column_values_spring] = reduced_springs

    return df


def get_full_springs(springs: np.ndarray, elevations: np.ndarray, kind: str) -> pd.DataFrame:
    """
    Returns soil springs in created for the given model in one DataFrame.

    Parameters
    ----------
    springs : ndarray dim[nelem,2,2,spring_dim]
        Springs at top and bottom of element
    elevations : ndarray
        self.nodes_coordinates["x [m]"].values
    kind : str
        type of spring to extract. one of ["p-y", "m-t", "Hb-y", "Mb-t", "t-z"]

    Returns
    -------
    pd.DataFrame
        Soil springs
    """

    # get rid of the dimension dependent on the p-value, we keep max m-t spring
    if kind == "m-t":
        springs = springs.max(axis=3)

    spring_dim = springs.shape[-1]
    nelem = springs.shape[0]

    column_values_spring = [f"VAL {i}" for i in range(spring_dim)]

    id = np.repeat(np.arange(nelem), 4)
    x = np.repeat(repeat_inner(elevations), 2)

    if len(x) > 2:
        t_b = ["top", "top", "bottom", "bottom"] * int(nelem)

        df = pd.DataFrame(
            data={
                "Element no.": id,
                "Position": t_b,
                "Elevation [m]": x,
            }
        )
    else:
        df = pd.DataFrame(
            data={
                "Element no.": id,
                "Elevation [m]": x,
            }
        )

    df["type"] = kind.split("-") * int(len(x) / 2)
    df[column_values_spring] = np.reshape(springs, (-1, spring_dim))

    return df


@njit(cache=True)
def conic(
    x_u: float,
    n: float,
    k: float,
    y_u: float,
    output_length: int,
):
    # Create x vector with 10% extension
    x = np.array([0, 0.001, 0.005, 0.01]).astype(np.float32) * x_u
    x = np.append(x, np.linspace(0.02 * x_u, x_u, output_length - 5).astype(np.float32))
    x = np.append(x, 1.1 * x_u)

    a = 1 - 2 * n

    y = np.zeros((len(x)), dtype=np.float32)

    for i in range(len(x)):
        if abs(x[i] - x_u) < 1e-2:
            y[i] = y_u
        elif x[i] < x_u:
            b = 2 * n * x[i] / x_u - (1 - n) * (1 + x[i] * k / y_u)
            c = x[i] * (k / y_u) * (1 - n) - n * (x[i] ** 2 / x_u**2)

            y[i] = y_u * 2 * c / (-b + (b**2 - 4 * a * c) ** 0.5)
        else:
            y[i] = y_u

    return x, y
