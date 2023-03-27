# misc functions


import random
import numpy as np

from matplotlib.colors import CSS4_COLORS


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


def generate_color_string():
    colors = list(CSS4_COLORS.values())
    return colors[random.randint(0, len(colors) - 1)]


def repeat_inner(arr):
    arr = arr.reshape(-1, 1)

    arr_inner = arr[1:-1]
    arr_inner = np.tile(arr_inner, (2)).reshape(-1)

    return np.hstack([arr[0], arr_inner, arr[-1]])
