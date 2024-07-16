# Import libraries
import math as m
import numpy as np
from numba import njit

# maximum resistance values
@njit(cache=True)
def _Qmax_api_clay(
    Su: float,
) -> float:
    # Unit end-bearing [kPa]
    return 9 * Su


@njit(cache=True)
def _Qmax_api_sand(
    sig: float,
    delta: float,
) -> float:
    # important variables
    delta_table = np.array([0, 15, 20, 25, 30, 35, 100], dtype=np.float32)
    Nq_table = np.array([8, 8, 12, 20, 40, 50, 50], dtype=np.float32)
    Qmax_table = np.array([1900, 1900, 2900, 4800, 9600, 12000, 12000], dtype=np.float32)

    Nq = np.interp(delta, delta_table, Nq_table)
    Qmax = np.interp(delta, delta_table, Qmax_table)

    # Unit end-bearing [kPa]
    return min(Qmax, sig * Nq)


@njit(cache=True)
def _fmax_api_clay(
    sig: float,
    Su: float,
    alpha_limit: float,
) -> float:
    """Creates the maximum skin friction.

    The methdology follows the API clay method of axial capacity found in .

    Parameters
    ----------
    sig : float
        vertical effcitve stress in kPa.
    Su : float
        undrained shear strength in kPa.
    alpha_limit : float
        limit value for the skin friction normalized to undrained shear strength.


    Returns
    -------
    float
        unit skin friction in kPa.
    """

    # important variables
    if sig == 0.0:
        psi = Su / 0.001
    else:
        psi = Su / sig

    if psi > 1.0:
        alpha = min(0.5 * psi ** (-0.25), alpha_limit)
    else:
        alpha = min(0.5 * psi ** (-0.5), alpha_limit)

    # Unit skin friction [kPa]
    return alpha * Su


# SPRING FUNCTIONS --------------------------------------------

# API sand function
@njit(cache=True)
def _fmax_api_sand(
    sig: float,
    delta: float,
    K: float = 0.8,
) -> float:
    """Creates the maximum skin friction.

    The methdology follows the API sand method of axial capacity found in .

    Parameters
    ----------
    sig : float
        vertical effcitve stress in kPa.
    delta: float
        interface friction angle in degrees
    K: float
        coefficient of lateral pressure.
        (0.8 for open-ended piles and 1.0 for cloased-ended)

    Returns
    -------
    float
        unit skin friction in kPa.
    """

    # important variables
    delta_table = np.array([0, 15, 20, 25, 30, 35, 100], dtype=np.float32)
    fs_max_table = np.array([47.8, 47.8, 67, 81.3, 95.7, 114.8, 114.8], dtype=np.float32)

    # limit unit skin friction according to API ref page 59
    fs_max = np.interp(delta, delta_table, fs_max_table)

    # Unit skin friction [kPa]
    return min(fs_max, K * sig * m.tan(delta * m.pi / 180.0))
