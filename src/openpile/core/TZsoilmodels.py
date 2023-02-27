"""

"""
# Import libraries
import math as m
import numpy as np
from numba import njit, prange
from random import random

# API sand function
@njit(parallel=True, cache=True)
def API_sand(sig:float, X:float, phi:float, D:float, Neq:float=1.0, ymax:float=0.2, output_length:int = 20):
    """
    Creates the API sand t-z curve from relevant input. 
    
    ---------
    input:
        sig: float    
            Vertical effective stress [unit: kPa]
        X: float
            Depth of the curve w.r.t. mudline [unit: m]
        phi: float      
            internal angle of friction of the sand layer [unit: degrees]
        D: float
            Pile width [unit: m]
        Neq: float, by default 1.0
            Number of equivalent cycles [unit: -]
        ymax: float, by default 0.2
            maximum value of y, default is 20% of the pile width
        output_length: int, by default 20
            Number of discrete point along the springs
    ---------
    Returns curve with 2 vectors:
        t: numpy 1darray
            t vector [unit: kPa/metre of pile length]
        z: numpy 1darray 
            z vector [unit: m]
    ---------    
    """

    return t, z

# API sand function
@njit(parallel=True, cache=True)
def API_clay(sig:float, X:float, Su:float, D:float, output_length:int = 20):
    """
    Creates the API clay t-z curve from relevant input. 
    

    ---------
    input:
        sig: float    
            Vertical effective stress [unit: kPa]
        X: float
            Depth of the curve w.r.t. mudline [unit: m]
        Su : float      
            Undrained shear strength [unit: kPa]
        D: float
            Pile width [unit: m]
        output_length: int, by default 20
            Number of discrete point along the springs
    ---------
    Returns curve with 2 vectors:
        t: numpy 1darray
            t vector [unit: kPa/metre of pile length]
        z: numpy 1darray 
            z vector [unit: m]
    ---------    
    """    
    
    return t, z