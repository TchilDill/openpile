import matplotlib.pyplot as plt
import openpile.utils.py_curves as py
import pytest 
import math as m
import numpy as np

import openpile.utils.py_curves as py

@pytest.fixture
def make_pmax_api_sand():
    def make(
        sig: float,
        X: float,
        phi: float,
        D: float,
    ):
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
        return min(C3 * sig * D, C1 * sig * X + C2 * sig * D)

    return make

@pytest.fixture
def make_pmax_api_clay():
    def make(
        sig: float,
        X: float,
        Su: float,
        D: float,
        J: float,
    ):

        return min(9*Su*X, (3 * Su + sig) * D + J * Su * X)
        
    return make


@pytest.mark.parametrize("xsigma",[0,75,200])
@pytest.mark.parametrize("xX",[0.1,5,20])
@pytest.mark.parametrize("xphi",[27,35,40])
@pytest.mark.parametrize("xD",[0.2,1.5,5.0,10.0])
@pytest.mark.parametrize("xkind",["static","cyclic"])
@pytest.mark.parametrize("xwater",[True,False])
def test_api_sand(make_pmax_api_sand,xsigma,xX, xphi, xD, xkind, xwater):
    #create spring
    y, p = py.api_sand(sig=xsigma,X=xX,phi=xphi, D=xD, kind=xkind, below_water_table=xwater)
    #helper fct
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    #check if sorted
    assert is_sorted(y)
    assert is_sorted(p)
    #check if origin is (0,0)
    assert p[0] == 0.0
    assert y[0] == 0.0
    #check pmax
    if xkind == "cyclic":
        A = 0.9
    elif xkind=="static":
        A = max(0.9, 3 - 0.8 * xX / xD)
    assert m.isclose(p[-1], A*make_pmax_api_sand(xsigma,xX,xphi,xD), rel_tol=0.01, abs_tol=0.1)


@pytest.mark.parametrize("xsigma",[0,75,200])
@pytest.mark.parametrize("xX",[0.1,5,20])
@pytest.mark.parametrize("xSu",[0.01,50,125,300])
@pytest.mark.parametrize("xe50",[0.005,0.0125,0.03])
@pytest.mark.parametrize("xD",[0.2,1.5,5.0,10.0])
@pytest.mark.parametrize("xJ",[0.25,0.5])
@pytest.mark.parametrize("xkind",["static","cyclic"])
def test_api_clay(make_pmax_api_clay,xsigma, xX, xSu, xe50, xD, xJ, xkind):
    #create spring
    y, p = py.api_clay(sig=xsigma, X=xX, Su=xSu, eps50=xe50, D=xD, J=xJ, kind=xkind)
    #helper fct
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    #check if sorted
    assert is_sorted(y)
    #check if origin is (0,0)
    assert p[0] == 0.0
    assert y[0] == 0.0
    #check initial stiffness
    pmax = make_pmax_api_clay(xsigma,xX,xSu,xD,xJ)
    kini = 0.23*pmax/(0.25*xe50*xD)
    assert m.isclose( (p[1]-p[0])/(y[1]-y[0]), kini, rel_tol=0.02 )
    #calculate pmax
    #check residual and max p
    Xr = max((6 * xD) / (xsigma / xX * xD / xSu + xJ), 2.5 * xD)

    factor = min(1.0,xX/Xr)
        
    if xkind=="cyclic" and xSu > 96:
        pres = 0.5*pmax*factor
        pu = 0.5*pmax
    elif xkind=="cyclic" and xSu <= 96:
        pres = 0.7185*pmax*factor
        pu = 0.7185*pmax
    elif xkind=="static":
        pres = pmax
        pu = pmax
    assert m.isclose(p[-1],pres, rel_tol=0.01, abs_tol=0.1)
    assert m.isclose(np.max(p), pu, rel_tol=0.01, abs_tol=0.1)

