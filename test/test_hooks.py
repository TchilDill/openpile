import pytest
from openpile.construct import Pile, SoilProfile, Layer, Model
from openpile.soilmodels import API_sand
from openpile.winkler import winkler
from openpile.utils.hooks import durkhop

import math as m


@pytest.fixture
def create_pile():
    return Pile.create_tubular(
        name="<pile name>", top_elevation=0, bottom_elevation=-40, diameter=7, wt=0.050
    )


@pytest.fixture
def create_cyclic_soilprofile():
    # Create a 40m deep offshore Soil Profile with a 15m water column
    sp = SoilProfile(
        name="Offshore Soil Profile",
        top_elevation=0,
        water_line=15,
        layers=[
            Layer(
                name="medium dense sand",
                top=0,
                bottom=-40,
                weight=18,
                lateral_model=API_sand(phi=33, kind="cyclic"),
            ),
        ],
    )
    return sp


@pytest.fixture
def create_static_soilprofile():
    # Create a 40m deep offshore Soil Profile with a 15m water column
    sp = SoilProfile(
        name="Offshore Soil Profile",
        top_elevation=0,
        water_line=15,
        layers=[
            Layer(
                name="medium dense sand",
                top=0,
                bottom=-40,
                weight=18,
                lateral_model=API_sand(phi=33, kind="static"),
            ),
        ],
    )
    return sp


@pytest.fixture
def create_duhrkop_soilprofile():
    # Create a 40m deep offshore Soil Profile with a 15m water column
    sp = SoilProfile(
        name="Offshore Soil Profile",
        top_elevation=0,
        water_line=15,
        layers=[
            Layer(
                name="medium dense sand",
                top=0,
                bottom=-40,
                weight=18,
                lateral_model=API_sand(phi=33, kind="cyclic", p_multiplier=durkhop(D=7, ra=0.3)),
            ),
        ],
    )
    return sp


def test_APIcylic_equals_durkhop_ra_03(
    create_pile, create_cyclic_soilprofile, create_duhrkop_soilprofile
):

    # Create Model 1
    M1 = Model(name="<model name>", pile=create_pile, soil=create_cyclic_soilprofile)
    # Apply bottom fixity along x-axis
    M1.set_support(elevation=-40, Tz=True)
    # Apply axial and lateral loads
    M1.set_pointload(elevation=0, Mx=-300e3, Py=10e3)
    R1 = winkler(M1)

    # Create Model 2
    M2 = Model(name="<model name>", pile=create_pile, soil=create_duhrkop_soilprofile)
    # Apply bottom fixity along x-axis
    M2.set_support(elevation=-40, Tz=True)
    # Apply axial and lateral loads
    M2.set_pointload(elevation=0, Mx=-300e3, Py=10e3)
    R2 = winkler(M2)

    assert m.isclose(
        R1.details()["Max. deflection [m]"], R2.details()["Max. deflection [m]"], rel_tol=0.01
    )


def test_pisa_custom_sand_py():

    from openpile.utils.py_curves import custom_pisa_sand, dunkirk_sand
    from openpile.utils.hooks import PISA_depth_variation

    # plot original dunkirk sand curve
    params = {"sig": 50, "X": 5, "Dr": 75, "G0": 50e3, "D": 6, "L": 20}
    y, p = dunkirk_sand(**params)
    # load PISA dunkirk sand depth variation functions
    funcs = PISA_depth_variation.dunkirk_sand_py_pisa_norm_param(D=6, L=20, Dr=75)
    # plot same curve with custom pisa sand curve
    params = {
        "sig": 50,
        "G0": 50e3,
        "D": 6,
        "X_ult": funcs["py_X"](5),
        "Y_ult": funcs["py_Y"](5),
        "n": funcs["py_n"](5),
        "k": funcs["py_k"](5),
    }
    y_custom, p_custom = custom_pisa_sand(**params)

    for i in range(len(y)):
        assert m.isclose(y[i], y_custom[i])
        assert m.isclose(p[i], p_custom[i])


def test_pisa_custom_sand_mt():

    from openpile.utils.mt_curves import custom_pisa_sand, dunkirk_sand
    from openpile.utils.hooks import PISA_depth_variation

    # plot original dunkirk sand curve
    params = {"sig": 50, "X": 5, "Dr": 75, "G0": 50e3, "D": 6, "L": 20, "p": 100}
    y, p = dunkirk_sand(**params)
    # load PISA dunkirk sand depth variation functions
    funcs = PISA_depth_variation.dunkirk_sand_mt_pisa_norm_param(L=20, Dr=75)
    # plot same curve with custom pisa sand curve
    params = {
        "sig": 50,
        "G0": 50e3,
        "D": 6,
        "p": 100,
        "X_ult": funcs["mt_X"](5),
        "Y_ult": funcs["mt_Y"](5),
        "n": funcs["mt_n"](5),
        "k": funcs["mt_k"](5),
    }
    y_custom, p_custom = custom_pisa_sand(**params)

    for i in range(len(y)):
        assert m.isclose(y[i], y_custom[i])
        assert m.isclose(p[i], p_custom[i])


def test_pisa_custom_clay_py():

    from openpile.utils.py_curves import custom_pisa_clay, cowden_clay
    from openpile.utils.hooks import PISA_depth_variation

    # plot original dunkirk sand curve
    params = {"X": 5, "Su": 75, "G0": 50e3, "D": 6}
    y, p = cowden_clay(**params)
    # load PISA dunkirk sand depth variation functions
    funcs = PISA_depth_variation.cowden_clay_py_pisa_norm_param(D=6)
    # plot same curve with custom pisa sand curve
    params = {
        "Su": 75,
        "G0": 50e3,
        "D": 6,
        "X_ult": funcs["py_X"](5),
        "Y_ult": funcs["py_Y"](5),
        "n": funcs["py_n"](5),
        "k": funcs["py_k"](5),
    }
    y_custom, p_custom = custom_pisa_clay(**params)

    for i in range(len(y)):
        assert m.isclose(y[i], y_custom[i])
        assert m.isclose(p[i], p_custom[i])


def test_pisa_custom_clay_mt():

    from openpile.utils.mt_curves import custom_pisa_clay, cowden_clay
    from openpile.utils.hooks import PISA_depth_variation

    # plot original dunkirk sand curve
    params = {"X": 5, "Su": 75, "G0": 50e3, "D": 6}
    y, p = cowden_clay(**params)
    # load PISA dunkirk sand depth variation functions
    funcs = PISA_depth_variation.cowden_clay_mt_pisa_norm_param(D=6)
    # plot same curve with custom pisa sand curve
    params = {
        "Su": 75,
        "G0": 50e3,
        "D": 6,
        "X_ult": funcs["mt_X"](5),
        "Y_ult": funcs["mt_Y"](5),
        "n": funcs["mt_n"](5),
        "k": funcs["mt_k"](5),
    }
    y_custom, p_custom = custom_pisa_clay(**params)

    for i in range(len(y)):
        assert m.isclose(y[i], y_custom[i])
        assert m.isclose(p[i], p_custom[i])
