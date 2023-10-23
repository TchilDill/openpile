import pytest
from openpile.construct import Pile, SoilProfile, Layer, Model
from openpile.soilmodels import API_sand
from openpile.analyze import winkler
from openpile.utils.multipliers import durkhop

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
    M1.set_support(elevation=-40, Tx=True)
    # Apply axial and lateral loads
    M1.set_pointload(elevation=0, Mz=-300e3, Py=30e3)
    R1 = winkler(M1)

    # Create Model 2
    M2 = Model(name="<model name>", pile=create_pile, soil=create_duhrkop_soilprofile)
    # Apply bottom fixity along x-axis
    M2.set_support(elevation=-40, Tx=True)
    # Apply axial and lateral loads
    M2.set_pointload(elevation=0, Mz=-300e3, Py=30e3)
    R2 = winkler(M2)

    assert m.isclose(
        R1.details()["Max. deflection [m]"], R2.details()["Max. deflection [m]"], rel_tol=0.01
    )
