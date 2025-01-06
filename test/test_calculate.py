import pytest
import numpy as np
import math as m

from pydantic import ValidationError

from openpile.construct import Pile, SoilProfile, Layer, Model
from openpile.soilmodels import API_sand

def test_entrapped_soil_weight_above_water_table():
    """calculate the weight of the soil inside a pile that is above water table"""

    # the special diameter and wall thickness is calculated and applied such that
    # a metre long of pile with this diameter ie quivalent
    # to one cubic metre
    special_diameter = (4 / m.pi) ** 0.5
    special_wallthickness = 0.01
    soil_weight = 18

    # a pile with the special diameter and an unreasonably thin wall thickness
    p = Pile.create_tubular(
        name="<pile name>",
        top_elevation=0,
        bottom_elevation=-10,
        diameter=special_diameter + (2 * special_wallthickness),
        wt=special_wallthickness,
    )

    # Create a 40m deep offshore Soil Profile with a 15m water column
    sp = SoilProfile(
        name="Offshore Soil Profile",
        top_elevation=0,
        water_line=-15,
        layers=[
            Layer(
                name="medium dense sand",
                top=0,
                bottom=-20,
                weight=soil_weight,
                lateral_model=API_sand(
                    phi=33,
                    kind="cyclic",
                    extension="mt_curves",
                ),
            ),
        ],
    )

    model = Model(name="", pile=p, soil=sp)
    # check
    assert m.isclose(model.entrapped_soil_weight, soil_weight * p.length)


def test_entrapped_soil_weight_below_water_table():
    """calculate the weight of the soil inside a pile that is submerged in water"""

    # the special diameter and wall thickness is calculated and applied such that
    # a metre long of pile with this diameter ie quivalent
    # to one cubic metre
    special_diameter = (4 / m.pi) ** 0.5
    special_wallthickness = 0.01
    soil_weight = 18

    # a pile with the special diameter and an unreasonably thin wall thickness
    p = Pile.create_tubular(
        name="<pile name>",
        top_elevation=0,
        bottom_elevation=-10,
        diameter=special_diameter + (2 * special_wallthickness),
        wt=special_wallthickness,
    )

    # Create a 40m deep offshore Soil Profile with a 15m water column
    sp = SoilProfile(
        name="Offshore Soil Profile",
        top_elevation=0,
        water_line=10,
        layers=[
            Layer(
                name="medium dense sand",
                top=0,
                bottom=-20,
                weight=soil_weight,
                lateral_model=API_sand(
                    phi=33,
                    kind="cyclic",
                    extension="mt_curves",
                ),
            ),
        ],
    )

    model = Model(name="", pile=p, soil=sp)
    # check
    assert m.isclose(model.entrapped_soil_weight, (soil_weight-10) * p.length)


def test_submerged_effective_pile_weight():

    #  the special diameter and wall thickness is calculated and applied such that
    # a metre long of pile with this diameter ie quivalent
    # to one cubic metre
    special_diameter = 10 / m.pi
    special_wallthickness = 0.001
    steel_weight = 78

    # a pile with the special diameter and an unreasonably thin wall thickness
    p = Pile.create_tubular(
        name="<pile name>",
        top_elevation=0,
        bottom_elevation=-100,
        diameter=special_diameter,
        wt=special_wallthickness,
    )

    # Create a 40m deep offshore Soil Profile with a 15m water column
    sp = SoilProfile(
        name="Offshore Soil Profile",
        top_elevation=0,
        water_line=15,
        layers=[
            Layer(
                name="medium dense sand",
                top=0,
                bottom=-100,
                weight=18,
                lateral_model=API_sand(
                    phi=33,
                    kind="cyclic",
                    extension="mt_curves",
                ),
            ),
        ],
    )

    model = Model(name="", pile=p, soil=sp)

    # check
    assert m.isclose(
        model.effective_pile_weight, (steel_weight - 10), abs_tol=0.1
    )


def test_half_submerged_effective_pile_weight():

    #  the special diameter and wall thickness is calculated and applied such that
    # a metre long of pile with this diameter ie quivalent
    # to one cubic metre
    special_diameter = 10 / m.pi
    special_wallthickness = 0.001
    steel_weight = 78

    # a pile with the special diameter and an unreasonably thin wall thickness
    p = Pile.create_tubular(
        name="<pile name>",
        top_elevation=0,
        bottom_elevation=-100,
        diameter=special_diameter,
        wt=special_wallthickness,
    )

    # Create a 40m deep offshore Soil Profile with a 15m water column
    sp = SoilProfile(
        name="Offshore Soil Profile",
        top_elevation=0,
        water_line=-50,
        layers=[
            Layer(
                name="medium dense sand",
                top=0,
                bottom=-100,
                weight=18,
                lateral_model=API_sand(
                    phi=33,
                    kind="cyclic",
                    extension="mt_curves",
                ),
            ),
        ],
    )

    model = Model(name="", pile=p, soil=sp)

    # check
    target_weight = 0.5 * ((steel_weight - 10) + steel_weight)
    assert m.isclose(model.effective_pile_weight, target_weight, abs_tol=0.1)
