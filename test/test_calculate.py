from openpile import construct
from openpile.soilmodels import API_clay
import pytest
import numpy as np
import math as m

from pydantic import ValidationError

from openpile.construct import Pile, SoilProfile, Layer, Model
from openpile.soilmodels import API_clay, API_sand, API_clay_axial
from openpile.calculate import entrapped_soil_weight


def test_entrapped_soil_weight():
    """calculate the weight of the soil inside the pile
    """

    # the special diameter and wall thickness is calculated and applied such that 
    # a metre long of pile with this diameter ie quivalent 
    # to one cubic metre
    special_diameter = (4/m.pi)**0.5
    special_wallthickness = 0.01
    soil_weight = 18

    # a pile with the special diameter and an unreasonably thin wall thickness
    p = Pile.create_tubular(
        name="<pile name>", 
        top_elevation=0, 
        bottom_elevation=-10, 
        diameter=special_diameter+(2*special_wallthickness), 
        wt=special_wallthickness
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

    # Create Model
    M = Model(name="<model name>", pile=p, soil=sp)

    # check
    assert m.isclose(entrapped_soil_weight(M), soil_weight*10)
