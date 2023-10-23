import pytest
from openpile.construct import Pile, SoilProfile, Layer, Model
from openpile.soilmodels import API_sand
from openpile.analyze import winkler
from openpile.utils.multipliers import durkhop

import math as m


class TestUsage:
    def test_ex1_create_a_pile(self):

        # Create a pile instance with two sections of respectively 10m and 30m length.
        pile = Pile(
            name="",
            kind="Circular",
            material="Steel",
            top_elevation=0,
            pile_sections={
                "length": [10, 30],
                "diameter": [7.5, 7.5],
                "wall thickness": [0.07, 0.08],
            },
        )

        # Override young's modulus
        pile.E = 250e6
        # Check young's modulus (value in kPa)
        print(pile.E)
        250000000.0
        # Override second moment of area across first section [in meters^4]
        pile.set_I(value=1.11, section=1)
        # Check updated second moment of area
        print(pile)
        # Override pile's width or pile's diameter [in meters]
        pile.width = 2.22
        # Check updated width or diameter
        # Override pile's area  [in meters^2]
        pile.area = 1.0

    def test_ex2_create_a_spring(self):

        # import p-y curve for api_sand from openpile.utils
        from openpile.utils.py_curves import api_sand

        y, p = api_sand(
            sig=50,  # vertical stress in kPa
            X=5,  # depth in meter
            phi=35,  # internal angle of friction
            D=5,  # the pile diameter
            below_water_table=True,  # use initial subgrade modulus under water
            kind="static",  # static curve
        )

    def test_ex3_create_laer(self):

        from openpile.construct import Layer
        from openpile.soilmodels import API_clay

        # Create a layer
        layer1 = Layer(
            name="Soft Clay",
            top=0,
            bottom=-10,
            weight=18,
            lateral_model=API_clay(Su=[30, 35], eps50=[0.01, 0.02], kind="static"),
        )

        print(layer1)

    def test_ex_4_create_soil_profile(self):

        from openpile.construct import SoilProfile, Layer
        from openpile.soilmodels import API_sand, API_clay

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
                    weight=18,
                    lateral_model=API_sand(phi=33, kind="cyclic"),
                ),
                Layer(
                    name="firm clay",
                    top=-20,
                    bottom=-40,
                    weight=18,
                    lateral_model=API_clay(Su=[50, 70], eps50=0.015, kind="cyclic"),
                ),
            ],
        )

        print(sp)

    def test_ex5_create_model(self):

        from openpile.construct import Pile, SoilProfile, Layer, Model
        from openpile.soilmodels import API_clay, API_sand

        p = Pile.create_tubular(
            name="<pile name>", top_elevation=0, bottom_elevation=-40, diameter=7.5, wt=0.075
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
                    weight=18,
                    lateral_model=API_sand(phi=33, kind="cyclic"),
                ),
                Layer(
                    name="firm clay",
                    top=-20,
                    bottom=-40,
                    weight=18,
                    lateral_model=API_clay(Su=[50, 70], eps50=0.015, kind="cyclic"),
                ),
            ],
        )

        # Create Model
        M = Model(name="<model name>", pile=p, soil=sp)

        # Apply bottom fixity along x-axis
        M.set_support(elevation=-40, Tx=True)
        # Apply axial and lateral loads
        M.set_pointload(elevation=0, Px=-20e3, Py=5e3)

        # Run analysis
        from openpile.analyze import winkler

        Result = winkler(M)

        # plot the results
        Result.plot()
