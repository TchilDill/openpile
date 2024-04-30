from openpile import construct
from openpile.soilmodels import API_clay
import pytest
import numpy as np
import math as m

from pydantic import ValidationError


class TestPile:
    def test_main_constructor(self):
        """Main constructor of Pile object, check if pile.data
        has even entries and that steel E value is 210GPa
        """
        # create a steel and circular pile
        pile = construct.Pile(
            name="",
            material="Steel",
            pile_sections=[
                construct.CircularPileSection(
                    top_elevation=0, 
                    bottom_elevation=-10, 
                    diameter=7.5, 
                    thickness=0.07
                ),
                construct.CircularPileSection(
                    top_elevation=-10, 
                    bottom_elevation=-40, 
                    diameter=7.5, 
                    thickness=0.08
                ),
            ],
        )


        # check Young modulus is indeed Steel
        assert pile.E == 210e6
        # check even numbers of row for dataframe
        assert pile.data.values.shape[0] % 2 == 0

    def test_pile_width(self):
        pile = construct.Pile(
            name="",
            material="Steel",
            pile_sections=[
                construct.CircularPileSection(
                    top_elevation=0.1, 
                    bottom_elevation=-9.9, 
                    diameter=8.0, 
                    thickness=0.07
                ),
                construct.CircularPileSection(
                    top_elevation=-9.9, 
                    bottom_elevation=-39.9, 
                    diameter=8.0, 
                    thickness=0.08
                ),
            ],
        )

        assert pile.width.mean() == 8.00

    def test_pile_length(self):
        pile = construct.Pile(
            name="",
            material="Steel",
            pile_sections=[
                construct.CircularPileSection(
                    top_elevation=10, 
                    bottom_elevation=-12, 
                    diameter=7.5, 
                    thickness=0.07
                ),
                construct.CircularPileSection(
                    top_elevation=-12, 
                    bottom_elevation=-28, 
                    diameter=8.5, 
                    thickness=0.08
                ),
            ],
        )

        assert pile.length == 38.0

    def test_pile_bottom(self):
        pile = construct.Pile(
            name="",
            material="Steel",
            pile_sections=[
                construct.CircularPileSection(
                    top_elevation=10, 
                    bottom_elevation=-12, 
                    diameter=7.5, 
                    thickness=0.07
                ),
                construct.CircularPileSection(
                    top_elevation=-12, 
                    bottom_elevation=-28, 
                    diameter=8.5, 
                    thickness=0.08
                ),
            ],
        )

        assert pile.bottom_elevation == -28.0

    def test_pile_area(self):
        """check pile area"""
        # Create a pile instance with two sections of respectively 10m and 30m length.
        pile = construct.Pile(
            name="",
            material="Steel",
            pile_sections=[
                construct.CircularPileSection(
                    top_elevation=10, 
                    bottom_elevation=0, 
                    diameter=1.0, 
                    thickness=0.5
                ),
                construct.CircularPileSection(
                    top_elevation=0, 
                    bottom_elevation=-30, 
                    diameter=1.0, 
                    thickness=0.5
                ),
            ],
        )

        assert pile.pile_sections[0].get_area() == 0.25 * m.pi
        assert pile.pile_sections[1].get_area() == 0.25 * m.pi
        assert pile.tip_area == 0.25 * m.pi
        assert pile.tip_footprint == 0.25 * m.pi

class TestLayer:
    def test_constructor(self):
        try:
            layer = construct.Layer(
                name="Soft Clay",
                top=0,
                bottom=-10,
                weight=19,
                lateral_model=API_clay(Su=[30, 35], eps50=[0.01, 0.02], kind="cyclic"),
            )
        except Exception:
            assert False, "Constructor did not work"

    def test_wrong_order_top_and_bottom(self):
        with pytest.raises(ValidationError):
            layer = construct.Layer(
                name="Soft Clay",
                top=-10,
                bottom=0,
                weight=19,
                lateral_model=API_clay(Su=[30, 35], eps50=[0.01, 0.02], kind="cyclic"),
            )

    def test_equal_top_and_bottom(self):
        with pytest.raises(ValidationError):
            layer = construct.Layer(
                name="Soft Clay",
                top=-5,
                bottom=-5,
                weight=19,
                lateral_model=API_clay(Su=[30, 35], eps50=[0.01, 0.02], kind="cyclic"),
            )

    def test_wrong_unit_weight(self):
        """Constructor should not accept a soil unit weight equal to 10 or less as
        it would be less dense than water and likely the user made a mistake with
        the effective unit weight.
        """
        with pytest.raises(ValidationError):
            layer = construct.Layer(
                name="Soft Clay",
                top=0,
                bottom=-10,
                weight=10,
                lateral_model=API_clay(Su=[30, 35], eps50=[0.01, 0.02], kind="static"),
            )


class TestSoilProfile:
    def test_wrong_layers_elevations_gap(self):
        with pytest.raises(ValidationError):
            sp = construct.SoilProfile(
                name="",
                top_elevation=0,
                water_line=0,
                layers=[
                    construct.Layer(
                        name="",
                        top=0,
                        bottom=-10,
                        weight=20,
                    ),
                    construct.Layer(
                        name="",
                        top=-15,
                        bottom=-30,
                        weight=20,
                    ),
                ],
            )

    def test_wrong_layers_elevations_overlap(self):
        with pytest.raises(ValidationError):
            sp = construct.SoilProfile(
                name="",
                top_elevation=0,
                water_line=0,
                layers=[
                    construct.Layer(
                        name="",
                        top=0,
                        bottom=-10,
                        weight=20,
                    ),
                    construct.Layer(
                        name="",
                        top=-8,
                        bottom=-30,
                        weight=20,
                    ),
                ],
            )

    def test_wrong_top_elevation(self):
        with pytest.raises(ValidationError):
            sp = construct.SoilProfile(
                name="",
                top_elevation=10,
                water_line=0,
                layers=[
                    construct.Layer(
                        name="",
                        top=11,
                        bottom=-10,
                        weight=20,
                    ),
                    construct.Layer(
                        name="",
                        top=-10,
                        bottom=-30,
                        weight=20,
                    ),
                ],
            )

    def test_wrong_bottom_elevation(self):
        sp = construct.SoilProfile(
            name="",
            top_elevation=10,
            water_line=0,
            layers=[
                construct.Layer(
                    name="",
                    top=10,
                    bottom=-10,
                    weight=20,
                ),
                construct.Layer(
                    name="",
                    top=-10,
                    bottom=-30,
                    weight=20,
                ),
            ],
        )

        assert sp.bottom_elevation == -30


class TestModel:
    def test_constructor_wo_soil_models(self):
        """
        check that model can still be created with no lateral or axial models"""
        try:
            model = construct.Model(
                name="",
                pile=construct.Pile(
                    name="",
                    top_elevation=20,
                    material="Steel",
                    pile_sections={"length": [50], "diameter": [7], "wall thickness": [0.08]},
                ),
                soil=construct.SoilProfile(
                    name="",
                    top_elevation=10,
                    water_line=0,
                    layers=[
                        construct.Layer(
                            name="",
                            top=10,
                            bottom=-10,
                            weight=20,
                        ),
                        construct.Layer(
                            name="",
                            top=-10,
                            bottom=-30,
                            weight=20,
                        ),
                    ],
                ),
            )
        except Exception:
            assert False, f"Constructor does not work"

        # check that p_y springs are all zero
        assert not np.all(model.get_py_springs()[["VAL 0", "VAL 1", "VAL 2"]].values)

        # # check that m_t springs are all zero
        # assert not np.all( model.get_mt_springs()[['VAL 0','VAL 1','VAL 2']].values )
