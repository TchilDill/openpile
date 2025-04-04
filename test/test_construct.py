from openpile import construct
from openpile.soilmodels import API_clay, API_sand, API_clay_axial, API_sand_axial
from openpile.materials import PileMaterial

import pytest
import numpy as np
import math as m

from pydantic import ValidationError


@pytest.fixture
def offshore_wind_pile1():
    # create a steel and circular pile
    return construct.Pile(
        name="",
        material="Steel",
        sections=[
            construct.CircularPileSection(top=0, bottom=-10, diameter=7.5, thickness=0.07),
            construct.CircularPileSection(top=-10, bottom=-40, diameter=7.5, thickness=0.08),
        ],
    )


@pytest.fixture
def circular_slender_pile():
    # create a steel and circular pile
    return construct.Pile(
        name="",
        material="Steel",
        sections=[
            construct.CircularPileSection(top=0, bottom=-10, diameter=0.7, thickness=0.02),
            construct.CircularPileSection(top=-10, bottom=-40, diameter=0.7, thickness=0.02),
        ],
    )


@pytest.fixture
def circular_slender_pile():
    # create a steel and circular pile
    return construct.Pile(
        name="",
        material="Steel",
        sections=[
            construct.CircularPileSection(top=0, bottom=-10, diameter=0.7, thickness=0.02),
            construct.CircularPileSection(top=-10, bottom=-40, diameter=0.7, thickness=0.02),
        ],
    )


class TestPileSection:
    def test_pile_section(self):
        section = construct.CircularPileSection(top=10, bottom=5, diameter=9.4, thickness=0.31)
        assert section.width == 9.4
        assert section.top == 10
        assert section.bottom == 5
        assert section.length == 5

    def test_elevations(self):
        with pytest.raises(ValueError):
            construct.CircularPileSection(top=-10, bottom=5, diameter=9.4, thickness=0.31)

    def test_default_thickness(self):
        section = construct.CircularPileSection(top=10, bottom=5, diameter=9)
        assert section.thickness == 4.5


class TestPile:
    def test_main_constructor(self, offshore_wind_pile1):
        """Main constructor of Pile object, check if pile.data
        has even entries and that steel E value is 210GPa
        """

        # check Young modulus is indeed Steel
        assert offshore_wind_pile1.E == 210e6
        # check even numbers of row for dataframe
        assert offshore_wind_pile1.data.values.shape[0] % 2 == 0

    def test_custom_material(self):
        pile = construct.Pile(
            name="",
            material="Steel",
            sections=[
                construct.CircularPileSection(top=0.1, bottom=-9.9, diameter=8.0, thickness=0.07),
            ],
        )

        assert pile.material.name == "Steel"
        assert isinstance(pile.material, PileMaterial)

    def test_pile_width(self):
        pile = construct.Pile(
            name="",
            material="Steel",
            sections=[
                construct.CircularPileSection(top=0.1, bottom=-9.9, diameter=8.0, thickness=0.07),
                construct.CircularPileSection(top=-9.9, bottom=-39.9, diameter=8.0, thickness=0.08),
            ],
        )

        assert pile.sections[0].width == 8.0
        assert pile.sections[1].width == 8.0

    def test_pile_length(self):
        pile = construct.Pile(
            name="",
            material="Steel",
            sections=[
                construct.CircularPileSection(top=10, bottom=-12, diameter=7.5, thickness=0.07),
                construct.CircularPileSection(top=-12, bottom=-28, diameter=8.5, thickness=0.08),
            ],
        )

        assert pile.length == 38.0

    def test_pile_bottom(self):
        pile = construct.Pile(
            name="",
            material="Steel",
            sections=[
                construct.CircularPileSection(top=10, bottom=-12, diameter=7.5, thickness=0.07),
                construct.CircularPileSection(top=-12, bottom=-28, diameter=8.5, thickness=0.08),
            ],
        )

        assert pile.bottom_elevation == -28.0

    def test_pile_area(self):
        """check pile area"""
        # Create a pile instance with two sections of respectively 10m and 30m length.
        pile = construct.Pile(
            name="",
            material="Steel",
            sections=[
                construct.CircularPileSection(top=10, bottom=0, diameter=1.0, thickness=0.5),
                construct.CircularPileSection(top=0, bottom=-30, diameter=1.0, thickness=0.5),
            ],
        )

        assert pile.sections[0].area == 0.25 * m.pi
        assert pile.sections[1].area == 0.25 * m.pi
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
    def test_boundary_conditions_simple_beam(self, circular_slender_pile):
        """this test should show that specific nodes are created where x2mesh is used."""
        model = construct.Model(
            name="",
            pile=circular_slender_pile,
            boundary_conditions=[
                construct.BoundaryFixation(
                    elevation=circular_slender_pile.top_elevation, y=True, z=True
                ),
                construct.BoundaryFixation(
                    elevation=circular_slender_pile.bottom_elevation, y=True
                ),
                construct.BoundaryDisplacement(
                    elevation=0.5
                    * (
                        circular_slender_pile.top_elevation + circular_slender_pile.bottom_elevation
                    ),
                    y=0.1,
                ),
            ],
        )

        results = model.solve()
        assert results.deflection["Deflection [m]"].max() == 0.1

    def test_boundary_conditions_fixed_end_beam(self, circular_slender_pile):
        """this test checks that the beam calculation with one end fixed and
        the other end loaded by a transverse force results in the correct known solution"""
        model = construct.Model(
            name="",
            pile=circular_slender_pile,
            boundary_conditions=[
                construct.BoundaryFixation(
                    elevation=circular_slender_pile.bottom_elevation, x=True, y=True, z=True
                ),
                construct.BoundaryForce(elevation=circular_slender_pile.top_elevation, y=1.0),
            ],
        )

        results = model.solve()
        assert m.isclose(
            results.deflection["Deflection [m]"][0],
            circular_slender_pile.length**3
            / (
                3
                * circular_slender_pile.E
                * circular_slender_pile.sections[0].second_moment_of_area
            ),
            abs_tol=1e-5,
        )

    def test_constructor_with_x2mesh(self, circular_slender_pile):
        """this test should show that specific nodes are created where x2mesh is used."""
        model = construct.Model(name="", pile=circular_slender_pile, x2mesh=[-10.38, -10.918])

        assert any(model.nodes_coordinates["z [m]"] == -10.38)
        assert any(model.nodes_coordinates["z [m]"] == -10.918)

    def test_constructor_wo_soil_models(self):
        """
        check that model can still be created with no lateral or axial models"""
        try:
            model = construct.Model(
                name="",
                pile=construct.Pile.create_tubular(
                    name="",
                    top_elevation=20,
                    bottom_elevation=-30,
                    material="Steel",
                    diameter=7,
                    wt=0.08,
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
        assert not np.all(
            model.get_distributed_lateral_springs()[["VAL 0", "VAL 1", "VAL 2"]].values
        )

        # # check that m_t springs are all zero
        assert not np.all(
            model.get_distributed_rotational_springs()[["VAL 0", "VAL 1", "VAL 2"]].values
        )

        assert not np.all(model.get_base_rotational_spring()[["VAL 0", "VAL 1", "VAL 2"]].values)

        assert not np.all(model.get_base_shear_spring()[["VAL 0", "VAL 1", "VAL 2"]].values)

    def test_entrapped_soil_weight_above_water_table(self):
        """calculate the weight of the soil inside a pile that is above water table"""

        # the special diameter and wall thickness is calculated and applied such that
        # a metre long of pile with this diameter ie quivalent
        # to one cubic metre
        special_diameter = (4 / m.pi) ** 0.5
        special_wallthickness = 0.01
        soil_weight = 18

        # a pile with the special diameter and an unreasonably thin wall thickness
        p = construct.Pile.create_tubular(
            name="<pile name>",
            top_elevation=0,
            bottom_elevation=-10,
            diameter=special_diameter + (2 * special_wallthickness),
            wt=special_wallthickness,
        )

        # Create a 40m deep offshore Soil Profile with a 15m water column
        sp = construct.SoilProfile(
            name="Offshore Soil Profile",
            top_elevation=0,
            water_line=-15,
            layers=[
                construct.Layer(
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

        model = construct.Model(name="", pile=p, soil=sp)
        # check
        assert m.isclose(model.entrapped_soil_weight, soil_weight * p.length)

    def test_entrapped_soil_weight_below_water_table(self):
        """calculate the weight of the soil inside a pile that is submerged in water"""

        # the special diameter and wall thickness is calculated and applied such that
        # a metre long of pile with this diameter ie quivalent
        # to one cubic metre
        special_diameter = (4 / m.pi) ** 0.5
        special_wallthickness = 0.01
        soil_weight = 18

        # a pile with the special diameter and an unreasonably thin wall thickness
        p = construct.Pile.create_tubular(
            name="<pile name>",
            top_elevation=0,
            bottom_elevation=-10,
            diameter=special_diameter + (2 * special_wallthickness),
            wt=special_wallthickness,
        )

        # Create a 40m deep offshore Soil Profile with a 15m water column
        sp = construct.SoilProfile(
            name="Offshore Soil Profile",
            top_elevation=0,
            water_line=10,
            layers=[
                construct.Layer(
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

        model = construct.Model(name="", pile=p, soil=sp)
        # check
        assert m.isclose(model.entrapped_soil_weight, (soil_weight - 10) * p.length)

    def test_submerged_effective_pile_weight(self):

        #  the special diameter and wall thickness is calculated and applied such that
        # a metre long of pile with this diameter ie quivalent
        # to one cubic metre
        special_diameter = 10 / m.pi
        special_wallthickness = 0.001
        steel_weight = 78

        # a pile with the special diameter and an unreasonably thin wall thickness
        p = construct.Pile.create_tubular(
            name="<pile name>",
            top_elevation=0,
            bottom_elevation=-100,
            diameter=special_diameter,
            wt=special_wallthickness,
        )

        # Create a 40m deep offshore Soil Profile with a 15m water column
        sp = construct.SoilProfile(
            name="Offshore Soil Profile",
            top_elevation=0,
            water_line=15,
            layers=[
                construct.Layer(
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

        model = construct.Model(name="", pile=p, soil=sp)

        # check
        assert m.isclose(model.effective_pile_weight, (steel_weight - 10), abs_tol=0.1)

    def test_half_submerged_effective_pile_weight(self):

        #  the special diameter and wall thickness is calculated and applied such that
        # a metre long of pile with this diameter ie quivalent
        # to one cubic metre
        special_diameter = 10 / m.pi
        special_wallthickness = 0.001
        steel_weight = 78

        # a pile with the special diameter and an unreasonably thin wall thickness
        p = construct.Pile.create_tubular(
            name="<pile name>",
            top_elevation=0,
            bottom_elevation=-100,
            diameter=special_diameter,
            wt=special_wallthickness,
        )

        # Create a 40m deep offshore Soil Profile with a 15m water column
        sp = construct.SoilProfile(
            name="Offshore Soil Profile",
            top_elevation=0,
            water_line=-50,
            layers=[
                construct.Layer(
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

        model = construct.Model(name="", pile=p, soil=sp)

        # check
        target_weight = 0.5 * ((steel_weight - 10) + steel_weight)
        assert m.isclose(model.effective_pile_weight, target_weight, abs_tol=0.1)

    def test_axial_capacity_same_as_winkler(self):

        p = construct.Pile.create_tubular(
            name="", top_elevation=0, bottom_elevation=-20, diameter=7.5, wt=0.075
        )
        # Create a 40m deep offshore Soil Profile with a 15m water column
        sp = construct.SoilProfile(
            name="Offshore Soil Profile",
            top_elevation=0,
            water_line=15,
            layers=[
                construct.Layer(
                    name="medium dense sand",
                    top=0,
                    bottom=-20,
                    weight=18,
                    axial_model=API_sand_axial(delta=28),
                ),
            ],
        )
        # Create Model
        M = construct.Model(name="Settlement", pile=p, soil=sp)
        # Apply bottom fixity along lateral axis
        M.set_support(elevation=-20, Ty=True)
        M.set_support(elevation=0, Ty=True)
        # Apply axial and lateral loads
        M.set_pointdisplacement(elevation=0, Tz=-1)
        # Run analysis
        result = M.solve()

        # test
        assert m.isclose(
            -(M.tip_resistance + M.shaft_resistance[0]), result.forces["N [kN]"][0], rel_tol=1e-3
        )
