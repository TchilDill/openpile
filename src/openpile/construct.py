"""
`Construct` module
==================

The `construct` module is used to construct all objects that 
form the inputs to calculations in openpile. 


These objects include:

- the Pile
- the SoilProfile
  - the Layer
- the Model

**Usage**

>>> from openpile.construct import Pile, SoilProfile, Layer, Model

"""

import math as m
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Union
from typing_extensions import Literal
from pydantic import (
    BaseModel,
    Field,
    root_validator,
    validator,
    PositiveFloat,
    confloat,
    conlist,
    constr,
    Extra,
    ValidationError,
)
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics
import openpile.core.validation as validation
import openpile.soilmodels as soilmodels

from openpile.core import misc

from openpile.soilmodels import ConstitutiveModel

from openpile.core.misc import generate_color_string


class PydanticConfig:
    arbitrary_types_allowed = True
    extra = Extra.forbid
    post_init_call = "after_validation"


@dataclass(config=PydanticConfig)
class Pile:
    """
    A class to create the pile.

    Parameters
    ----------
    name : str
        Pile/Structure's name.
    top_elevation : float
        top elevation of the pile. Note that this elevation provides a reference point to
        know where the pile is located, especially with respect to other object such as a SoilProfile.
    pile_sections : Dict[str, List[float]]
        argument that stores the relevant data of each pile segment.
        Below are the needed keys for the available piles:
        - kind:'Circular' >> keys:['length', 'diameter', 'wall thickness']
    kind : Literal["Circular",]
        type of pile or type of cross-section. by default "Circular"
    material : Literal["Steel",]
        material the pile is made of. by default "Steel"


    Example
    -------

    >>> from openpile.construct import Pile

    >>> # Create a pile instance with two sections of respectively 10m and 30m length.
    >>> pile = Pile(name = "",
    >>>         kind='Circular',
    >>>         material='Steel',
    >>>         top_elevation = 0,
    >>>         pile_sections={
    >>>             'length':[10,30],
    >>>             'diameter':[7.5,7.5],
    >>>             'wall thickness':[0.07, 0.08],
    >>>         }
    >>>     )
    """

    #: name of the pile
    name: str
    #: select the type of pile, can be of ('Circular', )
    kind: Literal["Circular"]
    #: select the type of material the pile is made of, can be of ('Steel', )
    material: Literal["Steel"]
    #: top elevation of the pile according to general vertical reference set by user
    top_elevation: float
    #: pile geometry made of a dictionary of lists. the structure of the dictionary depends on the type of pile selected.
    #: There can be as many sections as needed by the user. The length of the listsdictates the number of pile sections.
    pile_sections: Dict[str, List[PositiveFloat]]

    def __post_init__(self):
        # check that dict is correctly entered
        validation.pile_sections_must_be(self)

        # Create material specific specs for given material
        # if steel
        if self.material == "Steel":
            # unit weight
            self._uw = 78.0  # kN/m3
            # young modulus
            self._young_modulus = 210.0e6  # kPa
            # Poisson's ratio
            self._nu = 0.3
        else:
            raise UserWarning

        self._shear_modulus = self._young_modulus / (2 + 2 * self._nu)

        # create pile data used by openpile for mesh and calculations.
        # Create top and bottom elevations
        elevation = []
        # add bottom of section i and top of section i+1 (essentially the same values)
        for idx, val in enumerate(self.pile_sections["length"]):
            if idx == 0:
                elevation.append(self.top_elevation)
                elevation.append(elevation[-1] - val)
            else:
                elevation.append(elevation[-1])
                elevation.append(elevation[-1] - val)

        # create sectional properties

        # spread
        diameter = []
        # add top and bottom of section i (essentially the same values)
        for idx, val in enumerate(self.pile_sections["diameter"]):
            diameter.append(val)
            diameter.append(diameter[-1])

        # thickness
        thickness = []
        # add top and bottom of section i (essentially the same values)
        for idx, val in enumerate(self.pile_sections["wall thickness"]):
            thickness.append(val)
            thickness.append(thickness[-1])

        # Area & second moment of area
        area = []
        second_moment_of_area = []
        # add top and bottom of section i (essentially the same values)
        for _, (d, wt) in enumerate(
            zip(self.pile_sections["diameter"], self.pile_sections["wall thickness"])
        ):
            # calculate area
            if self.kind == "Circular":
                A = m.pi / 4 * (d**2 - (d - 2 * wt) ** 2)
                I = m.pi / 64 * (d**4 - (d - 2 * wt) ** 4)
                area.append(A)
                area.append(area[-1])
                second_moment_of_area.append(I)
                second_moment_of_area.append(second_moment_of_area[-1])
            else:
                # not yet supporting other kind
                raise ValueError()

        # Create pile data
        self.data = pd.DataFrame(
            data={
                "Elevation [m]": elevation,
                "Diameter [m]": diameter,
                "Wall thickness [m]": thickness,
                "Area [m2]": area,
                "I [m4]": second_moment_of_area,
            }
        )

    def __str__(self):
        return self.data.to_string()

    @property
    def bottom_elevation(self) -> float:
        """
        Bottom elevation of the pile [m VREF].
        """
        return self.top_elevation - sum(self.pile_sections["length"])

    @property
    def length(self) -> float:
        """
        Pile length [m].
        """
        return sum(self.pile_sections["length"])

    @property
    def volume(self) -> float:
        """
        Pile volume [m3].
        """
        A = self.data["Area [m2]"].values[1:]
        L = np.abs(np.diff(self.data["Elevation [m]"].values))
        return round((A * L).sum(), 2)

    @property
    def weight(self) -> float:
        """
        Pile weight [kN].
        """
        return round(self.volume * self._uw, 2)

    @property
    def E(self) -> float:
        """
        Young modulus of the pile material [kPa]. Thie value does not vary across and along the pile.
        """
        try:
            return self._young_modulus
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            print(e)

    @E.setter
    def E(self, value: float) -> None:
        try:
            self._young_modulus = value
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            print(e)

    @property
    def I(self) -> float:
        """
        Second moment of area of the pile [m4].

        The user can use the method :py:meth:`openpile.construct.Pile.set_I` to customise the second
        moment of area for different sections of the pile.
        """
        try:
            return self.data["I [m4]"]
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            print(e)

    @property
    def width(self) -> float:
        """
        Width of the pile [m]. (Used to compute soil springs)
        """
        try:
            return self.data.loc[:, "Diameter [m]"]
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            print(e)

    @width.setter
    def width(self, value: float) -> None:
        try:
            self.data.loc[:, "Diameter [m]"] = value
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            print(e)

    @property
    def area(self) -> float:
        "Sectional area of the pile [m2]"
        try:
            return self.data.loc[:, "Area [m2]"]
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            print(e)

    @area.setter
    def area(self, value: float) -> None:
        try:
            self.data.loc[:, "Area [m2]"] = value
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            print(e)

    @classmethod
    def create(
        cls,
        name: str,
        top_elevation: float,
        pile_sections: Dict[str, List[float]],
        kind: Literal[
            "Circular",
        ] = "Circular",
        material: Literal[
            "Steel",
        ] = "Steel",
    ):
        """A method to create the pile.

        Parameters
        ----------
        name : str
            Pile/Structure's name.
        top_elevation : float
            top elevation of the pile. Note that this elevation provides a reference point to
            know where the pile is located, especially with respect to other object such as a SoilProfile.
        pile_sections : Dict[str, List[float]]
            argument that stores the relevant data of each pile segment.
            Below are the needed keys for the available piles:
            - kind:'Circular' >> keys:['length', 'diameter', 'wall thickness']
        kind : Literal["Circular",]
            type of pile or type of cross-section. by default "Circular"
        material : Literal["Steel",]
            material the pile is made of. by default "Steel"

        Returns
        -------
        openpile.construct.Pile
            a Pile instance with embedded postprocessing to perform calculations with openpile.
        """

        obj = cls(
            name=name,
            kind=kind,
            material=material,
            top_elevation=top_elevation,
            pile_sections=pile_sections,
        )

        warnings.warn(
            "\nThe method Pile.create() will be removed in version 1.0.0."
            "\nPlease use the base class to create a Pile instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return obj

    @classmethod
    def create_tubular(
        cls,
        name: str,
        top_elevation: float,
        bottom_elevation: float,
        diameter: float,
        wt: float,
        material: str = "Steel",
    ):
        """A method to simplify the creation of a Pile instance.
        This method creates a circular and hollow pile of constant diameter and wall thickness.

        Parameters
        ----------
        name : str
            Pile/Structure's name.
        top_elevation : float
            top elevation of the pile [m VREF]
        bottom_elevation : float
            bottom elevation of the pile [m VREF]
        diameter : float
            pile diameter [m]
        wt : float
            pile's wall thickness [m]
        material : Literal["Steel",]
            material the pile is made of. by default "Steel"

        Returns
        -------
        openpile.construct.Pile
            a Pile instance.
        """

        obj = cls(
            name=name,
            kind="Circular",
            material=material,
            top_elevation=top_elevation,
            pile_sections={
                "length": [
                    (top_elevation - bottom_elevation),
                ],
                "wall thickness": [
                    wt,
                ],
                "diameter": [
                    diameter,
                ],
            },
        )

        return obj

    def set_I(self, value: float, section: int) -> None:
        """set second moment of area for a particular section of the pile.

        Parameters
        ----------
        value : float
            new second moment of area [m4].
        section : int
            section number for which to set new second moment of area

        """
        try:
            length = len(self.data["I [m4]"].values)
            if section * 2 > length:
                print("section number is too large")
            elif section < 1:
                print("section number must be 1 or above")
            else:
                self.data.loc[section * 2 - 2, "I [m4]"] = value
                self.data.loc[section * 2 - 1, "I [m4]"] = value
        except AttributeError:
            print("Please first create the pile with the Pile.create() method")
        except Exception as e:
            raise Exception

    def plot(self, assign=False):
        """Creates a plot of the pile with the properties.

        Parameters
        ----------
        assign : bool, optional
            this parameter can be set to True to return the figure, by default False

        Returns
        -------
        matplotlib.pyplot.figure
            only return the object if assign=True

        Example
        -------

        .. image:: _static/plots/Pile_plot.png
           :width: 70%

        """
        fig = graphics.pile_plot(self)
        return fig if assign else None


@dataclass(config=PydanticConfig)
class Layer:
    """A class to create a layer.

    The Layer stores information on the soil parameters of the layer as well
    as the relevant/representative constitutive model (aka. the soil spring).

    Parameters
    ----------
    name : str
        Name of the layer, use for printout.
    top : float
        top elevation of the layer in [m].
    bottom : float
        bottom elevation of the layer in [m].
    weight : float
        total unit weight in [kN/m3], cannot be lower than 10.
    lateral_model : ConstitutiveModel
        Lateral soil model of the layer, by default None.
    axial_model : ConstitutiveModel
        Axial soil model of the layer, by default None.
    color : str
        soil layer color in HEX format (e.g. '#000000'), by default None.
        If None, the color is generated randomly.


    Example
    -------

    >>> from openpile.construct import Layer
    >>> from openpile.core.soilmodels import API_clay

    >>> # Create a layer with increasing values of Su and eps50
    >>> layer1 = Layer(name='Soft Clay',
                   top=0,
                   bottom=-10,
                   weight=19,
                   lateral_model=API_clay(Su=[30,35], eps50=[0.01, 0.02], Neq=100),
                   )

    >>> # show layer
    >>> print(layer1)
    Name: Soft Clay
    Elevation: (0.0) - (-10.0) m
    Weight: 19.0 kN/m3
    Lateral model: 	API clay
        Su = 30.0-35.0 kPa
        eps50 = 0.01-0.02
        Cyclic, N = 100 cycles
    Axial model: None
    """

    #: name of the layer, use for printout
    name: str
    #: top elevaiton of the layer
    top: float
    #: bottom elevaiton of the layer
    bottom: float
    #: unit weight in kN of the layer
    weight: confloat(gt=10.0)
    #: Lateral constitutive model of the layer
    lateral_model: Optional[ConstitutiveModel] = None
    #: Axial constitutive model of the layer
    axial_model: Optional[ConstitutiveModel] = None
    #: Layer's color when plotted
    color: Optional[constr(min_length=7, max_length=7)] = None

    def __post_init__(self):
        if self.color is None:
            self.color = generate_color_string("earth")

    def __str__(self):
        return f"Name: {self.name}\nElevation: ({self.top}) - ({self.bottom}) m\nWeight: {self.weight} kN/m3\nLateral model: {self.lateral_model}\nAxial model: {self.axial_model}"

    @root_validator
    def check_elevations(cls, values):  # pylint: disable=no-self-argument
        if not values["top"] > values["bottom"]:
            print("Bottom elevation is higher than top elevation")
            raise ValueError
        else:
            return values


@dataclass(config=PydanticConfig)
class SoilProfile:
    """
    A class to create the soil profile. A soil profile consist of a ground elevation (or top elevation)
    with one or more layers of soil.

    Additionally, a soil profile can include discrete information at given elevation such as CPT
    (Cone Penetration Test) data. Not Implemented yet!

    Parameters
    ----------
    name : str
        Name of the soil profile, used for printout and plots.
    top_elevation : float
        top elevation of the soil profile in [m VREF].
    water_line : float
        elevation of the water table in [m VREF].
    layers : list[Layer]
        list of layers for the soil profile.
    cpt_data : np.ndarray
        cpt data table with
        1st col: elevation [m],
        2nd col: cone resistance [kPa],
        3rd col: sleeve friction [kPa],
        4th col: pore pressure u2 [kPa].

    Example
    -------

    >>> from openpile.construct import SoilProfile, Layer
    >>> from openpile.core.soilmodels import API_sand, API_clay

    >>> # Create a two-layer soil profile
    >>> sp = SoilProfile(
    >>>     name="BH01",
    >>>     top_elevation=0,
    >>>     water_line=0,
    >>>     layers=[
    >>>         Layer(
    >>>             name='Layer0',
    >>>             top=0,
    >>>             bottom=-20,
    >>>             weight=18,
    >>>             lateral_model= API_sand(phi=30, Neq=100)
    >>>         ),
    >>>         Layer( name='Layer1',
    >>>                 top=-20,
    >>>                 bottom=-40,
    >>>                 weight=19,
    >>>                 lateral_model= API_clay(Su=50, eps50=0.01, Neq=100),)
    >>>     ]
    >>> )

    >>> # Check soil profile content
    >>> print(sp)
    Layer 1
    ------------------------------
    Name: Layer0
    Elevation: (0.0) - (-20.0) m
    Weight: 18.0 kN/m3
    Lateral model: 	API sand
        phi = 30.0°
        Cyclic, N = 100 cycles
    Axial model: None
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Layer 2
    ------------------------------
    Name: Layer1
    Elevation: (-20.0) - (-40.0) m
    Weight: 19.0 kN/m3
    Lateral model: 	API clay
        Su = 50.0 kPa
        eps50 = 0.01
        Cyclic, N = 100 cycles
    Axial model: None
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """

    #: name of soil profile / borehole / location
    name: str
    #: top of ground elevation with respect to the model reference elevation datum
    top_elevation: float
    #: water elevation (this can refer to sea elevation of water table)
    water_line: float
    #: soil layers to consider in the soil propfile
    layers: List[Layer]
    #: Cone Penetration Test data with folloeing structure:
    #: 1st col: elevation[m],
    #: 2nd col: cone resistance[kPa],
    #: 3rd col: sleeve friction [kPa]
    #: 4th col: pore pressure u2 [kPa]
    #: (the cpt data outside the soil profile boundaries will be ignored)
    cpt_data: Optional[np.ndarray] = None

    @root_validator
    def check_layers_elevations(cls, values):  # pylint: disable=no-self-argument
        layers = values["layers"]

        top_elevations = np.array([x.top for x in layers], dtype=float)
        bottom_elevations = np.array([x.bottom for x in layers], dtype=float)
        idx_sort = np.argsort(top_elevations)

        top_sorted = top_elevations[idx_sort][::-1]
        bottom_sorted = bottom_elevations[idx_sort][::-1]

        # check no overlap
        if top_sorted[0] != values["top_elevation"]:
            raise ValueError("top_elevation not matching uppermost layer's elevations.")

        for i in range(len(top_sorted) - 1):
            if not m.isclose(top_sorted[i + 1], bottom_sorted[i], abs_tol=0.001):
                raise ValueError("Layers' elevations overlap.")

        return values

    @root_validator
    def check_multipliers_in_lateral_model(cls, values):
        def check_multipliers_callable(multiplier, ground_level, top, bottom, type):
            # if not a float, it must be a callable, then we check for Real Positive float
            if not isinstance(multiplier, float):
                # defines depth below ground to check
                depths = ground_level - np.linspace(start=top, stop=bottom, num=100)
                # check if positive real float is returned
                for depth in depths:
                    result = multiplier(depth)
                    if not isinstance(result, float):
                        TypeError(
                            f"One or more results of the {type}-multiplier callable is not a float"
                        )
                        return None
                    else:
                        if type in ["p", "m"]:
                            if result < 0.0:
                                print(
                                    f"One or more results of the {type}-multiplier callable is negative"
                                )
                                return None
                        elif type in ["y", "t"]:
                            if not result > 0.0:
                                ValueError(
                                    f"One or more results of the {type}-multiplier callable is not strictly positive"
                                )
                                return None

        layers = values["layers"]

        for layer in layers:
            if layer.lateral_model is not None:
                # check p-multipliers
                check_multipliers_callable(
                    layer.lateral_model.p_multiplier,
                    values["top_elevation"],
                    layer.top,
                    layer.bottom,
                    "p",
                )
                # check y-multipliers
                check_multipliers_callable(
                    layer.lateral_model.y_multiplier,
                    values["top_elevation"],
                    layer.top,
                    layer.bottom,
                    "y",
                )
                # check m-multipliers
                check_multipliers_callable(
                    layer.lateral_model.m_multiplier,
                    values["top_elevation"],
                    layer.top,
                    layer.bottom,
                    "m",
                )
                # check t-multipliers
                check_multipliers_callable(
                    layer.lateral_model.t_multiplier,
                    values["top_elevation"],
                    layer.top,
                    layer.bottom,
                    "t",
                )

        return values

    def __post_init__(self):
        pass

    def __str__(self):
        """List all layers in table-like format"""
        out = ""
        i = 0
        for layer in self.layers:
            i += 1
            out += f"Layer {i}\n" + "-" * 30 + "\n"
            out += f"{layer}\n" + "~" * 30 + "\n"
        return out

    @property
    def bottom_elevation(self) -> float:
        """
        Bottom elevation of the soil profile [m VREF].
        """
        return self.top_elevation - sum([abs(x.top - x.bottom) for x in self.layers])

    def plot(self, assign=False):
        """Creates a plot illustrating the stratigraphy.

        Parameters
        ----------
        assign : bool, optional
            this parameter can be set to True to return the figure, by default False

        Returns
        -------
        matplotlib.pyplot.figure
            only return the object if assign=True

        Example
        -------

        .. image:: _static/plots/SoilProfile_plot.png
           :scale: 70%
        """
        fig = graphics.soil_plot(self)
        return fig if assign is True else None


@dataclass(config=PydanticConfig)
class Model:
    """
    A class to create a Model.

    A Model is constructed based on the pile geometry/data primarily.
    Additionally, a soil profile can be fed to the Model, and soil springs can be created.

    Parameters
    ----------
    name : str
        Name of the model
    pile : Pile
        Pile instance to be included in the model.
    soil : Optional[SoilProfile], optional
        SoilProfile instance, by default None.
    element_type : str, optional
        can be of ['EulerBernoulli','Timoshenko'], by default 'Timoshenko'.
    x2mesh : List[float], optional
        additional elevations to be included in the mesh, by default none.
    coarseness : float, optional
        maximum distance in meters between two nodes of the mesh, by default 0.5.
    distributed_lateral : bool, optional
        include distributed lateral springs, by default True.
    distributed_moment : bool, optional
        include distributed moment springs, by default False.
    base_shear : bool, optional
        include lateral spring at pile toe, by default False.
    base_moment : bool, optional
        include moment spring at pile toe, by default False.


    Example
    -------

    >>> from openpile.construct import Pile, Model, Layer
    >>> from openpile.core.soilmodels import API_sand

    >>> # create pile
    >>> p = Pile(name = "WTG01",
    >>> 		kind='Circular',
    >>> 		material='Steel',
    >>> 		top_elevation = 0,
    >>> 		pile_sections={
    >>> 			'length':[10,30],
    >>> 			'diameter':[7.5,7.5],
    >>> 			'wall thickness':[0.07, 0.08],
    >>> 		}
    >>> 	)

    >>> # Create Soil Profile
    >>> sp = SoilProfile(
    >>> 	name="BH01",
    >>> 	top_elevation=0,
    >>> 	water_line=0,
    >>> 	layers=[
    >>> 		Layer(
    >>> 			name='Layer0',
    >>> 			top=0,
    >>> 			bottom=-40,
    >>> 			weight=18,
    >>> 			lateral_model= API_sand(phi=30, Neq=100)
    >>> 		),
    >>> 	]
    >>> )

    >>> # Create Model
    >>> M = Model(name="Example", pile=p, soil=sp)

    >>> # create Model without soil maximum 5 metres apart.
    >>> Model_without_soil = Model(name = "Example without soil", pile=p, coarseness=5)
    >>> # create Model with nodes maximum 1 metre apart with soil profile
    >>> Model_with_soil = Model(name = "Example with soil", pile=p, soil=sp, coarseness=1)

    """

    #: model name
    name: str
    #: pile instance that the Model should consider
    pile: Pile
    #: soil profile instance that the Model should consider
    soil: Optional[SoilProfile] = None
    #: type of beam elements
    element_type: Literal["Timoshenko", "EulerBernoulli"] = "Timoshenko"
    #: x coordinates values to mesh as nodes
    x2mesh: List[float] = Field(default_factory=list)
    #: mesh coarseness, represent the maximum accepted length of elements
    coarseness: float = 0.5
    #: whether to include p-y springs in the calculations
    distributed_lateral: bool = True
    #: whether to include m-t springs in the calculations
    distributed_moment: bool = True
    #: whether to include Hb-y spring in the calculations
    base_shear: bool = True
    #: whether to include Mb-t spring in the calculations
    base_moment: bool = True
    #: whether to include t-z springs in the calculations
    distributed_axial: bool = False
    #: whether to include Q-z spring in the calculations
    base_axial: bool = False

    @root_validator(skip_on_failure=True)
    def soil_and_pile_bottom_elevation_match(cls, values):  # pylint: disable=no-self-argument
        if values["soil"] is None:
            pass
        else:
            if values["pile"].bottom_elevation < values["soil"].bottom_elevation:
                raise UserWarning("The pile ends deeper than the soil profile.")
        return values

    def __post_init__(self):
        def check_springs(arr):
            check_nan = np.isnan(arr).any()
            check_negative = (arr < 0).any()

            return check_nan or check_negative

        def get_coordinates() -> pd.DataFrame:
            # Primary discretisation over x-axis
            x = np.array([], dtype=np.float16)
            # add get pile relevant sections
            x = np.append(x, self.pile.data["Elevation [m]"].values)
            # add soil relevant layers and others
            if self.soil is not None:
                soil_elevations = np.array(
                    [x.top for x in self.soil.layers] + [x.bottom for x in self.soil.layers],
                    dtype=float,
                )
                if any(soil_elevations < self.pile.bottom_elevation):
                    soil_elevations = np.append(self.pile.bottom_elevation, soil_elevations)
                    soil_elevations = soil_elevations[soil_elevations >= self.pile.bottom_elevation]
                x = np.append(x, soil_elevations)
            # add user-defined elevation
            x = np.append(x, self.x2mesh)

            # get unique values and sort in reverse order
            x = np.unique(x)[::-1]

            # Secondary discretisation over x-axis depending on coarseness factor
            x_secondary = np.array([], dtype=np.float16)
            for i in range(len(x) - 1):
                spacing = x[i] - x[i + 1]
                new_spacing = spacing
                divider = 1
                while new_spacing > self.coarseness:
                    divider += 1
                    new_spacing = spacing / divider
                new_x = x[i] - (
                    np.arange(start=1, stop=divider) * np.tile(new_spacing, (divider - 1))
                )
                x_secondary = np.append(x_secondary, new_x)

            # assemble x- coordinates
            x = np.append(x, x_secondary)
            x = np.unique(x)[::-1]

            # dummy y- coordinates
            y = np.zeros(shape=x.shape)

            # create dataframe coordinates
            nodes = pd.DataFrame(
                data={
                    "x [m]": x,
                    "y [m]": y,
                },
                dtype=float,
            ).round(3)
            nodes.index.name = "Node no."

            element = pd.DataFrame(
                data={
                    "x_top [m]": x[:-1],
                    "x_bottom [m]": x[1:],
                    "y_top [m]": y[:-1],
                    "y_bottom [m]": y[1:],
                },
                dtype=float,
            ).round(3)
            element.index.name = "Element no."

            return nodes, element

            # function doing the work

        def get_soil_profile() -> pd.DataFrame:
            top_elevations = [x.top for x in self.soil.layers]
            bottom_elevations = [x.bottom for x in self.soil.layers]
            soil_weights = [x.weight for x in self.soil.layers]

            idx_sort = np.argsort(top_elevations)[::-1]

            top_elevations = [top_elevations[i] for i in idx_sort]
            soil_weights = [soil_weights[i] for i in idx_sort]
            bottom_elevations = [bottom_elevations[i] for i in idx_sort]

            # #calculate vertical stress
            # v_stress = [0.0,]
            # for uw, top, bottom in zip(soil_weights, top_elevations, bottom_elevations):
            #     v_stress.append(v_stress[-1] + uw*(top-bottom))

            # elevation in model w.r.t to x axis
            x = top_elevations

            return pd.DataFrame(
                data={"Top soil layer [m]": x, "Unit Weight [kN/m3]": soil_weights},
                dtype=np.float64,
            )

        def create_springs() -> np.ndarray:
            # dim of springs
            spring_dim = 15

            # Allocate array
            py = np.zeros(shape=(self.element_number, 2, 2, spring_dim), dtype=np.float32)
            mt = np.zeros(
                shape=(self.element_number, 2, 2, spring_dim, spring_dim), dtype=np.float32
            )
            Hb = np.zeros(shape=(1, 1, 2, spring_dim), dtype=np.float32)
            Mb = np.zeros(shape=(1, 1, 2, spring_dim), dtype=np.float32)

            tz = np.zeros(shape=(self.element_number, 2, 2, 15), dtype=np.float32)

            # fill in spring for each element
            for layer in self.soil.layers:
                elements_for_layer = self.soil_properties.loc[
                    (self.soil_properties["x_top [m]"] <= layer.top)
                    & (self.soil_properties["x_bottom [m]"] >= layer.bottom)
                ].index

                # py curve
                if layer.lateral_model is None:
                    pass
                else:
                    # Set local layer parameters for each element of the layer
                    for i in elements_for_layer:
                        # vertical effective stress
                        sig_v = self.soil_properties[
                            ["sigma_v top [kPa]", "sigma_v bottom [kPa]"]
                        ].iloc[i]
                        # elevation
                        elevation = self.soil_properties[["x_top [m]", "x_bottom [m]"]].iloc[i]
                        # depth from ground
                        depth_from_ground = (
                            self.soil_properties[["xg_top [m]", "xg_bottom [m]"]].iloc[i]
                        ).abs()
                        # pile width
                        pile_width = self.element_properties["Diameter [m]"].iloc[i]

                        # p-y curves
                        if (
                            layer.lateral_model.spring_signature[0] and self.distributed_lateral
                        ):  # True if py spring function exist

                            # calculate springs (top and) for each element
                            for j in [0, 1]:
                                (py[i, j, 1], py[i, j, 0]) = layer.lateral_model.py_spring_fct(
                                    sig=sig_v[j],
                                    X=depth_from_ground[j],
                                    layer_height=(layer.top - layer.bottom),
                                    depth_from_top_of_layer=(layer.top - elevation[j]),
                                    D=pile_width,
                                    L=(self.soil.top_elevation - self.pile.bottom_elevation),
                                    below_water_table=elevation[j] <= self.soil.water_line,
                                    output_length=spring_dim,
                                )

                        if (
                            layer.lateral_model.spring_signature[2] and self.distributed_moment
                        ):  # True if mt spring function exist

                            # calculate springs (top and) for each element
                            for j in [0, 1]:
                                (mt[i, j, 1], mt[i, j, 0]) = layer.lateral_model.mt_spring_fct(
                                    sig=sig_v[j],
                                    X=depth_from_ground[j],
                                    layer_height=(layer.top - layer.bottom),
                                    depth_from_top_of_layer=(layer.top - elevation[j]),
                                    D=pile_width,
                                    L=(self.soil.top_elevation - self.pile.bottom_elevation),
                                    below_water_table=elevation[j] <= self.soil.water_line,
                                    output_length=spring_dim,
                                )

                    # check if pile tip is within layer
                    if (
                        layer.top >= self.pile.bottom_elevation
                        and layer.bottom <= self.pile.bottom_elevation
                    ):

                        # Hb curve
                        sig_v_tip = self.soil_properties["sigma_v bottom [kPa]"].iloc[-1]

                        if layer.lateral_model.spring_signature[1] and self.base_shear:

                            # calculate Hb spring
                            (Hb[0, 0, 1], Hb[0, 0, 0]) = layer.lateral_model.Hb_spring_fct(
                                sig=sig_v_tip,
                                X=-self.pile.bottom_elevation,
                                layer_height=(layer.top - layer.bottom),
                                depth_from_top_of_layer=(layer.top - self.pile.bottom_elevation),
                                D=pile_width,
                                L=(self.soil.top_elevation - self.pile.bottom_elevation),
                                below_water_table=self.pile.bottom_elevation
                                <= self.soil.water_line,
                                output_length=spring_dim,
                            )

                        # Mb curve
                        if layer.lateral_model.spring_signature[3] and self.base_moment:

                            (Mb[0, 0, 1], Mb[0, 0, 0]) = layer.lateral_model.Mb_spring_fct(
                                sig=sig_v_tip,
                                X=-self.pile.bottom_elevation,
                                layer_height=(layer.top - layer.bottom),
                                depth_from_top_of_layer=(layer.top - self.pile.bottom_elevation),
                                D=pile_width,
                                L=(self.soil.top_elevation - self.pile.bottom_elevation),
                                below_water_table=self.pile.bottom_elevation
                                <= self.soil.water_line,
                                output_length=spring_dim,
                            )

            if check_springs(py):
                print("py springs have negative or NaN values.")
                print(
                    """if using PISA type springs, this can be due to parameters behind out of the parameter space.
                Please check that: 2 < L/D < 6.
                """
                )
            if check_springs(mt):
                print("mt springs have negative or NaN values.")
                print(
                    """if using PISA type springs, this can be due to parameters behind out of the parameter space.
                Please check that: 2 < L/D < 6.
                """
                )
            if check_springs(Hb):
                print("Hb spring has negative or NaN values.")
                print(
                    """if using PISA type springs, this can be due to parameters behind out of the parameter space.
                Please check that: 2 < L/D < 6.
                """
                )
            if check_springs(Mb):
                print("Mb spring has negative or NaN values.")
                print(
                    """if using PISA type springs, this can be due to parameters behind out of the parameter space.
                Please check that: 2 < L/D < 6.
                """
                )
            return py, mt, Hb, Mb, tz

        # creates mesh coordinates
        self.nodes_coordinates, self.element_coordinates = get_coordinates()
        self.element_number = int(self.element_coordinates.shape[0])

        # creates element structural properties
        # merge Pile.data and self.coordinates
        self.element_properties = pd.merge_asof(
            left=self.element_coordinates.sort_values(by=["x_top [m]"]),
            right=self.pile.data.sort_values(by=["Elevation [m]"]),
            left_on="x_top [m]",
            right_on="Elevation [m]",
            direction="forward",
        ).sort_values(by=["x_top [m]"], ascending=False)
        # add young modulus to data
        self.element_properties["E [kPa]"] = self.pile.E
        # delete Elevation [m] column
        self.element_properties.drop("Elevation [m]", inplace=True, axis=1)
        # reset index
        self.element_properties.reset_index(inplace=True, drop=True)

        # create soil properties
        if self.soil is not None:
            self.soil_properties = pd.merge_asof(
                left=self.element_coordinates[["x_top [m]", "x_bottom [m]"]].sort_values(
                    by=["x_top [m]"]
                ),
                right=get_soil_profile().sort_values(by=["Top soil layer [m]"]),
                left_on="x_top [m]",
                right_on="Top soil layer [m]",
                direction="forward",
            ).sort_values(by=["x_top [m]"], ascending=False)
            # add elevation of element w.r.t. ground level
            self.soil_properties["xg_top [m]"] = (
                self.soil_properties["x_top [m]"] - self.soil.top_elevation
            )
            self.soil_properties["xg_bottom [m]"] = (
                self.soil_properties["x_bottom [m]"] - self.soil.top_elevation
            )
            # add vertical stress at top and bottom of each element
            condition_below_water_table = self.soil_properties["x_top [m]"] <= self.soil.water_line
            self.soil_properties["Unit Weight [kN/m3]"][condition_below_water_table] = (
                self.soil_properties["Unit Weight [kN/m3]"][condition_below_water_table] - 10.0
            )
            s = (
                self.soil_properties["x_top [m]"] - self.soil_properties["x_bottom [m]"]
            ) * self.soil_properties["Unit Weight [kN/m3]"]
            self.soil_properties["sigma_v top [kPa]"] = np.insert(
                s.cumsum().values[:-1],
                np.where(self.soil_properties["x_top [m]"].values == self.soil.top_elevation)[0],
                0.0,
            )
            self.soil_properties["sigma_v bottom [kPa]"] = s.cumsum()
            # reset index
            self.soil_properties.reset_index(inplace=True, drop=True)

            # Create arrays of springs
            (
                self._py_springs,
                self._mt_springs,
                self._Hb_spring,
                self._Mb_spring,
                self._tz_springs,
            ) = create_springs()

        # Initialise nodal global forces with link to nodes_coordinates (used for force-driven calcs)
        self.global_forces = self.nodes_coordinates.copy()
        self.global_forces["Px [kN]"] = 0
        self.global_forces["Py [kN]"] = 0
        self.global_forces["Mz [kNm]"] = 0

        # Initialise nodal global displacement with link to nodes_coordinates (used for displacement-driven calcs)
        self.global_disp = self.nodes_coordinates.copy()
        self.global_disp["Tx [m]"] = 0
        self.global_disp["Ty [m]"] = 0
        self.global_disp["Rz [rad]"] = 0

        # Initialise nodal global support with link to nodes_coordinates (used for defining boundary conditions)
        self.global_restrained = self.nodes_coordinates.copy()
        self.global_restrained["Tx"] = False
        self.global_restrained["Ty"] = False
        self.global_restrained["Rz"] = False

    @property
    def embedment(self) -> float:
        """Pile embedment length [m].

        Returns
        -------
        float (or None if no SoilProfile is present)
            Pile embedment
        """
        if self.soil is None:
            return None
        else:
            return self.soil.top_elevation - self.pile.bottom_elevation

    @property
    def top(self) -> float:
        """top elevation of the model [m].

        Returns
        -------
        float
        """
        if self.soil is None:
            return self.pile.top_elevation
        else:
            return max(self.pile.top_elevation, self.soil.top_elevation, self.soil.water_line)

    @property
    def bottom(self) -> float:
        """bottom elevation of the model [m].

        Returns
        -------
        float
        """

        if self.soil is None:
            return self.pile.bottom_elevation
        else:
            return min(self.pile.bottom_elevation, self.soil.bottom_elevation)

    def get_structural_properties(self) -> pd.DataFrame:
        """
        Returns a table with the structural properties of the pile sections.
        """
        try:
            return self.element_properties
        except AttributeError:
            print("Data not found. Please create Model with the Model.create() method.")
        except Exception as e:
            print(e)

    def get_soil_properties(self) -> pd.DataFrame:
        """
        Returns a table with the soil main properties and soil models of each element.
        """
        try:
            return self.soil_properties
        except AttributeError:
            print("Data not found. Please create Model with the Model.create() method.")
        except Exception as e:
            print(e)

    def get_pointload(self, output=False, verbose=True):
        """
        Returns the point loads currently defined in the mesh via printout statements.

        Parameters
        ----------
        output : bool, optional
            If true, it returns the printout statements as a variable, by default False.
        verbose : float, optional
            if True, printout statements printed automaically (ideal for use with iPython), by default True.
        """
        out = ""
        try:
            for idx, elevation, _, Px, Py, Mz in self.global_forces.itertuples(name=None):
                if any([Px, Py, Mz]):
                    string = f"\nLoad applied at elevation {elevation} m (node no. {idx}): Px = {Px} kN, Py = {Py} kN, Mx = {Mz} kNm."
                    if verbose is True:
                        print(string)
                    out += f"\nLoad applied at elevation {elevation} m (node no. {idx}): Px = {Px} kN, Py = {Py} kN, Mx = {Mz} kNm."
            if output is True:
                return out
        except Exception:
            print("No data found. Please create the Model first.")
            raise

    def set_pointload(
        self,
        elevation: float = 0.0,
        Py: float = None,
        Px: float = None,
        Mz: float = None,
    ):
        """
        Defines the point load(s) at a given elevation.

        .. note:
            If run several times at the same elevation, the loads are overwritten by the last command.


        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0.
        Py : float, optional
            Shear force in kN, by default None.
        Px : float, optional
            Normal force in kN, by default None.
        Mz : float, optional
            Bending moment in kNm, by default None.
        """

        # identify if one node is at given elevation or if load needs to be split
        nodes_elevations = self.nodes_coordinates["x [m]"].values
        # check if corresponding node exist
        check = np.isclose(nodes_elevations, np.tile(elevation, nodes_elevations.shape), atol=0.001)

        try:
            if any(check):
                # one node correspond, extract node
                node_idx = int(np.where(check == True)[0])
                # apply loads at this node
                if Px is not None:
                    self.global_forces.loc[node_idx, "Px [kN]"] = Px
                if Py is not None:
                    self.global_forces.loc[node_idx, "Py [kN]"] = Py
                if Mz is not None:
                    self.global_forces.loc[node_idx, "Mz [kNm]"] = Mz
            else:
                if (
                    elevation > self.nodes_coordinates["x [m]"].iloc[0]
                    or elevation < self.nodes_coordinates["x [m]"].iloc[-1]
                ):
                    print(
                        "Load not applied! The chosen elevation is outside the mesh. The load must be applied on the structure."
                    )
                else:
                    print(
                        "Load not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the Model."
                    )
        except Exception:
            print("\n!User Input Error! Please create Model first with the Model.create().\n")
            raise

    def set_pointdisplacement(
        self,
        elevation: float = 0.0,
        Ty: float = None,
        Tx: float = None,
        Rz: float = None,
    ):
        """
        Defines the displacement at a given elevation.

        .. note::
            for defining supports, this function should not be used, rather use `.set_support()`.

        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0.
        Ty : float, optional
            Translation along y-axis, by default None.
        Tx : float, optional
            Translation along x-axis, by default None.
        Rz : float, optional
            Rotation around z-axis, by default None.
        """

        try:
            # identify if one node is at given elevation or if load needs to be split
            nodes_elevations = self.nodes_coordinates["x [m]"].values
            # check if corresponding node exist
            check = np.isclose(
                nodes_elevations, np.tile(elevation, nodes_elevations.shape), atol=0.001
            )

            if any(check):
                # one node correspond, extract node
                node_idx = int(np.where(check == True)[0])
                # apply displacements at this node
                if Tx is not None:
                    self.global_disp.loc[node_idx, "Tx [m]"] = Tx
                    self.global_restrained.loc[node_idx, "Tx"] = Tx > 0.0
                if Ty is not None:
                    self.global_disp.loc[node_idx, "Ty [m]"] = Ty
                    self.global_restrained.loc[node_idx, "Ty"] = Ty > 0.0
                if Rz is not None:
                    self.global_disp.loc[node_idx, "Rz [rad]"] = Rz
                    self.global_restrained.loc[node_idx, "Rz"] = Rz > 0.0
                # set restrain at this node

            else:
                if (
                    elevation > self.nodes_coordinates["x [m]"].iloc[0]
                    or elevation < self.nodes_coordinates["x [m]"].iloc[-1]
                ):
                    print(
                        "Support not applied! The chosen elevation is outside the mesh. The support must be applied on the structure."
                    )
                else:
                    print(
                        "Support not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the Model."
                    )
        except Exception:
            print("\n!User Input Error! Please create Model first with the Model.create().\n")
            raise

    def set_support(
        self,
        elevation: float = 0.0,
        Ty: bool = False,
        Tx: bool = False,
        Rz: bool = False,
    ):
        """
        Defines the supports at a given elevation. If True, the relevant degree of freedom is restrained.

        .. note:
            If run several times at the same elevation, the support are overwritten by the last command.


        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0.
        Ty : bool, optional
            Translation along y-axis, by default False.
        Tx : bool, optional
            Translation along x-axis, by default False.
        Rz : bool, optional
            Rotation around z-axis, by default False.
        """

        try:
            # identify if one node is at given elevation or if load needs to be split
            nodes_elevations = self.nodes_coordinates["x [m]"].values
            # check if corresponding node exist
            check = np.isclose(
                nodes_elevations, np.tile(elevation, nodes_elevations.shape), atol=0.001
            )

            if any(check):
                # one node correspond, extract node
                node_idx = int(np.where(check == True)[0])
                # apply loads at this node
                self.global_restrained.loc[node_idx, "Tx"] = Tx
                self.global_restrained.loc[node_idx, "Ty"] = Ty
                self.global_restrained.loc[node_idx, "Rz"] = Rz
            else:
                if (
                    elevation > self.nodes_coordinates["x [m]"].iloc[0]
                    or elevation < self.nodes_coordinates["x [m]"].iloc[-1]
                ):
                    print(
                        "Support not applied! The chosen elevation is outside the mesh. The support must be applied on the structure."
                    )
                else:
                    print(
                        "Support not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the Model."
                    )
        except Exception:
            print("\n!User Input Error! Please create Model first with the Model.create().\n")
            raise

    def get_py_springs(self, kind: str = "node") -> pd.DataFrame:
        """Table with p-y springs computed for the given Model.

        Posible to extract the springs at the node level (i.e. spring at each node)
        or element level (i.e. top and bottom springs at each element)

        Parameters
        ----------
        kind : str
            can be of ("node", "element").

        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with p-y springs, i.e. p-value [kN/m] and y-value [m].
        """
        if self.soil is None:
            return None
        else:
            if kind == "element":
                return misc.get_full_springs(
                    springs=self._py_springs,
                    elevations=self.nodes_coordinates["x [m]"].values,
                    kind="p-y",
                )
            elif kind == "node":
                return misc.get_reduced_springs(
                    springs=self._py_springs,
                    elevations=self.nodes_coordinates["x [m]"].values,
                    kind="p-y",
                )
            else:
                return None

    def get_mt_springs(self, kind: str = "node") -> pd.DataFrame:
        """Table with m-t (rotational) springs computed for the given Model.

        Posible to extract the springs at the node level (i.e. spring at each node)
        or element level (i.e. top and bottom springs at each element)

        Parameters
        ----------
        kind : str
            can be of ("node", "element").

        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with m-t springs, i.e. m-value [kNm] and t-value [-].
        """
        if self.soil is None:
            return None
        else:
            if kind == "element":
                return misc.get_full_springs(
                    springs=self._mt_springs,
                    elevations=self.nodes_coordinates["x [m]"].values,
                    kind="m-t",
                )
            elif kind == "node":
                return misc.get_reduced_springs(
                    springs=self._mt_springs,
                    elevations=self.nodes_coordinates["x [m]"].values,
                    kind="m-t",
                )
            else:
                return None

    def get_Hb_spring(self) -> pd.DataFrame:
        """Table with Hb (base shear) spring computed for the given Model.


        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with Hb spring, i.e. H-value [kN] and y-value [m].
        """
        if self.soil is None:
            return None
        else:
            spring_dim = self._Hb_spring.shape[-1]

            column_values_spring = [f"VAL {i}" for i in range(spring_dim)]

            df = pd.DataFrame(
                data={
                    "Node no.": [self.element_number + 1] * 2,
                    "Elevation [m]": [self.pile.bottom_elevation] * 2,
                }
            )
            df["type"] = ["Hb", "y"]
            df[column_values_spring] = self._Hb_spring.reshape(2, spring_dim)

            return df

    def get_Mb_spring(self) -> pd.DataFrame:
        """Table with Mb (base moment) spring computed for the given Model.


        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with Mb spring, i.e. M-value [kNn] and t-value [-].
        """
        if self.soil is None:
            return None
        else:
            spring_dim = self._Hb_spring.shape[-1]

            column_values_spring = [f"VAL {i}" for i in range(spring_dim)]

            df = pd.DataFrame(
                data={
                    "Node no.": [self.element_number + 1] * 2,
                    "Elevation [m]": [self.pile.bottom_elevation] * 2,
                }
            )
            df["type"] = ["Mb", "t"]
            df[column_values_spring] = self._Hb_spring.reshape(2, spring_dim)

            return df

    def plot(self, assign=False):
        """Create a plot of the model with the mesh and boundary conditions.

        Parameters
        ----------
        assign : bool, optional
            this parameter can be set to True to return the figure, by default False.

        Returns
        -------
        matplotlib.pyplot.figure
            only return the object if assign=True.

        Examples
        --------

        *Plot without SoilProfile fed to the model:*

        .. image:: _static/plots/Model_no_soil_plot.png
           :scale: 70%

        *Plot with SoilProfile fed to the model:*

        .. image:: _static/plots/Model_with_soil_plot.png
           :scale: 70%
        """
        fig = graphics.connectivity_plot(self)
        return fig if assign else None

    @classmethod
    def create(
        cls,
        name: str,
        pile: Pile,
        soil: Optional[SoilProfile] = None,
        element_type: Literal["Timoshenko", "EulerBernoulli"] = "Timoshenko",
        x2mesh: List[float] = Field(default_factory=list),
        coarseness: float = 0.5,
        distributed_lateral: bool = True,
        distributed_moment: bool = True,
        distributed_axial: bool = False,
        base_shear: bool = True,
        base_moment: bool = True,
        base_axial: bool = False,
    ):
        """A method to create the Model.

        Parameters
        ----------
        name : str
            Name of the model
        pile : Pile
            Pile instance to be included in the model
        soil : Optional[SoilProfile], optional
            SoilProfile instance, by default None
        element_type : str, optional
            can be of ['EulerBernoulli','Timoshenko'], by default 'Timoshenko'
        x2mesh : List[float], optional
            additional elevations to be included in the mesh, by default none
        coarseness : float, optional
            maximum distance in meters between two nodes of the mesh, by default 0.5
        distributed_lateral : bool, optional
            include distributed lateral springs, by default True
        distributed_moment : bool, optional
            include distributed moment springs, by default False
        base_shear : bool, optional
            include lateral spring at pile toe, by default False
        base_moment : bool, optional
            include moment spring at pile toe, by default False

        Returns
        -------
        openpile.construct.Model
            a Model instance with a Pile structure and optionally a SoilProfile
        """

        obj = cls(
            name=name,
            pile=pile,
            soil=soil,
            element_type=element_type,
            x2mesh=x2mesh,
            coarseness=coarseness,
            distributed_lateral=distributed_lateral,
            distributed_moment=distributed_moment,
            distributed_axial=distributed_axial,
            base_shear=base_shear,
            base_moment=base_moment,
            base_axial=base_axial,
        )

        warnings.warn(
            "\nThe method Model.create() will be removed in version 1.0.0."
            "\nPlease use the base class to create a Pile instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return obj

    def __str__(self):
        return self.element_properties.to_string()
