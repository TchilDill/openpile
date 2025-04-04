"""
`Construct` module
==================

The `construct` module is used to construct all objects that 
form the inputs to calculations in openpile. 


These objects include:

- Pile 
  - PileMaterial
  - PileSection
- SoilProfile
- Layer
- Model

**Usage**

>>> from openpile.construct import Pile, SoilProfile, Layer, Model

"""


import math as m
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics

from openpile.materials import PileMaterial
import openpile.materials as materials
from openpile.core import misc, _model_build
from openpile.soilmodels import LateralModel, AxialModel
from openpile.core.misc import generate_color_string
from openpile.core._model_build import (
    check_springs,
    get_soil_properties,
    apply_bc,
    validate_bc,
    parameter2elements,
)

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from typing_extensions import Literal, Annotated, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    InstanceOf,
    Field,
    model_validator,
    computed_field,
)
from functools import cached_property

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


class AbstractPile(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class AbstractLayer(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class AbstractSoilProfile(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class AbstractModel(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class PileSection(BaseModel, ABC):
    """
    An abstract Pile Segment is a section of a pile.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    @abstractmethod
    def top_elevation(self) -> float:
        pass

    @property
    @abstractmethod
    def bottom_elevation(self) -> float:
        pass

    @property
    @abstractmethod
    def footprint(self) -> float:
        pass

    @property
    @abstractmethod
    def length(self) -> float:
        pass

    @property
    @abstractmethod
    def area(self) -> float:
        pass

    @property
    @abstractmethod
    def entrapped_area(self) -> float:
        pass

    @property
    @abstractmethod
    def outer_perimeter(self) -> float:
        pass

    @property
    @abstractmethod
    def inner_perimeter(self) -> float:
        pass

    @abstractmethod
    def get_volume(self, length) -> float:
        pass

    @abstractmethod
    def get_entrapped_volume(self, length) -> float:
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        pass

    @property
    @abstractmethod
    def second_moment_of_area(self) -> float:
        pass


class CircularPileSection(PileSection):
    """A circular section of a pile.

    Parameters
    ----------
    top : float
        the top elevation of the circular section, in meters
    bottom : float
        the bottom elevation of the circular section, in meters
    diameter : float
        the diameter of the circular section, in meters
    thickness : Optional[float], optional
        the wall thickness of the circular section if the section is hollow, in meters,
        by default None which means the section is solid.

    """

    top: float
    bottom: float
    diameter: Annotated[float, Field(gt=0)]
    thickness: Optional[Annotated[float, Field(gt=0)]] = None

    @model_validator(mode="after")
    def get_proper_thickness(self):
        if self.thickness is None:
            self.thickness = self.diameter / 2
        return self

    @model_validator(mode="after")
    def check_elevations(self):
        if self.bottom >= self.top:
            raise ValueError(
                f"Bottom elevation ({self.bottom}) must be less than top elevation ({self.top})."
            )
        return self

    @property
    def top_elevation(self) -> float:
        return self.top

    @property
    def bottom_elevation(self) -> float:
        return self.bottom

    @property
    def length(self) -> float:
        return self.top - self.bottom

    @property
    def area(self) -> float:
        return (self.diameter**2 - (self.diameter - 2 * self.thickness) ** 2) * m.pi / 4

    @property
    def entrapped_area(self) -> float:
        return (self.diameter - 2 * self.thickness) ** 2 * m.pi / 4

    @property
    def outer_perimeter(self) -> float:
        return self.diameter * m.pi

    @property
    def inner_perimeter(self) -> float:
        return (self.diameter - 2 * self.thickness) * m.pi

    @property
    def footprint(self) -> float:
        return self.diameter**2 * m.pi / 4

    @property
    def width(self) -> float:
        return self.diameter

    @property
    def second_moment_of_area(self) -> float:
        return (self.diameter**4 - (self.diameter - 2 * self.thickness) ** 4) * m.pi / 64

    def get_volume(self, length) -> float:
        return self.area * length

    def get_entrapped_volume(self, length) -> float:
        return length * (self.diameter - 2 * self.thickness) ** 2 * m.pi / 4


class Pile(AbstractPile):
    #: name of the pile
    name: str
    #: There can be as many sections as needed by the user. The length of the listsdictates the number of pile sections.
    sections: List[InstanceOf[PileSection]]
    #: select the type of material the pile is made of, can be of ('Steel', 'Concrete') or a material created from openpile.materials.PileMaterial.custom()
    material: Union[Literal["Steel", "Concrete"], PileMaterial]
    """
    A class to create the pil.e.

    Parameters
    ----------
    name : str
        Pile/Structure's name.
    sections : List[PileSection]
        argument that stores the relevant data of each pile segment. numbering of sections is made from uppermost elevation and 0-indexed.
    material : Literal["Steel",]
        material the pile is made of. by default "Steel"


    Example
    -------

    >>> from openpile.construct import Pile, CircularPileSection
    >>> # Create a pile instance with two sections of respectively 10m and 30m length.
    >>> pile = Pile(name = "",
    ...         material='Steel',
    ...         sections=[
    ...             CircularPileSection(
    ...                 top=0, 
    ...                 bottom=-10, 
    ...                 diameter=7.5, 
    ...                 thickness=0.07
    ...             ),
    ...             CircularPileSection(
    ...                 top=-10, 
    ...                 bottom=-40, 
    ...                 diameter=7.5, 
    ...                 thickness=0.08
    ...             ),
    ...         ]
    ...     )

    One can also create a pile from other constructors such as: create_tubular(), that creates a ciruclar hollow pile of one unique section.

    >>> from openpile.construct import Pile
    >>> pile = Pile.create_tubular(name = "",
    ...         top_elevation = 0,
    ...         bottom_elevation = -40,
    ...         diameter=7.5,
    ...         wt=0.07,
    ...         )
    """

    # check that we have at least one pile section
    @model_validator(mode="after")
    def sections_must_be_provided(self):
        if len(self.sections) == 0:
            raise ValueError("No pile sections provided.")
        return self

    # check that dict is correctly entered
    @model_validator(mode="after")
    def sections_must_not_overlap(self):
        self.sections = sorted(self.sections, key=lambda x: -x.top_elevation)
        for i, segment in enumerate(self.sections):
            if i == 0:
                pass
            else:
                previous_segment = self.sections[i - 1]
                if segment.top_elevation != previous_segment.bottom_elevation:
                    raise ValueError(
                        f"Pile sections are not consistent. Pile section No. {i} and No. {i-1} do not connect."
                    )
        return self

    # check that dict is correctly entered
    @model_validator(mode="after")
    def check_materials(self):
        if self.material == "Steel":
            self.material = materials.steel
        elif self.material == "Concrete":
            self.material = materials.concrete
        return self

    @property
    def top_elevation(self) -> float:
        return self.sections[0].top_elevation

    @property
    def data(self) -> pd.DataFrame:
        # create pile data used by openpile for mesh and calculations.
        # Create top and bottom elevations
        return pd.DataFrame(
            data={
                "Elevation [m]": [
                    x for x in self.sections for x in [x.top_elevation, x.bottom_elevation]
                ],
                "Width [m]": [x.width for x in self.sections for x in [x, x]],
                "Area [m2]": [x.area for x in self.sections for x in [x, x]],
                "I [m4]": [x.second_moment_of_area for x in self.sections for x in [x, x]],
                "Entrapped Area [m2]": [x.entrapped_area for x in self.sections for x in [x, x]],
                "Outer Perimeter [m]": [x.outer_perimeter for x in self.sections for x in [x, x]],
                "Inner Perimeter [m]": [x.inner_perimeter for x in self.sections for x in [x, x]],
            }
        )

    def __str__(self):
        if self.shape == "Circular":
            return pd.DataFrame(
                data={
                    "Elevation [m]": [
                        x for x in self.sections for x in [x.top_elevation, x.bottom_elevation]
                    ],
                    "Diameter [m]": [x.width for x in self.sections for x in [x, x]],
                    "Wall thickness [m]": [x.thickness for x in self.sections for x in [x, x]],
                    "Area [m2]": [x.area for x in self.sections for x in [x, x]],
                    "I [m4]": [x.second_moment_of_area for x in self.sections for x in [x, x]],
                }
            ).to_string()
        else:
            return pd.DataFrame(
                data={
                    "Elevation [m]": [
                        x for x in self.sections for x in [x.top_elevation, x.bottom_elevation]
                    ],
                    "Width [m]": [x.width for x in self.sections for x in [x, x]],
                    "Area [m2]": [x.area for x in self.sections for x in [x, x]],
                    "I [m4]": [x.second_moment_of_area for x in self.sections for x in [x, x]],
                }
            ).to_string()

    @property
    def bottom_elevation(self) -> float:
        """
        Bottom elevation of the pile [m VREF].
        """
        return self.sections[-1].bottom_elevation

    @property
    def length(self) -> float:
        """
        Pile length [m].
        """
        return self.top_elevation - self.bottom_elevation

    @property
    def volume(self) -> float:
        """
        Pile volume [m3].
        """
        return sum([x.area * x.length for x in self.sections])

    @property
    def weight(self) -> float:
        """
        Pile weight [kN].
        """
        return self.volume * self.material.unitweight

    @property
    def G(self) -> float:
        """
        Shear modulus of the pile material [kPa]. Thie value does not vary across and along the pile.
        """
        return self.material.shear_modulus

    @property
    def E(self) -> float:
        """
        Young modulus of the pile material [kPa]. Thie value does not vary across and along the pile.
        """
        return self.material.young_modulus

    @property
    def tip_area(self) -> float:
        "Sectional area at the bottom of the pile [m2]"
        return self.sections[-1].area

    @property
    def tip_footprint(self) -> float:
        "footprint area at the bottom of the pile [m2]"
        return self.sections[-1].footprint

    @property
    def inner_volume(self) -> float:
        """the inner volume in [m3] of the pile from the model object."""

        z_top = np.array([x.top_elevation for x in self.sections])
        z_bottom = np.array([x.bottom_elevation for x in self.sections])
        L = z_top - z_bottom
        area_inside = np.array([x.entrapped_area for x in self.sections])

        return np.sum(area_inside * L)

    @classmethod
    def create_tubular(
        cls,
        name: str,
        top_elevation: float,
        bottom_elevation: float,
        diameter: float,
        wt: float,
        material: Union[Literal["Steel", "Concrete"], PileMaterial] = "Steel",
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
        material : Literal["Steel", "Concrete"] or an instance of openpile.materials.PileMaterial
            material the pile is made of. by default "Steel"

        Returns
        -------
        openpile.construct.Pile
            a Pile instance.
        """

        obj = cls(
            name=name,
            material=material,
            sections=[
                CircularPileSection(
                    top=top_elevation,
                    bottom=bottom_elevation,
                    diameter=diameter,
                    thickness=wt,
                )
            ],
        )
        return obj

    @property
    def shape(self):
        if all([isinstance(x, CircularPileSection) for x in self.sections]):
            return "Circular"
        else:
            "Undefined"


class Layer(AbstractLayer):
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
    lateral_model : LateralModel
        Lateral soil model of the layer, by default None.
    axial_model : AxialModel
        Axial soil model of the layer, by default None.
    color : str
        soil layer color in HEX format (e.g. '#000000'), by default None.
        If None, the color is generated randomly.


    Example
    -------

    >>> from openpile.construct import Layer
    >>> from openpile.soilmodels import API_clay
    >>> # Create a layer with increasing values of Su and eps50
    >>> layer1 = Layer(name='Soft Clay',
    ...            top=0,
    ...            bottom=-10,
    ...            weight=19,
    ...            lateral_model=API_clay(Su=[30,35], eps50=[0.01, 0.02], kind='static'),
    ...            )
    >>> # show layer
    >>> print(layer1) # doctest: +NORMALIZE_WHITESPACE
    Name: Soft Clay
    Elevation: (0.0) - (-10.0) m
    Weight: 19.0 kN/m3
    Lateral model: 	API clay
        Su = 30.0-35.0 kPa
        eps50 = 0.01-0.02
        static curves
        ext: None
    Axial model: None
    """

    #: name of the layer, use for printout
    name: str
    #: top elevaiton of the layer
    top: float
    #: bottom elevaiton of the layer
    bottom: float
    #: unit weight in kN of the layer
    weight: Annotated[float, Field(gt=10.0)]
    #: Lateral constitutive model of the layer
    lateral_model: Optional[LateralModel] = None
    #: Axial constitutive model of the layer
    axial_model: Optional[AxialModel] = None
    #: Layer's color when plotted
    color: Optional[Annotated[str, Field(min_length=7, max_length=7)]] = None

    def model_post_init(self, *args, **kwargs):
        if self.color is None:
            self.color = generate_color_string("earth")
        return self

    def __str__(self):
        return f"Name: {self.name}\nElevation: ({self.top}) - ({self.bottom}) m\nWeight: {self.weight} kN/m3\nLateral model: {self.lateral_model}\nAxial model: {self.axial_model}"

    @model_validator(mode="after")
    def check_elevations(self):
        if not self.top > self.bottom:
            raise ValueError("Bottom elevation is higher than top elevation")
        else:
            return self


class SoilProfile(AbstractSoilProfile):
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
    >>> # import objects
    >>> from openpile.construct import SoilProfile, Layer
    >>> from openpile.soilmodels import API_sand, API_clay
    >>> # Create a two-layer soil profile
    >>> sp = SoilProfile(
    ...     name="BH01",
    ...     top_elevation=0,
    ...     water_line=0,
    ...     layers=[
    ...         Layer(
    ...             name='Layer0',
    ...             top=0,
    ...             bottom=-20,
    ...             weight=18,
    ...             lateral_model= API_sand(phi=30, kind='cyclic')
    ...         ),
    ...         Layer( name='Layer1',
    ...                 top=-20,
    ...                 bottom=-40,
    ...                 weight=19,
    ...                 lateral_model= API_clay(Su=50, eps50=0.01, kind='static'),)
    ...     ]
    ... )
    >>> # Check soil profile content
    >>> print(sp) # doctest: +NORMALIZE_WHITESPACE
    Layer 1
    ------------------------------
    Name: Layer0
    Elevation: (0.0) - (-20.0) m
    Weight: 18.0 kN/m3
    Lateral model: 	API sand
        phi = 30.0°
        cyclic curves
        ext: None
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
        static curves
        ext: None
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

    @model_validator(mode="after")
    def check_layers_elevations(self):

        top_elevations = np.array([x.top for x in self.layers], dtype=float)
        bottom_elevations = np.array([x.bottom for x in self.layers], dtype=float)
        idx_sort = np.argsort(top_elevations)

        top_sorted = top_elevations[idx_sort][::-1]
        bottom_sorted = bottom_elevations[idx_sort][::-1]

        # check no overlap
        if top_sorted[0] != self.top_elevation:
            raise ValueError("top_elevation not matching uppermost layer's elevations.")

        for i in range(len(top_sorted) - 1):
            if not m.isclose(top_sorted[i + 1], bottom_sorted[i], abs_tol=0.001):
                raise ValueError("Layers' elevations overlap.")

        return self

    @model_validator(mode="after")
    def check_multipliers_in_lateral_model(self):
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

        for layer in self.layers:
            if layer.lateral_model is not None:
                # check p-multipliers
                check_multipliers_callable(
                    layer.lateral_model.p_multiplier,
                    self.top_elevation,
                    layer.top,
                    layer.bottom,
                    "p",
                )
                # check y-multipliers
                check_multipliers_callable(
                    layer.lateral_model.y_multiplier,
                    self.top_elevation,
                    layer.top,
                    layer.bottom,
                    "y",
                )
                # check m-multipliers
                check_multipliers_callable(
                    layer.lateral_model.m_multiplier,
                    self.top_elevation,
                    layer.top,
                    layer.bottom,
                    "m",
                )
                # check t-multipliers
                check_multipliers_callable(
                    layer.lateral_model.t_multiplier,
                    self.top_elevation,
                    layer.top,
                    layer.bottom,
                    "t",
                )

        return self

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

    def plot(self, ax=None):
        """Creates a plot illustrating the stratigraphy.

        Parameters
        ----------
        ax : axis handle from matplotlib figure, optional
            if None, a new axis handle is created

        Returns
        -------
        matplotlib.pyplot.figure
            only return the object if assign=True

        Example
        -------

        .. plot::
            :context: reset
            :include-source: False

            from openpile.construct import Pile, CircularPileSection
            # Create a pile instance with two sections of respectively 10m and 30m length.
            p = Pile(name = "",
                    material='Steel',
                    sections=[
                        CircularPileSection(
                            top=0,
                            bottom=-10,
                            diameter=7.5,
                            thickness=0.07
                        ),
                        CircularPileSection(
                            top=-10,
                            bottom=-40,
                            diameter=7.5,
                            thickness=0.08
                        ),
                    ]
            )

            from openpile.construct import SoilProfile, Layer

            # save the figure generated by SoilProfile.plot()
            sp = SoilProfile(
                name="<soil profile name>",
                top_elevation=0,
                water_line=8.0,
                layers=[
                    Layer(name="<layer name>", top=0, bottom=-4, weight=18),
                    Layer(name="<layer name>", top=-4, bottom=-8, weight=18),
                    Layer(name="<layer name>", top=-8, bottom=-20, weight=18),
                    Layer(name="<layer name>", top=-20, bottom=-40, weight=18),
                ],
            )
            sp.plot()
        """
        ax = graphics.soil_plot(self, ax=ax)


class BoundaryFixation(BaseModel):
    """
    A class to create a boundary condition where support is fixed.

    Parameters
    ----------
    elevation : str
        Elevation of the boundary condition [m VREF]
    x : bool
        Fix the boundary condition in the x-direction
    y : bool
        Fix the boundary condition in the y-direction
    z : bool
        Fix the boundary condition in the z-direction
    """

    elevation: float
    x: Optional[bool] = None
    y: Optional[bool] = None
    z: Optional[bool] = None


class BoundaryDisplacement(BaseModel):
    """
    A class to create a boundary condition where displacement is given.

    Parameters
    ----------
    elevation : str
        Elevation of the boundary condition [m VREF]
    x : float
        Apply displacement in the x-direction [m]
    y : float
        Apply displacement in the y-direction [m]
    z : float
        Apply displacement in the z-direction [m]
    """

    elevation: float
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class BoundaryForce(BaseModel):
    """
    A class to create a boundary condition where force is given.

    Parameters
    ----------
    elevation : str
        Elevation of the boundary condition [m VREF]
    x : float
        Apply force in the x-direction [kN]
    y : float
        Apply force in the y-direction [kN]
    z : float
        Apply force in the z-direction [kN]
    """

    elevation: float
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class Model(AbstractModel):
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
    boundary_conditions : list[BoundaryFix, BoundaryDisp, BoundaryForce], optional
        list of boundary conditions to be included in the model, by default None.
        Boundary conditions can be added when instantiating the model with Boundary... objects or via the methods:
        `.set_pointload()`, `.set_pointdisplacement()`, `.set_support()`
    soil : Optional[SoilProfile], optional
        SoilProfile instance, by default None.
    element_type : str, optional
        can be of ['EulerBernoulli','Timoshenko'], by default 'Timoshenko'.
    x2mesh : List[float], optional
        additional elevations to be included in the mesh, by default none.
    coarseness : float, optional
        maximum distance in meters between two nodes of the mesh, by default 0.5. A value lower than 0.01 is not accepted due to computational purposes.
    distributed_lateral : bool, optional
        include distributed lateral springs, by default True.
    distributed_moment : bool, optional
        include distributed moment springs, by default False.
    base_shear : bool, optional
        include lateral spring at pile toe, by default False.
    base_moment : bool, optional
        include moment spring at pile toe, by default False.
    distributed_axial : bool, optional
        include distributed axial springs, by default True.
    base_axial : bool, optional
        include base axial springs, by default True.
    plugging : bool, optional
        whether the pile is plugged or unplugged, by default False.


    Example
    -------

    >>> from openpile.construct import Pile, Model, Layer
    >>> from openpile.soilmodels import API_sand
    >>> # create pile
    ... p = Pile(name = "",
    ...          material='Steel',
    ...          sections=[
    ...             CircularPileSection(
    ...                 top=0,
    ...                 bottom=-10,
    ...                 diameter=7.5,
    ...                 thickness=0.07
    ...             ),
    ...             CircularPileSection(
    ...                 top=-10,
    ...                 bottom=-40,
    ...                 diameter=7.5,
    ...                 thickness=0.08
    ...             ),
    ...          ]
    ...     )

    >>> # Create Soil Profile
    >>> sp = SoilProfile(
    ... 	name="BH01",
    ... 	top_elevation=0,
    ... 	water_line=0,
    ... 	layers=[
    ... 		Layer(
    ... 			name='Layer0',
    ... 			top=0,
    ... 			bottom=-40,
    ... 			weight=18,
    ... 			lateral_model= API_sand(phi=30, kind='cyclic')
    ... 		),
    ... 	]
    ... )
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
    #: boundary conditions of the model
    boundary_conditions: List[Union[BoundaryFixation, BoundaryForce, BoundaryDisplacement]] = Field(
        default_factory=list
    )
    #: soil profile instance that the Model should consider
    soil: Optional[SoilProfile] = None
    #: type of beam elements
    element_type: Literal["Timoshenko", "EulerBernoulli"] = "Timoshenko"
    #: x coordinates values to mesh as nodes
    x2mesh: List[float] = Field(default_factory=list)
    #: mesh coarseness, represent the maximum accepted length of elements
    coarseness: Annotated[float, Field(ge=0.01)] = 0.5
    #: whether to include p-y springs in the calculations
    distributed_lateral: bool = True
    #: whether to include m-t springs in the calculations
    distributed_moment: bool = True
    #: whether to include Hb-y spring in the calculations
    base_shear: bool = True
    #: whether to include Mb-t spring in the calculations
    base_moment: bool = True
    #: whether to include t-z springs in the calculations
    distributed_axial: bool = True
    #: whether to include Q-z spring in the calculations
    base_axial: bool = True
    #: plugging
    plugging: bool = None

    @model_validator(mode="after")
    def max_coarseness(self):
        multiplier = 5
        if self.pile.length >= multiplier * self.coarseness:
            return self
        else:
            raise ValueError(
                f"the coarseness factor is too high, please decrease it to at least a value of {np.floor(self.pile.length*100/multiplier)/100}"
            )

    @model_validator(mode="after")
    def soil_and_pile_bottom_elevation_match(self):
        if self.soil is None:
            pass
        else:
            if self.pile.bottom_elevation < self.soil.bottom_elevation:
                raise ValueError("The pile ends deeper than the soil profile.")
        return self

    @model_validator(mode="after")
    def bc_validation(self):
        # validate_bc(self.boundary_conditions, BoundaryDisplacement)
        # validate_bc(self.boundary_conditions, BoundaryForce)
        # validate_bc(self.boundary_conditions, BoundaryFixation)
        return self

    @computed_field
    # @cached_property
    def soil_properties(self) -> Union[Dict[str, np.ndarray], None]:
        return get_soil_properties(self.pile, self.soil, self.x2mesh, self.coarseness)

    @computed_field
    # @cached_property
    def element_properties(self) -> Dict[str, np.ndarray]:
        # creates element structural properties
        # merge Pile.data and coordinates
        element_properties = pd.merge_asof(
            left=self.element_coordinates.sort_values(by=["z_top [m]"]),
            right=self.pile.data.sort_values(by=["Elevation [m]"]),
            left_on="z_top [m]",
            right_on="Elevation [m]",
            direction="forward",
        ).sort_values(by=["z_top [m]"], ascending=False)
        # add young modulus to data
        element_properties["E [kPa]"] = self.pile.material.young_modulus
        # delete Elevation [m] column
        element_properties.drop("Elevation [m]", inplace=True, axis=1)
        # reset index
        element_properties.reset_index(inplace=True, drop=True)

        return element_properties

    @computed_field
    @cached_property
    def nodes_coordinates(self) -> Dict[str, np.ndarray]:
        return _model_build.get_coordinates(self.pile, self.soil, self.x2mesh, self.coarseness)[0]

    @computed_field
    @cached_property
    def element_coordinates(self) -> Dict[str, np.ndarray]:
        return _model_build.get_coordinates(self.pile, self.soil, self.x2mesh, self.coarseness)[1]

    @computed_field
    @property
    def global_forces(self) -> Dict[str, np.ndarray]:

        validate_bc(self.boundary_conditions, BoundaryForce)

        # Initialise nodal global forces with link to nodes_coordinates (used for force-driven calcs)
        df = self.nodes_coordinates.copy()
        df["Pz [kN]"] = 0
        df["Py [kN]"] = 0
        df["Mx [kNm]"] = 0

        nodes_elevations = df["z [m]"].values

        df["Pz [kN]"], df["Py [kN]"], df["Mx [kNm]"] = apply_bc(
            nodes_elevations,
            df["Pz [kN]"].values,
            df["Py [kN]"].values,
            df["Mx [kNm]"].values,
            self.boundary_conditions,
            BoundaryForce,
            "Load",
        )

        return df

    @computed_field
    @property
    def global_disp(self) -> Dict[str, np.ndarray]:

        validate_bc(self.boundary_conditions, BoundaryDisplacement)

        # Initialise nodal global displacement with link to nodes_coordinates (used for displacement-driven calcs)
        df = self.nodes_coordinates.copy()
        df["Tz [m]"] = 0.0
        df["Ty [m]"] = 0.0
        df["Rx [rad]"] = 0.0

        nodes_elevations = df["z [m]"].values

        df["Tz [m]"], df["Ty [m]"], df["Rx [rad]"] = apply_bc(
            nodes_elevations,
            df["Tz [m]"].values,
            df["Ty [m]"].values,
            df["Rx [rad]"].values,
            self.boundary_conditions,
            BoundaryDisplacement,
            "Displacement",
        )

        return df

    @computed_field
    @property
    def global_restrained(self) -> Dict[str, np.ndarray]:

        validate_bc(self.boundary_conditions, BoundaryFixation)

        # Initialise nodal global support with link to nodes_coordinates (used for defining boundary conditions)
        df = self.nodes_coordinates.copy()
        df["Tz"] = False
        df["Ty"] = False
        df["Rx"] = False

        nodes_elevations = df["z [m]"].values

        df["Tz"], df["Ty"], df["Rx"] = apply_bc(
            nodes_elevations,
            df["Tz"].values,
            df["Ty"].values,
            df["Rx"].values,
            self.boundary_conditions,
            BoundaryFixation,
            "Fixity",
        )
        return df

    @property
    def element_number(self) -> int:
        return self.element_properties.shape[0]

    def model_post_init(self, *args, **kwargs):
        def create_springs() -> np.ndarray:
            # dim of springs
            spring_dim = 15
            # springs dim for axial
            tz_springs_dim = 15
            qz_spring_dim = 15

            # Allocate array
            py = np.zeros(shape=(self.element_number, 2, 2, spring_dim), dtype=np.float32)
            mt = np.zeros(
                shape=(self.element_number, 2, 2, spring_dim, spring_dim), dtype=np.float32
            )
            Hb = np.zeros(shape=(1, 1, 2, spring_dim), dtype=np.float32)
            Mb = np.zeros(shape=(1, 1, 2, spring_dim), dtype=np.float32)

            # allocate array for axial springs
            tz = np.zeros(shape=(self.element_number, 2, 2, tz_springs_dim), dtype=np.float32)
            qz = np.zeros(shape=(1, 1, 2, qz_spring_dim), dtype=np.float32)

            soil_prop = self.soil_properties

            # fill in spring for each element
            for layer in self.soil.layers:
                elements_for_layer = soil_prop.loc[
                    (soil_prop["z_top [m]"] <= layer.top)
                    & (soil_prop["z_bottom [m]"] >= layer.bottom)
                ].index

                for i in elements_for_layer:
                    # Set local layer parameters for each element of the layer
                    # vertical effective stress
                    sig_v = soil_prop[["sigma_v top [kPa]", "sigma_v bottom [kPa]"]].iloc[i].values
                    # elevation
                    elevation = soil_prop[["z_top [m]", "z_bottom [m]"]].iloc[i].values
                    # depth from ground
                    depth_from_ground = (
                        (soil_prop[["zg_top [m]", "zg_bottom [m]"]].iloc[i]).abs().values
                    )
                    # pile width
                    pile_width = self.element_properties["Width [m]"].iloc[i]
                    perimeter_out = self.element_properties["Outer Perimeter [m]"].iloc[i]
                    perimeter_in = self.element_properties["Outer Perimeter [m]"].iloc[i]

                    sig_v_tip = soil_prop["sigma_v bottom [kPa]"].iloc[-1]

                    # t-z curves
                    if layer.axial_model is not None:

                        if self.distributed_axial:  # True if tz spring function exist
                            # calculate springs (top and bottom) for each element
                            for j in [0, 1]:
                                (tz[i, j, 1], tz[i, j, 0]) = layer.axial_model.tz_spring_fct(
                                    sig=sig_v[j],
                                    X=depth_from_ground[j],
                                    circumference_out=perimeter_out,
                                    circumference_in=perimeter_in,
                                    layer_height=(layer.top - layer.bottom),
                                    depth_from_top_of_layer=(layer.top - elevation[j]),
                                    D=pile_width,
                                    # TODO add wall thickness for CPT methods?
                                    L=(self.soil.top_elevation - self.pile.bottom_elevation),
                                    below_water_table=elevation[j] <= self.soil.water_line,
                                    output_length=tz_springs_dim,
                                )

                        if (
                            layer.top >= self.pile.bottom_elevation
                            and layer.bottom <= self.pile.bottom_elevation
                            and self.base_axial
                        ):
                            # calculate Qz spring
                            (qz[0, 0, 1], qz[0, 0, 0]) = layer.axial_model.Qz_spring_fct(
                                sig=sig_v_tip,
                                tip_area=self.pile.tip_area,
                                footprint=self.pile.tip_footprint,
                                layer_height=(layer.top - layer.bottom),
                                depth_from_top_of_layer=(layer.top - self.pile.bottom_elevation),
                                D=pile_width,
                                L=(self.soil.top_elevation - self.pile.bottom_elevation),
                                below_water_table=self.pile.bottom_elevation
                                <= self.soil.water_line,
                                output_length=qz_spring_dim,
                            )

                    # py curve
                    if layer.lateral_model is not None:

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
                            if layer.lateral_model.spring_signature[1] and self.base_shear:

                                # calculate Hb spring
                                (Hb[0, 0, 1], Hb[0, 0, 0]) = layer.lateral_model.Hb_spring_fct(
                                    sig=sig_v_tip,
                                    X=(self.soil.top_elevation - self.soil.bottom_elevation),
                                    layer_height=(layer.top - layer.bottom),
                                    depth_from_top_of_layer=(
                                        layer.top - self.pile.bottom_elevation
                                    ),
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
                                    X=(self.soil.top_elevation - self.soil.bottom_elevation),
                                    layer_height=(layer.top - layer.bottom),
                                    depth_from_top_of_layer=(
                                        layer.top - self.pile.bottom_elevation
                                    ),
                                    D=pile_width,
                                    L=(self.soil.top_elevation - self.pile.bottom_elevation),
                                    below_water_table=self.pile.bottom_elevation
                                    <= self.soil.water_line,
                                    output_length=spring_dim,
                                )

            # ensure springs are oriented correctly with respect to z-axis
            # going down is compression and should be negative in "z" values
            tz[:, :, :] = tz[:, :, :] * (-1)
            qz[:, :, :] = qz[:, :, :] * (-1)

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

            return py, mt, Hb, Mb, tz, qz

        if self.soil is not None:
            # Create arrays of springs
            (
                self._py_springs,
                self._mt_springs,
                self._Hb_spring,
                self._Mb_spring,
                self._tz_springs,
                self._qz_spring,
            ) = create_springs()

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

    @property
    def effective_pile_weight(self) -> float:
        if self.soil is not None:
            submerged_element = self.element_properties["z_top [m]"].values <= self.soil.water_line

            elem_z_top = self.element_properties["z_top [m]"].values
            elem_z_bottom = self.element_properties["z_bottom [m]"].values
            V = (elem_z_top - elem_z_bottom) * parameter2elements(
                self.pile.sections, lambda x: x.area, elem_z_top, elem_z_bottom
            )
            W = np.zeros(shape=V.shape)
            W[submerged_element] = V[submerged_element] * (self.pile.material.unitweight - 10)
            W[~submerged_element] = V[~submerged_element] * (self.pile.material.unitweight)
            return W.sum()

        else:
            raise Exception(
                "Model must be linked to a soil profile, use `openpile.construct.Pile.weight instead.`"
            )

    @property
    def shaft_resistance(self) -> float:
        "the shaft resistances [kN] in compression and tension respectively calculated from the provided axial models along the pile."

        if self.soil is None:
            raise Exception("Model must be linked to a soil profile with provided axial models.")
        else:
            # influence zones
            influence = np.abs(np.gradient(self.nodes_coordinates["z [m]"].values))
            influence = influence / 2
            influence

            # Shaft resistance calc compression
            comp = np.sum(
                np.abs(
                    np.min(self._tz_springs[:, 0, 0, :], axis=1) * influence[:-1]
                    + np.min(self._tz_springs[:, 1, 0, :], axis=1) * influence[1:]
                )
            )
            # Shaft resistance calc tension
            tens = np.sum(
                np.abs(
                    np.max(self._tz_springs[:, 0, 0, :], axis=1) * influence[:-1]
                    + np.max(self._tz_springs[:, 1, 0, :], axis=1) * influence[1:]
                )
            )

            return comp, tens

    @property
    def tip_resistance(self) -> float:
        "the end bearing resistance [kN] calculated from the provided axial model at tip elevation"
        if self.soil is not None:
            return float(np.abs(np.min(self._qz_spring[0, 0, 0, :])))
        else:
            raise Exception("Model must be linked to a soil profile with provided axial models.")

    @property
    def entrapped_soil_weight(self) -> float:
        """calculates total weight of soil inside the pile. (Unit: kN)"""

        if self.soil is not None:

            # weight water in kN/m3
            uw_water = 10

            # soil volume
            elem_z_top = self.element_properties["z_top [m]"].values
            elem_z_bottom = self.element_properties["z_bottom [m]"].values
            L = elem_z_top - elem_z_bottom
            area_inside = parameter2elements(
                self.pile.sections, lambda x: x.entrapped_area, elem_z_top, elem_z_bottom
            )
            Vi = area_inside * L
            # element mid-point elevation
            elevation = 0.5 * (
                self.soil_properties["z_top [m]"] + self.soil_properties["z_bottom [m]"]
            )
            # soil weight for each element where we have soil and pile
            elem_number = int(self.element_properties.shape[0])
            element_sw = np.zeros(elem_number)

            for layer in self.soil.layers:
                elements_for_layer = self.soil_properties.loc[
                    (self.soil_properties["z_top [m]"] <= layer.top)
                    & (self.soil_properties["z_bottom [m]"] >= layer.bottom)
                ].index

                # Set local layer parameters for each element of the layer
                for i in elements_for_layer:
                    # Calculate inner soil weight
                    element_sw[i] = (
                        layer.weight * Vi[i]
                        if elem_z_top[i] > self.soil.water_line
                        else (layer.weight - uw_water) * Vi[i]
                    )
            return element_sw.sum()
        else:
            raise Exception(
                "Model must be linked to a soil profile, the argument `soil` must be provided."
            )

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

    def _update_bc(self, elevation, x, y, z, BC_class):
        # list existing
        existing_bcs = [
            bc
            for bc in self.boundary_conditions
            if bc.elevation == elevation and isinstance(bc, BC_class)
        ]

        # check if elevation already exists for the provided axis
        if len(existing_bcs) == 1:
            # check if elevation already exists for the provided axis and modify boundary condition object
            if x is not None:
                existing_bcs[0].x = x
            if y is not None:
                existing_bcs[0].y = y
            if z is not None:
                existing_bcs[0].z = z
        else:
            self.boundary_conditions.append(BC_class(elevation=elevation, x=x, y=y, z=z))

    def set_pointload(
        self,
        *,
        elevation: float = 0.0,
        Py: float = None,
        Pz: float = None,
        Mx: float = None,
    ):
        """
        Defines the point load(s) at a given elevation.

        .. note:
            If run several times at the same elevation, the loads along relevant axes are overwritten by the last command.

        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0.
        Py : float, optional
            Shear force in kN, by default None.
        Pz : float, optional
            Normal force in kN, by default None.
        Mx : float, optional
            Bending moment in kNm, by default None.
        """
        self._update_bc(elevation, z=Pz, y=Py, x=Mx, BC_class=BoundaryForce)

    def set_pointdisplacement(
        self,
        elevation: float = 0.0,
        Ty: float = None,
        Tz: float = None,
        Rx: float = None,
    ):
        """
        Defines the displacement at a given elevation.

        .. note::
            for defining supports, this function should not be used, rather use `.set_support()`.

        .. note:
            If run several times at the same elevation, the displacements along relevant axes are overwritten by the last command.

        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0.
        Ty : float, optional
            Translation along y-axis, by default None.
        Tz : float, optional
            Translation along z-axis, by default None.
        Rx : float, optional
            Rotation around x-axis, by default None.
        """
        self._update_bc(elevation, z=Tz, y=Ty, x=Rx, BC_class=BoundaryDisplacement)

    def set_support(
        self,
        elevation: float = 0.0,
        Ty: bool = None,
        Tz: bool = None,
        Rx: bool = None,
    ):
        """
        Defines the supports at a given elevation. If True, the relevant degree of freedom is restrained.

        .. note:
            If run several times at the same elevation, the support along relevant axes are overwritten by the last command.


        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0.
        Ty : bool, optional
            Translation along y-axis, by default None.
        Tz : bool, optional
            Translation along z-axis, by default None.
        Rx : bool, optional
            Rotation around x-axis, by default None.
        """
        self._update_bc(elevation, z=Tz, y=Ty, x=Rx, BC_class=BoundaryFixation)

    def get_distributed_axial_springs(self, kind: str = "lumped") -> pd.DataFrame:
        """Table with t-z springs computed for the given Model with t-value [kN/m] and y-value [m].

        Posible to extract the springs as typical structural springs (which are also the raw
        springs used in the model) or element level (i.e. top and bottom springs at each element)

        Parameters
        ----------
        kind : str
            can be of ("lumped", "distributed").

        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with t-z springs
        """
        if self.soil is None:
            return None
        else:
            if kind == "distributed":
                return misc.get_distributed_soil_springs(
                    springs=self._tz_springs,
                    elevations=self.nodes_coordinates["z [m]"].values,
                    kind="t-z",
                )
            elif kind == "lumped":
                return misc.get_lumped_soil_springs(
                    springs=self._tz_springs,
                    elevations=self.nodes_coordinates["z [m]"].values,
                    kind="t-z",
                )
            else:
                return None

    def get_distributed_lateral_springs(self, kind: str = "lumped") -> pd.DataFrame:
        """Table with p-y springs computed for the given Model with p-value [kN/m] and y-value [m].

        Posible to extract the springs as typical structural springs (which are also the raw
        springs used in the model) or element level (i.e. top and bottom springs at each element)

        Parameters
        ----------
        kind : str
            can be of ("lumped", "distributed").

        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with p-y springs
        """
        if self.soil is None:
            return None
        else:
            if kind == "distributed":
                return misc.get_distributed_soil_springs(
                    springs=self._py_springs,
                    elevations=self.nodes_coordinates["z [m]"].values,
                    kind="p-y",
                )
            elif kind == "lumped":
                return misc.get_lumped_soil_springs(
                    springs=self._py_springs,
                    elevations=self.nodes_coordinates["z [m]"].values,
                    kind="p-y",
                )
            else:
                return None

    def get_distributed_rotational_springs(self, kind: str = "lumped") -> pd.DataFrame:
        """Table with m-t (rotational) springs computed for the given Model with m-value [kNm] and t-value [radians]

        Posible to extract the springs as typical structural springs (which are also the raw
        springs used in the model) or element level (i.e. top and bottom springs at each element)

        Parameters
        ----------
        kind : str
            can be of ("lumped", "distributed").

        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with m-t springs.
        """
        if self.soil is None:
            return None
        else:
            if kind == "distributed":
                return misc.get_soil_springs(
                    springs=self._mt_springs,
                    elevations=self.nodes_coordinates["z [m]"].values,
                    kind="m-t",
                )
            elif kind == "lumped":
                return misc.get_lumped_soil_springs(
                    springs=self._mt_springs,
                    elevations=self.nodes_coordinates["z [m]"].values,
                    kind="m-t",
                )
            else:
                return None

    def get_base_shear_spring(self) -> pd.DataFrame:
        """Table with Hb (base shear) spring computed for the given Model with Hb-value [kN] and y-value [m].

        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with Hb spring.
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

    def get_base_axial_spring(self) -> pd.DataFrame:
        """Table with Q-z (base axial) spring computed for the given Model with Q-value [kN] and z-value [m].

        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with Q-z spring.
        """
        if self.soil is None:
            return None
        else:
            spring_dim = self._qz_spring.shape[-1]

            column_values_spring = [f"VAL {i}" for i in range(spring_dim)]

            df = pd.DataFrame(
                data={
                    "Node no.": [self.element_number + 1] * 2,
                    "Elevation [m]": [self.pile.bottom_elevation] * 2,
                }
            )
            df["type"] = ["q", "z"]
            df[column_values_spring] = self._qz_spring.reshape(2, spring_dim)

            return df

    def get_base_rotational_spring(self) -> pd.DataFrame:
        """Table with Mb (base moment) spring computed for the given Model with M-value [kNn] and t-value [radians].


        Returns
        -------
        pd.DataFrame (or None if no SoilProfile is present)
            Table with Mb spring, i.e.
        """
        if self.soil is None:
            return None
        else:
            spring_dim = self._Mb_spring.shape[-1]

            column_values_spring = [f"VAL {i}" for i in range(spring_dim)]

            df = pd.DataFrame(
                data={
                    "Node no.": [self.element_number + 1] * 2,
                    "Elevation [m]": [self.pile.bottom_elevation] * 2,
                }
            )
            df["type"] = ["Mb", "t"]
            df[column_values_spring] = self._Mb_spring.reshape(2, spring_dim)

            return df

    def plot(self, ax=None):
        """Create a plot of the model with the mesh and boundary conditions.

        Parameters
        ----------
        ax : axis handle from matplotlib figure, optional
            if None, a new axis handle is created

        Examples
        --------

        *Plot without SoilProfile fed to the model:*

        .. plot::
            :context: close-figs
            :include-source: False

            from openpile.construct import Model

            # save the figure generated by Model.plot()
            m = Model(name="<Model name>", pile=p)
            m.set_pointload(elevation=0, Py=-500)
            m.set_support(elevation=-40, Tz=True, Ty=True)
            m.set_support(elevation=-10, Ty=True)
            m.plot()

        *Plot with SoilProfile fed to the model:*

        .. plot::
            :context: close-figs
            :include-source: False

            m = Model(name="<Model name>", pile=p, soil=sp)
            m.set_pointload(elevation=0, Py=500)
            m.plot()
        """
        graphics.connectivity_plot(self, ax=ax)

    def solve(self):
        """
        Solves the boundary conditions by calling either:
         - :py:func:`openpile.analyze.beam` if no SoilProfile is provided
         - :py:func:`openpile.analyze.winkler` if a SoilProfile is provided

        Returns
        -------
        WinklerResult
            objects that stores results of the analysis.
        """
        from openpile.winkler import beam, winkler

        return beam(self) if self.soil is None else winkler(self)

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
