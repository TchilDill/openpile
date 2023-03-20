"""
`Construct` module
==================

The `construct` module is used to construct all objects that 
form the inputs to calculations in openpile. 


This include:

- the pile
- the soil profile
- the Model

"""

import math as m
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from typing_extensions import Literal
from pydantic import BaseModel, Field, root_validator, validator, PositiveFloat, confloat, conlist, Extra
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics
import openpile.utils.validation as validation
import openpile.core.soilmodels as soilmodels

from openpile.utils import misc

from openpile.core.soilmodels import ConstitutiveModel

from openpile.utils.misc import generate_color_string


class PydanticConfig:
    arbitrary_types_allowed = True
    extra=Extra.forbid

@dataclass(config=PydanticConfig)
class Pile:
    """
    A class to create the pile.

    Pile instances include the pile geometry and data. Following
    the initialisation of a pile, a Pandas dataframe is created 
    which can be read by the following command:
    
    .. note::
        The classmethod :py:meth:`openpile.construct.Pile.create` shall be used to create a Pile instance.
        
        This method ensures that the post-initialization of the `Pile` instance and post processing is done.  
    
    Example
    -------
    
    >>> from openpile.construct import Pile
        
    >>> # Create a pile instance with two sections of respectively 10m and 30m length.
    >>> pile = Pile.create(kind='Circular',
    >>>         material='Steel',
    >>>         top_elevation = 0,
    >>>         pile_sections={
    >>>             'length':[10,30],
    >>>             'diameter':[7.5,7.5],
    >>>             'wall thickness':[0.07, 0.08],
    >>>         } 
    >>>     )
    >>> # Print the pile data 
    >>> print(pile)
        Elevation [m]  Diameter [m]  Wall thickness [m]  Area [m2]     I [m4]
    0            0.0           7.5                0.07   1.633942  11.276204
    1          -10.0           7.5                0.07   1.633942  11.276204
    2          -10.0           7.5                0.08   1.864849  12.835479
    3          -40.0           7.5                0.08   1.864849  12.835479
    >>> # Override young's modulus
    >>> pile.E = 250e6
    >>> # Check young's modulus    
    >>> print(pile.E)
    250000000.0
    >>> # Override second moment of area across whole pile
    >>> pile.I = 1.11
    >>> # Check updated second moment of area
    >>> print(pile)
        Elevation [m]  Diameter [m] Wall thickness [m] Area [m2]  I [m4]
    0            0.0           7.5               <NA>      <NA>    1.11
    1          -10.0           7.5               <NA>      <NA>    1.11
    2          -10.0           7.5               <NA>      <NA>    1.11
    3          -40.0           7.5               <NA>      <NA>    1.11
    >>> # Override pile's width or pile's diameter
    >>> pile.width = 2.22
    >>> # Check updated width or diameter
    >>> print(pile)   
        Elevation [m]  Diameter [m] Wall thickness [m] Area [m2]  I [m4]
    0            0.0          2.22               <NA>      <NA>    1.11
    1          -10.0          2.22               <NA>      <NA>    1.11
    2          -10.0          2.22               <NA>      <NA>    1.11
    3          -40.0          2.22               <NA>      <NA>    1.11  
    
    """
    #: select the type of pile, can be of ('Circular', )
    kind: Literal['Circular']
    #: select the type of material the pile is made of, can be of ('Steel', )
    material: Literal['Steel']
    #: top elevation of the pile according to general vertical reference set by user
    top_elevation: float
    #: pile geometry made of a dictionary of lists. the structure of the dictionary depends on the type of pile selected.
    #: There can be as many sections as needed by the user. The length of the listsdictates the number of pile sections. 
    pile_sections: Dict[str, List[PositiveFloat]]
    
    def _postinit(self):
        
        # check that dict is correctly entered
        validation.pile_sections_must_be(self)
        
        # Create material specific specs for given material        
        # if steel
        if self.material == 'Steel':
            # unit weight
            self._uw = 78.0 # kN/m3
            # young modulus
            self._young_modulus = 210.0e6 #kPa
            # Poisson's ratio
            self._nu = 0.3
        else:
            raise UserWarning
        
        self._shear_modulus = self._young_modulus / (2+2*self._nu)
        
        # create pile data used by openpile for mesh and calculations.
        # Create top and bottom elevations
        elevation = []
        #add bottom of section i and top of section i+1 (essentially the same values) 
        for idx, val in enumerate(self.pile_sections['length']):
            if idx == 0:
                elevation.append(self.top_elevation)
                elevation.append(elevation[-1] - val)
            else:
                elevation.append(elevation[-1])
                elevation.append(elevation[-1] - val)

        #create sectional properties
        
        #spread
        diameter = []
        #add top and bottom of section i (essentially the same values) 
        for idx, val in enumerate(self.pile_sections['diameter']):
            diameter.append(val)
            diameter.append(diameter[-1])

        #thickness
        thickness = []
        #add top and bottom of section i (essentially the same values) 
        for idx, val in enumerate(self.pile_sections['wall thickness']):
            thickness.append(val)
            thickness.append(thickness[-1])
           
        #Area & second moment of area
        area = []
        second_moment_of_area = []
        #add top and bottom of section i (essentially the same values) 
        for _, (d, wt) in enumerate(zip(self.pile_sections['diameter'],self.pile_sections['wall thickness'])):
            # calculate area
            if self.kind == 'Circular':
                A = m.pi / 4 * (d**2 - (d-2*wt)**2)
                I = m.pi / 64 * (d**4 - (d-2*wt)**4)
                area.append(A)
                area.append(area[-1])
                second_moment_of_area.append(I)
                second_moment_of_area.append(second_moment_of_area[-1])
            else:
                #not yet supporting other kind
                raise ValueError()
    
        # Create pile data     
        self.data = pd.DataFrame(data = {
            'Elevation [m]' :  elevation,
            'Diameter [m]' : diameter,
            'Wall thickness [m]' : thickness,  
            'Area [m2]' : area,
            'I [m4]': second_moment_of_area,
            }
        )
    
    def __str__(self):
        return self.data.to_string()
      
    @classmethod  
    def create(cls, kind: Literal['Circular'], material: Literal['Steel'], top_elevation: float,  pile_sections: Dict[str, List[float]] ):
        """
        A method to create the pile. This function provides a 2-in-1 command where:
        
        - a `Pile` instance is created
        - the `._postinit()` method is run and creates all additional pile data necessary.

        """
        obj = cls( kind = kind, material = material, top_elevation=top_elevation,  pile_sections = pile_sections)
        obj._postinit()

        return obj

    @property
    def bottom_elevation(self) -> float: 
        """
        Bottom elevation of the pile.
        """
        return self.top_elevation - sum(self.pile_sections['length']) 

    @property
    def length(self) -> float: 
        """
        Pile length.
        """
        return sum(self.pile_sections['length']) 
          
              
    @property
    def E(self) -> float: 
        """
        Young modulus of the pile material. Thie value does not vary across and along the pile.
        """
        try:
            return self._young_modulus
        except AttributeError:
            print('Please first create the pile with the Pile.create() method')
        except Exception as e:
            print(e)
    
    @E.setter
    def E(self, value: float) -> None:
        try:
            self._young_modulus = value
        except AttributeError:
            print('Please first create the pile with the Pile.create() method')
        except Exception as e:
            print(e)
            
    @property
    def I(self) -> float:
        """
        Second moment of area of the pile. 
        
        If user-defined, the whole
        second moment of area of the pile is overriden. 
        """
        try:
            return self.data['I [m4]'].mean()
        except AttributeError:
            print('Please first create the pile with the Pile.create() method')
        except Exception as e:
            print(e)
            
    @I.setter
    def I(self, value: float) -> None:
        try:
            self.data.loc[:,'I [m4]'] = value
            self.data.loc[:,['Area [m2]' ,'Wall thickness [m]']] = pd.NA  
        except AttributeError:
            print('Please first create the pile with the Pile.create() method')
        except Exception as e:
            print(e)  

    @property
    def width(self) -> float:
        """
        Width of the pile. (Used to compute soil springs)
        """
        try:
            return self.data['Diameter [m]'].mean()
        except AttributeError:
            print('Please first create the pile with the Pile.create() method')
        except Exception as e:
            print(e)
            
    @width.setter
    def width(self, value: float) -> None:
        try:
            self.data.loc[:,'Diameter [m]'] = value
            self.data.loc[:,['Area [m2]' ,'Wall thickness [m]']] = pd.NA  
        except AttributeError:
            print('Please first create the pile with the Pile.create() method')
        except Exception as e:
            print(e)


@dataclass(config=PydanticConfig)
class Layer:
    """A class to create a layer. The Layer stores information on the soil parameters
    of the layer as well as the constitutive model (aka. the soil spring) that is 
    deemed representative. 

    Example
    -------
    
    >>> # Create a layer
    >>> layer1 = Layer(name='Soft Clay',
    >>>            top=0,
    >>>            bottom=-10,
    >>>            weight=9,
    >>>            lateralmodel=APIclay(Su=[30,35], eps50=[0.01, 0.02], Neq=100), 
    >>>         )
    
    >>> # show layer
    >>> print(layer1)
    Soft Clay
    9.0 kN/m3
    API clay
    Su = 30.0-35.0 kPa
    eps50 = 0.01-0.02
    Cyclic
    """
    name: str
    top: float
    bottom: float
    weight: Union[PositiveFloat, conlist(PositiveFloat, min_items=1, max_items=2)]
    lateral_model: Optional[ConstitutiveModel] = None
    axial_model: Optional[ConstitutiveModel] = None
    color: Optional[str] = None

    def __post_init__(self):
        if self.color is None:
            self.color = generate_color_string()

    def __str__(self):
        return f"Name: {self.name}\nElevation: ({self.top}) - ({self.bottom}) m\nWeight: {self.weight} kN/m3\nLateral model: {self.lateral_model}\nAxial model: {self.axial_model}"

    @root_validator
    def check_elevations(cls, values): #pylint: disable=no-self-argument
        if not values['top'] > values['bottom']:
            print('Bottom elevation is higher than top elevation')
            raise ValueError
        else:
            return values
    

@dataclass(config=PydanticConfig)
class SoilProfile:
    """
    A class to create the soil profile. A soil profile consist of a ground elevation (or top elevation)
    with one or more layers of soil. 

    Additionally, a soil profile can include discrete information at given elevation such as CPT 
    (Cone Penetration Test) data

    Example
    -------

    >>> from openpile.construct import SoilProfile, Layer, APIsand
        
    """
    #: name of soil profile / borehole / location
    name: str
    #: top of ground elevation with respect to the model reference elevation datum
    top_elevation: float
    #: water elevation (this can refer to sea elevation of water table)
    water_elevation: float
    #: soil layers to consider in the soil propfile 
    layers: List[Layer]
    #: Cone Penetration Test data with folloeing structure: 
    #: 1st col: elevation[m], 2nd col: cone resistance[MPa], 3rd col: pore pressure u2 [MPa] 
    #: (the cpt data cannot be given outside the soil profile boundaries defined by the layers)
    cpt_data: Optional[np.ndarray] = None
    
    @root_validator
    def check_layers_elevations(cls,values): # pylint: disable=no-self-argument
        layers = values['layers']
        
        top_elevations = np.array([x.top for x in layers], dtype=float)
        bottom_elevations = np.array([x.bottom for x in layers], dtype=float)  
        idx_sort = np.argsort(top_elevations)        
        
        top_sorted = top_elevations[idx_sort][::-1]     
        bottom_sorted = bottom_elevations[idx_sort][::-1] 
        
        #check no overlap
        if top_sorted[0] != values['top_elevation']:
            raise ValueError("top_elevation not matching uppermost layer's elevations.")
        
        for i in range(len(top_sorted)-1):            
            if not m.isclose(top_sorted[i+1], bottom_sorted[i], abs_tol=0.001):
                raise ValueError("Layers' elevations overlap.")
    
        return values
    
    def __str__(self):
        """List all layers in table-like format"""
        out = ""
        i = 0
        for layer in self.layers:
            i += 1
            out += f"Layer {i}\n" + "-"*30 + "\n"
            out += f"{layer}\n" + "~"*30 + "\n"
        return out
    
    def _postinit(self):
        pass
    
    @property
    def bottom_elevation(self) -> float: 
        """
        Bottom elevation of the soil profile.
        """
        return self.top_elevation - sum([abs(x.top - x.bottom) for x in self.layers]) 
    
    @classmethod
    def create(cls, name: str, top_elevation: float, water_elevation: float, layers: List[Layer], cpt_data: Optional[np.ndarray] = None):
        
        obj = cls(name=name, top_elevation=top_elevation, water_elevation=water_elevation, layers=layers, cpt_data=cpt_data)
        obj._postinit()
        
        return obj  



@dataclass(config=PydanticConfig)
class Model:
    """
    A class to create the Model.

    The Model is constructed based on the pile geometry/data primarily.
    Additionally, a soil profile can be fed to the Model, and soil springs can be created. 
        
    Example
    -------
    
    >>> from openpile.construct import Pile, Model    
    
    >>> # create Model without soil maximum 5 metres apart.
    >>> Model_without_soil = Model.create(pile=p, coarseness=5)
    >>> # create Model with nodes maximum 1 metre apart with soil springs
    >>> Model_with_soil = Model.create(pile=p, soil=sp, coarseness=1)

    """
    #: pile instance that the Model should consider
    pile: Pile
    #: soil profile instance that the Model should consider
    soil: Optional[SoilProfile] = None
    #: "EB" for Euler-Bernoulli or "T" for Timoshenko
    element_type: Literal['Timoshenko', 'EulerBernoulli'] = 'Timoshenko'
    #: x coordinates values to mesh as nodes
    x2mesh: List[float] = Field(default_factory=list)
    #: mesh coarseness, represent the maximum accepted length of elements
    coarseness: float = 0.5
    #: whether to include p-y springs in the calculations
    distributed_lateral: bool = True
    #: whether to include m-t springs in the calculations
    distributed_moment: bool = False
    #: whether to include Hb-y springs in the calculations
    base_shear: bool = False
    #: whether to include Mb-t springs in the calculations
    base_moment: bool = False
    
    @root_validator
    def soil_and_pile_bottom_elevation_match(cls, values): # pylint: disable=no-self-argument
        if values['pile'].bottom_elevation < values['soil'].bottom_elevation:
            raise UserWarning('The pile ends deeper than the soil profile.')
        return values
    
    def get_structural_properties(self) -> pd.DataFrame:
        """
        Returns a table with the structural properties of the pile sections.
        """
        try: 
            return self.element_properties
        except AttributeError:
            print('Data not found. Please create Model with the Model.create() method.')
        except Exception as e:
            print(e)

    def get_soil_properties(self) -> pd.DataFrame:
        """
        Returns a table with the soil main properties and soil models of each element.
        """
        try:
            return self.soil_properties
        except AttributeError:
            print('Data not found. Please create Model with the Model.create() method.')
        except Exception as e:
            print(e)

    def _postinit(self):
        
        def get_coordinates() -> pd.DataFrame:
            
            # Primary discretisation over x-axis
            x = np.array([],dtype=np.float16)
            # add get pile relevant sections
            x = np.append(x,self.pile.data['Elevation [m]'].values)
            # add soil relevant layers and others
            if self.soil is not None:
                soil_elevations = np.array([x.top for x in self.soil.layers]+[x.bottom for x in self.soil.layers], dtype=float)
                x = np.append(x,soil_elevations)
            # add user-defined elevation
            x = np.append(x, self.x2mesh)
            
            # get unique values and sort in reverse order 
            x = np.unique(x)[::-1]
            
            # Secondary discretisation over x-axis depending on coarseness factor
            x_secondary = np.array([],dtype=np.float16)
            for i in range(len(x)-1):
                spacing = x[i] - x[i+1]
                new_spacing = spacing
                divider = 1
                while new_spacing > self.coarseness:
                    divider +=1
                    new_spacing = spacing/divider
                new_x = x[i] - (np.arange(start=1, stop=divider) * np.tile(new_spacing, (divider-1) ))                
                x_secondary = np.append(x_secondary, new_x)
                
            # assemble x- coordinates 
            x = np.append(x,x_secondary)
            x = np.unique(x)[::-1]
            
            # dummy y- coordinates
            y = np.zeros(shape= x.shape)
            
            #create dataframe coordinates
            nodes = pd.DataFrame(
                data = {
                    'x [m]': x,
                    'y [m]': y,
                }, dtype= float
            ).round(3)
            nodes.index.name = 'Node no.'
            
            element = pd.DataFrame(
                data = {
                    'x_top [m]': x[:-1],
                    'x_bottom [m]': x[1:],
                    'y_top [m]': y[:-1],
                    'y_bottom [m]': y[1:],
                }, dtype=float
            ).round(3)
            element.index.name = 'Element no.'
            
            return nodes, element
        
            # function doing the work
    
        def get_soil_profile() -> pd.DataFrame:
            top_elevations = [x.top for x in self.soil.layers]
            bottom_elevations = [x.bottom for x in self.soil.layers]
            soil_weights =  [x.weight for x in self.soil.layers]
            
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
            # elevation w.r.t. ground elevation
            xg = [z - x[0] for z in x]
            
            return pd.DataFrame(data={'Top soil layer [m]': x,
                                      "Unit Weight [kN/m3]": soil_weights}, dtype=np.float64)
    
        def create_springs() -> np.ndarray:
            
            #dim of springs
            spring_dim = 15
            
            # Allocate array
            py = np.zeros(shape=(self.element_number,2,2,spring_dim),dtype=np.float32)
            mt = np.zeros(shape=(self.element_number,2,2,spring_dim),dtype=np.float32)
            Hb = np.zeros(shape=(1,1,2,spring_dim),dtype=np.float32)
            Mb = np.zeros(shape=(1,1,2,spring_dim),dtype=np.float32)
            
            tz = np.zeros(shape=(self.element_number,2,2,15),dtype=np.float32)

            # fill in spring for each element
            for layer in self.soil.layers: 
                elements_for_layer = self.soil_properties.loc[
                    (self.soil_properties['x_top [m]'] <= layer.top) 
                     & (self.soil_properties['x_bottom [m]'] >= layer.bottom)
                    ].index
                #py curve
                if layer.lateral_model.spring_signature[0]: #True if py spring function exist
                    for i in elements_for_layer:
                        sig_v = self.soil_properties[['sigma_v top [kPa]','sigma_v bottom [kPa]']].iloc[i]
                        elevation = self.soil_properties[['x_top [m]', 'x_bottom [m]']].iloc[i]
                        depth_from_ground = (self.soil_properties[['xg_top [m]','xg_bottom [m]']].iloc[i]).abs()
                        pile_width = self.element_properties['Diameter [m]'].iloc[i]

                        # top and bottom of element
                        for j in [0,1]:
                            py[i, j, 0, :], py[i, j, 1, :] = layer.lateral_model.py_spring_fct(
                                sig=sig_v[j],
                                X = depth_from_ground[j],
                                layer_height = (layer.top - layer.bottom),
                                depth_from_top_of_layer = (layer.top - elevation[j]),
                                D = pile_width, 
                                L = self.pile.length,
                                output_length = spring_dim)
            
            return py, mt, Hb, Mb, tz
              
        # creates mesh coordinates
        self.nodes_coordinates, self.element_coordinates = get_coordinates()
        self.element_number = int(self.element_coordinates.shape[0])
        
        # creates element structural properties
        #merge Pile.data and self.coordinates
        self.element_properties = pd.merge_asof(left= self.element_coordinates.sort_values(by=['x_top [m]']),
                                                right= self.pile.data.sort_values(by=['Elevation [m]']),
                                                left_on='x_top [m]',
                                                right_on='Elevation [m]',
                                                direction='forward').sort_values(by=['x_top [m]'],ascending=False)
        #add young modulus to data
        self.element_properties['E [kPa]'] = self.pile.E
        #delete Elevation [m] column
        self.element_properties.drop('Elevation [m]', inplace=True, axis=1)
        #reset index
        self.element_properties.reset_index(inplace=True, drop=True)

        # create soil properties
        self.soil_properties = pd.merge_asof(left= self.element_coordinates[['x_top [m]','x_bottom [m]']].sort_values(by=['x_top [m]']),
                                                right= get_soil_profile().sort_values(by=['Top soil layer [m]']),
                                                left_on='x_top [m]',
                                                right_on='Top soil layer [m]',
                                                direction='forward').sort_values(by=['x_top [m]'],ascending=False)
        # add elevation of element w.r.t. ground level
        self.soil_properties['xg_top [m]'] = self.soil_properties['x_top [m]'] - self.soil.top_elevation
        self.soil_properties['xg_bottom [m]'] = self.soil_properties['x_bottom [m]'] - self.soil.top_elevation
        # add vertical stress at top and bottom of each element
        s =  (self.soil_properties['x_top [m]'] - self.soil_properties['x_bottom [m]']) * self.soil_properties["Unit Weight [kN/m3]"]
        self.soil_properties['sigma_v top [kPa]'] = np.insert(s.cumsum().values[:-1], 
                                                              np.where(self.soil_properties['x_top [m]'].values==self.soil.top_elevation)[0], 
                                                              0.0)
        self.soil_properties['sigma_v bottom [kPa]'] = s.cumsum()
        #reset index
        self.soil_properties.reset_index(inplace=True, drop=True)


        # Initialise nodal global forces with link to nodes_coordinates (used for force-driven calcs)
        self.global_forces = self.nodes_coordinates.copy()
        self.global_forces['Px [kN]'] = 0
        self.global_forces['Py [kN]'] = 0
        self.global_forces['Mz [kNm]'] = 0

        # Initialise nodal global displacement with link to nodes_coordinates (used for displacement-driven calcs)
        self.global_disp = self.nodes_coordinates.copy()
        self.global_disp['Tx [m]'] = 0
        self.global_disp['Ty [m]'] = 0
        self.global_disp['Rz [rad]'] = 0
        
        # Initialise nodal global support with link to nodes_coordinates (used for defining boundary conditions)
        self.global_restrained = self.nodes_coordinates.copy()
        self.global_restrained['Tx'] = False
        self.global_restrained['Ty'] = False
        self.global_restrained['Rz'] = False


        self.py_springs, self.mt_springs, self.Hb_spring, self.Mb_spring, self.tz_springs = create_springs()
    
    

    def soil_springs(self, kind:Literal['p-y','m-t','Hb-y','Mb-t','t-z']) -> pd.DataFrame: 
        """
        Returns soil springs created for the given model in one DataFrame.

        Parameters
        ----------
        kind : str
            type of spring to extract.

        Returns
        -------
        pd.DataFrame
            Soil springs
        """
        
        def springs_to_df(springs:np.ndarray, flag) -> pd.DataFrame:
            spring_dim = springs.shape[-1]
            column_values_spring = [f"VAL {i}" for i in range(spring_dim)]
            
            id = np.repeat(np.arange(self.element_number),4)
            x = np.repeat(misc.repeat_inner(self.nodes_coordinates['x [m]'].values),2)
            
            if len(x) > 2:
                t_b = ["top","bottom"]*int(len(x)/2)
            
                df = pd.DataFrame(data = {
                    "Element no.":id,
                    "Position":t_b,
                    "Elevation [m]": x, 
                    })
            else:
                df = pd.DataFrame(data = {
                    "Element no.":id,
                    "Elevation [m]": x, 
                    })

            df['type'] = flag.split("-")*int(len(x)/2)
            df[column_values_spring] = np.reshape(springs, (-1,spring_dim))

            return df
        
        # main part of function
        if self.soil is None:
            RuntimeError('No soil found. Please create the Model first with soil.')
        else:
            if kind == 'p-y':
                springs_df = springs_to_df(self.py_springs, flag=kind)
            elif kind == 'm-t':
                springs_df = springs_to_df(self.mt_springs, flag=kind)
            elif kind == 'Hb-y':
                springs_df = springs_to_df(self.Hb_spring, flag=kind)
            elif kind == 'Mb-t':
                springs_df = springs_to_df(self.Mb_spring, flag=kind)
            elif kind == 't-z':
                springs_df = springs_to_df(self.tz_springs, flag=kind)
            else:
                ValueError("kind should be one of ['p-y','m-t','Hb-y','Mb-t','t-z']")

        return springs_df
    
    
    def get_pointload(self, output = False, verbose = True):
        """
        Returns the point loads currently defined in the mesh via printout statements.
        
        Parameters
        ----------
        output : bool, optional
            If true, it returns the printout statements as a variable, by default False
        verbose : float, optional
            if True, printout statements printed automaically (ideal for use with iPython), by default True
        """
        out = ""
        try:
            for (idx, elevation, _, Px, Py, Mz) in self.global_forces.itertuples(name=None):
                if any([Px, Py, Mz]):
                    string = f"\nLoad applied at elevation {elevation} m (node no. {idx}): Px = {Px} kN, Py = {Py} kN, Mx = {Mz} kNm."
                    if verbose is True:
                        print(string)
                    out += f"\nLoad applied at elevation {elevation} m (node no. {idx}): Px = {Px} kN, Py = {Py} kN, Mx = {Mz} kNm."
            if output is True:
                return out
        except Exception:
            print('No data found. Please create the Model first.')
            raise
    
        
    def set_pointload(self, elevation:float=0.0, Py:float=0.0, Px:float=0.0, Mz:float=0.0):
        """
        Defines the point load(s) at a given elevation. 
        
        .. note:
            If run several times at the same elevation, the loads are overwritten by the last command.


        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0
        Py : float, optional
            Shear force in kN, by default 0.0
        Px : float, optional
            Normal force in kN, by default 0.0
        Mz : float, optional
            Bending moment in kNm, by default 0.0
        """
    
        #identify if one node is at given elevation or if load needs to be split
        nodes_elevations = self.nodes_coordinates['x [m]'].values
        # check if corresponding node exist
        check = np.isclose(nodes_elevations ,np.tile(elevation, nodes_elevations.shape),atol=0.001)
        
        try:
            if any(check):
                #one node correspond, extract node
                node_idx = int(np.where(check == True)[0])
                # apply loads at this node
                self.global_forces.loc[node_idx,'Px [kN]'] = Px
                self.global_forces.loc[node_idx,'Py [kN]'] = Py
                self.global_forces.loc[node_idx,'Mz [kNm]'] = Mz
            else:
                if elevation > self.nodes_coordinates['x [m]'].iloc[0] or elevation < self.nodes_coordinates['x [m]'].iloc[-1]:
                    print("Load not applied! The chosen elevation is outside the mesh. The load must be applied on the structure.")
                else:
                    print("Load not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the Model.")
        except Exception:
            print("\n!User Input Error! Please create Model first with the Model.create().\n")
            raise


    def set_support(self, elevation:float=0.0, Ty:bool=False, Tx:bool=False, Rz:bool=False):
        """
        Defines the supports at a given elevation. If True, the relevant degree of freedom is restrained.
        
        .. note:
            If run several times at the same elevation, the support are overwritten by the last command.


        Parameters
        ----------
        elevation : float, optional
            the elevation must match the elevation of a node, by default 0.0
        Ty : bool, optional
            Translation along y-axis, by default False
        Tx : bool, optional
            Translation along x-axis, by default False
        Rz : bool, optional
            Rotation around z-axis, by default False
        """

        try:
            #identify if one node is at given elevation or if load needs to be split
            nodes_elevations = self.nodes_coordinates['x [m]'].values
            # check if corresponding node exist
            check = np.isclose(nodes_elevations ,np.tile(elevation, nodes_elevations.shape),atol=0.001)
            
            if any(check):
                #one node correspond, extract node
                node_idx = int(np.where(check == True)[0])
                # apply loads at this node
                self.global_restrained.loc[node_idx,'Tx'] = Tx
                self.global_restrained.loc[node_idx,'Ty'] = Ty
                self.global_restrained.loc[node_idx,'Rz'] = Rz
            else:
                if elevation > self.nodes_coordinates['x [m]'].iloc[0] or elevation < self.nodes_coordinates['x [m]'].iloc[-1]:
                    print("Support not applied! The chosen elevation is outside the mesh. The support must be applied on the structure.")
                else:
                    print("Support not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the Model.")
        except Exception:
            print("\n!User Input Error! Please create Model first with the Model.create().\n")
            raise

        
    def plot(self, assign = False):
        fig = graphics.connectivity_plot(self)
        return fig if assign else None


    @classmethod
    def create(cls, 
               pile: Pile, 
               soil: Optional[SoilProfile] = None, 
               element_type: Literal['Timoshenko', 'EulerBernoulli'] = 'Timoshenko', 
               x2mesh: List[float] = Field(default_factory=list), 
               coarseness: float = 0.5,
               py_springs: bool = True,
               mt_springs: bool = False,
               Hb_spring: bool = False,
               Mb_spring: bool = False
               ):
        
        obj = cls(pile=pile, soil=soil, element_type=element_type, x2mesh=x2mesh, coarseness=coarseness)
        obj._postinit()
        
        return obj
    

    def __str__(self):
        return self.element_properties.to_string()

