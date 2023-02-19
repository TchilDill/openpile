"""
`Construct` module
==================

The `construct` module is used to construct all objects that 
form the inputs to calculations in openpile. 


This include:

- the pile
- the soil profile
- the mesh

"""
import math as m
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field, root_validator
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics
from openpile.utils import validation as validation

class PydanticConfig:
    arbitrary_types_allowed = True

@dataclass(config=PydanticConfig)
class Pile:
    """
    A class to create the pile.

    Pile instances include the pile geometry and data. Following
    the initialisation of a pile, a Pandas dataframe is created 
    which can be read by the following command:
    
    Example
    -------
    >>> from openpile.construct import Pile
     
    >>> # Create a pile instance with two sections of respectively 10m and 30m length.
    >>> pile = Pile(kind='Circular',
    >>>         material='Steel',
    >>>         top_elevation = 0,
    >>>         pile_sections={
    >>>             'length':[10,30],
    >>>             'diameter':[7.5,7.5],
    >>>             'wall thickness':[0.07, 0.08],
    >>>         } 
    >>>     )
    >>> # Create the pile secondary data
    >>> pile.create()
    >>> # Print the pile data 
    >>> print(pile.data)
    Elevation [m]  Diameter [m]  Wall thickness [m]  Area [m2]     I [m4]
    0            0.0           7.5                0.08   1.633942  11.276204
    1          -10.0           7.5                0.08   1.633942  11.276204
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
    >>> print(pile.data)
    Elevation [m]  Diameter [m] Wall thickness [m] Area [m2]  I [m4]
    0            0.0           7.5               <NA>      <NA>    1.11
    1          -10.0           7.5               <NA>      <NA>    1.11
    2          -10.0           7.5               <NA>      <NA>    1.11
    3          -40.0           7.5               <NA>      <NA>    1.11  
    
    >>> # Override pile's width or pile's diameter
    >>> pile.width = 2.22
    >>> # Check updated width or diameter
    >>> print(pile.data)   
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
    pile_sections: Dict[str, List[float]]
    
    def create(self):
        
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
            raise ValueError()
        
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
            #calculate area
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
    
    def show(self):
        print(self.data.to_string())
      
    @property
    def E(self) -> float: 
        """Young modulus of the pile material. Constant for the entire pile.

        """
        try:
            return self._young_modulus
        except Exception as exc:
            raise NameError('Please first create the pile with .create() method') from exc

    
    @E.setter
    def E(self, value: float) -> None:
        try:
            self._young_modulus = value
        except Exception as exc:
            raise NameError('Please first create the pile with .create() method') from exc

            
    @property
    def I(self) -> float:
        """Second moment of area of the pile. 
        
        If user-defined, the whole
        second moment of area of the pile is overriden. 
        """
        try:
            return self.data['I [m4]'].mean()
        except Exception as exc:
            raise NameError('Please first create the pile with .create() method') from exc
    
    @I.setter
    def I(self, value: float) -> None:
        self.data.loc[:,'I [m4]'] = value
        self.data.loc[:,['Area [m2]' ,'Wall thickness [m]']] = pd.NA    

    @property
    def width(self) -> float:
        """Width of the pile. Used to compute soil springs.
        
        """
        try:
            return self.data['Diameter [m]'].mean()
        except Exception as exc:
            raise NameError('Please first create the pile with .create() method') from exc

    @width.setter
    def width(self, value: float) -> None:
        try:
            self.data.loc[:,'Diameter [m]'] = value
            self.data.loc[:,['Area [m2]' ,'Wall thickness [m]']] = pd.NA  
        except Exception as exc:
            raise NameError('Please first create the pile with .create() method') from exc

@dataclass(config=PydanticConfig)
class SoilProfile:
    pass

@dataclass(config=PydanticConfig)
class Mesh:
    """
    A class to create the mesh.

    The mesh is constructed based on the pile geometry and data primarily.
    As such it is possible to not feed any soil profile to the mesh, relying then on restrained degree(s) of freedom.
    
    The soil profile can be supplemented such that soil springs can be computed. After creating the mesh object, the soil springs can be
    generated via the `ssis()` method.
    
    Example
    -------
    
    """
    #: pile instance that the mesh should consider
    pile: Pile
    #: soil profile instance that the mesh should consider
    soil: Optional[SoilProfile] = None
    #: "EB" for Euler-Bernoulli or "T" for Timoshenko
    element_type: Literal['Timoshenko', 'EulerBernoulli'] = 'Timoshenko'
    #: x coordinates values to mesh as nodes
    x2mesh: List[float] = Field(default_factory=list)
    #: mesh coarseness, represent the maximum accepted length of elements
    coarseness: float = 0.5
    
    def get_structural_properties(self):
        try: 
            return self.element_properties
        except Exception as exc:
            print('Data not found. Please create mesh first.')
            raise Exception from exc

    def get_soil_properties(self):
        try:
            return self.soil_properties
        except Exception as exc:
            print('Data not found. Please create mesh first.')
            raise Exception from exc

    def create(self):
        
        def get_coordinates() -> pd.DataFrame:
            
            # Primary discretisation over x-axis
            x = np.array([],dtype=np.float16)
            # add get pile relevant sections
            x = np.append(x,self.pile.data['Elevation [m]'].values)
            # add soil relevant layers and others
            if self.soil is not None:
                x = np.append(x,self.soil.data['Elevation [m]'].values)
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
        
        # creates mesh coordinates
        self.nodes_coordinates, self.element_coordinates = get_coordinates()
        self.element_number = int(self.element_coordinates.shape[0])
        
        # creates element structural properties
        #merge Pile.data and self.coordinates
        self.element_properties = pd.merge_asof(left= self.element_coordinates.sort_values(by=['x_top [m]']),
                                                right= self.pile.data.drop_duplicates(subset='Elevation [m]',keep='last').sort_values(by=['Elevation [m]']),
                                                left_on='x_top [m]',
                                                right_on='Elevation [m]',
                                                direction='forward').sort_values(by=['x_top [m]'],ascending=False)
        self.element_properties['E [kPa]'] = self.pile.E
        
        # create element soil properties
        if self.soil is None:
            self.soil_properties = None
        else:
            self.soil_properties = pd.DataFrame()

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

    def ssis(self) -> pd.DataFrame: 
        
        # function doing the work
        def create_springs():
            pass
        
        # main part of function
        try: 
            if self.soil is None:
                RuntimeError('No soil found. Please create the mesh first.')
            else: 
                self.soil_springs = create_springs()
        except Exception:
            print('No soil found. Please first create the mesh first.')
            raise
    
    def get_pointload(self, output = False, verbose = True):
        """_summary_

        _extended_summary_
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
            print('No data found. Please create the mesh first.')
            raise
        
    def set_pointload(self, elevation:float=0.0, Py:float=0.0, Px:float=0.0, Mz:float=0.0):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        elevation : float, optional
            _description_, by default 0.0
        Py : float, optional
            _description_, by default 0.0
        Px : float, optional
            _description_, by default 0.0
        Mz : float, optional
            _description_, by default 0.0
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
                    print("Load not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the mesh.")
        except Exception:
            print("\n!User Input Error! Please create mesh first with the Mesh.create().\n")
            raise

    def set_support(self, elevation:float=0.0, Ty:bool=False, Tx:bool=False, Rz:bool=False):
            """_summary_

            _extended_summary_

            Parameters
            ----------
            elevation : float, optional
                _description_, by default 0.0
            Ty : float, optional
                _description_, by default 0.0
            Tx : float, optional
                _description_, by default 0.0
            Rz : float, optional
                _description_, by default 0.0
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
                        print("Support not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the mesh.")
            except Exception:
                print("\n!User Input Error! Please create mesh first with the Mesh.create().\n")
                raise
      
    def show(self):
        print(self.data.to_string())
        
    def plot(self, assign = False):
        fig = graphics.connectivity_plot(self)
        return fig if assign else None

#@validate_arguments decorator not nedded as it is already embedded in the Pile class
def create_pile( kind: Literal['Circular'], material: Literal['Steel'], top_elevation: float,  pile_sections: Dict[str, List[float]] ):
    """_summary_

    A function to create the pile. This function provides a 2-in-1 command where:
    
    - a `Pile` instance is created
    - the `.create()` method is run right away and creates all additional pile data necessary.

    Pile instances include the pile geometry and data.
    
    Example
    -------
    >>> from openpile.construct import Pile
     
    >>> # Create a pile instance with two sections of respectively 10m and 30m length.
    >>> pile = create_pile(kind='Circular',
    >>>                 material='Steel',
    >>>                 top_elevation = 0,
    >>>                 pile_sections={
    >>>                 'length':[10,30],
    >>>                 'diameter':[7.5,7.5],
    >>>                 'wall thickness':[0.07, 0.08],
    >>>             }
    >>>         )

    See Also
    --------
    openpile.construct.Pile

    Parameters
    ----------
    kind : str
        select the type of pile, can be of ('Circular', )
    material : str
        select the type of material the pile is made of, can be of ('Steel', )
    top_elevation : float
        _description_
    pile_sections : dict
        _description_

    Returns
    -------
    _type_
        _description_
    """
    obj = Pile( kind = kind, material = material, top_elevation=top_elevation,  pile_sections = pile_sections)
    obj.create()

    return obj

#@validate_arguments decorator not nedded as it is already embedded in the Mesh class
def create_mesh(pile: Pile, soil: Optional[SoilProfile] = None, element_type: Literal['Timoshenko', 'EulerBernoulli'] = 'Timoshenko', x2mesh: List[float] = Field(default_factory=list), coarseness: float = 0.5):
    
    obj = Mesh(pile=pile, soil=soil, element_type=element_type, x2mesh=x2mesh, coarseness=coarseness)
    obj.create()
    
    return obj

if __name__ == "__main__":

    from openpile.construct import Pile, Mesh
    
    #Check pile
    MP01 = Pile(kind='Circular',
                material='Steel',
                top_elevation = 0,
                pile_sections={
                    'length':[10,30],
                    'diameter':[9.5,7.5],
                    'wall thickness':[0.1, 0.08],
                } 
            )
    # Create the pile secondary data
    MP01.create()
    # Print the pile data 
    print(MP01.data)
    # print(MP01.E)
    # MP01.E = 250e6
    # print(MP01.E)
    # MP01.I = 1.11
    # print(MP01.data)
    # MP01.width = 2.22
    # print(MP01.data)
    # from openpile.utils.txt import txt_pile
    # print(txt_pile(MP01))
    MP01_mesh = Mesh(pile=MP01, element_type="Timoshenko")
    MP01_mesh.create()
    MP01_mesh.set_pointload(elevation = -2.1, Px=500)
    MP01_mesh.get_pointload()