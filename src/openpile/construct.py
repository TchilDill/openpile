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
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

from openpile.utils import validation as validate

class PydanticConfig:
    arbitrary_types_allowed = True

@dataclass(config=PydanticConfig)
class Pile:
    """
    A class to create the pile.

    Pile instances include the pile geometry and data. Following
    the initialisation of a pile, a Pandas dataframe is created 
    which can be read by the following command:
    
    **Example**
    
    >>> from openpile.construct import Pile
     
    >>> # Create a pile instance with two sections of respectively 10m and 30m length.
    >>> pile = Pile(type='Circular',
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
    type: Literal['Circular']
    #: select the type of material the pile is made of, can be of ('Steel', )
    material: Literal['Steel']
    #: top elevation of the pile according to general vertical reference set by user
    top_elevation: float
    #: pile geometry made of a dictionary of lists. the structure of the dictionary depends on the type of pile selected.
    #: There can be as many sections as needed by the user. The length of the listsdictates the number of pile sections. 
    pile_sections: Dict[str, List[float]] = Field(default_factory=Dict)
    
    def create(self):
        
        # check that dict is correctly entered
        validate.pile_sections_must_be(self)
        
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
            if self.type == 'Circular':
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
    
    **Example**

    """
    #: pile instance that the mesh should consider
    pile: Pile
    #: soil profile instance that the mesh should consider
    soil: Optional[SoilProfile] = None
    #: "EB" for Euler-Bernoulli or "T" for Timoshenko
    element_type: Literal['Timoshenko', 'EulerBernoulli'] = 'Timoshenko'
    #: z coordinates values to mesh as nodes
    z2mesh: List[float] = Field(default_factory=list)
    #: mesh coarseness, represent the maximum accepted length of elements
    coarseness: float = 0.2
    
    def get_structural_properties(self):
        pass

    def get_soil_properties(self):
        pass

    def create(self):
        
        def get_coordinates() -> pd.DataFrame:
            
            # Primary discretisation over z-axis
            z = np.array([],dtype=np.float16)
            # add get pile relevant sections
            z = np.append(z,self.pile.data['Elevation [m]'].values)
            # add soil relevant layers and others
            if self.soil is not None:
                z = np.append(z,self.soil.data['Elevation [m]'].values)
            # add user-defined elevation
            z = np.append(z, self.z2mesh)
            
            # get unique values and sort in reverse order 
            z = np.unique(z)[::-1]
            
            # Secondary discretisation over z-axis depending on coarseness factor
            z_secondary = np.array([],dtype=np.float16)
            for i in range(len(z)-1):
                spacing = z[i] - z[i+1]
                new_spacing = spacing
                divider = 1
                while new_spacing > self.coarseness:
                    divider +=1
                    new_spacing = spacing/divider
                new_z = z[i] - (np.arange(start=1, stop=divider) * np.tile(new_spacing, (divider-1) ))   
                
                z_secondary = np.append(z_secondary, new_z)
                
            # assemble z- coordinates 
            z = np.append(z,z_secondary)
            z = np.unique(z)[::-1]
            
            # dummy y- coordinates
            y = np.zeros(shape= z.shape)
            
            #create dataframe coordinates
            coordinates = pd.DataFrame(
                data = {
                    'z': z,
                    'y': y,
                }, dtype= float
            ).round(3)
            coordinates.index.name = 'Node no.'
            
            return coordinates
        
        
        # creates mesh coordinates
        nodes_coordinates = get_coordinates()
        self.coordinates = pd.DataFrame(
            data = {
                'z_top [m]': nodes_coordinates['z'].values[:-1],
                'z_bottom [m]':nodes_coordinates['z'].values[1:],
                'y_top [m]': nodes_coordinates['y'].values[:-1],
                'y_bottom [m]':nodes_coordinates['y'].values[1:],
            }, dtype=float
        )
        self.coordinates.index.name = 'Element no.'
        
        # creates element structural properties
        #merge Pile.data and self.coordinates
        self.element_properties = pd.merge_asof(left= self.coordinates.sort_values(by=['z_top [m]']),
                                                right= self.pile.data.drop_duplicates(subset='Elevation [m]',keep='last').sort_values(by=['Elevation [m]']),
                                                left_on='z_top [m]',
                                                right_on='Elevation [m]',
                                                direction='forward').sort_values(by=['z_top [m]'],ascending=False)
        self.element_properties['E [kPa]'] = self.pile.E
        
        # create element soil properties
        
    def ssis(self) -> pd.DataFrame: 
        pass

if __name__ == "__main__":

    from openpile.construct import Pile, Mesh
    
    #Check pile
    MP01 = Pile(type='Circular',
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
    print(MP01_mesh.element_properties)
