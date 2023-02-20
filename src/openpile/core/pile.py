import math as m
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field, root_validator
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics
import openpile.utils.validation as validation

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
    
    def __str__(self):
        return self.data.to_string()
      
    @classmethod  
    def create(cls, kind: Literal['Circular'], material: Literal['Steel'], top_elevation: float,  pile_sections: Dict[str, List[float]] ):
        """_summary_

        A method to create the pile. This function provides a 2-in-1 command where:
        
        - a `Pile` instance is created
        - the `._postinit()` method is run and creates all additional pile data necessary.

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
        obj = cls( kind = kind, material = material, top_elevation=top_elevation,  pile_sections = pile_sections)
        obj._postinit()

        return obj
      
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