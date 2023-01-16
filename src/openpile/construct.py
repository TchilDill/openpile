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
from typing import List, Optional
from pydantic import BaseModel, validator, Field
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Pile:
    """
    A class to create the pile.

    Pile instances include the pile geometry and data. Following
    the initialisation of a pile, a Pandas dataframe is created 
    which can be read by the following command:
    
    **Example**
    
    >>> import openpile as op
    >>> # Create a pile instance
    >>> pile = Pile(type='Circular',
    >>>             material='Steel',
    >>>             top_elevation = 0,
    >>>             length_spread_thickness=[ 
    >>>                 [ 5, 10.0, 0.06],
    >>>                 [30, 10.0, 0.08],
    >>>             ],
    >>>         )

    >>> # print the dataframe
    >>> print(pile.data)
        Elevation [m]  Spread [m]  Thickness [m]  Area [m2]     I [m4]
    0              0       10.0           0.08   1.873646  23.141213
    1             -5       10.0           0.08   1.873646  23.141213
    2             -5       10.0           0.08   2.493168  30.669955
    3            -35       10.0           0.08   2.493168  30.669955
    
    >>> # Override young's modulus
    >>> pile.E = 250e6
    >>> # Check young's modulus    
    >>> print(pile.E)
    250000000.0
    
    >>> # Override second moment of area across whole pile
    >>> pile.I = 1.11
    >>> # Check updated second moment of area
    >>> print(pile.data)
        Elevation [m]  Spread [m] Thickness [m] Area [m2]  I [m4]
    0              0       10.0          <NA>      <NA>    1.11
    1             -5       10.0          <NA>      <NA>    1.11
    2             -5       10.0          <NA>      <NA>    1.11
    3            -35       10.0          <NA>      <NA>    1.11    
    
    >>> # Override pile's width or pile's diameter
    >>> pile.Spread = 2.22
    >>> # Check updated width or diameter
    >>> print(pile.data)   
       Elevation [m]  Spread [m] Thickness [m] Area [m2]  I [m4]
    0              0       10.0          <NA>      <NA>    1.11
    1             -5       10.0          <NA>      <NA>    1.11
    2             -5       10.0          <NA>      <NA>    1.11
    3            -35       10.0          <NA>      <NA>    1.11    
    """
    #: select the type of pile, can be of ('Circular', )
    type: str 
    #: select the type of material the pile is made of, can be of ('Steel', )
    material: str 
    #: top elevation of the pile according to general vertical reference set by user
    top_elevation: float
    #: pile geometry where each list describes a section of the pile,
    #: there can be as many sections as needed by the user.
    #:
    #: A pile with a unique section of 30m long, 6m diameter, and 80mm wall 
    #: thicknnes would look like what is given in the example above.
    length_spread_thickness: List[List[float]]
    
    @validator('type', always=True)
    def _type_must_equal(cls, v):
        accepted_values = ['Circular',]
        if v not in accepted_values:
            raise ValueError("Pile type must be one of the following: \n - " + '\n - '.join(accepted_values))  
        return v
    
    @validator('material', always=True)
    def _material_must_equal(cls, v):
        accepted_values = ['Steel',]
        if v not in accepted_values:
            raise ValueError("Pile material must be one of the following: \n - " + '\n - '.join(accepted_values))  
        return v
    
    @validator('length_spread_thickness', always=True)
    def _check_length_spread_thickness(cls, v):
        for idx, sublist in enumerate(v):
            if len(sublist) != 3: 
                raise ValueError("The input parameter `length_spread_thickness` for openpile.pile shall be a list of lists, the latter composed of 3 items, the first one for the length of the section, the second for the diameter of the section, and the third for the wall thickness. "+f"{len(sublist)} items were given in the pile section no. {idx+1}")  
            if sublist[2] >= sublist[1]/2:
                raise ValueError("The wall thickness cannot be larger than half the spread of the pile")
        return v
    
    def __post_init__(self):
        """Function that performs additional steps after initialisation.
        """

        # Create material specific specs for given material        
        # if steel
        if self.material == 'Steel':
            # unit weight
            self._UW = 78.0 # kN/m3
            # young modulus
            self._Young_modulus = 210.0e6 #kPa
        else:
            ValueError() 
        
        
        # Create top and bottom elevations
        elevation = []
        #add bottom of section i and top of section i+1 (essentially the same values) 
        for section_id, section_values in enumerate(self.length_spread_thickness):
            if section_id == 0:
                elevation.append(self.top_elevation)
                elevation.append(elevation[-1] - section_values[0])
            else:
                elevation.append(elevation[-1])
                elevation.append(elevation[-1] - section_values[0])

        #create sectional properties
        
        #spread
        spread = []
        #add top and bottom of section i (essentially the same values) 
        for _, section_values in enumerate(self.length_spread_thickness):
            spread.append(section_values[1])
            spread.append(spread[-1])

        #thickness
        thickness = []
        #add top and bottom of section i (essentially the same values) 
        for _, section_values in enumerate(self.length_spread_thickness):
            thickness.append(section_values[2])
            thickness.append(thickness[-1])
           
        #Area & second moment of area
        area = []
        second_moment_of_area = []
        #add top and bottom of section i (essentially the same values) 
        for _, section_values in enumerate(self.length_spread_thickness):
            #calculate area
            diam = section_values[1] #diameter or width
            thickness = section_values[2] #wall thickness
            if self.type == 'Circular':
                A = m.pi / 4 * (diam**2 - (diam-2*thickness)**2)
                I = m.pi / 64 * (diam**4 - (diam-2*thickness)**4)
                area.append(A)
                area.append(area[-1])
                second_moment_of_area.append(I)
                second_moment_of_area.append(second_moment_of_area[-1])
            else:
                #not yet supporting other kind
                ValueError()
        
        # second moment of area
        second_moment_of_area = []
        #add top and bottom of section i (essentially the same values) 
        for _, section_values in enumerate(self.length_spread_thickness):
            #calculate area
            diam = section_values[1] #diameter or width
            thickness = section_values[2] #wall thickness
            if self.type == 'Circular':
                I = m.pi / 64 * (diam**4 - (diam-2*thickness)**4)
                second_moment_of_area.append(I)
                second_moment_of_area.append(second_moment_of_area[-1])
            else:
                #not yet supporting other kind
                ValueError()        
        
        # Create pile data     
        self.data = pd.DataFrame(data = {
            'Elevation [m]' :  elevation,
            'Spread [m]' : spread,
            'Thickness [m]' : thickness,  
            'Area [m2]' : area,
            'I [m4]': second_moment_of_area,
            }
        )    
        
    @property
    def E(self):
        """Young modulus of the pile material. Constant for the entire pile.

        """
        return self._Young_modulus
    
    @E.setter
    def E(self, value: float) -> None:
        try:
            self._Young_modulus = value
        except TypeError:
            raise('Value must be a float')
            
    @property
    def I(self):
        """Second moment of area of the pile. 
        
        If user-defined, the whole
        second moment of area of the pile is overriden. 
        """
        return self.data['I [m4]'].mean()
    
    @I.setter
    def I(self, value: float) -> None:
        self.data.loc[:,'I [m4]'] = value
        self.data.loc[:,['Area [m2]' ,'Thickness [m]']] = pd.NA    

    @property
    def Spread(self):
        """Width of the pile. Used to compute soil springs.
        
        """
        return self.data['Spread [m]'].mean()
    
    @Spread.setter
    def Spread(self, value: float) -> None:
        self.data.loc[:,'Spread [m]'] = value
        self.data.loc[:,['Area [m2]' ,'Thickness [m]']] = pd.NA    



if __name__ == "__main__":
    MP01 = Pile(type='Circular',
                material='Steel',
                top_elevation = 0,
                length_spread_thickness=[ 
                    [ 5, 10.0, 0.06],
                    [30, 10.0, 0.08],
                ],
            )
    print(MP01.data)
    MP01.E = 250e6
    print(MP01.E)
    MP01.I = 1.11
    print(MP01.data)
    MP01.Spread = 2.22
    print(MP01.data)

