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
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, validator, Field
from pydantic.dataclasses import dataclass

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
    >>> pile = op.construct.Pile(
    >>>    top_elevation = 0,
    >>>    height_width_thickness = [
    >>>        [30, 6.0, 0.08],
    >>>    ]
    >>> )
    >>> # print the dataframe
    >>> pile.print_table()
    
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
    height_width_thickness: List[List[float]]
    
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
    
    @validator('height_width_thickness', always=True)
    def _check_height_width_thickness(cls, v):
        for idx, sublist in enumerate(v):
            if len(sublist) != 3: 
                raise ValueError("The input parameter `height_width_thickness` for openpile.pile shall be a list of lists, the latter composed of 3 items, the first one for the height of the section, the second for the diameter of the section, and the third for the wall thickness. "+f"{len(sublist)} items were given in the pile section no. {idx+1}")  
            if sublist[2] >= sublist[1]/2:
                raise ValueError("The wall thickness cannot be larger than half the width of the pile")
        return v
    
    def __post_init__(self):
        top_elev = []
        bot_elev = [] 
        for i in range(len(self.height_width_thickness)):
            if i == 0:
                top_elev.append(self.top_elevation)
                bot_elev.append(self.top_elevation - self.height_width_thickness[i][0])
            else:
                top_elev.append(bot_elev[-1])
                bot_elev.append(bot_elev[-1] - self.height_width_thickness[i][0])
            
        pile_dict = {
            'top_elevation' :  top_elev,
            'bottom_elevation' : bot_elev,
            'width' : [x[1] for x in self.height_width_thickness],
            'wall thickness' : [x[2] for x in self.height_width_thickness]
            
        }
        self.data = pd.DataFrame(data=pile_dict)
          
if __name__ == "__main__":
    MP01 = Pile(type='Circular',
                material='Steel',
                top_elevation = 0,
                height_width_thickness=[ 
                    [ 5, 10.0, 0.06],
                    [30, 10.0, 0.08],
                ],
            )
    print(MP01.data)

