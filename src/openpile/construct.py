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

from typing import List, Optional
from pydantic import BaseModel, validator, Field

class Pile(BaseModel):
    """_summary_

    A class to create the pile.

    Pile instances include the pile geometry and data. Following
    the initialisation of a pile, a Pandas dataframe is created 
    which can be read by the following command:
    
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

    Parameters
    ----------
    type : _type_
        _description_

    """
    type: Optional[str] = 'Circular' 
    material: Optional[str] = 'Steel'
    top_elevation: float
    height_width_thickness: List[List[float]]
    
    @validator('type')
    def _type_must_equal(cls, v) -> None:
        accepted_values = ['Circular',]
        if v not in accepted_values:
            raise ValueError("Pile type must be one of the following: \n - " + '\n - '.join(accepted_values))  
    
    @validator('material')
    def _material_must_equal(cls, v) -> None:
        accepted_values = ['Steel',]
        if v not in accepted_values:
            raise ValueError("Pile material must be one of the following: \n - " + '\n - '.join(accepted_values))  
    
    @validator('height_width_thickness')
    def _check_height_width_thickness(cls, v) -> None:
        for idx, sublist in enumerate(v):
            if len(sublist) != 3: 
                raise ValueError("The input parameter `height_width_thickness` for openpile.pile shall be a list of lists, the latter composed of 3 items, the first one for the height of the section, the second for the diameter of the section, and the third for the wall thickness. "+f"{len(sublist)} items were given in the pile section no. {idx+1}")  
            if sublist[2] >= sublist[1]/2:
                raise ValueError("The wall thickness cannot be larger than half the width of the pile")
            
if __name__ == "__main__":
    MP01 = Pile(type='Circular',
                material='Steel',
                top_elevation = 0,
                height_width_thickness=[  
                    [ 5, 10.0, 0.06],
                    [30, 10.0, 0.08],
                ],
            )

