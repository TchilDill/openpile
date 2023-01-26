# ----------------------------------------------------------------
#                              PILE
# ----------------------------------------------------------------

class UserInputError(Exception):
    """Custom error for wrong inputs to openpile"""
    pass

def pile_topelevation_must_be(obj):
    if not isinstance(obj.top_elevation, (int,float)):
        raise UserInputError("'top_elevation must be a number'")    

def pile_type_must_be(obj):
    if not isinstance(obj.type, str):
        raise UserInputError("'type' must be a string")
    
    accepted_values = ['Circular',]
    if obj.type not in accepted_values:
        raise UserInputError("Pile type must be one of the following: \n - " + '\n - '.join(accepted_values))

def pile_material_must_be(obj):
    if not isinstance(obj.material, str):
        raise UserInputError("'material' must be a string")    

    accepted_values = ['Steel',]
    if obj.material not in accepted_values:
        raise UserInputError("Pile material must be one of the following: \n - " + '\n - '.join(accepted_values))

def pile_pile_sections_must_be(obj):
    if not isinstance(obj.pile_sections, dict):
        raise UserInputError("'pile_sections' must be a dictionary")
    
    if obj.type == 'Circular':
        reference_list = ['diameter', 'length', 'wall thickness']
        sorted_list = list(obj.pile_sections.keys())
        sorted_list.sort()
        if sorted_list != reference_list:
            raise UserInputError("pile_sections must have all and only the following keys: \n - " + '\n - '.join(reference_list))
        for idx, (_, sublist) in enumerate(obj.pile_sections.items()):                
            if not isinstance(sublist,list):
                raise UserInputError("'pile_section' must be a dictionary of lists")
            for value in sublist:
                if not isinstance(value,(int,float)):
                    raise UserInputError("'values in dictionary can only be numbers'")
            
            if idx == 0:                     
                reference_length = len(sublist)
            else:
                if len(sublist) != reference_length:
                        raise ValueError("length of lists in pile_sections must be the same")
    
        for i in range(reference_length):
            if obj.pile_sections['diameter'][i]/2 < obj.pile_sections['wall thickness'][i]: 
                raise ValueError("The wall thickness cannot be larger than half the diameter of the pile")
            
            
# ----------------------------------------------------------------
#                              MESH
# ----------------------------------------------------------------

