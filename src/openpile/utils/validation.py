"""
The validation module gathers all utilised functions/class/objects serving
the purpose of validation.
"""

class UserInputError(Exception):
    """Custom error for wrong inputs to openpile"""

# ----------------------------------------------------------------
#                              PILE
# ----------------------------------------------------------------

def pile_sections_must_be(obj):
    if not isinstance(obj.pile_sections, dict):
        raise UserInputError("openpile.construct.Pile.pile_sections must be a dictionary")
    
    if obj.type == 'Circular':
        reference_list = ['diameter', 'length', 'wall thickness']
        sorted_list = list(obj.pile_sections.keys())
        sorted_list.sort()
        if sorted_list != reference_list:
            raise UserInputError("openpile.construct.Pile.pile_sections must have all and only the following keys: \n - " + '\n - '.join(reference_list))
        for idx, (_, sublist) in enumerate(obj.pile_sections.items()):                
            if not isinstance(sublist,list):
                raise UserInputError("openpile.construct.Pile.pile_sections must be a dictionary of lists")
            for value in sublist:
                if not isinstance(value,(int,float)):
                    raise UserInputError("values in openpile.construct.Pile.pile_sections can only be numbers")
            
            if idx == 0:                     
                reference_length = len(sublist)
            else:
                if len(sublist) != reference_length:
                    raise ValueError("length of lists in openpile.construct.Pile.pile_sections must be the same")
    
        for i in range(reference_length):
            if obj.pile_sections['diameter'][i]/2 < obj.pile_sections['wall thickness'][i]: 
                raise ValueError("The wall thickness cannot be larger than half the diameter of the pile")


def param_must_be_type(parameter, parameter_name, parameter_type, type_name):
    if not isinstance(parameter, parameter_type):
        raise UserInputError(f"Wrong type for {parameter_name}. Please provide a {type_name}")


def str_must_be_one_of_those(param: str, param_name:str, accepted_values: list):
    param_must_be_type(param, param_name, str, 'string')
    if param not in accepted_values:
        raise UserInputError(f"Value in {param_name} type must be one of the following: \n - " + '\n - '.join(accepted_values))

def must_be_numbers_in_list(values, values_name):
    
    for value in values:
        if not isinstance(value,(float,int)):
            raise UserInputError(f"values in {values_name} can only be numbers")
