"""
The validation module gathers all utilised functions/class/objects serving
the purpose of validation.
"""

import numpy as np


class UserInputError(Exception):
    """Custom error for wrong inputs to openpile"""


class InvalidBoundaryConditionsError(Exception):
    """Custom error for wrong inputs to boundary conditions"""


# ----------------------------------------------------------------
#                              PILE
# ----------------------------------------------------------------


def param_must_be_type(parameter, parameter_name, parameter_type, type_name):
    if not isinstance(parameter, parameter_type):
        raise UserInputError(f"Wrong type for {parameter_name}. Please provide a {type_name}")


def str_must_be_one_of_those(param: str, param_name: str, accepted_values: list):
    param_must_be_type(param, param_name, str, "string")
    if param not in accepted_values:
        raise UserInputError(
            f"Value in {param_name} type must be one of the following: \n - "
            + "\n - ".join(accepted_values)
        )


def must_be_numbers_in_list(values, values_name):
    for value in values:
        if not isinstance(value, (float, int)):
            raise UserInputError(f"values in {values_name} can only be numbers")


def check_boundary_conditions(model):
    """
    Check if boundary conditions are satisfactory to solve the system of equations.

    Parameters
    ----------
    model: openppile.construct.Model object
        object crated from openpile
    """
    # rename vars
    restrained_dof = model.global_restrained
    loaded_dof = model.global_forces

    # count BC in [Translation over z-axis,Translation over y-axis,Rotation around x-axis]
    restrained_count_Rx = np.count_nonzero(restrained_dof["Rx"])
    restrained_count_Ty = np.count_nonzero(restrained_dof["Ty"])
    restrained_count_Tz = np.count_nonzero(restrained_dof["Tz"])
    restrained_count_total = restrained_count_Rx + restrained_count_Ty + restrained_count_Tz

    loaded_count_Rx = np.count_nonzero(loaded_dof["Mx [kNm]"])
    loaded_count_Ty = np.count_nonzero(loaded_dof["Py [kN]"])
    loaded_count_Tz = np.count_nonzero(loaded_dof["Pz [kN]"])
    loaded_count_total = loaded_count_Rx + loaded_count_Ty + loaded_count_Tz

    if restrained_count_total == 0 and model.soil is None:
        raise InvalidBoundaryConditionsError("No support conditions are provided.")

    if loaded_count_total == 0:
        raise InvalidBoundaryConditionsError("No load conditions are provided.")

    if model.soil is None:
        # normally loaded beam
        if loaded_count_Tz > 0 and restrained_count_Tz > 0:
            # support in x axis is given and load over x is given --> correct BC
            pass
        elif loaded_count_Tz > 0 and restrained_count_Tz == 0:
            # support in x axis is given and load over x is given --> correct BC
            raise InvalidBoundaryConditionsError(
                "Support conditions in normal direction not provided."
            )

        # laterally-loaded beam
        if (restrained_count_Ty + restrained_count_Rx) >= 2 and loaded_count_Ty > 0:
            pass
            # support in y and z axes are given and load over y is given --> correct BC
        elif (restrained_count_Ty + restrained_count_Rx) >= 2 and loaded_count_Rx > 0:
            pass
            # support in y and z axes are given and load over z is given --> correct BC
        elif (loaded_count_Rx + loaded_count_Ty) == 0:
            # no trasnverse load --> no need to give any error
            pass
        else:
            raise InvalidBoundaryConditionsError("Support conditions against bending not provided.")
    else:
        raise InvalidBoundaryConditionsError("Soil in mesh not yet possible.")
