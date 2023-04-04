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


def pile_sections_must_be(obj):
    if not isinstance(obj.pile_sections, dict):
        raise UserInputError("openpile.construct.Pile.pile_sections must be a dictionary")

    if obj.kind == "Circular":
        reference_list = ["diameter", "length", "wall thickness"]
        sorted_list = list(obj.pile_sections.keys())
        sorted_list.sort()
        if sorted_list != reference_list:
            raise UserInputError(
                "openpile.construct.Pile.pile_sections must have all and only the following keys: \n - "
                + "\n - ".join(reference_list)
            )
        for idx, (_, sublist) in enumerate(obj.pile_sections.items()):
            if not isinstance(sublist, list):
                raise UserInputError(
                    "openpile.construct.Pile.pile_sections must be a dictionary of lists"
                )
            for value in sublist:
                if not isinstance(value, (int, float)):
                    raise UserInputError(
                        "values in openpile.construct.Pile.pile_sections can only be numbers"
                    )

            if idx == 0:
                reference_length = len(sublist)
            else:
                if len(sublist) != reference_length:
                    raise ValueError(
                        "length of lists in openpile.construct.Pile.pile_sections must be the same"
                    )

        for i in range(reference_length):
            if obj.pile_sections["diameter"][i] / 2 < obj.pile_sections["wall thickness"][i]:
                raise ValueError(
                    "The wall thickness cannot be larger than half the diameter of the pile"
                )


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
        Mehs object crated from openpile
    """
    # rename vars
    restrained_dof = model.global_restrained
    loaded_dof = model.global_forces

    # count BC in [Translation over z-axis,Translation over y-axis,Rotation around x-axis]
    restrained_count_Rz = np.count_nonzero(restrained_dof["Rz"])
    restrained_count_Ty = np.count_nonzero(restrained_dof["Ty"])
    restrained_count_Tx = np.count_nonzero(restrained_dof["Tx"])
    restrained_count_total = restrained_count_Rz + restrained_count_Ty + restrained_count_Tx

    loaded_count_Rz = np.count_nonzero(loaded_dof["Mz [kNm]"])
    loaded_count_Ty = np.count_nonzero(loaded_dof["Py [kN]"])
    loaded_count_Tx = np.count_nonzero(loaded_dof["Px [kN]"])
    loaded_count_total = loaded_count_Rz + loaded_count_Ty + loaded_count_Tx

    if restrained_count_total == 0:
        raise InvalidBoundaryConditionsError("No support conditions are provided.")

    if loaded_count_total == 0:
        raise InvalidBoundaryConditionsError("No load conditions are provided.")

    if model.soil is None:
        # normally loaded beam
        if loaded_count_Tx > 0 and restrained_count_Tx > 0:
            # support in x axis is given and load over x is given --> correct BC
            pass
        elif loaded_count_Tx > 0 and restrained_count_Tx == 0:
            # support in x axis is given and load over x is given --> correct BC
            raise InvalidBoundaryConditionsError(
                "Support conditions in normal direction not provided."
            )

        # laterally-loaded beam
        if (restrained_count_Ty + restrained_count_Rz) >= 2 and loaded_count_Ty > 0:
            pass
            # support in y and z axes are given and load over y is given --> correct BC
        elif (restrained_count_Ty + restrained_count_Rz) >= 2 and loaded_count_Rz > 0:
            pass
            # support in y and z axes are given and load over z is given --> correct BC
        elif (loaded_count_Rz + loaded_count_Ty) == 0:
            # no trasnverse load --> no need to give any error
            pass
        else:
            raise InvalidBoundaryConditionsError("Support conditions against bending not provided.")
    else:
        raise InvalidBoundaryConditionsError("Soil in mesh not yet possible.")
