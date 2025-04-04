import pandas as pd
import numpy as np


def validate_bc(bc_list, bc_cls):
    """
    helper function to validate boundary condition
    """
    err = False
    flag = []
    # check if bcs overlap
    bc_checklist = set([(bc.elevation, bc.x) for bc in bc_list if isinstance(bc, bc_cls)])
    if len(bc_checklist) > 1:
        if len(set([bc[0] for bc in bc_checklist])) != len([bc[0] for bc in bc_checklist]):
            flag.append("x")
            err = True
    bc_checklist = set([(bc.elevation, bc.y) for bc in bc_list if isinstance(bc, bc_cls)])
    if len(bc_checklist) > 1:
        if len(set([bc[0] for bc in bc_checklist])) != len([bc[0] for bc in bc_checklist]):
            flag.append("y")
            err = True
    bc_checklist = set([(bc.elevation, bc.z) for bc in bc_list if isinstance(bc, bc_cls)])
    if len(bc_checklist) > 1:
        if len(set([bc[0] for bc in bc_checklist])) != len([bc[0] for bc in bc_checklist]):
            flag.append("z")
            err = True
    if err:
        raise ValueError(
            f"Multiple boundary conditions ({bc_cls.__name__}) are given along {', and '.join(flag)}. axis and at same elevation."
        )


def apply_bc(nodes_elevations, zglobal, yglobal, xglobal, bc_list, bc_cls, output_text):
    """
    helper function to apply boundary condition
    """
    # apply boundary condition
    for bc in bc_list:
        if isinstance(bc, bc_cls):
            check = np.isclose(
                nodes_elevations, np.tile(bc.elevation, nodes_elevations.shape), atol=0.001
            )
            if any(check):
                # one node correspond, extract node
                node_idx = next((i for i, x in enumerate(check) if x == True), None)
                # apply loads at this node
                if bc.x:
                    xglobal[node_idx] = bc.x
                if bc.y:
                    yglobal[node_idx] = bc.y
                if bc.z:
                    zglobal[node_idx] = bc.z
            else:
                if bc.elevation > nodes_elevations[0] or bc.elevation < nodes_elevations[-1]:
                    print(
                        f"{output_text} not applied! The chosen elevation is outside the mesh. The {output_text} must be applied on the structure."
                    )
                else:
                    print(
                        f"{output_text} not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the Model."
                    )
    return zglobal, yglobal, xglobal


def parameter2elements(objects: list, key: callable, elem_top: list, elem_bottom: list):
    """converts a list of pile sections into a list of elements

    Objects must be either a list of pile sections (openpile.construct.PileSection) or a list of soil layers (openpile.construct.Layer)

    """

    elem_top = np.array(elem_top)
    elem_bottom = np.array(elem_bottom)
    # create a NaN array with same array as elem_z_top
    elem_param = np.full(elem_top.size, np.nan)

    for obj in objects:
        top_limit, bottom_limit = obj.top, obj.bottom
        idx = np.where((elem_top <= top_limit) & (elem_bottom >= bottom_limit))[0]
        elem_param[idx] = key(obj)

    return elem_param


def get_tip_sig_v_eff(
    tip_elevation: float,
    water_elevation: float,
    layers: list,  # List[openpile.construct.Layer]
    sig_v_mudline: float = 0,
    water_unit_weight: float = 10.0,
):
    """Calculates the effective vertical stress at the pile tip"""
    sig_v_tip = sig_v_mudline
    water_unit_weight = 10.0

    for layer in sorted(layers, key=lambda x: -x.top):
        buoyant_weight = layer.weight - water_unit_weight
        if tip_elevation <= layer.bottom:
            if water_elevation <= layer.bottom:
                w = layer.weight
            elif water_elevation < layer.top:
                w = (
                    layer.weight * (layer.top - water_elevation)
                    + buoyant_weight * (water_elevation - layer.bottom)
                ) / (layer.top - layer.bottom)
            else:
                w = buoyant_weight

            sig_v_tip += w * (layer.top - layer.bottom)

        elif tip_elevation < layer.top and water_elevation > layer.bottom:
            if water_elevation <= tip_elevation:
                w = layer.weight
            elif water_elevation < layer.top:
                w = (
                    layer.weight * (layer.top - water_elevation)
                    + buoyant_weight * (water_elevation - tip_elevation)
                ) / (layer.top - tip_elevation)
            else:
                w = buoyant_weight

            sig_v_tip += w * (layer.top - tip_elevation)

        return sig_v_tip


def check_springs(arr):
    check_nan = np.isnan(arr).any()
    check_negative = (arr < 0).any()

    return check_nan or check_negative


def get_coordinates(pile, soil, x2mesh, coarseness) -> pd.DataFrame:
    # Primary discretisation over z-axis
    z = np.array([], dtype=np.float16)
    # add get pile relevant sections
    z = np.append(z, pile.data["Elevation [m]"].values)
    # add soil relevant layers and others
    if soil is not None:
        soil_elevations = np.array(
            [x.top for x in soil.layers] + [x.bottom for x in soil.layers],
            dtype=float,
        )
        if any(soil_elevations < pile.bottom_elevation):
            soil_elevations = np.append(pile.bottom_elevation, soil_elevations)
            soil_elevations = soil_elevations[soil_elevations >= pile.bottom_elevation]
        z = np.append(z, soil_elevations)
    # add user-defined elevation
    if x2mesh is None:
        x2mesh = []
    z = np.append(z, x2mesh)

    # get unique values and sort in reverse order
    z = np.unique(z)[::-1]

    # Secondary discretisation over z-axis depending on coarseness factor
    z_secondary = np.array([], dtype=np.float16)
    for i in range(len(z) - 1):
        spacing = z[i] - z[i + 1]
        new_spacing = spacing
        divider = 1
        while new_spacing > coarseness:
            divider += 1
            new_spacing = spacing / divider
        new_z = z[i] - (np.arange(start=1, stop=divider) * np.tile(new_spacing, (divider - 1)))
        z_secondary = np.append(z_secondary, new_z)

    # assemble x- coordinates
    z = np.append(z, z_secondary)
    z = np.unique(z)[::-1]

    # dummy y- coordinates
    y = np.zeros(shape=z.shape)

    # create dataframe coordinates
    nodes = pd.DataFrame(
        data={
            "z [m]": z,
            "y [m]": y,
        },
        dtype=float,
    ).round(4)
    nodes.index.name = "Node no."

    element = pd.DataFrame(
        data={
            "z_top [m]": z[:-1],
            "z_bottom [m]": z[1:],
            "y_top [m]": y[:-1],
            "y_bottom [m]": y[1:],
        },
        dtype=float,
    ).round(4)
    element.index.name = "Element no."

    return nodes, element


def get_soil_profile(soil) -> pd.DataFrame:
    top_elevations = [x.top for x in soil.layers]
    bottom_elevations = [x.bottom for x in soil.layers]
    soil_weights = [x.weight for x in soil.layers]

    idx_sort = np.argsort(top_elevations)[::-1]

    top_elevations = [top_elevations[i] for i in idx_sort]
    soil_weights = [soil_weights[i] for i in idx_sort]
    bottom_elevations = [bottom_elevations[i] for i in idx_sort]

    # elevation in model w.r.t to x axis
    x = top_elevations

    return pd.DataFrame(
        data={"Top soil layer [m]": x, "Unit Weight [kN/m3]": soil_weights},
        dtype=np.float64,
    )


def get_soil_properties(pile, soil, x2mesh, coarseness):
    # dummy allocation
    soil_properties = None
    # create soil properties
    if soil is not None:
        element_coordinates = get_coordinates(pile, soil, x2mesh, coarseness)[1]
        soil_properties = pd.merge_asof(
            left=element_coordinates[["z_top [m]", "z_bottom [m]"]].sort_values(by=["z_top [m]"]),
            right=get_soil_profile(soil).sort_values(by=["Top soil layer [m]"]),
            left_on="z_top [m]",
            right_on="Top soil layer [m]",
            direction="forward",
        ).sort_values(by=["z_top [m]"], ascending=False)
        # add elevation of element w.r.t. ground level
        soil_properties["zg_top [m]"] = soil_properties["z_top [m]"] - soil.top_elevation
        soil_properties["zg_bottom [m]"] = soil_properties["z_bottom [m]"] - soil.top_elevation
        # add vertical stress at top and bottom of each element
        condition_below_water_table = soil_properties["z_top [m]"] <= soil.water_line
        soil_properties.loc[condition_below_water_table, "Unit Weight [kN/m3]"] = (
            soil_properties["Unit Weight [kN/m3]"][condition_below_water_table] - 10.0
        )
        s = (soil_properties["z_top [m]"] - soil_properties["z_bottom [m]"]) * soil_properties[
            "Unit Weight [kN/m3]"
        ]
        soil_properties["sigma_v top [kPa]"] = np.insert(
            s.cumsum().values[:-1],
            np.where(soil_properties["z_top [m]"].values == soil.top_elevation)[0],
            0.0,
        )
        soil_properties["sigma_v bottom [kPa]"] = s.cumsum()
        # reset index
        soil_properties.reset_index(inplace=True, drop=True)

        return soil_properties
