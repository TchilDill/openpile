
import pandas as pd
import numpy as np


def check_springs(arr):
            check_nan = np.isnan(arr).any()
            check_negative = (arr < 0).any()

            return check_nan or check_negative

def get_coordinates(pile, soil, x2mesh, coarseness) -> pd.DataFrame:
    # Primary discretisation over x-axis
    x = np.array([], dtype=np.float16)
    # add get pile relevant sections
    x = np.append(x, pile.data["Elevation [m]"].values)
    # add soil relevant layers and others
    if soil is not None:
        soil_elevations = np.array(
            [x.top for x in soil.layers] + [x.bottom for x in soil.layers],
            dtype=float,
        )
        if any(soil_elevations < pile.bottom_elevation):
            soil_elevations = np.append(pile.bottom_elevation, soil_elevations)
            soil_elevations = soil_elevations[soil_elevations >= pile.bottom_elevation]
        x = np.append(x, soil_elevations)
    # add user-defined elevation
    if x2mesh is None:
         x2mesh = []
    x = np.append(x, x2mesh)

    # get unique values and sort in reverse order
    x = np.unique(x)[::-1]

    # Secondary discretisation over x-axis depending on coarseness factor
    x_secondary = np.array([], dtype=np.float16)
    for i in range(len(x) - 1):
        spacing = x[i] - x[i + 1]
        new_spacing = spacing
        divider = 1
        while new_spacing > coarseness:
            divider += 1
            new_spacing = spacing / divider
        new_x = x[i] - (
            np.arange(start=1, stop=divider) * np.tile(new_spacing, (divider - 1))
        )
        x_secondary = np.append(x_secondary, new_x)

    # assemble x- coordinates
    x = np.append(x, x_secondary)
    x = np.unique(x)[::-1]

    # dummy y- coordinates
    y = np.zeros(shape=x.shape)

    # create dataframe coordinates
    nodes = pd.DataFrame(
        data={
            "x [m]": x,
            "y [m]": y,
        },
        dtype=float,
    ).round(3)
    nodes.index.name = "Node no."

    element = pd.DataFrame(
        data={
            "x_top [m]": x[:-1],
            "x_bottom [m]": x[1:],
            "y_top [m]": y[:-1],
            "y_bottom [m]": y[1:],
        },
        dtype=float,
    ).round(3)
    element.index.name = "Element no."

    return nodes, element

    # function doing the work

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

def get_all_properties(pile, soil, x2mesh, coarseness) -> pd.DataFrame:
    # creates mesh coordinates
    nodes_coordinates, element_coordinates = get_coordinates(pile, soil, x2mesh, coarseness)

    # creates element structural properties
    # merge Pile.data and coordinates
    element_properties = pd.merge_asof(
        left=element_coordinates.sort_values(by=["x_top [m]"]),
        right=pile.data.sort_values(by=["Elevation [m]"]),
        left_on="x_top [m]",
        right_on="Elevation [m]",
        direction="forward",
    ).sort_values(by=["x_top [m]"], ascending=False)
    # add young modulus to data
    element_properties["E [kPa]"] = pile.E
    # delete Elevation [m] column
    element_properties.drop("Elevation [m]", inplace=True, axis=1)
    # reset index
    element_properties.reset_index(inplace=True, drop=True)

    # create soil properties
    if soil is not None:
        soil_properties = pd.merge_asof(
            left=element_coordinates[["x_top [m]", "x_bottom [m]"]].sort_values(
                by=["x_top [m]"]
            ),
            right=get_soil_profile(soil).sort_values(by=["Top soil layer [m]"]),
            left_on="x_top [m]",
            right_on="Top soil layer [m]",
            direction="forward",
        ).sort_values(by=["x_top [m]"], ascending=False)
        # add elevation of element w.r.t. ground level
        soil_properties["xg_top [m]"] = (
            soil_properties["x_top [m]"] - soil.top_elevation
        )
        soil_properties["xg_bottom [m]"] = (
            soil_properties["x_bottom [m]"] - soil.top_elevation
        )
        # add vertical stress at top and bottom of each element
        condition_below_water_table = soil_properties["x_top [m]"] <= soil.water_line
        soil_properties["Unit Weight [kN/m3]"][condition_below_water_table] = (
            soil_properties["Unit Weight [kN/m3]"][condition_below_water_table] - 10.0
        )
        s = (
            soil_properties["x_top [m]"] - soil_properties["x_bottom [m]"]
        ) * soil_properties["Unit Weight [kN/m3]"]
        soil_properties["sigma_v top [kPa]"] = np.insert(
            s.cumsum().values[:-1],
            np.where(soil_properties["x_top [m]"].values == soil.top_elevation)[0],
            0.0,
        )
        soil_properties["sigma_v bottom [kPa]"] = s.cumsum()
        # reset index
        soil_properties.reset_index(inplace=True, drop=True)

    return soil_properties, element_properties, nodes_coordinates, element_coordinates
