"""
`calculate` module
==================

The `calculate` module is used to run various functions outside the scope of the `analyze` module. 

Every function from this module returns an `openpile.compute.CalculateResult` object. 

"""

import pandas as pd
import numpy as np
import math as m


class CalculateResult:
    _values: tuple


def _pile_element_surface(model):
    """calculates outer and inner surface of pile elements.

    Parameters
    ----------
    model : openpile.construct.Model

    Returns
    -------
    np.ndarray
        outside surface
    np.ndarray
        inside surface
    """
    perimeter_outside = model.element_properties["Diameter [m]"].values * m.pi
    perimeter_inside = (
        model.element_properties["Diameter [m]"].values
        - 2 * model.element_properties["Wall thickness [m]"]
    ) * m.pi
    L = (
        model.element_properties["x_top [m]"].values
        - model.element_properties["x_bottom [m]"].values
    )

    return perimeter_outside * L, perimeter_inside * L


def _pile_inside_volume(model):
    area_inside = (
        (
            model.element_properties["Diameter [m]"].values
            - 2 * model.element_properties["Wall thickness [m]"]
        )
        ** 2
        * m.pi
        / 4
    )
    L = (
        model.element_properties["x_top [m]"].values
        - model.element_properties["x_bottom [m]"].values
    )

    return area_inside * L


def effective_pile_weight(model):
    """Calculates the pile weight in the model with consideration of buoyancy

    Parameters
    ----------
    model : openpile.construct.Model
        OpenPile Model object

    Returns
    -------
    float
        pile weight in kN

    Raises
    ------
    Exception
        if soil profile does not exist

    See also
    --------
    `openpile.construct.Pile.weight`
    """

    if model.soil is not None:
        submerged_element = model.element_properties["x_bottom [m]"].values < model.soil.water_line

        L = (
            model.element_properties["x_top [m]"].values
            - model.element_properties["x_bottom [m]"].values
        )
        V = L * model.element_properties["Area [m2]"].values
        W = np.zeros(shape=V.shape)
        W[submerged_element] = V[submerged_element] * (model.pile._uw - 10)
        W[~submerged_element] = V[~submerged_element] * (model.pile._uw)

        return W.sum()

    else:
        raise Exception(
            "Model must be linked to a soil profile, use `openpile.construct.Pile.weight instead.`"
        )


def bearingcapacity(model, kind):
    if kind == "API":
        if not all([x.bearingcapacity_signature == "API" for x in model.soil.layer.axial_model]):
            ValueError("All axial models must be API compliant.")

    plugged = shaft_resistance(model, outer_shaft=True, inner_shaft=False)
    +unit_end_bearing(model) * model.pile.tip_area
    -entrapped_soil_weight(model)
    # - model.

    # unplugged =

    # isplugged =

    return {
        "Compression": {
            "shaft_resistance": 0,
        }
    }


def unit_end_bearing(
    model,
) -> float:

    for layer in model.soil.layers:
        if layer.axial_model is None:
            pass
        else:
            # check if pile tip is within layer
            if (
                layer.top >= model.pile.bottom_elevation
                and layer.bottom <= model.pile.bottom_elevation
            ):
                # vertical effective stress at pile tip
                sig_v_tip = (model.soil_properties["sigma_v bottom [kPa]"].iloc[-1],)

                # Calculate unit tip resistance with effective area
                return (
                    layer.axial_model.unit_tip_resistance(
                        sig=sig_v_tip,
                        depth_from_top_of_layer=(
                            model.soil.top_elevation - model.soil.bottom_elevation
                        ),
                        layer_height=(layer.top - layer.bottom),
                    )
                    * layer.axial_model.Q_multiplier
                )

    return 0.0


def entrapped_soil_weight(model) -> float:
    """calculates total weight of soil inside the pile. (Unit: kN)

    Parameters
    ----------
    model : openpile.construct.Model
        OpenPile Model to assess

    Returns
    -------
    float
        value of entrapped total  weight of soil inside the pile in unit:kN
    """
    # weight water in kN/m3
    uw_water = 10

    # soil volume
    Vi = _pile_inside_volume(model)
    # element mid-point elevation
    elevation = 0.5 * (model.soil_properties["x_top [m]"] + model.soil_properties["x_bottom [m]"])
    # soil weight for each element where we have soil and pile
    element_sw = np.zeros(model.element_number)

    for layer in model.soil.layers:
        elements_for_layer = model.soil_properties.loc[
            (model.soil_properties["x_top [m]"] <= layer.top)
            & (model.soil_properties["x_bottom [m]"] >= layer.bottom)
        ].index

        # Set local layer parameters for each element of the layer
        for i in elements_for_layer:
            # Calculate inner soil weight
            element_sw[i] = (
                layer.weight * Vi[i]
                if elevation[i] <= model.soil.water_line
                else (layer.weight - uw_water) * Vi[i]
            )

    return element_sw.sum()


def shaft_resistance(
    model,
) -> float:
    """Calculates shaft resistance of the pile based on the axial models assigned to the SoilProfile layers. (Unit: kN)

    Parameters
    ----------
    model : openpile.construct.Model
        OpenPile Model to assess
    outer_shaft : bool, optional
        outer shaft resistance toggle switch, by default True
    inner_shaft : bool, optional
        inner shaft resistance toggle switch, by default True

    Returns
    -------
    float
        value of shaft resistance in unit:kN
    """
    # pile element surfaces
    So, Si = _pile_element_surface(model)

    # get vertical effective stress
    sigveff = 0.5 * (
        model.soil_properties["sigma_v top [kPa]"] + model.soil_properties["sigma_v bottom [kPa]"]
    )

    # depth from ground
    depth_from_ground = (
        0.5 * (model.soil_properties["xg_top [m]"] + model.soil_properties["xg_bottom [m]"])
    ).abs()

    # shaft resistance for each element where we have soil and pile
    element_fs = np.zeros((2, model.element_number))

    for layer in model.soil.layers:
        elements_for_layer = model.soil_properties.loc[
            (model.soil_properties["x_top [m]"] <= layer.top)
            & (model.soil_properties["x_bottom [m]"] >= layer.bottom)
        ].index

        if layer.axial_model is None:
            pass
        else:
            # Set local layer parameters for each element of the layer
            for i in elements_for_layer:
                # depth from ground
                depth_from_ground = (
                    (model.soil_properties[["xg_top [m]", "xg_bottom [m]"]].iloc[i]).abs().mean()
                )

                # Calculate outer shaft resistance
                element_fs[0, i] = (
                    layer.axial_model.unit_shaft_friction(
                        sig=sigveff[i],
                        depth_from_top_of_layer=depth_from_ground,
                        layer_height=(layer.top - layer.bottom),
                    )
                    * layer.axial_model.unit_shaft_signature(So[i], Si[i])["out"]
                    * So[i]
                    * layer.axial_model.t_multiplier
                )
                # Calculate inner shaft resistance
                element_fs[1, i] = (
                    layer.axial_model.unit_shaft_friction(
                        sig=sigveff[i],
                        depth_from_top_of_layer=depth_from_ground,
                        layer_height=(layer.top - layer.bottom),
                    )
                    * layer.axial_model.unit_shaft_signature(So[i], Si[i])["in"]
                    * Si[i]
                    * layer.axial_model.t_multiplier
                )

    if outer_shaft is False:
        element_fs[0, :] = 0.0
    if inner_shaft is False:
        element_fs[1, :] = 0.0

    return element_fs.sum()
