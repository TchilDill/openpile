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


def bearingcapacity(
    model,
) -> CalculateResult:

    def pile_element_surface(model):
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
        perimeter_outside = model.element_properties['Diameter [m]'].values * m.pi
        perimeter_inside = (model.element_properties['Diameter [m]'].values -2*model.element_properties['Wall thickness [m]']) * m.pi
        L = (model.element_properties['x_top [m]'].values - model.element_properties['x_bottom [m]'].values)

        return perimeter_outside*L, perimeter_inside*L

    def pile_inside_volume(model):
        area_inside = (model.element_properties['Diameter [m]'].values -2*model.element_properties['Wall thickness [m]'])**2 * m.pi/4
        L = (model.element_properties['x_top [m]'].values - model.element_properties['x_bottom [m]'].values)
        
        return area_inside*L


    # DEFINE MAIN ELEMEMT PROPERTIES, perimeters, area, inside volume

    # pile element surfaces
    So, Si = pile_element_surface(model)

    # soil volume 
    Vi = pile_inside_volume(model)

    # get vertical effective stress
    sigveff = 0.5*(model.soil_properties["sigma_v top [kPa]"]+ model.soil_properties["sigma_v bottom [kPa]"])
    elevation = 0.5*(model.soil_properties["x_top [m]"]+model.soil_properties[ "x_bottom [m]"])

    # depth from ground
    depth_from_ground = (0.5*(model.soil_properties["xg_top [m]"]+model.soil_properties["xg_bottom [m]"])).abs()

    elements_where_soil_meets_pile = model.soil_properties.loc[
                    (model.soil_properties["x_top [m]"] <= model.soil.top_elevation)
                    & (model.soil_properties["x_bottom [m]"] >= model.soil.top_elevation)
                ].index
    
    # unit shaft resistance for each element where we have soil and pile
    element_fs = np.zeros(model.element_number)
    # unit shaft resistance for each element where we have soil and pile
    element_Q = np.zeros(model.element_number)

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
                # vertical effective stress
                sig_v = model.soil_properties[
                    ["sigma_v top [kPa]", "sigma_v bottom [kPa]"]
                ].iloc[i].mean()
                # elevation
                elevation = model.soil_properties[["x_top [m]", "x_bottom [m]"]].iloc[i].mean()
                # depth from ground
                depth_from_ground = (model.soil_properties[["xg_top [m]", "xg_bottom [m]"]].iloc[i]).abs().mean()

                # Calculate unit shaft resistance
                element_fs[i] = layer.axial_model.unit_shaft_friction(
                    sig=sig_v, 
                    depth_from_top_of_layer=depth_from_ground, 
                    layer_height=(layer.top - layer.bottom),
                )

                # Calculate unit tip resistance
                element_Q[i] = layer.axial_model.unit_tip_resistance(
                    sig=sig_v, 
                    depth_from_top_of_layer=depth_from_ground, 
                    layer_height=(layer.top - layer.bottom),
                )