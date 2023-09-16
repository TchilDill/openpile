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

    # # pile width
    # pile_width = model.element_properties["Diameter [m]"]
    # # wall thickness
    # wall_thickness = model.element_properties["Wall thickness [m]"]

    # for layer in model.soil.layers:
    #     elements_for_layer = model.soil_properties.loc[
    #                 (model.soil_properties["x_top [m]"] <= layer.top)
    #                 & (model.soil_properties["x_bottom [m]"] >= layer.bottom)
    #             ].index

    #     if layer.axial_model is None:
    #         pass
    #     else:
    #         # Set local layer parameters for each element of the layer
    #         for i in elements_for_layer:
    #             pass
