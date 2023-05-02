---
API
---

.. automodule:: openpile.construct
    :members:
    :exclude-members: __init__, PydanticConfig, soil_and_pile_bottom_elevation_match, create, check_elevations, check_layers_elevations

.. automodule:: openpile.soilmodels
    :members: 
    :exclude-members: __init__, PydanticConfig, PydanticConfigFrozen, ConstitutiveModel, 
                      LateralModel, AxialModel, py_spring_fct, mt_spring_fct, Hb_spring_fct, 
                      Mb_spring_fct, spring_signature

.. automodule:: openpile.analyses
    :members: 
    :exclude-members: PydanticConfig, structural_forces_to_df, disp_to_df, Result


`utils` module
==============

.. automodule:: openpile.utils.py_curves
    :members:
    :exclude-members: random


.. automodule:: openpile.utils.mt_curves
    :members:
    :exclude-members: random


.. automodule:: openpile.utils.tz_curves
    :members:
    :exclude-members: random


.. automodule:: openpile.utils.qz_curves
    :members:
    :exclude-members: random


The `Result` class
==================

.. autoclass:: openpile.analyses.Result
    :members:
    :exclude-members: name, displacements, forces, Config, __init__

    Usage
    -----

    The `Result` class is created by any analyses from the :py:mod:`openpile.analyses` module.

    As such the user can use the following properties and/or methods for any return values of an analysis. 