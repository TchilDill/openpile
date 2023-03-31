---
API
---

.. automodule:: openpile.construct
    :members:
    :exclude-members: __init__, PydanticConfig

.. automodule:: openpile.analyses
    :members: 
    :exclude-members: PydanticConfig, structural_forces_to_df, disp_to_df, Result

The `Result` class
==================

.. autoclass:: openpile.analyses.Result
    :members:
    :exclude-members: name, displacements, forces, Config, __init__

    Usage
    -----

    The `Result` class is created by any analyses from the :py:mod:`openpile.analyses` module.

    As such the user can use the following properties and/or methods for any return values of an analysis. 