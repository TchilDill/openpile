Axial soil models
=================

The following axial models are included in openpile. 

* API sand
* API Clay


API model
---------

The axial  model calculates skin friction along the pile and end-bearing at pile tip.

OpenPile's use of this model is done by calling the following class in a layer:

* in coarse-grained materials, :py:class:`openpile.soilmodels.API_ax_sand`
* in fine-grained materials, :py:class:`openpile.soilmodels.API_ax_clay`

This soil model then provides soil springs as given by the function(s) below and depending on the type of material:

* :py:func:`openpile.utils.tz_curves.api_sand`
* :py:func:`openpile.utils.qz_curves.api_sand`
* :py:func:`openpile.utils.tz_curves.api_clay`
* :py:func:`openpile.utils.qz_curves.api_clay`


Distributed springs - t-z curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The backbone curve is computed via the piecewise formulation 
by [API2000]_ or through 
Kraft's formulation that captures small-strain 
characteristics of the soil [KrRK81]_ in the backbone curve.

.. note::
    Kraft's formulation is accessed by the user by stipulating the small-strain shear 
    stiffness of the soil, :math:`G_0`

**API Sand**


**API Clay**


Base spring - Q-z curve
^^^^^^^^^^^^^^^^^^^^^^^

The maximum resistance is calculated as follows:

* API clay: :math:`Q_{max} = 9 S_u`
  where :math:`S_u` is the clay undrained shear strength.
* API sand: :math:`Q_{max} = N_q \sigma^\prime_v`
  where :math:`\sigma^\prime_v` is the overburden effective stress and :math:`N_q` is 
  the end-bearing factor depending on the interface friction angle :math:`\delta = \phi - 5` given in [API2000]_.


The backbone curve is computed via the piecewise formulation 
by [API2000]_.
