Axial soil models
=================

The following axial models are included in openpile. 

* API sand
* API Clay


API model
---------

The axial model calculates skin friction along the pile and end-bearing at pile tip.

OpenPile's use of this model is done by calling the following class in a layer:

* in coarse-grained materials, :py:class:`openpile.soilmodels.API_sand_axial`
* in fine-grained materials, :py:class:`openpile.soilmodels.API_clay_axial`

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

For sand, the API guidelines provide methods to estimate the resistance offered by sandy soils along the pile. 
These springs are defined based on the following considerations:

1. **Unit Skin Friction** :math:`f_s`: This is the frictional resistance per unit area along the pile shaft. It depends on the effective overburden pressure and the soil-pile interface properties.

     .. math::
     
       f_s = \sigma_v^\prime \cdot K \cdot tan(\delta) < f_{s,\text{max}}
     
     where:
     
     - \sigma_v^\prime is the effective vertical stress at the depth considered.
     - K is the coefficient of horizontal earth pressure (typically ranges from 0.4 to 1.0 for sands).
     - \delta is the angle of friction between the pile and the sand, often taken as a fraction of the soil's internal friction angle :math:`\varphi` (usually :math:`\delta = 0.7 \varphi` to :math:`\varphi`).
     - Limit Skin Friction :math:`f_{s,\text{max}}` is the maximum unit skin friction that can be mobilized. It is typically given by empirical correlations or laboratory tests. The following is assumed in OpenPile:

      .. list-table:: Correlation between interface friction angle and shaft friction 
        :header-rows: 0

        * - :math:`\delta` [degrees]
          - 15
          - 20
          - 25
          - 30
          - 35
        * - :math:`f_{s,\texttt{max}}` [kPa]
          - 47.8
          - 67
          - 81.3
          - 95.7
          - 114.8

  2. A backbone curve computed via the piecewise formulation seen in [API2000]_.


**API Clay**

For clay, the API guidelines describe the axial soil springs in a manner that accounts for the undrained shear strength of the clay. 
The springs are characterized as follows:

1. **Unit Skin Friction** :math:`f_s`: For clay, this is based on the undrained shear strength :math:`S_u` of the soil and a factor that accounts for the adhesion between the clay and the pile.

     .. math::
     
        f_s = \alpha \cdot S_u < f_{s,\text{max}}
     
     where:
     
     - :math:`\alpha` is the adhesion factor, which depends on the type of clay and the pile material. 
       It typically ranges from 0.5 to 1.0 for soft clays and 0.3 to 0.6 for stiff clays.
       As per the API guidelines, this adhesion factor can be calculated as:
     - :math:`S_u` is the undrained shear strength of the clay.
     - Limit Skin Friction :math:`f_{s,\text{max}}` is the maximum unit skin friction for clay, 
       which can be directly related to the undrained shear strength and the adhesion factor. 
       In general, the limit skin friction is set to the undrained shear strength.

  2. A backbone curve computed via the piecewise formulation seen in [API2000]_.


Base spring - Q-z curve
^^^^^^^^^^^^^^^^^^^^^^^

The maximum resistance is calculated as follows:

* API clay: :math:`Q_{max} = 9 S_u`
  where :math:`S_u` is the clay undrained shear strength.
* API sand: :math:`Q_{max} = N_q \sigma^\prime_v`
  where :math:`\sigma_v^\prime` is the overburden effective stress and :math:`N_q` is 
  the end-bearing factor depending on the interface friction angle :math:`\varphi`, see below table.

  +---------------------------+------+------+------+------+-------+
  | :math:`\varphi` [degrees] | 15.0 | 20.0 | 25.0 | 30.0 | 35.0  |
  +---------------------------+------+------+------+------+-------+
  | :math:`N_q` [kPa]         | 8.0  | 12.0 | 20.0 | 40.0 | 50.0  |
  +---------------------------+------+------+------+------+-------+
  | :math:`Q_{max}` [kPa]     | 1900 | 2900 | 4800 | 9600 | 12000 |
  +---------------------------+------+------+------+------+-------+


The backbone curve is computed via the piecewise formulation 
by [API2000]_.
