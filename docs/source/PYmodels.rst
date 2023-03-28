--------------
PY soil models
--------------

The following PY models are included in openpile. 

* :ref:`API-sand` 
* :ref:`API-clay` 

The function :py:func:`openpile.utils.py_curves.[PY soil model]` generates the p-y curve for 
the relevant PY soil model.

Furthermore, the user can include the PY soil models discussed here in a soil profile's :py:class:`openpile.construct.Layer` 
by calling the class :py:class:`openpile.soilmodels.[PY soil model]` 

This part of the documentation discusses the theory and calculations. 
Please refer to the API or Usage sections for more practical information.

.. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
.. _API-sand:

API sand
========

Introduction 
------------

.. rubric:: References 

.. [MuOn84] Murchison, J.M., and O'Neill, M.,W., 1984. *Evaluation of p-y relationships in cohesionless soils.* In Proceedings of Analysis and Design of Pile Foundations, San Francisco, October 1-5, pp. 174-191. 
.. [DNV-RP-C212] DNVGL RP-C212, Recommended Practice, Geotechnical design.
.. [API2GEO-2011] API, April 2011. Geotechnical and Foundation Design Considerations, ANSI/API Recommended Practice 2GEO, First Edition, American Petroleum Institute, 103 p.


p- y formulation
----------------

The p-y formulation called API sand is based on the work conducted by
O'neill and Murchison (see [MuOn84]_).  

The API sand formulation is presented in both the API and DNVGL standards,
see, [DNV-RP-C212]_ and [API2GEO-2011]_


Granular soils are modelled by the sand p-y model as described 
with the following backbone formula:

.. math::

    p = A P_{max}  \tanh \left( \frac{k X}{A P_{max} }  y \right) 

where:

* :math:`A` is a factor to account for static of cyclic loading 
* :math:`P_{max}` is the ultimate resistance of the p-y curve 
* :math:`k` is the initial modulus of subgrade reaction
* :math:`X` is the depth below mudline of the p-y curve.

Factor A
--------

The factor A takes into account whether the curve represent 
static(also called monotonic) or cycling loading and is equal to:

.. math::

    A = 
    \begin{cases} 
    \begin{split}
    0.9 & \text{  for cyclic loading} \\ 
    \\
    3 - 0.8 \frac{X}{D} \ge 0.9 & \text{  for static loading}
        \end{split}
      \end{cases}

where:

* :math:`D` is the pile diameter. 
 
Initial subgrade reaction
-------------------------

The factor k is the initial modulus of subgrade reaction, which can be 
approximated by the following equation in which the output is given in kN/m³ 
and where :math:`\phi` is inserted in degrees: 

.. math::

    k = 
    \begin{cases} 
    \begin{split}
    \max \left(197.8 \cdot \phi^2 - 10232 \cdot \phi + 136820 , 5400 \right) & \text{ ,  below water table} \\ 
    \\
    \max \left(215.3 \cdot \phi^2 - 8232 \cdot \phi + 63657 , 5400 \right) & \text{ ,  above water table}
    \end{split}
    \end{cases}

The equation is a fit to the recommended values in [DNV-RP-C212]_.  The correspondence 
of this fit is illustrated in below figure:

![k_vs_phi.jpg](/_static/API_sand/k_vs_phi.jpg)


Ultimate resistance
-------------------
   
The coefficients C1, C2, and C3 depend on the friction angle :math:`\phi` as shown 
in below figure.

![C_coeffs_graph.jpg](/_static/API_sand/C_coeffs_graph.jpg)



.. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
.. _API-clay:

API clay
========


