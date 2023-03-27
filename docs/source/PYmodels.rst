--------------
PY soil models
--------------

The following PY models are included in openpile. 

* :ref:`API-sand` 
* :ref:`API-clay` 


.. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
.. _API-sand:

API sand
========

Introduction 
------------

This class generates the p-y curve for API sand with
specificed input given in the following sections.

References
----------

- Ref. /1/. Murchison, J.M., and O'Neill, M.,W., 1984. *Evaluation of p-y relationships in cohesionless soils.* In Proceedings of Analysis and Design of Pile Foundations, San Francisco, October 1-5, pp. 174-191. 
- Ref. /2/. DNVGL RP-C212, Recommended Practice, Geotechnical design.
- Ref. /3/. API, April 2011. Geotechnical and Foundation Design Considerations, ANSI/API Recommended Practice 2GEO, First Edition, American Petroleum Institute, 103 p.

$p$-$y$ formulation
-------------------

General
^^^^^^^

The $p$-$y$ formulation called API sand is based on the work conducted by
O'neill and Murchison (see Ref. /1/).  

The API sand formulation is presented in both the API and DNVGL standards,
see, Ref. /2/ and Ref. /3/.

Equations
^^^^^^^^^

Granular soils are modelled by the sand p-y model as described with the following formula:

![formula01.jpg](_static/API_sand/formula01.jpg)

the coefficients C1, C2, and C3 depend on the friction angle $\phi\prime$ as shown 
in below figure.

![C_coeffs_graph.jpg](_static/API_sand/C_coeffs_graph.jpg)

The factor A takes into account static and cycling loading and is equal to:

![A_factor_def.jpg](_static/API_sand/A_factor_def.jpg)

The factor k is the initial modulus of subgrade reaction, which can be 
approximated by the following equation in which the output is given in kN/m³ 
and where $\phi\prime$ is inserted in degrees: 

$k = \max \left(197.8 \cdot \phi\prime^2 - 10232 \cdot \phi\prime + 136820 , 5400 \right)$

The equation is a fit to the recommended values in Ref. /2/.  The correspondence 
of this fit is illustrated in below figure:

![k_vs_phi.jpg](_static/API_sand/k_vs_phi.jpg)



.. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
.. _API-clay:

API clay
========