
Introduction to soil models
===========================

The function :py:func:`openpile.utils.py_curves.[<py spring>]` generates the lateral springs (or p-y curve/spring) 
for the relevant soil model. Similar modules exist for other types of springs:

* rotational springs (found in :py:mod:`openpile.utils.mt_curves`)
* axial springs (found in :py:mod:`openpile.utils.tz_curves`)
* base lateral spring (found in :py:mod:`openpile.utils.Hb_curves`)
* base rotational spring (found in :py:mod:`openpile.utils.Mb_curves`)
* base axial spring (found in :py:mod:`openpile.utils.qz_curves`)


Furthermore, the user can include the those springs discussed in a soil profile's :py:class:`openpile.construct.Layer` 
by calling a soil mode, found in :py:class:`openpile.soilmodels.[<soil model>]` 

The rest of the below documentation discusses the soil models available to the user, 
the theory behind and calculations. 
Please refer to the :ref:`ApplicationProgrammingInterface` for more details and :ref:`usage` for more practical information.


.. rubric:: References 

.. [MuOn83] Murchison, J.M., and O'Neill, M.,W., 1983. *An Evaluation of p-y Relationships 
    in Sands.* Rserach Report No. GT.DF02-83, Department of Civil Engineering, 
    University of Houston, Houston, Texas, May, 1983.
.. [MuOn84] Murchison, J.M., and O'Neill, M.,W., 1984. *Evaluation of p-y relationships 
    in cohesionless soils.* In Proceedings of Analysis and Design of Pile Foundations, 
    San Francisco, October 1-5, pp. 174-191.
.. [DNV-RP-C212] DNVGL RP-C212. *Recommended Practice, Geotechnical design*.
    Edition 2019-09 - Amended 2021-09.
.. [API2000] API, December 2000. *Recommended Practice for Planning, Designing, and Constructing 
    Fixed Offshore Platforms - Working Stress Design (RP 2A-WSD)*, Twenty-First Edition.
.. [Matl70] Matlock, H. (1970). *Correlations for Design of Laterally Loaded Piles in Soft Clay*. 
    Offshore Technology Conference Proceedings, Paper No. OTC 1204, Houston, Texas. 
.. [BaCA06] Battacharya,  S.,  Carrington,  T.  M.  and  Aldridge,  T.  R.  (2006),  
    *Design  of  FPSO  Piles  against  Storm  Loading*. Proceedings Annual Offshore Technology 
    Conference, OTC17861, Houston, Texas, May, 2006.
.. [KrRK81] Kraft, L.M., Ray, R.P., and Kagawa, T. (1981). *Theoretical t-z curves*. 
    Journal of the Geotechnical Engineering Division, ASCE, Vol. 107, No. GT11, pp. 1543-1561.
.. [BHBG20] Byrne, B. W., Houlsby, G. T., Burd, H. J., Gavin, K. G., Igoe, D. J. P., 
    Jardine, R. J., Martin, C. M., McAdam, R. A., Potts, D. M., Taborda, D. M. G. & Zdravkovic ́, L. (2020). 
    PISA design model for monopiles for offshore wind turbines: application 
    to a stiff glacial clay till. Géotechnique, https://doi.org/10.1680/ jgeot.18.P.255.
.. [BTZA20] Burd, H. J., Taborda, D. M. G., Zdravkovic ́, L., Abadie, C. N., Byrne, B. W., 
    Houlsby, G. T., Gavin, K. G., Igoe, D. J. P., Jardine, R. J., Martin, C. M., McAdam, R. A., 
    Pedro, A. M. G. & Potts, D. M. (2020). PISA design model for monopiles for offshore wind 
    turbines: application to a marine sand. Géotechnique, https://doi.org/10.1680/jgeot.18.P.277.
.. [BABH20] Burd, H. J., Abadie, C. N., Byrne, B. W., Houlsby, G. T., Martin, C. M., McAdam, R. A., 
    Jardine, R.J., Pedro, A.M., Potts, D.M., Taborda, D.M., Zdravković, L., and Andrade, M.P. 
    (2020). Application of the PISA Design Model to Monopiles Embedded in Layered Soils. 
    Géotechnique 70(11): 1-55. https://doi.org/10.1680/jgeot.20.PISA.009
.. [Rees97] Reese, L.C. (1997), Analysis of Laterally Loaded Piles in Weak Rock, Journal of Geotechnical
    and Geoenvironmental Engineering, ASCE, vol. 123 (11) Nov., ASCE, pp. 1010-1017.
.. [SøIA10] Sorensen, S.P.H. & Ibsen, L.B. & Augustesen, A.H. (2010), Effects of diameter on 
    initial stiffness of p-y curves for large-diameter piles in sand, Numerical Methods in 
    Geotechnical Engineering, CRC Press, pp. 907-912.
.. [Søre12] Sorensen, S.P.H. (2012), Soil-Structure Interaction For Nonslender, Large-Diameter 
    Offshore Monopiles. PhD Thesis, Department of Civil Engineering, Aalborg University, Denmark.
