Changelog
---------

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/),
and [PEP 440](https://www.python.org/dev/peps/pep-0440/).

## [1.0.0] - 2024-06-xx

*OpenPile's kernel now accounts for axial soil springs when running a Winkler analysis. The axial soil springs are by default turned on (`base_axial` and `distributed_axial` arguments of `openpile.construct.Model`) and considered if an `axial_model` is fed to a `openpile.construct.Layer`*

The minimum python version to use with Openpile >= 1.0.0 is python 3.8.

### Added
- extensions of API-type lateral soil models are now considered in the string output when printing out the model or the entire soil profile, i.e. when running `print(openpile.construct.Layer)`
- the `PileMaterial` class in the brand new `openpile.materials` module is now used to determine the material of the structure. Such material can now be customised by users by creating a new instance of `openpile.materials.PileMaterial`.


### Modified

- `openpile.construct.Model.get_py_springs()` and other related methods to extract springs have been updated to the following naming style: `.get_distributed_lateral_springs()`, see documentation for more details. 
- The API clay model available until v0.7.1 has been decoupled to form the `API_clay` and the mofidied_Matlock models, new function and new model can be seen here: `Openpile.utils.py_curves.modified_Matlock()` and `Openpile.soilmodels.Modified_Matlock_clay`. Such decision was made to make it clearer on what model is used when running an analysis.
- functions found in `openpile.calculate` are now dependent on `openpile.construct.Pile` and `openpile.construct.SoilProfile` instead of `openpile.construct.Model`.

### Deleted
- The functions `openpile.analyze.simple_beam_analysis()` and `openpile.analyze.simple_winkler_analysis()` are now removed from the code.
- The method `openpile.construct.Pile.set_I()` method has been deleted, the second moment of area of a given pile segment can be changed by calling a Custom PileSection Geometry.
- The property `openpile.construct.Pile.E` cannot be overriden anymore, instead use the new feature where any material can be provided to the pile via `openpile.materials.PileMaterial`.
- The method `openpile.construct.Model.get_pointload()` is now deprecated and cannot be used anymore. Use the property `openpile.construct.Model.boundary_conditions` instead to list all boundary conditions.

## [0.7.1] - 2023-11-21

### Fixed

- maximum resistance in API_clay where Pmax was wrongly calculated based on depth instead of diameter, credits to Zhenghui Qiu for spotting this.


## [0.7.0] - 2023-11-12

### Added

- Added soil models:
  - `openpile.soilmodels.Bothkennar_clay` from the PISA joint-industry project

## [0.6.0] - 2023-10-23

### Added 
- added properties to `Pile` object: 
  - `openpile.construct.Pile.tip_area` and
  - `openpile.construct.Pile.tip_footprint`
- added soil springs: 
  - `openpile.utils.py_curves.custom_pisa_sand` and `openpile.utils.py_curves.custom_pisa_clay`
  - `openpile.utils.mt_curves.custom_pisa_sand` and `openpile.utils.mt_curves.custom_pisa_clay`
  - `openpile.utils.Hb_curves.custom_pisa_sand` and `openpile.utils.Hb_curves.custom_pisa_clay`
  - `openpile.utils.Mb_curves.custom_pisa_sand` and `openpile.utils.Mb_curves.custom_pisa_clay`
- added soil models:
  - `openpile.soilmodels.Bothkennar_clay` from the PISA joint-industry project
  - `openpile.soilmodels.Custom_pisa_sand` and `openpile.soilmodels.Custom_pisa_clay`, these models can be used to customise PISA formulations based on external sources, such as an FE model. 
- new functions to calculate Dunkirk Sand and Cowden Clay normalized parameters, these functions are in the module: `openpile.utils.multipliers` and are the following: `get_cowden_clay_(..)_norm_param()` and `get_dunkirk_sand_(..)_norm_param()`.

### Fixed 
- Stress stiffness matrix modified for not yielding negative values due to axial force component. 


## [0.5.0] - 2023-08-02  
- added soil spring `openpile.utils.py_curves.reese_weakrock`
- added soil model `openpile.soilmodels.Reese_weakrock`

## [0.4.0] - 2023-07-30 
- Updates in documentation
- new methods available for `openpile.analyze.Result` class:
  - new method `winkler()` replacing `simple_winkler_analysis()`, the use of the latter triggers a deprecation warning when used.
  - new method `beam()` replacing `simple_beam_analysis()`, the use of the latter triggers a deprecation warning when used.
  - new method `openpile.analyze.details()` that provides summary of an `AnalyzeResult` object.
- new methods available for `openpile.construct.Model` class:
  - `openpile.construct.Model.get_py_springs()`
  - `openpile.construct.Model.get_mt_springs()`
  - `openpile.construct.Model.get_Hb_spring()`
  - `openpile.construct.Model.get_Mb_spring()`
- new feature which allow user to enter a function in place of a float for springs multipliers when creating `SoilModel` objects. the function must take as input a PositiveFloat representing the depth below ground level, and as output the multiplier that shall be used by the soil spring for this depth.
- new `openpile.utils.multipliers` module that stores validated functions for use in multipliers in SoilModels objects.

## [0.3.3] - 2023-05-19 
- fix error in Dunkirk_sand rotational springs
- benchmarked Dunkirk sand soil model against literature from Burd et al (2020). 

## [0.3.2] - 2023-05-18 
- fixed bug in the kernel when applying base springs.
- clean up some part of the root directory
- benchmarked Cowden Clay soil model against literature from Byrne et al (2020).

## [0.3.1] - 2023-05-16 
- fixed bug in kernel that was amplifying the soil resistance and yielding unrealistic forces in the
  pile.

## [0.3.0] - 2023-05-02 
- new method to retrieve the p-y mobilisation of springs in Results via the `.py_mobilization()`
- update to the connectivity plot `openpile.construct.Model.plot()` that adds the soil profile to the plot 
  if a soil profile is fed to the model.
- tz- and Qz-curves following the well-known API standards are now included in `openpile.utils`
- updates to the documentation
- API p-y curves now officially unit tested

## [0.2.0] - 2023-04-24
- new Pile constructor `openpile.construct.Pile.create_tubular` creating a 
  circular and hollow steel pile of constant cross section.
- new properties for `openpile.construct.Pile`: `weight` and `volume`
- new `openpile.construct.Pile` method: `set_I()` to change the second moment of area of a given pile segment
- new `SoilProfile.plot()` method to visualize the soil profile
- API sand and API clay curves and models now accept `kind` instead of `Neq` arguments to differentiate between 
  static and cyclic curves
- create() methods in the construct module are now deprecated and should not be used anymore. Precisely, that is the 
  case for `openpile.  construct.Pile.create()` and `openpile.construct.Model.create()`. 

## [0.1.0] - 2023-04-10
### Added
- PISA sand and clay models (called Dunkirk_sand and Cowden_clay models)
- Rotational springs and base springs (shear and moment), see `utils` module
- New set of unit tests covering the `Construct` module, coverage is not 100%.

## [0.0.1] - 2023-03-31
### Notes
- first release of openpile with simple beam and simple winkler analysis with lateral springs

### Added
- `Construct` module with Pile, SoilProfile, Layer, and Model objects
- `utils` module with py curves
- `Analysis`modile with `simple_beam_analysis()` and `simple_winkler_analysis()`
- `Result` class that provides the user with plotting and Pandas Dataframe overview of results. 
