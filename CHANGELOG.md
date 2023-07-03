Changelog
---------

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/),
and [PEP 440](https://www.python.org/dev/peps/pep-0440/).


## [0.4.0] - 2023-XX-XX (not released yet) 
- Update in documentation
- fix bug in `openpile.analyses.simple_beam_analysis()`
- new methods available for `openpile.analyses.Result` class
- new methods available for `openpile.construct.Model` class:
  - `openpile.construct.Model.get_py_springs()`
  - `openpile.construct.Model.get_mt_springs()`
  - `openpile.construct.Model.get_Hb_spring()`
  - `openpile.construct.Model.get_Mb_spring()`

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
