Changelog
---------

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/),
and [PEP 440](https://www.python.org/dev/peps/pep-0440/).


## [0.2.0] - 2023-XX-XX (not releaset yet)
- new Pile constructor `openpile.construct.Pile.create_tubular` creating a 
  circular and hollow steel pile of constant cross section.
- new `openpile.construct.Pile` properties: `weight`, `volume`, 

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
