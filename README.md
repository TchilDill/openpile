# OpenPile

Open-source PILE software.

<!-- [![Python Support](https://img.shields.io/pypi/pyversions/openpile.svg)](https://pypi.org/project/openpile/) -->
[![License: LGPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
[![Downloads](https://static.pepy.tech/badge/openpile/month)](https://pepy.tech/project/openpile)

![Tests](https://github.com/TchilDill/openpile/actions/workflows/Test.yml/badge.svg) 
[![codecov](https://codecov.io/gh/TchilDill/Openpile/graph/badge.svg?token=HQERTZ09CV)](https://codecov.io/gh/TchilDill/Openpile)
[![Documentation Status](https://readthedocs.org/projects/openpile/badge/?version=latest)](https://openpile.readthedocs.io/en/latest/?badge=latest)


[![issues closed](https://img.shields.io/github/issues-closed/TchilDill/openpile)](https://github.com/TchilDill/openpile/issues)
[![PRs closed](https://img.shields.io/github/issues-pr-closed/TchilDill/openpile)](https://github.com/TchilDill/openpile/pulls)
[![GitHub last commit](https://img.shields.io/github/last-commit/TchilDill/openpile)](https://github.com/TchilDill/openpile/commits/master)
[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10116667.svg)](https://doi.org/10.5281/zenodo.10116667)

Please see [official documentation](https://openpile.readthedocs.io/en/latest/) for more information.

This package is an open source python library that provides users a tool box for geotechnical pile
calculations.

This package allows the user to:

* Use basic Euler-Bernoulli beam theory (as well as Timoshenko's variation) to compute 
  forces, deflection of a beam (or a pile) under adequate loading and 
  support conditions.
* Use Winkler's approach of a beam (or a pile) supported by linear or non-linear lateral and/or 
  rotational springs to compute forces and deflection of the pile based on recognised 
  soil models such as the widely used traditional API models for sand and clay or more recent models that stem from the PISA joint-industry project.

This library supports the following versions of python: 3.7-3.10.
Python 3.11 is not supported!

## Support

This package takes time and effort. You can support by buying me a coffee.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/TchillDill)

## Installation Instructions

**Prerequisites**:

* a version of python is installed on your machine (supported versions: 3.7-3.10)
* pip is installed in your environment.

```bash
pip install openpile
```

## Features

 * Python 3.7-3.10 support
 * Interactive structure perfectly suited for Jupyter Notebook 
 * Integrated data validation to prevent wrong inputs with pydantic
 * very fast computations fueled by the Numpy, Numba and Pandas libraries
 * Calculations
   * Beam calculation
   * Winkler model (i.e. beam supported by soil springs)
     * Load-driven analyses
     * Displacement-driven analyses 
   * Out-of-the-box computation of individual soil springs
   <!-- * Axial capacity calculations via integration -->
 * Friendly API interface with object-oriented approach
 * Matplotlib and Pandas libraries to facilitate post-processing of results. 

 ## Please share with the community

This library relies on community interactions. Please consider sharing a post about `OpenPile` and the value it can provide for researcher, academia and professionals.

[![GitHub Repo stars](https://img.shields.io/badge/share%20on-reddit-red?logo=reddit)](https://reddit.com/submit?url=https://github.com/TchilDill/openpile&title=openpile)
[![GitHub Repo stars](https://img.shields.io/badge/share%20on-twitter-03A9F4?logo=twitter)](https://twitter.com/share?url=https://github.com/TchilDill/openpile&t=openpile)
[![GitHub Repo stars](https://img.shields.io/badge/share%20on-linkedin-3949AB?logo=linkedin)](https://www.linkedin.com/shareArticle?url=https://github.com/TchilDill/openpile&title=openpile)