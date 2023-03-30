# openpile

Geotechnical super-toolbox for pile-related calculations.

<!-- [![Python Support](https://img.shields.io/pypi/pyversions/openpile.svg)](https://pypi.org/project/openpile/) -->
[![License: LGPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)

[![Visual Studio](https://badgen.net/badge/icon/visualstudio?icon=visualstudio&label)](https://visualstudio.microsoft.com)

![Tests](https://github.com/TchilDill/openpile/actions/workflows/Test.yml/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/openpile/badge/?version=latest)](https://openpile.readthedocs.io/en/latest/?badge=latest)
[![Github All Releases](https://img.shields.io/github/downloads/TchilDill/openpile/total.svg)]()

[![issues closed](https://img.shields.io/github/issues-closed/TchilDill/openpile)](https://github.com/TchilDill/openpile/issues)
[![PRs closed](https://img.shields.io/github/issues-pr-closed/TchilDill/openpile)](https://github.com/TchilDill/openpile/pulls)
[![GitHub last commit](https://img.shields.io/github/last-commit/TchilDill/openpile)](https://github.com/TchilDill/openpile/commits/master)
[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)


Please see [official documentation](https://openpile.readthedocs.io/en/latest/) for more information.

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
 * Calculations
   * Beam calculation
   * Winkler model (i.e. beam supported by soil springs)
     * Load-driven analyses
     * Displacement-driven analyses 
   * Out-of-the-box computation of individual soil springs
   <!-- * Axial capacity calculations via integration -->
 * Friendly API interface with  object-oriented approach
 * Fully integrated output with python environment with Matplotlib and Pandas libraries. 