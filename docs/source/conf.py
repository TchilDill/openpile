# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'openpile'
copyright = '2023, Guillaume Melin'
author = 'Guillaume Melin'

import sys
from pathlib import Path
pypath = Path(__file__).parents[2]
#add path
sys.path.insert(0, str(Path(pypath / 'src')))
from openpile import __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
                'sphinx.ext.autodoc',
                "sphinx.ext.githubpages",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for LaTeX output ------------------------------------------------
latex_engine = 'pdflatex'
numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']