import subprocess, os, sys

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# To import sphinx extensions we've put in the repository:
sys.path.insert(0, os.path.abspath('../sphinxext'))

# -- Project information -----------------------------------------------------

project = 'RandBLAS'
copyright = "2024, Riley Murray, Burlen Loring"
author = "Riley Murray, Burlen Loring"

# -- General configuration ---------------------------------------------------


if not os.path.exists('../build/html'):
    os.makedirs('../build/html')

subprocess.call('doxygen --version', shell=True)
subprocess.call('doxygen', shell=True)

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# pip install sphinxcontrib-bibtex breathe
extensions = [
    'breathe',
    'sphinx.ext.mathjax',
    'mathmacros'
]

bibtex_bibfiles = ['bibliography.bib']

# Configuring Breathe
breathe_projects = {
    "RandBLAS": "../build/xml"
}
breathe_default_project = "RandBLAS"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

table_styling_embed_css = False

html_theme_path = [sphinx_rtd_theme.get_html_theme_path(), "../themes"]
extensions += ['sphinx_rtd_theme','sphinx_design']
html_theme = 'randblas_rtd'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['../themes/randblas_rtd/static']

# html_css_files = [
#     'theme_overrides.css'  # overrides for wide tables in RTD theme
# ]

# Add custom JavaScript file
html_js_files = [
    'custom.js',
]

# numfig = True
math_numfig = True
math_eqref_format = "Eq. {number}"  # use a non-breaking-space unicode character.
numfig_secnum_depth = 1