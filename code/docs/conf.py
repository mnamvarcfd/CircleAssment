# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the src directory to the Python path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Myocardial Blood Flow'
copyright = '2025, Your Name'
author = 'Your Name'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Core autodoc functionality
    'sphinx.ext.autosummary',  # Generate autosummary tables
    'sphinx.ext.napoleon',     # Support for NumPy/Google style docstrings
    'sphinx.ext.viewcode',     # Add source code links
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Napoleon settings for NumPy/Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# GitHub Pages configuration
html_baseurl = ''  # Set to repository name if deploying to username.github.io/repo-name/docs
html_context = {
    'display_github': True,
    'github_user': 'mnamvarcfd',
    'github_repo': 'CircleAssment',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}
