# Configuration file for the Sphinx documentation builder.

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -- Project information -----------------------------------------------------
project = 'GraphEm Rapids'
copyright = '2026, Alexander Kolpakov and Igor Rivin'
author = 'Alexander Kolpakov and Igor Rivin'
release = '0.3.0.dev0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
nitpicky = True
nitpick_ignore = [
    ('py:class', 'adjacency'),
    ('py:class', 'degrees'),
    ('py:class', 'logging.Logger'),
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_title = 'GraphEm RAPIDS documentation'
html_static_path = []

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

linkcheck_timeout = 20
linkcheck_retries = 2
