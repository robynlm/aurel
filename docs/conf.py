import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))
import aurel.core as core

# Configuration file for the Sphinx documentation builder.

project = 'aurel'
copyright = '2025, Robyn L. Munoz'
author = 'Robyn L. Munoz'
release = 'May 2025'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_math_dollar', 
    'sphinx.ext.mathjax',
]

#napoleon_use_rtype = False
mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}
rst_prolog = """
.. role:: raw-latex(raw)
   :format: latex html
"""

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
myst_enable_extensions = ["amsmath"]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

autodoc_typehints = "none"

def skip_member_handler(app, what, name, obj, skip, options):
    # List of functions to skip
    functions_to_skip = ["func"]+list(core.descriptions.keys())
    # Check if the function belongs to the specified modules and should be skipped
    if ((name in functions_to_skip) 
        and (getattr(obj, "__module__", None) in ["aurel.core", "aurel.coreanalytic"])):
        print(f"Skipping: {name} in module {obj.__module__}")
        return True  # Skip this function

    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member_handler)

import subprocess

def generate_rst_files():
    subprocess.run(["python3", "source/generate_rst.py"])

generate_rst_files()