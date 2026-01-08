import os
import sys
import re
import subprocess

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))
import aurel.core as core
import aurel.time as time_module

# Generate list of estimation functions for documentation
est_functions_keys = ', '.join([f"``{key}``" for key in time_module.est_functions.keys()])

# Configuration file for the Sphinx documentation builder.

project = 'aurel'
copyright = '2025, Robyn L. Munoz'
author = 'Robyn L. Munoz'
release = 'May 2025'

# Also update pyproject.toml [project.optional-dependencies.docs]
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_math_dollar', 
    'sphinx.ext.mathjax',
    'myst_nb',
    'sphinx_design'
]

# notebooks, rendering latex maths
nb_execution_mode = "off"

# MyST-NB configuration for LaTeX rendering
myst_enable_extensions = ["amsmath", "dollarmath"]
myst_dmath_double_inline = True

mathjax3_config = {
    "tex": {
        "inlineMath": [['$', '$'], ['\\(', '\\)']],
        "displayMath": [['$$', '$$'], ["\\[", "\\]"]],
    }
}
rst_prolog = """
.. role:: raw-latex(raw)
   :format: latex html

.. |est_functions_keys| replace:: {est_keys}
""".format(est_keys=est_functions_keys)

exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    'notebooks/docinspect.ipynb',
    'source/README.md'
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/aurel.png'  # Add your logo here
html_favicon = '_static/favicon.png'
html_css_files = [
    'custom.css',
]
html_js_files = [
    'custom.js',
]

autodoc_typehints = "none"

# Render LateX in notebook outputs
def process_notebook_source(app, docname, source):
    """Simple test function to replace LaTeX outputs with 'Hi <3'."""
    if source and len(source) > 0 and docname.startswith('notebooks/'):
        content = source[0]
        brackets = r'\[\s*"([^"]*?)"\s*\]'
        # First pass: Convert display_data to stream
        display_data_pattern = r'{\s*"data":\s*{\s*"text/latex":\s*'+brackets+r',\s*"text/plain":\s*\[\s*"<IPython\.core\.display\.Latex object>"\s*\]\s*},\s*"metadata":\s*{[^}]*},\s*"output_type":\s*"display_data"\s*}'
        
        def convert_to_stream(match):
            latex_content = match.group(1)
            
            # Convert to stream output format
            stream_output = f'''{{
         "name": "stdout",
         "output_type": "stream",
         "text": [
          "{latex_content}"
         ]
        }}'''
            
            return stream_output
        
        content = re.sub(display_data_pattern, convert_to_stream, content, flags=re.DOTALL)
        # Second pass: Simple approach - just merge any two consecutive stdout streams
        # Much more specific pattern that only matches the text array content
        c = r'[^\[\]]*'
        nest = r'(?:\['+c+r'\]'+c+r')*'
        brackets = r'\['+c+nest+r'\]'  # Matches single quoted string in array
        simple_merge_pattern = r'({\s*"name":\s*"stdout",\s*"output_type":\s*"stream",\s*"text":\s*'+brackets+r'\s*})\s*,\s*({\s*"name":\s*"stdout",\s*"output_type":\s*"stream",\s*"text":\s*'+brackets+r'\s*})'
        
        def simple_merge(match):
            block1 = match.group(1)
            block2 = match.group(2)
            nest = r'(?:'+c+r'|'+c+r'\['+c+r'\]'+c+r')*'
            capture_pattern = r'"text":\s*\[('+nest+r')\]'
            text1_match = re.search(capture_pattern, block1, re.DOTALL)
            text2_match = re.search(capture_pattern, block2, re.DOTALL)

            if text1_match and text2_match:
                text1 = text1_match.group(1).strip()
                text2 = text2_match.group(1).strip()
                if not text1.endswith('\\n'):
                    text1 = text1[:-1] + '\\n"'
                combined = text1 + ",      " + text2
                return f'''{{
         "name": "stdout",
         "output_type": "stream",
         "text": [
          {combined}
         ]
        }}'''
            return match.group(0)  # Return original if extraction fails
        
        # Keep applying until no more merges possible
        prev_content = ""
        while prev_content != content:
            prev_content = content
            content = re.sub(simple_merge_pattern, simple_merge, content, flags=re.DOTALL)
        # Replace the original source with the modified content
        source[0] = content

def setup(app):
    app.connect('source-read', process_notebook_source)

subprocess.run(["python3", "source/generate_rst.py"], check=True)

subprocess.run(["python3", "source/generate_rstsymbolic.py"], check=True)
