"""Generate RST documentation from hierarchical descriptions.yml."""

import inspect
import os
import sys

import yaml

# Get absolute paths based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
src_dir = os.path.join(project_root, 'src')

# Add the src directory to the Python path
sys.path.insert(0, src_dir)
import aurel.core as core  # noqa: E402

# File paths
descriptions_file = os.path.join(src_dir, 'aurel/data/descriptions.yml')
output_file = os.path.join(script_dir, 'core.rst')

# Load hierarchical structure from descriptions.yml
with open(descriptions_file) as f:
    categories = yaml.safe_load(f)

# Keep track of processed variables
varsdone = []

# Extract assumed variables by looking for assumption messages
assumed = []
for key in core.descriptions:
    if 'assume' in core.descriptions[key].lower():
        assumed += [key]

#
allfunctions = []
for name, _ in inspect.getmembers(core.AurelCore, inspect.isfunction):
    allfunctions.append(name)

# Process categories hierarchy
def process_category(f, category_name, category_data, varsdone, level=0):
    """Process a category and write to file."""
    f.write(category_name + "\n")

    # Determine heading level
    if level == 0:
        underline = "=" * len(category_name)
    elif level == 1:
        underline = "-" * len(category_name)
    else:
        underline = "^" * len(category_name)
    f.write(underline + "\n\n")

    if isinstance(category_data, dict):
        for subcat_name, subcat_data in category_data.items():
            # Skip metadata fields
            if subcat_name in ('category', 'subcategory'):
                continue
            elif subcat_name == 'note':
                f.write(f"{subcat_data}\n\n")
                continue

            # Check if this is a variable (has string description) or subcategory
            if isinstance(subcat_data, str):
                # It's a variable
                if ((subcat_name in core.descriptions) and
                    (subcat_name not in varsdone) and
                    (subcat_name in allfunctions)):
                    # Add a reference label that matches what Sphinx expects
                    # for the class method
                    f.write(f".. _aurel.core.AurelCore.{subcat_name}:\n\n")
                    # Link directly to the source code in _modules
                    f.write(
                        f"`{subcat_name} <../_modules/aurel/core.html"
                        f"#AurelCore.{subcat_name}>`_: "
                        f"{core.descriptions[subcat_name]}\n\n"
                    )
                    varsdone.append(subcat_name)
            else:
                # It's a subcategory
                process_category(f, subcat_name, subcat_data, varsdone, level=level + 1)

# Start writing the .rst file
with open(output_file, "w") as f:
    f.write("aurel.core\n")
    f.write("##########\n\n")
    f.write(".. automodule:: aurel.core\n\n")

    # Add hidden section FIRST for viewcode anchors (wrapped in hidden div)
    # This creates the autodoc anchors that [source] links need
    members_list = ", ".join(core.descriptions.keys())
    f.write(".. raw:: html\n\n")
    f.write("   <div style='display: none;' aria-hidden='true'>\n\n")
    f.write(".. autoclass:: aurel.core.AurelCore\n")
    f.write(f"   :members: {members_list}\n")
    f.write("   :noindex:\n\n")
    f.write(".. raw:: html\n\n")
    f.write("   </div>\n\n")

    f.write(".. _descriptions_list:\n\n")
    # Add a section for functions listed in `descriptions`
    f.write("descriptions\n")
    f.write("************\n\n")



    f.write(".. _assumed_quantities:\n\n")
    f.write("Assumed quantities\n")
    f.write("==================\n\n")
    f.write(
        'If not defined, vacuum Minkowski is assumed for the definition '
        'of the following quantities:\n\n'
    )
    f.write(
        r'**alpha**: $\alpha = 1$, the lapse, to change this do '
        r'**AurelCore.data["alpha"] = ...** before running calculations' +
        "\n\n"
    )
    f.write(
        r"**dtalpha**: $\partial_t \alpha = 0$, "
        r"the time derivative of the lapse" + "\n\n"
    )
    f.write(
        r"**betax, betay, betaz**: $\beta^i = 0$, "
        r"the shift vector with spatial indices up" + "\n\n"
    )
    f.write(
        r"**dtbetax, dtbetay, dtbetaz**: $\partial_t \beta^i = 0$, "
        r"the time derivative of the shift vector with spatial indices up" +
        "\n\n"
    )
    f.write(
        r"**gxx, gxy, gxz, gyy, gyz, gzz**: $g_{ij} = \delta_{ij}$, "
        r"the spatial components of the spacetime metric with indices down" +
        "\n\n"
    )
    f.write(
        r"**kxx, kxy, kxz, kyy, kyz, kzz**: $K_{ij} = 0$, "
        r"the spatial components of the extrinsic curvature "
        r"with indices down" + "\n\n"
    )
    f.write(r"**rho0**: $\rho_0 = 0$, the rest-mass energy density"+"\n\n")
    f.write(r"**press**: $p = 0$, the fluid pressure"+"\n\n")
    f.write(r"**eps**: $\epsilon = 0$, the fluid specific internal energy"+"\n\n")
    f.write(r"**w_lorentz**: $W = 1$, the Lorentz factor"+"\n\n")
    f.write(
        r"**velx, vely, velz**: $v^i = 0$, "
        r"the Eulerian fluid three velocity with spatial indices up" +
        "\n\n"
    )
    assumed_done = ['alpha', 'dtalpha', 'betax', 'betay', 'betaz',
                    'dtbetax', 'dtbetay', 'dtbetaz',
                    'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz',
                    'kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz',
                    'rho0', 'press', 'eps',
                    'w_lorentz', 'velx', 'vely', 'velz']
    missing = set(assumed) - set(assumed_done)
    if missing:
        raise RuntimeError(
            f"Documentation generation failed: {len(missing)} "
            f"assumed variable(s) not documented.\n"
            f"Missing variables: {', '.join(sorted(missing))}\n"
            f"Please update the assumed quantities list in "
            f"docs/source/generate_rst.py"
        )

    # Process all top-level categories
    for cat_name, cat_data in categories.items():
        process_category(f, cat_name, cat_data, varsdone)

    if len(varsdone) != len(core.descriptions):
        missing = set(core.descriptions.keys()) - set(varsdone)
        raise RuntimeError(
            f"Documentation generation failed: {len(missing)} "
            f"variable(s) not categorized.\n"
            f"Missing variables: {', '.join(sorted(missing))}\n"
            f"Please update the categorization in docs/source/generate_rst.py"
        )

    # Add a section for other functions (AurelCore members that aren't in descriptions)
    f.write("AurelCore Methods\n")
    f.write("*****************\n\n")

    # Build exclude-members list from descriptions keys
    exclude_list = ", ".join(core.descriptions.keys())

    f.write(".. autoclass:: aurel.core.AurelCore\n")
    f.write("   :show-inheritance:\n")
    f.write("   :members:\n")
    f.write(f"   :exclude-members: {exclude_list}\n\n")
