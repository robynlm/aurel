"""Generate RST documentation for symbolic aurel functions."""

import inspect
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))
import aurel.coresymbolic as coresymbolic

# Directory to save the generated .rst files
output_dir = "."

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# File to write the documentation
output_file = os.path.join(output_dir, "source/coresymbolic.rst")

varsdone = []

def print_subsec(title, subsecvars, allfunctions, varsdone):
    """Print subsection with function documentation to RST file."""
    if title != "":
        f.write(title+"\n")
        f.write("-"*len(title)+"\n\n")
    for name in list(coresymbolic.symbolic_descriptions.keys()):
        if ((name in allfunctions)
            and (name in subsecvars)
            and (name not in varsdone)):
            # Add a reference label that matches what Sphinx expects
            # for the class method
            f.write(f".. _aurel.core.AurelCoreSymbolic.{name}:\n\n")
            # Link directly to the source code in _modules
            f.write(
                f"`{name} <../_modules/aurel/coresymbolic.html"
                f"#AurelCoreSymbolic.{name}>`_: "
                f"{coresymbolic.symbolic_descriptions[name]}\n\n"
            )
            varsdone.append(name)
    return varsdone

# Start writing the .rst file
with open(output_file, "w") as f:
    f.write("aurel.coresymbolic\n")
    f.write("##################\n\n")
    f.write(".. automodule:: aurel.coresymbolic\n\n")

    # Add hidden section FIRST for viewcode anchors (wrapped in hidden div)
    # This creates the autodoc anchors that [source] links need
    members_list = ", ".join(coresymbolic.symbolic_descriptions.keys())
    f.write(".. raw:: html\n\n")
    f.write("   <div style='display: none;' aria-hidden='true'>\n\n")
    f.write(".. autoclass:: aurel.coresymbolic.AurelCoreSymbolic\n")
    f.write(f"   :members: {members_list}\n")
    f.write("   :noindex:\n\n")
    f.write(".. raw:: html\n\n")
    f.write("   </div>\n\n")

    f.write(".. _symbolic_descriptions_list:\n\n")
    # Add a section for functions listed in `descriptions`
    f.write("symbolic_descriptions\n")
    f.write("*********************\n\n")
    allfunctions = []
    for name, _ in inspect.getmembers(
        coresymbolic.AurelCoreSymbolic, inspect.isfunction
    ):
        allfunctions.append(name)

    f.write(".. _symbolic_assumed_quantities:\n\n")
    f.write("Assumed quantities\n")
    f.write("==================\n\n")
    f.write(
        'If not defined, vacuum Minkowski is assumed for the definition '
        'of the following quantities:\n\n'
    )
    print_subsec("", ['gdown'], allfunctions, varsdone)

    f.write("Callable quantities\n")
    f.write("===================\n\n")
    print_subsec(
        "", list(coresymbolic.symbolic_descriptions.keys()),
        allfunctions, varsdone
    )

    if len(varsdone) != len(coresymbolic.symbolic_descriptions):
        missing = (
            set(coresymbolic.symbolic_descriptions.keys()) - set(varsdone)
        )
        raise RuntimeError(
            f"Documentation generation failed: {len(missing)} "
            f"variable(s) not categorized.\n"
            f"Missing variables: {', '.join(sorted(missing))}\n"
            f"Please update the categorization in "
            f"docs/source/generate_rstsymbolic.py"
        )

    # Add a section for other functions (AurelCore members that aren't in descriptions)
    f.write("AurelCoreSymbolic Methods\n")
    f.write("*************************\n\n")

    # Build exclude-members list from descriptions keys
    exclude_list = ", ".join(coresymbolic.symbolic_descriptions.keys())

    f.write(".. autoclass:: aurel.coresymbolic.AurelCoreSymbolic\n")
    f.write("   :show-inheritance:\n")
    f.write("   :members:\n")
    f.write(f"   :exclude-members: {exclude_list}\n\n")
