"""
descriptions_utils.py.

This module provides utilities for loading and processing variable descriptions
used throughout the aurel project. Descriptions are stored in a hierarchical
YAML file that combines categories with descriptions.
"""

import os
import re

import yaml


def _assumption(keyname, assumption_text):
    """Generate assumption message for a variable.

    Parameters
    ----------
    keyname : str
        The name of the variable
    assumption_text : str
        The assumption being made (e.g., "$g_{xx}=1$")

    Returns
    -------
    str
        Formatted assumption message
    """
    message = (f" I assume {assumption_text}, "
               f"if not then please define "
               f"AurelCore.data['{keyname}'] = ... ")
    return message


def _extract_descriptions(data):
    """Extract variable descriptions from 2-level hierarchical structure.

    The structure is: Category -> Subcategory -> Variables (or Category -> Variables)
    Metadata fields (category, subcategory, note) are ignored.

    Parameters
    ----------
    data : dict
        The hierarchical data structure

    Returns
    -------
    dict
        Flat dictionary mapping variable names to descriptions
    """
    descriptions = {}

    for _, category_value in data.items():
        if isinstance(category_value, dict):
            for key, value in category_value.items():
                # Skip metadata fields
                if key in ('category', 'subcategory', 'note'):
                    continue

                if isinstance(value, dict):
                    # This is a subcategory, iterate through its items
                    for var_name, var_desc in value.items():
                        # Skip metadata fields
                        if var_name in ('category', 'subcategory', 'note'):
                            continue
                        descriptions[var_name] = var_desc
                elif isinstance(value, str):
                    # Direct variable in category (no subcategory)
                    descriptions[key] = value

    return descriptions


def load_descriptions():
    """Load descriptions from YAML file and process ASSUME markers.

    Reads descriptions.yml which contains a hierarchical structure combining
    categories with variable descriptions. Automatically converts
    [ASSUME:key:text] markers (one per description max).

    Returns
    -------
    dict
        Flat dictionary mapping variable names to their descriptions
    """
    # Get the directory containing this file (utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to aurel/ directory then into data/
    yaml_path = os.path.join(os.path.dirname(current_dir), 'data', 'descriptions.yml')

    # Load YAML file
    with open(yaml_path) as f:
        hierarchical_data = yaml.safe_load(f)

    # Extract flat descriptions from hierarchical structure
    descriptions = _extract_descriptions(hierarchical_data)

    # Process ASSUME markers (one per description)
    # Pattern: [ASSUME:keyname:assumption_text]
    assume_pattern = re.compile(r'\[ASSUME:([^:]+):([^\]]+)\]')

    for key, value in descriptions.items():
        # Find ASSUME marker in the description (should be at most one)
        match = assume_pattern.search(value)
        if match:
            keyname, assumption_text = match.groups()
            marker = f"[ASSUME:{keyname}:{assumption_text}]"
            replacement = _assumption(keyname, assumption_text)
            value = value.replace(marker, replacement)
            descriptions[key] = value

    return descriptions


def load_symbolic_descriptions():
    """Load symbolic descriptions from YAML file.

    Reads symbolic_descriptions.yml which contains a simple flat structure
    mapping variable names directly to descriptions.

    Returns
    -------
    dict
        Dictionary mapping variable names to their descriptions
    """
    # Get the directory containing this file (utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to aurel/ directory then into data/
    yaml_path = os.path.join(
        os.path.dirname(current_dir), 'data', 'symbolic_descriptions.yml'
    )

    # Load YAML file
    with open(yaml_path) as f:
        descriptions = yaml.safe_load(f)

    return descriptions
