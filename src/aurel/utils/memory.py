"""Memory size calculation utilities.

This module provides functions to calculate the memory size of various
Python objects, including numpy arrays and collections of arrays. It also
includes a utility to format byte sizes into human-readable strings.
"""

import sys

import numpy as np


def get_size(obj):
    """Get the memory size of an object in bytes.

    This function handles different object types appropriately:
    - For numpy arrays, uses .nbytes to get actual data size
    - For lists/tuples of arrays, sums the sizes recursively
    - For other objects, uses sys.getsizeof

    Parameters
    ----------
    obj : object
        The object to measure.

    Returns
    -------
    int
        Size in bytes.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.ones((100, 100))
    >>> size = get_size(arr)
    >>> size == arr.nbytes
    True
    """
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.nbytes

    # Handle lists/tuples of arrays (common in aurel for tensor components)
    if isinstance(obj, (list, tuple)):
        return sum(get_size(item) for item in obj)

    # Handle dictionaries (recursive)
    if isinstance(obj, dict):
        return sum(get_size(k) + get_size(v) for k, v in obj.items())

    # For other objects, fall back to sys.getsizeof (NOTE: this is shallow
    # and may not account for all referenced objects but we tried to cover
    # common cases above)
    return sys.getsizeof(obj)


def format_size(size_bytes):
    """Format byte size into human-readable string.

    Parameters
    ----------
    size_bytes : int or float
        Size in bytes.

    Returns
    -------
    str
        Human-readable size string (e.g., "1.5 MB", "2.3 GB").

    Examples
    --------
    >>> format_size(1024)
    '1.00 KB'
    >>> format_size(1024 * 1024)
    '1.00 MB'
    >>> format_size(1.5 * 1024 * 1024 * 1024)
    '1.50 GB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
