"""Jupyter notebook detection and related utilities."""

import sys


def is_notebook():
    """Check if code is running in a Jupyter notebook environment.

    Returns
    -------
    bool
        True if running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        return 'IPKernelApp' in get_ipython().config
    except (ImportError, AttributeError):
        return False


def get_tqdm():
    """Get appropriate tqdm function based on environment.

    Returns notebook version of tqdm if in a notebook environment,
    otherwise returns the standard CLI version.

    Returns
    -------
    tqdm : function
        The appropriate tqdm progress bar function.
    disable : bool
        Whether to disable tqdm (when output is redirected).
    """
    in_notebook = is_notebook()
    is_terminal = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
    disable = not in_notebook and not is_terminal

    if in_notebook:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm
    else:
        from tqdm import tqdm

    return tqdm, disable
