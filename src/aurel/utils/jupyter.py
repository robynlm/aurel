"""Jupyter notebook detection and related utilities."""



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
