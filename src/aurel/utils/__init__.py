"""Internal utility functions for the aurel package."""

from .jupyter import is_notebook
from .memory import get_size, format_size

__all__ = ['is_notebook', 'get_size', 'format_size']
