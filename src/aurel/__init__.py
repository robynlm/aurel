"""Aurel: A Python package for automatic relativistic calculations."""

__version__ = "0.9.3"
from .core import *
from .coresymbolic import *
from .finitedifference import *
from .maths import *
from .numerical import *
from .reading import *
from .time import *


def helloworld():
    """Print a welcome message."""
    print("Hello! Welcome to Aurel!", flush=True)
