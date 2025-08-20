"""Tests for checking docstring coverage across the aurel package."""

import inspect
import numpy as np
import aurel
import pytest

class TestDocstrings:
    """Test docstring coverage."""
    aurel_elements = [getattr(aurel, name) for name in dir(aurel) 
         if ((inspect.isclass(getattr(aurel, name)) 
              or inspect.isfunction(getattr(aurel, name)))
             and getattr(aurel, name).__module__.startswith('aurel'))]
    
    @pytest.mark.parametrize("module", aurel_elements)
    def test_docstrings(self, module):
        missing_docs = []

        if inspect.isclass(module):
            # Check classes 
            methods = inspect.getmembers(module, predicate=inspect.isfunction)
            for name, func in methods:
                # For every function in the class
                doc = func.__doc__
                if not doc or not doc.strip():
                    missing_docs.append(name)
        else:
            # Check functions
            doc = module.__doc__
            if not doc or not doc.strip():
                missing_docs.append(module.__name__)
        
        assert not missing_docs, f"Missing docstrings: {missing_docs}"