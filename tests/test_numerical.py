"""
Tests for the numerical module.

These tests verify numerical methods work correctly.
"""

import numpy as np

from aurel import numerical


class TestDichotomy:
    """Tests for the dichotomy (bisection) method."""

    def test_dichotomy_linear_function(self):
        """Test dichotomy with a simple linear function."""
        # Find x such that f(x) = 2*x + 3 = 10
        # Solution: x = 3.5
        def f(x):
            return 2*x + 3

        result = numerical.dichotomy(10.0, f, 0.0, 10.0, 1e-6)
        assert np.isclose(result, 3.5, atol=1e-5)

    def test_dichotomy_quadratic_function(self):
        """Test dichotomy with a quadratic function."""
        # Find x such that f(x) = x^2 = 16
        # Solution: x = 4 (positive root)
        def f(x):
            return x**2

        result = numerical.dichotomy(16.0, f, 0.0, 10.0, 1e-6)
        assert np.isclose(result, 4.0, atol=1e-5)

    def test_dichotomy_exponential_function(self):
        """Test dichotomy with an exponential function."""
        # Find x such that f(x) = e^x = e^2
        # Solution: x = 2
        def f(x):
            return np.exp(x)

        result = numerical.dichotomy(np.exp(2), f, 0.0, 5.0, 1e-6)
        assert np.isclose(result, 2.0, atol=1e-5)

    def test_dichotomy_trigonometric_function(self):
        """Test dichotomy with a trigonometric function."""
        # Find x such that f(x) = sin(x) = 0.5
        # Solution: x = pi/6 ≈ 0.5236 (in range [0, pi/2])
        def f(x):
            return np.sin(x)

        result = numerical.dichotomy(0.5, f, 0.0, np.pi/2, 1e-6)
        assert np.isclose(result, np.pi/6, atol=1e-5)

    def test_dichotomy_tight_tolerance(self):
        """Test dichotomy converges with tight tolerance."""
        # Find x such that f(x) = x^3 = 27
        # Solution: x = 3
        def f(x):
            return x**3

        result = numerical.dichotomy(27.0, f, 0.0, 10.0, 1e-10)
        assert np.isclose(result, 3.0, atol=1e-9)

