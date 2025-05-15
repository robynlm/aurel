"""
numerical.py

This module contains numerical methods for various calculations.
So far it only contains the bisection method for finding roots of a function.
"""

def dichotomy(y_wanted, function, lower_bound, upper_bound, tolerance):
    """
    Find the root of a function using the bisection method.

    Numerically solving for x: function(x) = y_wanted

    Parameters
    ----------
    y_wanted : float
        The value of the function to find the root for.
    function : callable
        The function for which to find the root.
    lower_bound : float
        The lower bound of the interval to search for the root.
    upper_bound : float
        The upper bound of the interval to search for the root.
    tolerance : float
        The tolerance for the convergence of the method.
    Returns
    -------
    float
        The x value for which the function is equal to y_wanted.
    """
    x_low = lower_bound
    x_upp = upper_bound
    x_mid = (x_low + x_upp) / 2
    y_low = function(x_low)
    y_upp = function(x_upp)
    y_mid = function(x_mid)
    while abs(y_wanted / y_mid - 1) > tolerance:
        if y_wanted > y_mid:
            y_low = y_mid
            x_low = x_mid
            x_mid = (x_low + x_upp) / 2
            y_mid = function(x_mid)
        else:
            y_upp = y_mid
            x_upp = x_mid
            x_mid = (x_low + x_upp) / 2
            y_mid = function(x_mid)
    return x_mid