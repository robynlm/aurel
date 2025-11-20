"""
numerical.py

This module contains numerical methods for various calculations.
"""

import numpy as np
import scipy

def dichotomy(y_wanted, function, lower_bound, upper_bound, tolerance):
    """Find the root of a function using the bisection method.

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

def interpolate(val, grid_points, target_points, method='linear'):
    """Interpolate scalar field from one grid to a new grid.

    Parameters
    ----------
    val : ndarray
        Scalar field values on the regular grid. Shape should match the grid 
        dimensions.
    grid_points : tuple of ndarray
        Tuple of coordinate arrays defining the regular grid 
        coordinates. Each array should be 1D.
    target_points : tuple of ndarray
        Tuple of target coordinate positions. Arrays can be any shape but must
        have matching dimensions.
    method : str, optional
        Interpolation method used. See 
        `scipy.interpolate.RegularGridInterpolator documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_

    Returns
    -------
    ndarray
        Interpolated values at the target points, with the same shape as the 
        target coordinate arrays.
    """
    target_flat = []
    for x in target_points:
        target_flat += [x.flatten()]
    
    # Flatten target points and stack into (N, 3) array for interpolator
    target_shape = target_points[0].shape
    points_flat = np.stack(target_flat, axis=-1)
    
    # Check that all target points are within grid bounds
    for i, (grid, target) in enumerate(zip(grid_points, target_points)):
        grid_min, grid_max = grid.min(), grid.max()
        target_min, target_max = target.min(), target.max()
        if target_min < grid_min or target_max > grid_max:
            raise ValueError(
                f"Target points in dimension {i} are outside grid bounds. "
                f"Grid range: [{grid_min}, {grid_max}], "
                f"Target range: [{target_min}, {target_max}]"
            )
    
    # Create interpolator and evaluate
    interpolator = scipy.interpolate.RegularGridInterpolator(
        grid_points, val, method=method, bounds_error=False, fill_value=None)
    return interpolator(points_flat).reshape(target_shape)