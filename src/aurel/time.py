"""
Variable calculation and scalar estimation over time for the Aurel package.

This module provides a function to calculate variables over time series data
and applying statistical estimation functions to 3D arrays.
"""

#TODO: I got the warning:
"""
/its/home/rlm36/myenv/lib64/python3.9/site-packages/distributed/client.py:3362: UserWarning: Sending large graph of size 22.06 MiB.
This may cause some slowdown.
Consider loading the data with Dask directly
 or using futures or delayed objects to embed the data into the graph without repetition.
See also https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask for more information.
"""

import jax.numpy as jnp
from . import core
import inspect
import dask
from dask.distributed import Client

# Dictionary of available estimation functions for 3D arrays
est_functions = {
    'max': jnp.max,
    'mean': jnp.mean,
    'median': jnp.median,
    'min': jnp.min,
    'sum': jnp.sum,
    'std': jnp.std,
    'var': jnp.var,
    'maxabs': lambda array: jnp.max(jnp.abs(array)),
    'minabs': lambda array: jnp.min(jnp.abs(array)),
    'meanabs': lambda array: jnp.mean(jnp.abs(array)),
    'medianabs': lambda array: jnp.median(jnp.abs(array)),
    'sumabs': lambda array: jnp.sum(jnp.abs(array)),
    'stdabs': lambda array: jnp.std(jnp.abs(array)),
    'varabs': lambda array: jnp.var(jnp.abs(array)),
    'OD': lambda array: array[0, 0, 0],
    'UD': lambda array: array[-1, -1, -1],
}

def over_time(data, fd, vars=[], estimates=[], 
              nbr_processes=1, verbose=True, veryverbose=False):
    """Calculate variables from the data and store them in the data dictionary.
    
    This function processes time series data by creating AurelCore instances 
    for each time step, calculating specified variables, and optionally 
    applying statistical estimation functions to 3D arrays.
    
    Parameters
    ----------
    data : dict
        Dictionary containing time series data. Must include an 'it' key with
        iteration information. The function will add calculated variables
        to this dictionary.
    fd : str
        Finite difference class used to initialize AurelCore instances.
    vars : list of str, optional
        List of variable names to calculate. These variables will be computed
        using the AurelCore instance at each time step. If empty, no variable
        calculations are performed. Default is an empty list.
    estimates : list of str or dict, optional
        List containing estimation functions to apply to all 3D scalar arrays.
        Elements can be:
        - str: Names of predefined functions from est_functions 
          ('max', 'mean', 'median', 'min', 'OD', 'UD')
        - dict: Custom estimation functions with string keys (function names) 
          and functions that take a 3D array of shape (fd.Nx, fd.Ny, fd.Nz) 
          and return a scalar value.
        Default is an empty list (no estimation applied).
    verbose : bool, optional
        If True, prints debug information about the calculation process.
    veryverbose : bool, optional
        If True, provides more detailed debug information. Defaults to False.
    
    Returns
    -------
    dict
        Updated data dictionary with calculated variables. 
        If `vars` is provided, a new key is added containing a list of 
        values for each time step. 
        If `estimates` is provided, additional keys are added with format 
        '{variable}_{estimation_func}'.
    """
    
    # Clean vars list to remove any variables that are already in data
    cleaned_vars = []
    for v in vars:
        if v not in data:
            if v in list(core.descriptions.keys()):
                cleaned_vars += [v]
            else:
                print(f"Error: Variable '{v}' not in core.descriptions,"
                      +" skipping.", flush=True)
    vars = cleaned_vars
    
    # Clean estimate list to remove any estimations that are already in data
    cleaned_estimates = []
    if vars != []:
        # Then apply all estimates again
        for est_item in estimates:
            if isinstance(est_item, str):
                # If the estime is a string, check if it is in est_functions
                # and add it to cleaned_estimates
                if est_item in list(est_functions.keys()):
                    cleaned_estimates += [est_item]
                else:
                    print(f"Error: Estimation function '{est_item}' not "
                        + "in est_functions, skipping.", flush=True)
                    print(f"Available functions: {list(est_functions.keys())}",
                        flush=True)
                    print("You can add custom functions"
                        + " estimates=[{'function_name':"
                        + " user_defined_function}].", flush=True)
            elif isinstance(est_item, dict):
                # If the estimate is a dict, validate each function
                for func_name, function in est_item.items():
                    try:
                        validate_estimation_function(
                            function, func_name, fd, verbose=verbose)
                        cleaned_estimates += [{func_name: function}]
                    except ValueError as e:
                        print(f"Error: {e}")
                        print(f"Skipping function '{func_name}'")
                        continue
    else:
        # Only apply estimates that are not already in data
        estimates_done = []
        for v in list(data.keys()):
            if '_' in v:
                estimates_done += [v.split('_')[-1]]
        for est_item in estimates:
            if isinstance(est_item, str):
                if est_item not in estimates_done:
                    if est_item in list(est_functions.keys()):
                        cleaned_estimates += [est_item]
                    else:
                        print(f"Error: Estimation function '{est_item}' not "
                            + "in est_functions, skipping.", flush=True)
                        print(f"Available functions: {list(est_functions.keys())}",
                            flush=True)
                        print("You can add custom functions"
                            + " estimates=[{'function_name':"
                            + " user_defined_function}].", flush=True)
            elif isinstance(est_item, dict):
                for func_name, function in est_item.items():
                    if func_name not in estimates_done:
                        try:
                            validate_estimation_function(
                                function, func_name, fd, verbose=verbose)
                            cleaned_estimates += [{func_name: function}]
                        except ValueError as e:
                            print(f"Error: {e}")
                            print(f"Skipping function '{func_name}'")
                            continue
    estimates = cleaned_estimates
    
    # Only perform variable calculations if vars list is not empty
    if vars or estimates:
        
        # Transform dict of lists to a list of dicts
        keys = data.keys()
        data_list = [dict(zip(keys, values)) for values in zip(*data.values())]
        
        # Calculate first instance)
        data_list_i0, scalarkeys = process_single_timestep(
            data_list[0], fd, vars, estimates, verbose, None)
        
        # Calculate all the other
        if len(data_list) > 1:
            if verbose:
                print("Now doing the same, processing time steps in parallel "
                      +f"with Dask client and nbr_processes = {nbr_processes}",
                      flush=True)
            # Create a wrapper function that captures the fixed parameters
            def process_wrapper(data_dict):
                return process_single_timestep(
                    data_dict, fd, vars, estimates, veryverbose, scalarkeys)
            # Iterate in parallel through each time step in the data
            with Client(threads_per_worker=1, n_workers=nbr_processes) as client:
                futures = [client.submit(process_wrapper, item)
                           for item in data_list[1:]]
                results = client.gather(futures)
            # Combine and sort the results by 'it' key
            sorted_results = sorted(results, key=lambda x: x['it'])
            data_list = [data_list_i0] + sorted_results
        else:
            data_list = [data_list_i0]
        
        # Transform list of dicts to a dict of lists
        keys = data_list[0].keys()
        data = {key: [] for key in keys}
        for i in range(len(data_list)):
            for key in keys:
                data[key].append(data_list[i][key])
        for key in keys:
            data[key] = jnp.array(data[key])
        del data_list
    
        return data
    else:
        print("No new variables or estimations requested,"
               + " returning original data.", flush=True)
        return data

def process_single_timestep(data, fd, vars, estimates, 
                            verbose, scalarkeys):
    """Process a single time step for variable calculation and estimation.
    
    This function creates an AurelCore instance for the specified time step,
    calculates requested variables, and applies statistical estimation functions
    to 3D arrays if specified.
    
    Parameters
    ----------
    data : dict
        Dictionary containing variables of relevant iteration. 
        The function will add calculated variables
        to this dictionary.
    fd : str
        Finite difference class used to initialize AurelCore instances.
    vars : list of str
        List of variable names to calculate. These variables will be computed
        using the AurelCore instance at this time step. If empty, no variable
        calculations are performed.
    estimates : list of str or dict
        List containing estimation functions to apply to all 3D scalar arrays.
        Elements can be:
        - str: Names of predefined functions from est_functions
        - dict: Custom estimation functions with string keys (function names)
          and functions that take a 3D array of shape (fd.Nx, fd.Ny, fd.Nz) 
          and return a scalar value.
        If empty, no estimation applied.
    verbose : bool
        Passed to AurelCore for debug information and calculation process.
    scalarkeys : list of str
        List of keys in `data` that contain 3D scalar arrays. If None,
        the function will determine these keys automatically.
    
    Returns
    -------
    dict
        Updated data dictionary with calculated variables.
        If `vars` is provided, a new key is added containing a list of
        values for this time step.
        If `estimates` is provided, additional keys are added with format
        '{variable}_{estimation_func}'.
    list of str
        List of scalar keys if `scalarkeys` is None, otherwise returns None."""
    # ====== Calculate vars if requested
    if vars:
        # Create a new AurelCore instance for this time step
        rel = core.AurelCore(fd, verbose=verbose)
        
        # Set all existing data values for this time step 
        # (except variables to be calculated)
        for key, values in data.items():
            if key not in vars:
                rel.data[key] = values
        # Freeze the data for cache management
        rel.freeze_data()
        
        # Calculate and store each requested variable
        for v in vars:
            data[v] = rel[v]
        
        # Clean up AurelCore instance to free memory
        del rel

    # ====== Find all keys that contain 3D scalar arrays
    # (shape that has 3 dimensions)
    if scalarkeys is None:
        scalarkeys = []
        for key in data.keys():
            if len(jnp.shape(data[key])) == 3:
                scalarkeys += [key]
        return_scalarkeys = True
    else:
        return_scalarkeys = False
    
    # ====== Apply estimation functions if requested
    if estimates:
        
        # Process each item in the estimates list
        for est_item in estimates:
            if verbose:
                if isinstance(est_item, str):
                    print(f"Processing estimation item: {est_item}", 
                          flush=True)
                elif isinstance(est_item, dict):
                    for func_name, func in est_item.items():
                        print(f"Processing estimation item: {func_name}", 
                            flush=True)
            # For each scalar key, process the estimation functions
            for key in scalarkeys:
                if isinstance(est_item, str):
                    if key+'_'+est_item not in list(data.keys()):
                        # Handle predefined estimation from est_functions
                        # Apply estimation function and store the result
                        func = est_functions[est_item]
                        data[key+'_'+est_item] = func(data[key])
                elif isinstance(est_item, dict):
                    # Handle custom estimation functions from dictionary
                    for func_name, func in est_item.items():
                        if key+'_'+func_name not in list(data.keys()):
                            # Apply the custom estimation function and store 
                            data[key+'_'+func_name] = func(data[key])
    
    # Return the updated data and scalar keys if new information requested
    if return_scalarkeys:
        return data, scalarkeys
    else:
        return data
    
def validate_estimation_function(func, func_name, fd, verbose=True):
    """Validate that an estimation function has correct signature and behavior.
    
    Parameters
    ----------
    func : callable
        Function to validate
    func_name : str
        Name of the function for error messages
    fd : FiniteDifference
        Finite difference object to get grid dimensions
    verbose : bool, optional
        If True, prints debug information about the validation process.
        
    Returns
    -------
    bool
        True if function is valid
    """
    if verbose:
        print(f"Validating estimation function '{func_name}'...", flush=True)
    
    # Check if it's callable
    if not callable(func):
        raise ValueError(f"Estimation function '{func_name}' must be callable")
    
    # Check function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    if len(params) != 1:
        raise ValueError(
            f"Estimation function '{func_name}' must take exactly 1 parameter, "
            f"got {len(params)}: {params}"
        )
    
    # Test with dummy array
    test_array = jnp.ones((fd.Nx, fd.Ny, fd.Nz))
    
    try:
        result = func(test_array)
    except Exception as e:
        raise ValueError(
            f"Estimation function '{func_name}' failed when called with "
            f"array of shape ({fd.Nx}, {fd.Ny}, {fd.Nz}): {e}"
        )
    
    # Check return type and shape
    if not isinstance(result, (int, float, jnp.number)):
        if hasattr(result, 'shape') and result.shape != ():
            raise ValueError(
                f"Estimation function '{func_name}' must return a scalar "
                f"(int or float), got shape {result.shape}"
            )
        if not jnp.isscalar(result):
            raise ValueError(
                f"Estimation function '{func_name}' must return a scalar "
                f"(int or float), got type {type(result)}"
            )
    if verbose:
        print(f"✓ Custom function '{func_name}' validated successfully")
    return True