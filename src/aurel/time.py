"""
Variable calculation and scalar estimation over time for the Aurel package.

This module provides a function to calculate variables over time series data
and applying statistical estimation functions to 3D arrays.
"""

import jax.numpy as jnp
from . import core
from dask.distributed import Client

# Dictionary of available estimation functions for 3D arrays
est_functions = {
    'max': jnp.max,                         # Maximum value across entire array
    'mean': jnp.mean,                       # Mean value across entire array
    'median': jnp.median,                   # Median value across entire array
    'min': jnp.min,                         # Minimum value across entire array
    'OD': lambda array: array[0, 0, 0],     # Origin value
    'UD': lambda array: array[-1, -1, -1],  # Upper diagonal value
}

def over_time(data, fd, vars=[], estimate=[], 
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
    estimate : list of str or dict, optional
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
        If `estimate` is provided, additional keys are added with format 
        '{variable}_{estimation_func}'.
    """
    
    # Clean vars list to remove any variables that are already in data
    cleaned_vars = []
    for v in vars:
        if v not in data:
            cleaned_vars += [v]
    vars = cleaned_vars
    
    # Clean estimate list to remove any estimations that are already in data
    cleaned_estimate = []
    estimates_done = []
    for v in list(data.keys()):
        if '_' in v:
            estimates_done += [v.split('_')[-1]]
    for est_item in estimate:
         if isinstance(est_item, str):
            if est_item not in estimates_done:
                cleaned_estimate += [est_item]
            elif isinstance(est_item, dict):
                for func_name in est_item.keys():
                    if func_name not in estimates_done:
                        cleaned_estimate += [est_item]
    estimate = cleaned_estimate
    
    # Only perform variable calculations if vars list is not empty
    if vars or estimate:
        
        # Transform dict of lists to a list of dicts
        keys = data.keys()
        data_list = [dict(zip(keys, values)) for values in zip(*data.values())]
        
        # Calculate first instance)
        data_list_i0, scalarkeys = process_single_timestep(
            data_list[0], fd, vars, estimate, verbose, None)
        
        # Calculate all the other
        if len(data_list) > 1:
            if verbose:
                print("Now doing the same, processing time steps in parallel "
                      +f"with Dask client and nbr_processes = {nbr_processes}", 
                      flush=True)
            # Create a wrapper function that captures the fixed parameters
            def process_wrapper(data_dict):
                return process_single_timestep(
                    data_dict, fd, vars, estimate, veryverbose, scalarkeys)
            # Iterate in parallel through each time step in the data
            client = Client(threads_per_worker=1, n_workers=nbr_processes)
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

def process_single_timestep(data, fd, vars, estimate, 
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
    estimate : list of str or dict
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
        If `estimate` is provided, additional keys are added with format
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
    
    # ====== Apply estimation functions if requested
    if estimate:
        # Find all keys that contain 3D scalar arrays
        # (shape that has 3 dimensions)
        if scalarkeys is None:
            scalarkeys = []
            for key in data.keys():
                if len(jnp.shape(data[key])) == 3:
                    scalarkeys += [key]
            return_scalarkeys = True
        else:
            return_scalarkeys = False
        
        # For each scalar key, process the estimation functions
        for key in scalarkeys:
            # Process each item in the estimate list
            for est_item in estimate:
                if isinstance(est_item, str):
                    # Handle predefined estimation from est_functions
                    if est_item in est_functions.keys():
                        # Apply estimation function and store the result
                        func = est_functions[est_item]
                        data[key+'_'+est_item] = func(data[key])
                    else:
                        ValueError(
                            "Help, unknown estimation function:",
                            est_item)
                elif isinstance(est_item, dict):
                    # Handle custom estimation functions from dictionary
                    for func_name, func in est_item.items():
                        # Apply the custom estimation function and store 
                        data[key+'_'+func_name] = func(data[key])
    
    # Return the updated data and scalar keys if new information requested
    if return_scalarkeys:
        return data, scalarkeys
    else:
        return data