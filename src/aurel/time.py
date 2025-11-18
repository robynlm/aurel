"""
Variable calculation and scalar estimation over time for the Aurel package.

This module provides a function to calculate variables over time series data
and applying statistical estimation functions to 3D arrays.
"""
import numpy as np
from . import core
import inspect
import sys

# Check if running in a notebook
is_notebook = 'ipykernel' in sys.modules
# Check if stdout is a terminal (not redirected to a file)
is_terminal = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
disable_tqdm = not is_notebook and not is_terminal

# Import appropriate tqdm version
if is_notebook:
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm
else:
    from tqdm import tqdm

# Dictionary of available estimation functions for 3D arrays
est_functions = {
    'max': np.max,
    'mean': np.mean,
    'median': np.median,
    'min': np.min,
    'sum': np.sum,
    'std': np.std,
    'var': np.var,
    'maxabs': lambda array: np.max(np.abs(array)),
    'minabs': lambda array: np.min(np.abs(array)),
    'meanabs': lambda array: np.mean(np.abs(array)),
    'medianabs': lambda array: np.median(np.abs(array)),
    'sumabs': lambda array: np.sum(np.abs(array)),
    'stdabs': lambda array: np.std(np.abs(array)),
    'varabs': lambda array: np.var(np.abs(array)),
    'OD': lambda array: array[0, 0, 0],
    'UD': lambda array: array[-1, -1, -1],
}

def over_time(data, fd, vars=[], estimates=[], 
              verbose=True, veryverbose=False, **rel_kwargs):
    """Calculate variables from the data and store them in the data dictionary.
    
    This function processes time series data by creating AurelCore instances 
    for each time step, calculating specified variables, and optionally 
    applying estimation functions to 3D arrays.
    
    Parameters
    ----------
    data : dict
        Dictionary containing time series data. Must include an 'it' key with
        iteration information. This function will add calculated variables
        to this dictionary.
    fd : FiniteDifference
        Finite difference class used to initialize AurelCore instances.
    vars : list of str or dict, optional
        List of variable names to calculate. These variables will be computed
        using the AurelCore instance at each time step.
        Elements can be:

        - str: Names of predefined variables from core.descriptions
        - dict: Custom variable definitions with string keys (variable names)
          and functions that take an AurelCore instance and return the variable
          value.

        Default is an empty list, no variable calculations are performed.
    estimates : list of str or dict, optional
        List containing estimation functions to apply to all 3D scalar arrays.
        Elements can be:

        - str: Names of predefined functions from est_functions 
          ('max', 'mean', 'median', 'min', 'OD', 'UD', ...)
        - dict: Custom estimation functions with string keys (function names) 
          and functions that take a scalar 3D array of shape 
          (fd.Nx, fd.Ny, fd.Nz) and return a scalar value.
        
        Default is an empty list (no estimation applied).
    verbose : bool, optional
        If True, prints debug information about the calculation process.
    veryverbose : bool, optional
        If True, provides more detailed debug information. Defaults to False.
    **rel_kwargs : dict, optional
        Additional parameters passed to AurelCore initialization, such as:
        
        - Lambda : float, cosmological constant
        - tetrad_to_use : str, tetrad choice for Weyl calculations
    
    Returns
    -------
    dict
        Updated data dictionary with calculated variables. 
        If `vars` is provided, a new key is added containing a list of 
        values for each time step. 
        If `estimates` is provided, additional keys are added with format 
        '{variable}_{estimation_func}'.
    """

    # Check/define temporal key
    temporal_key = None
    for temp in ['it', 'iteration', 't', 'time']:
        if temp in list(data.keys()):
            temporal_key = temp
    if temporal_key is None:
        raise ValueError("Data dictionary must contain a temporal key: "
                         "'it', 'iteration', 't', or 'time'.")
    
    # Clean vars list to remove any variables that are already in data
    cleaned_vars = []
    for v in vars:
        if isinstance(v, str):
            if v not in data:
                if v in list(core.descriptions.keys()):
                    cleaned_vars += [v]
                else:
                    print(f"Error: Variable '{v}' not in core.descriptions,"
                        +" skipping.", flush=True)
        elif isinstance(v, dict):
            # If the variable is a dict, validate each function
            for func_name, function in v.items():
                if func_name not in data:
                    try:
                        validate_variable_function(
                            function, func_name, fd,
                            verbose, False, rel_kwargs)
                        cleaned_vars += [{func_name: function}]
                    except ValueError as e:
                        print(f"Error: {e}")
                        print(f"Skipping function '{func_name}'")
                        continue
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
                        print("Available functions: "
                              + f"{list(est_functions.keys())}",
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
    
    # Only perform variable calculations if lists are not empty
    if vars or estimates:
        
        # Transform dict of lists to a list of dicts
        keys = data.keys()
        input_data_list = [dict(zip(keys, values)) 
                           for values in zip(*data.values())]
        del data
        
        # Calculate first instance
        data_list_i0, scalarkeys = process_single_timestep(
            input_data_list[0], fd, vars, estimates, verbose, None, rel_kwargs)
        data_list = [data_list_i0]
        del data_list_i0
        
        # Calculate all the other
        if len(input_data_list) > 1:
            
            # Sequential processing
            if verbose:
                print("Now processing remaining time steps sequentially",
                    flush=True)
            
            # When redirected, replace \r with \n so progress appears on new lines
            tqdm_kwargs = {} if not disable_tqdm else {
                'file': type('', (), {
                    'write': lambda self, s: sys.stdout.write(s.replace('\r', '\n')), 
                    'flush': lambda self: sys.stdout.flush()})()
            }
            results = [process_single_timestep(item, fd, vars, estimates, 
                                                veryverbose, scalarkeys, rel_kwargs)
                        for item in tqdm(input_data_list[1:], **tqdm_kwargs)]
            # Combine and sort the results by temporal_key key
            data_list += results
        del input_data_list
        data_list_sorted = sorted(data_list, key=lambda x: x[temporal_key])
        del data_list
        
        # Transform list of dicts to a dict of lists
        keys = data_list_sorted[0].keys()
        data = {key: [] for key in keys}
        for i in range(len(data_list_sorted)):
            for key in keys:
                data[key].append(data_list_sorted[i][key])
        for key in keys:
            data[key] = np.array(data[key])
        del data_list_sorted
    
        if verbose:
            print("Done!", flush=True)
        return data
    else:
        print("No new variables or estimations requested,"
               + " returning original data.", flush=True)
        return data

def process_single_timestep(data, fd, vars, estimates, 
                            verbose, scalarkeys, rel_kwargs):
    """Process a single time step for variable calculation and estimation.
    
    This function creates an AurelCore instance for the specified time step,
    calculates requested variables, and applies estimation 
    functions to 3D arrays if specified.
    
    Parameters
    ----------
    data : dict
        Dictionary containing variables of relevant iteration. 
        The function will add calculated variables
        to this dictionary.
    fd : FiniteDifference
        Finite difference class used to initialize AurelCore instances.
    vars : list of str or dict
        List of variable names to calculate. These variables will be computed
        using the AurelCore instance at each time step.
        Elements can be:

        - str: Names of predefined variables from core.descriptions
        - dict: Custom variable definitions with string keys (variable names)
          and functions that take an AurelCore instance and return the variable
          value.

        If empty, no variable calculations are performed.
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
    rel_kwargs : dict
        Additional parameters passed to AurelCore initialization.
    
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
    
    if verbose:
        for temp in ['it', 'iteration', 't', 'time']:
            if temp in list(data.keys()):
                print(f"Processing {temp} = {data[temp]}", 
                      flush=True)
                break
    
    # ====== Calculate vars if requested
    if vars:
        # Create a new AurelCore instance for this time step
        rel = core.AurelCore(fd, verbose=verbose, **rel_kwargs)
        
        # Set all existing data values for this time step 
        # (except variables to be calculated)
        for key, values in data.items():
            if key not in vars:
                rel.data[key] = values
        # Freeze the data for cache management
        rel.freeze_data()

        # Implement custom variables if requested
        for v in vars:
            if isinstance(v, str):
                pass
            elif isinstance(v, dict):
                for func_name, function in v.items():
                    if verbose:
                        print(f"Calculating custom variable '{func_name}'...",
                              flush=True)
                    rel.data[func_name] = function(rel)
                    rel.var_importance[func_name] = 0 # Freeze this in
                    if verbose:
                        print(f"Calculated and freezed variable '{func_name}'"
                              + f" in AurelCore", flush=True)
        
        # Calculate and store each requested variable
        for v in vars:
            if isinstance(v, str):
                data[v] = rel[v]
            elif isinstance(v, dict):
                for func_name, function in v.items():
                    data[func_name] = rel[func_name]
        
        # Clean up AurelCore instance to free memory
        del rel

    # ====== Find all keys that contain 3D scalar arrays
    # (shape that has 3 dimensions)
    if scalarkeys is None:
        scalarkeys = []
        for key in data.keys():
            if len(np.shape(data[key])) == 3:
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
    """Validate that estimation function has correct signature and behaviour.
    
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
            f"Estimation function '{func_name}' must take exactly 1 parameter,"
            f" a scalar array, instead got {len(params)}: {params}"
        )
    
    # Test with dummy array
    test_array = np.ones((fd.Nx, fd.Ny, fd.Nz))
    
    try:
        result = func(test_array)
    except Exception as e:
        raise ValueError(
            f"Estimation function '{func_name}' failed when called with "
            f"array of shape ({fd.Nx}, {fd.Ny}, {fd.Nz}): {e}"
        )
    
    # Check return type and shape - must be scalar-like
    try:
        # Convert to Python scalar if possible evaluation
        if hasattr(result, 'item') and callable(result.item):
            result_value = result.item()
        else:
            result_value = result
    except:
        result_value = result
    
    # Check if result is scalar-like: Python scalar or 0-dimensional array
    is_scalar = (
        isinstance(result_value, (int, float, complex)) or  # Python scalars
        (hasattr(result, 'shape') and result.shape == ()) or  # 0-d arrays
        (hasattr(result, 'ndim') and result.ndim == 0)        # NumPy scalars
    )
    
    if not is_scalar:
        raise ValueError(
            f"Estimation function '{func_name}' must return a scalar, "
            f"got type {type(result)}"
        )
    if verbose:
        print(f"✓ Custom function '{func_name}' validated successfully")
    return True

def validate_variable_function(func, func_name, fd,
                               verbose, veryverbose, rel_kwargs):
    """Validate that variable function has correct signature and behaviour.
    
    Parameters
    ----------
    func : callable
        Function to validate
    func_name : str
        Name of the function for error messages
    fd : FiniteDifference
        Finite difference object to get grid dimensions
    verbose : bool
        If True, prints debug information about the validation process. 
    veryverbose : bool
        If True, prints aurelcore verbose information.
    rel_kwargs : dict
        Additional parameters passed to AurelCore initialization.
        
    Returns
    -------
    bool
        True if function is valid
    """
    if verbose:
        print(f"Validating variable function '{func_name}'...", flush=True)

    # Check if it's callable
    if not callable(func):
        raise ValueError(f"Variable function '{func_name}' must be callable")
    
    # Check function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    if len(params) != 1:
        raise ValueError(
            f"Variable function '{func_name}' must take exactly 1 parameter,"
            f" the AurelCore class, instead got {len(params)}: {params}"
        )
    
    # Test with dummy AurelCore instance
    rel = core.AurelCore(fd, verbose=veryverbose, **rel_kwargs)
    try:
        rel.data[func_name] = func(rel)
    except Exception as e:
        raise ValueError(
            f"Variable function '{func_name}' failed when called with "
            f"AurelCore instance: {e}"
        )
    
    if verbose:
        print(f"✓ Custom function '{func_name}' validated successfully")
    return True