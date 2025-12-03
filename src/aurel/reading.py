"""
reading.py

This module provides comprehensive functionality for reading and writing 
numerical relativity simulation data, with specialized support for 
Einstein Toolkit (ET) simulations. The module handles complex data structures 
including multiple restart files, chunked data, variable groupings, and 
different refinement levels.

Key functions relevant for Aurel: ``read_data`` and ``save_data``.
All other functions are auxiliary utilities for Einstein Toolkit data handling.

Core Einstein Toolkit Functionality
-----------------------------------
- **Parameter extraction**: Read simulation parameters from .par files
- **Iteration management**: List and track available simulation iterations
- **Content listing**: List available variables and their file associations
- **Data reading**: Read 3D simulation data from HDF5 files
- **Data chunking**: Join data chunks from distributed file systems
- **Caching**: Save/load data in per-iteration formats for faster access

Environment Setup
-----------------
Users must set the SIMLOC environment variable to the simulation directories:

    ``export SIMLOC="/path/to/simulations/"``

For multiple simulation locations, use colon separation:

    ``export SIMLOC="/path1:/path2:/path3"``
"""

import os
import numpy as np
import glob
import h5py
import re
import contextlib
import json

aurel_tensor_to_scalar = {
    'gammadown3': ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'],
    'Kdown3': ['kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz'],
    'betaup3': ['betax', 'betay', 'betaz'],
    'dtbetaup3': ['dtbetax', 'dtbetay', 'dtbetaz'],
    'velup3': ['velx', 'vely', 'velz'],
    'Momentumup3': ['Momentumx', 'Momentumy', 'Momentumz'],
    'Weyl_Psi': ['Weyl_Psi4r', 'Weyl_Psi4i'],
}

def transform_vars_tensor_to_scalar(var):
    """Transform Aurel tensor variable names to their scalar component names.

    This function decomposes tensor quantities into their individual scalar
    components for detailed analysis. For example, the 3-metric 'gammadown3'
    becomes ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'].

    Parameters
    ----------
    var : list of str
        List of Aurel tensor variable names to transform.

    Returns
    -------
    list of str
        List of scalar variable names. Tensor variables are expanded to their
        components, while scalar variables are kept as-is.
    """
    var_scalars = []
    for v in var:
        if v in list(aurel_tensor_to_scalar.keys()):
            var_scalars += aurel_tensor_to_scalar[v]
        else:
            var_scalars += [v]
    return var_scalars

def read_data(param, **kwargs):
    """Read simulation data from Aurel or Einstein Toolkit format files.

    This is the main entry point for reading numerical relativity simulation 
    data. It automatically detects the data format and routes to the 
    appropriate reading function. Supports both Einstein Toolkit HDF5 output 
    and Aurel's optimized per-iteration format.

    Parameters
    ----------
    param : dict
        Simulation parameters dictionary containing metadata about the 
        simulation. If 'simulation' key is present, treats as Einstein Toolkit
        data; otherwise treats as Aurel format. In Einstein Toolkit case, you
        can use the parameters function.

    Other Parameters
    ----------------
    it : list of int, optional
        Iteration numbers to read. Default [0].

    vars : list of str, optional  
        Variable names to read. Default [] (read all available).

    rl : int, optional
        Refinement level to read. Default 0 (coarsest level).

    restart : int, optional
        Specific restart number to read from. Default -1 (auto-detect).
        Set to specific value to read from that restart only.

    split_per_it : bool, optional
        For ET data: whether to use per-iteration files. Default True.
        When True, reads from cached files and fills missing data from ET 
        files, then saves newly read ET data to per-iteration files.
        When False, reads exclusively from ET files.

    usecheckpoints : bool, optional
        For ET data: whether to use checkpoint iterations if available. 
        Default False.

    verbose : bool, optional
        Print detailed progress information. Default False.

    veryverbose : bool, optional
        Print very detailed debugging information. Default False.

    Returns
    -------
    dict
        Dictionary containing the simulation data with keys::

            {
                'it': [iteration_numbers], # ints
                't': [time_values],        # floats or None if not available
                'var1': [data_arrays],     # arrays or Nones if not available
                'var2': [data_arrays],
                ...
            }
    """
    it = sorted(list(set(kwargs.get('it', [0]))))
    if len(it) == 0:
        raise ValueError(
            'it can not be an empty list. '
            + 'Please provide at least one iteration number to read.')

    # Determine data format and route to appropriate reader
    # Einstein Toolkit data has 'simulation' key in parameters
    if 'simulation' in param.keys():
        data = read_ET_data(param, **kwargs)
    else:
        # Aurel format data (per-iteration HDF5 files)
        data = read_aurel_data(param, **kwargs)
    return data

def read_aurel_data(param, **kwargs):
    """Read data from Aurel format simulation output files.

    Aurel format stores simulation data in per-iteration HDF5 files with the 
    naming convention 'it_<iteration>.hdf5'. Each file contains variables 
    organized by refinement level with keys like 'varname rl=0'.

    This format is optimized for easy parallelization across iterations.

    Parameters
    ----------
    param : dict
        Simulation parameters dictionary. Must contain either:
        - 'datapath': Direct path to directory with it_*.hdf5 files, OR
        - 'simulation', 'simpath', 'simname': For ET-style directory structure

    Other Parameters
    ----------------
    it : list of int, optional
        Iteration numbers to read. Default [0].
        
    vars : list of str, optional
        Variable names to read. Default [] (read all available variables).
        
    rl : int, optional
        Refinement level to read. Default 0.
        
    restart : int, optional
        Restart number (for ET-style directory structure). Default 0.
        
    verbose : bool, optional
        Print progress information. Default False.
        
    veryverbose : bool, optional
        Print detailed debugging information. Default False.

    Returns
    -------
    dict
        Dictionary with simulation data::

            {
                'it': [iteration_numbers], # ints
                't': [time_values],        # floats or None if not available
                'var1': [data_arrays],     # arrays or Nones if not available
                'var2': [data_arrays],
                ...
            }
    """
    # Extract reading parameters with defaults
    it = sorted(list(set(kwargs.get('it', [0]))))
    rl = kwargs.get('rl', 0)
    restart = kwargs.get('restart', 0)
    var = list(set(kwargs.get('vars', [])))
    verbose = kwargs.get('verbose', False)
    veryverbose = kwargs.get('veryverbose', False)

    # Determine if we should read all variables or specific ones
    if var == []:
        get_them_all = True
    else:
        get_them_all = False
        
    if veryverbose:
        print('read_aurel_data: looking for it ', it)
        if get_them_all:
            print('read_aurel_data: reading all variables available')
        else:
            print('read_aurel_data: reading variables {}'.format(var))

    # Construct data directory path based on parameter structure
    if 'simulation' in param.keys():
        # Einstein Toolkit style directory structure
        datapath = (param['simpath']
                    + param['simname']
                    + '/output-{:04d}/'.format(restart)
                    + param['simname']
                 + '/all_iterations/')
    else:
        # Direct path to Aurel format files
        datapath = param['datapath']
        
    # Always include time coordinate in variables to read
    var += ['t']
    # Initialize data dictionary with empty lists for each variable
    data = {'it': np.array(it), **{v: [] for v in var}}
    
    # Read data for each requested iteration
    for it_index, iit in enumerate(it):
        fname = '{}it_{}.hdf5'.format(datapath, int(iit))
        
        # Handle missing iteration files gracefully
        if not os.path.exists(fname):
            # Record None values for all variables at this iteration
            for key in var:
                data[key].append(None)
        else:
            # File exists - read data from HDF5 file
            with h5py.File(fname, 'r') as f:
                if get_them_all:
                    # Scan file for variables at this rl
                    for key in f.keys():
                        if ' rl={}'.format(rl) in key:
                            var += [key.split(' rl')[0]]
                    var = list(set(var))  # Remove duplicates
                    
                # Read each requested variable
                for key in var:
                    skey = key + ' rl={}'.format(rl)  # HDF5 dataset key
                    
                    # Ensure this variable exists in our data dictionary
                    if key not in data.keys():
                        data[key] = [None]*it_index # Fill iterations with None
                        
                    # Read data if dataset exists, otherwise store None
                    if skey in list(f.keys()):
                        data[key].append(np.array(f[skey]))
                    else:
                        data[key].append(None)
    return data

def save_data(param, data, **kwargs):
    """Save simulation data to Aurel format HDF5 files.

    Saves simulation data in the per-iteration format used by Aurel, where each
    iteration is stored in a separate HDF5 file named 'it_<iteration>.hdf5'.
    Variables are stored as datasets with keys like 'varname rl=<level>'.

    This format provides several advantages:

    - Fast random access to specific iterations
    - Parallel I/O capabilities 
    - Efficient storage for sparse temporal sampling

    Parameters
    ----------
    param : dict
        Simulation parameters dictionary containing path information.
        Should include either:
        - 'datapath': Direct path where files will be saved, OR  
        - 'simulation' 'simpath', 'simname': For ET-style directory structure

    data : dict
        Dictionary containing simulation data to save.
        Expected structure::

            {
                'it': [iteration_numbers],
                't': [time_values], 
                'var1': [data_arrays],
                'var2': [data_arrays],
                ...
            }

        Each variable should have one array per iteration.

    Other Parameters
    ----------------

    vars : list of str, optional
        Variables to save. Default [] (save all variables in data).
        Note: 'it' and 't' are automatically included if present.

    it : list of int, optional  
        Iterations to save. Default [0].

    rl : int, optional
        Refinement level for saved data. Default 0.
        Used in HDF5 dataset keys: 'varname rl=<rl>'
        Can just represent grids of different shapes.

    restart : int, optional
        Restart number (for ET-style paths). Default 0.

    Notes
    -----
    - Overwrites existing datasets with the same name
    - Skips variables with None values
    """
    vars = kwargs.get('vars', [])
    it = sorted(list(set(kwargs.get('it', [0]))))
    rl = kwargs.get('rl', 0)
    restart = kwargs.get('restart', 0)

    if vars == []:
        vars = list(data.keys())

    if 'it' not in vars and 'it' in data.keys():
        vars += ['it']
    if 't' not in vars and 't' in data.keys():
        vars += ['t']

    # check paths
    if 'simulation' in param.keys():
        # ET-style directory structure
        datapath = (param['simpath']
                 + param['simname']
                 + '/output-{:04d}/'.format(restart)
                 + param['simname']
                 + '/all_iterations/')
        lastpart = ''
    else:
        datapath = param['datapath']
        if not datapath.endswith('/'):
            lastpart = datapath.split('/')[-1]
            firstpart = datapath.split('/')[:-1]
            datapath = '/'.join(firstpart) + '/'
        else:
            lastpart = ''
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    datapath += lastpart

    # save the data
    for it_index, iit in enumerate(it):
        fname = '{}it_{}.hdf5'.format(datapath, int(iit))
        with h5py.File(fname, 'a') as f:
            for key in vars:
                if data[key] is not None:
                    skey = key + ' rl={}'.format(rl)
                    # TODO: are you sure about overwritting?
                    if skey in list(f.keys()):
                        del f[skey]
                    f.create_dataset(skey, data=data[key][it_index])

###############################################################################
###############################################################################
#
#                  Einstein Toolkit specific functions
#
###############################################################################
###############################################################################

# =============================================================================
#                   Aurel to Einstein Toolkit variables
# =============================================================================

aurel_to_ET_varnames = {
    'gammadown3' : ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'],
    'Kdown3' : ['kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz'],
    'Ktrace' : ['trK'],
    'alpha' : ['alp'],
    'dtalpha' : ['dtalp'],
    'betaup3' : ['betax', 'betay', 'betaz'],
    'dtbetaup3' : ['dtbetax', 'dtbetay', 'dtbetaz'],
    'rho0' : ['rho'],
    'eps' : ['eps'],
    'press' : ['press'],
    'w_lorentz' : ['w_lorentz'],
    'velup3' : ['vel[0]', 'vel[1]', 'vel[2]'],
    'velx' : ['vel[0]'],
    'vely' : ['vel[1]'],
    'velz' : ['vel[2]'],
    'Hamiltonian' : ['H'],
    'Momentumup3' : ['M1', 'M2', 'M3'],
    'Momentumx' : ['M1'],
    'Momentumy' : ['M2'],
    'Momentumz' : ['M3'],
    'Weyl_Psi' : ['Psi4r', 'Psi4i'],
    'Weyl_Psi4r' : ['Psi4r'],
    'Weyl_Psi4i' : ['Psi4i']
}
def transform_vars_aurel_to_ET(var):
    """Transform ET variable names to Aurel variable names.

    Parameters
    ----------
    var : list of str
        List of variable names to transform.

    Returns
    -------
    list of str
        List of transformed variable names.
    """
    var_ET = []
    for v in var:
        if v in list(aurel_to_ET_varnames.keys()):
            var_ET += aurel_to_ET_varnames[v]
        else:
            var_ET += [v]
    return var_ET

def transform_vars_ET_to_aurel_groups(vars):
    """Transform ET variable names to Aurel variable group names.

    Parameters
    ----------
    var : list of str
        List of variable names to transform.

    Returns
    -------
    list of str
        List of transformed variable names.
    """
    aurel_vars = []
    for newv, oldvs in  aurel_to_ET_varnames.items():
        if np.sum([v in vars for v in oldvs]) == len(oldvs):
            aurel_vars += [newv]
            for oldv in oldvs:
                vars.remove(oldv)
    aurel_vars += vars
    return aurel_vars

ET_to_aurel_varnames = {
    'rho':'rho0', 'alp':'alpha', 'dtalp':'dtalpha',
    'trK':'Ktrace', 'H':'Hamiltonian',
    'vel[0]':'velx', 'vel[1]':'vely', 'vel[2]':'velz',
    'M1':'Momentumx', 'M2':'Momentumy', 'M3':'Momentumz',
    'Psi4r':'Weyl_Psi4r', 'Psi4i':'Weyl_Psi4i'
}
def transform_vars_ET_to_aurel(var):
    """Transform ET variable name to Aurel variable name.

    Parameters
    ----------
    var : str
        String of variable name to transform.

    Returns
    -------
    str
        String of transformed variable name.
    """
    if var in list(ET_to_aurel_varnames.keys()):
        var = ET_to_aurel_varnames[var]
    return var

# =============================================================================
#                Reading data file names and hdf5 dataset keys
# =============================================================================

# Regex for HDF5 dataset keys
# Example: "ADMBASE::gxx it=0 tl=0 m=0 rl=0 c=0"
rx_key = re.compile(
    r"([^:]+)::(\S+) it=(\d+) tl=(\d+)( m=0)?( rl=(\d+))?( c=(\d+))?")

def parse_hdf5_key(key):
    """Parse HDF5 dataset key into its components.
    
    Parameters
    ----------
    key : str
        HDF5 dataset key in format like "admbase::gxx it=0 tl=0 m=0 rl=0 c=0"
    
    Returns
    -------
    dict or None
        Returns None if the key doesn't match the expected format.
        Otherwise returns dictionary with parsed components::

            {
                'thorn': str,        # e.g., 'admbase'
                'variable': str,     # e.g., 'gxx'
                'it': int,          # iteration number
                'tl': int,          # time level
                'm': int or None,   # m value (if present)
                'rl': int or None,  # refinement level (if present) 
                'c': int or None    # chunk number (if present)
            }
            
    """
    
    match = rx_key.match(key)
    if match:
        return {
            'thorn': match.group(1),
            'variable': match.group(2),
            'it': int(match.group(3)) if match.group(3) else None,
            'tl': int(match.group(4)) if match.group(4) else None,
            'm': 0 if match.group(5) else None,
            'rl': int(match.group(7)) if match.group(7) else None,
            'c': int(match.group(9)) if match.group(9) else None,
            'combined variable name': match.group(1) + '::' + match.group(2)
        }
    return None

# Regex for HDF5 filenames: matches both single-variable and group files
# Examples: "rho.xyz.h5", "admbase-metric.file_123.xyz.h5"
rx_h5file = re.compile(
    r"^(([a-zA-Z0-9_]+)-)?([a-zA-Z0-9\[\]_]+)(.xyz)?(.file_(\d+))?(.xyz)?.h5$")

rx_checkpoint = re.compile(
    r"^checkpoint\.chkpt\.it_(\d+)(\.file_(\d+))?.h5$")

def parse_h5file(filepath):
    """Parse HDF5 filename into its components.
    
    Parameters
    ----------
    filepath : str
        HDF5 file path or filename. Can be either:

        - Full absolute/relative path: "/path/to/admbase-metric.h5"
        
        - Just filename: "rho.xyz.h5"

    Returns
    -------
    dict or None
        Returns None if the filename doesn't match the expected format.
        Otherwise returns dictionary with parsed components::

            {
                'thorn_group': str or None, # e.g., 'admbase-' or None
                'thorn': str or None,       # e.g., 'admbase' or None  
                'variable_or_group': str,   # e.g., 'metric' or 'rho'
                'xyz_prefix': str or None,  # e.g., '.xyz' or None
                'chunk_number': int or None # e.g., 123 or None (from file_nbr)
                'xyz_suffix': str or None,  # e.g., '.xyz' or None
                'is_group_file': bool,      # True -> a grouped variable file
            }

        And if it is a checkpoint file::

            {
                'iteration': int,        # e.g., 1000
                'chunk_number': int or None # e.g., 123 or None (from file_nbr)
            }

    """
    
    # Extract just the filename if a full path was provided
    if os.path.sep in filepath:
        filename = os.path.basename(filepath)
    else:
        filename = filepath
    
    match_checkpoint = rx_checkpoint.match(filename)
    match = rx_h5file.match(filename)
    if match_checkpoint:
        iteration = int(match_checkpoint.group(1)) if match.group(1) else None
        chunk_number = int(match_checkpoint.group(3)) if match.group(3) else None
        return {'iteration': iteration,
                'chunk_number': chunk_number}
    elif match:
        thorn_group_with_dash = match.group(1)  # e.g., "admbase-" or None
        thorn_name = match.group(2)             # e.g., "admbase" or None
        variable_or_group_name = match.group(3) # e.g., "metric" or "rho"
        xyz_prefix = match.group(4)             # e.g., ".xyz" or None
        chunk_number = int(match.group(6)) if match.group(6) else None
                                                # e.g., "123" or None
        xyz_suffix = match.group(7)             # e.g., ".xyz" or None
        
        base_name = f"{thorn_group_with_dash}{variable_or_group_name}"
        return {
            'thorn_with_dash': thorn_group_with_dash,
            'thorn': thorn_name,
            'variable_or_group': variable_or_group_name,
            'base_name': base_name,
            'xyz_prefix': xyz_prefix,
            'chunk_number': chunk_number,
            'xyz_suffix': xyz_suffix,
            'group_file': thorn_group_with_dash is not None
        }
    return None

# =============================================================================
#           Reading parameters, iterations and list data content
# =============================================================================

def parameters(simname):
    """Read and parse Einstein Toolkit simulation parameters.

    Extracts comprehensive simulation metadata from Einstein Toolkit parameter
    files (.par), including grid configuration, restart information, and all
    thorn-specific parameters.

    The function automatically:
    - Locates parameter files
    - Counts restart files to determine simulation state
    - Calculates derived grid parameters (extents, grid points)
    - Parses thorn parameters with proper type conversion

    Parameters
    ----------
    simname : str
        Name of the simulation directory to analyze.
        Must exist in one of the SIMLOC directories.

    Returns
    -------
    dict
        Comprehensive parameter dictionary containing:

        Core Information:
            'simname' : str, Simulation name (input parameter)

            'simulation' : str, Data format identifier ('ET')

            'simpath' : str, Path to simulation root directory
            
        Grid Parameters:
            'xmin', 'xmax', 'Lx', 'Nx', 'dx' : float/int, minimum, maximum, 
            extent, grid points and spacing. Same for Y and Z-direction 
            parameters.
            
        Thorn Information:
            'list_of_thorns' : list of str, All active thorns in the simulation
            
        Parameter Values:
            All parameters from the .par file with appropriate types.
            Conflicting parameter names include thorn prefix.

    Notes
    -----
    The SIMLOC environment variable must be set:
    
        ``export SIMLOC="/path/to/simulations"``
        
    For multiple locations:
    
        ``export SIMLOC="/path1:/path2:/path3"``
    """
    parameters = {
        'simname':simname,
        'simulation':'ET'
        }

    # Looking for data
    founddata = False
    simlocs = os.environ.get('SIMLOC', '').split(':')
    if simlocs == ['']:
        raise ValueError(
            'Could not find environment variable SIMLOC. '
            +'Please set it to the path of your simulations.'
            +'`export SIMLOC="/path/to/simulations/"`')
    for simloc in simlocs:
        param_files = sorted(glob.glob(simloc + parameters['simname'] 
                                       + '/output-*/' 
                                       + parameters['simname'] + '.par'))
        if param_files:
            parampath = param_files[0]
            founddata = True
            parameters['simpath'] = simloc
            parameters['datapath'] = (simloc + parameters['simname'] 
                                      + '/output-0000/'
                                      + parameters['simname'] + '/')
            break
    if not founddata:
        raise ValueError('Could not find simulation parameter file for: ' 
                         + simname)

    # save all parameters
    with open(parampath, 'r') as f:
        lines = f.read().split('\n') # read file
    lines = [l.split('#')[0] for l in lines] # remove comments
    lines = [l for l in lines if l!=''] # remove empty lines

    list_of_thorns = []
    for line in lines:
        if '::' in line and '=' in line:
            # split line into thorn, variable and value
            # thorn::variable=value
            # split thorn and rest of line
            line = re.split('::', line)
            thorn = line[0]
            linerest = line[1]
            if len(line)>2:
                for l in line[2:]:
                    linerest += '::' + l
            # split rest of line into variable and value
            # TODO: case where value goes accross multiple lines
            linerest = re.split('=', linerest)
            variable = linerest[0]
            value = linerest[1]
            if len(linerest)>2:
                for l in linerest[2:]:
                    value += '='+l
            thorn = thorn.strip()
            variable = variable.strip()
            value = value.strip()
            
            # format value, number or string
            digitvalue = value.replace('.', '', 1).replace(
                '-', '', 1).replace('+', '', 1).replace('e', '', 1)
            if digitvalue.isdigit():
                if '.' in value or 'e' in value:
                    value = float(value)
                else:
                    value = int(value)
            else:
                if '"' in value:
                    value = value.split('"')[1]
                    
            # save variable and value
            if variable in [
                'verbose',
                'timelevels',
                'evolution_method',
                'out_every', 
                'one_file_per_group']:
                # different thorns use the same variable name
                parameters[thorn + '::' + variable] = value
            else:
                parameters[variable] = value

            # save thorn name
            list_of_thorns += [thorn]
        if 'ActiveThorns' in line:
            thorns = re.split('"', line)[1]
            list_of_thorns += re.split(' ', thorns)
    list_of_thorns = list(set(list_of_thorns)) # remove duplicates
    parameters['list_of_thorns'] = list_of_thorns
                
    # calculate extra grid info
    for coord in ['x', 'y', 'z']:
        L = parameters[coord+'max']-parameters[coord+'min']
        N = int(L/parameters['d'+coord]) - 1 # default not include lower point
        for key in parameters.keys():
            if key in ['boundary_shiftout_'+coord+'_lower', 
                       'boundary_shiftout_'+coord+'_upper']:
                N += parameters[key]
        if ('boundary_shiftout_'+coord+'_lower' 
            not in list(parameters.keys())):
            parameters[coord+'min'] += parameters['d'+coord]
        parameters['L'+coord] = L
        parameters['N'+coord] = N
    if 'max_refinement_levels' not in parameters.keys():
        parameters['max_refinement_levels'] = 1
    return parameters

def saveprint(it_file, some_str, verbose=True):
    """Save the string to the file and print it to the screen."""
    if verbose:
        print(some_str, flush=True)
    with contextlib.redirect_stdout(it_file):
        print(some_str)

def iterations(param, **kwargs):
    """Analyze and catalog available iterations across all simulation restarts.

    This function systematically scans Einstein Toolkit simulation output
    to determine what data is available at each iteration and refinement level
    across all restart files. It creates a comprehensive catalog saved to
    simpath/simname/iterations.txt for quick access by other analysis 
    functions.

    Parameters
    ----------
    param : dict
        Simulation parameters from `parameters()` function. Must contain:

        - 'simpath': Path to simulation directory (with trailing slash)
        - 'simname': Simulation name
        
    skip_last : bool, optional
        Skip the last restart directory. Default True.
        Useful to avoid interrupting running simulations.
        
    verbose : bool, optional
        Print detailed progress information. Default True.
        Shows contents of iterations.txt file and overall summary.

    Returns
    -------
    dict
        Dictionary containing iteration information for all restarts::
        
            {
                restart_number: {
                    'var available': [variable names],
                    'its available': [itmin, itmax], 
                    'rl = 0': [itmin, itmax, dit],
                    'rl = 1': [itmin, itmax, dit],
                    ...
                    'checkpoints': [it1, it2, ...]
                },
                ...
                'overall': {
                    'rl = 0': [[itmin, itmax, dit], ...],
                    'rl = 1': [[itmin, itmax, dit], ...],
                    ...
                }
            }

    Notes
    -----
    Creates/appends to '{simpath}/{simname}/iterations.txt' with format::
    
        === restart 0
        3D variables available: ['rho', 'alpha', 'gxx', ...]
        it = 0 -> 1000
        rl = 0 at it = np.arange(0, 1000, 2)
        rl = 1 at it = np.arange(0, 1000, 1)
        Checkpoints available at its: [0, 10, ...]
        === restart 1
        ...

    Also with verbose=True, prints the above and overall iterations from 
    collect_overall_iterations ::

        === Overall iterations
        rl = 0 at it = [np.arange(0, 2000, 2)]
        rl = 1 at it = [np.arange(0, 2000, 1)]
    """
    skip_last = kwargs.get('skip_last', True)
    verbose = kwargs.get('verbose', True)
    verbose_file = kwargs.get('verbose_file', True)
    
    # Open/create iterations catalog file
    it_filename = param['simpath']+param['simname']+'/iterations.txt'
    file_existed_before = os.path.isfile(it_filename)

    if not verbose and verbose_file:
        verbose_file = False

    if verbose:
        if not file_existed_before:
            print('Creating new iterations file: ' + it_filename, 
                  flush=True)
    
    with open(it_filename, "a+") as it_file:
        # Display existing content from previous runs
        it_file.seek(0)
        contents = it_file.read()

        # Create its_available dictionary to store iteration data
        if file_existed_before:
            its_available = read_iterations(
                param, skip_last=skip_last, verbose=verbose)
        else:
            its_available = {}

        # Print existing contents if verbose_file
        if verbose_file:
            print(contents, flush=True)

        # Determine all restarts available
        files = os.listdir(param['simpath'] + param['simname'])
        output_pattern = re.compile(r'^output-(\d+)$')
        relevant_files = [file for file in files 
                          if output_pattern.match(file)]
        all_restarts = sorted([int(fl.split('-')[1]) 
                                for fl in relevant_files])

        # Determine which restarts need processing
        # Cut off the active one
        if skip_last:
            all_restarts = all_restarts[:-1]

        # Cut off what has already been processed
        lines = contents.split("\n")
        restarts_done = [int(line.split("restart ")[1]) 
                         for line in lines 
                         if 'restart' in line]
        all_restarts = [rnbr for rnbr in all_restarts 
                        if rnbr not in restarts_done]
        if all_restarts == [] and restarts_done == []:
            if skip_last:
                raise ImportError(
                    'Nothing to process. Consider running with'
                    + ' skip_last=False to analyse the last restart'
                    + ' (if it is not an active restart).')
            else:
                raise ImportError('Nothing to process.')
        if verbose:
            print('Restarts to process: ' + str(all_restarts), flush=True)
            if all_restarts == [] and skip_last:
                print('Nothing new to process. Consider running with'
                      + ' skip_last=False to analyse the last restart'
                      + ' (if it is not an active restart).', flush=True)

        # Process each restart directory
        for restart in all_restarts:
            saveprint(it_file, ' === restart {}'.format(restart), 
                      verbose=verbose_file)
            its_available[restart] = {}
            
            # Analyze available variables in this restart
            vars_and_files = get_content(param, 
                                         verbose=verbose, restart=restart)
            vars_available = []
            for key in vars_and_files.keys():
                vars_available += list(key)
            
            # Error handling for empty restart directories
            if vars_available == []:
                # Path to this restart's output directory that was checked
                datapath = (param['simpath']+param['simname']
                            +'/output-{:04d}/'.format(restart)
                            +param['simname']+'/')
                saveprint(it_file, 'Could not find 3D data in ' + datapath)
            else:
                # Transform ET variable names to Aurel conventions
                aurel_vars_available = transform_vars_ET_to_aurel_groups(
                    vars_available)
                saveprint(it_file, '3D variables available: '
                          + str(aurel_vars_available), 
                          verbose=verbose_file)
                its_available[restart]['var available'] = aurel_vars_available
                
                # Select representative file for iteration analysis
                # Preferably a light file that only has one variable
                files_with_single_var = [
                    var for var in list(vars_and_files.keys()) 
                    if len(var)==1]
                if files_with_single_var == []:
                    for var_to_read in list(vars_and_files.keys()):
                        file_for_it = vars_and_files[var_to_read][0]
                        try:
                            if verbose:
                                print('Checking if ' + file_for_it
                                      + ' can be read', flush=True)
                            with h5py.File(file_for_it, 'r') as f:
                                fkeys = list(f.keys())
                            file_to_read = True
                            break
                        except OSError:
                            if verbose:
                                print('ERROR: Could not read ' 
                                      + file_for_it, flush=True)
                            file_to_read = False
                            continue
                else:
                    for var_to_read in files_with_single_var:
                        if 'NaNmask' not in var_to_read:
                            file_for_it = vars_and_files[var_to_read][0]
                            try:
                                if verbose:
                                    print('Checking if ' + file_for_it
                                        + ' can be read', flush=True)
                                with h5py.File(file_for_it, 'r') as f:
                                    fkeys = list(f.keys())
                                file_to_read = True
                                break
                            except OSError:
                                if verbose:
                                    print('ERROR: Could not read ' 
                                        + file_for_it, flush=True)
                                file_to_read = False
                                continue
                if file_to_read:
                    saveprint(it_file, 'Reading iterations in: '
                            + file_for_it, verbose=verbose_file)
                    with h5py.File(file_for_it, 'r') as f:
                        
                        # only consider one of the variables in this file
                        fkeys = list(f.keys())
                        for fk in fkeys:
                            varkey = parse_hdf5_key(fk)
                            if varkey is not None:
                                varkey = varkey['variable']
                                break
                        if verbose: print(f'Checking variable {varkey}')
                            
                        # all the keys of this variable
                        fkeys = [k for k in fkeys if varkey in k]
                        
                        # all the iterations
                        allits = np.sort([parse_hdf5_key(k)['it'] 
                                          for k in fkeys])
                        saveprint(
                            it_file, 
                            'it = {} -> {}'.format(
                                np.min(allits), np.max(allits)), 
                            verbose=verbose_file)
                        its_available[restart]['its available'] = [
                            np.min(allits), np.max(allits)]
                            
                        # maximum refinement level present
                        rlmax = np.max([parse_hdf5_key(k)['rl'] 
                                        for k in fkeys])
                        
                        # for each refinement level
                        for rl in range(rlmax+1):
                            # take the corresponding keys
                            keysrl = [k for k in fkeys 
                                    if parse_hdf5_key(k)['rl'] == rl]
        
                            if keysrl!=[]:
                                # Check if there are chunk numbers
                                if parse_hdf5_key(keysrl[0])['c'] is not None:
                                    cs = list(set([parse_hdf5_key(k)['c'] 
                                                for k in keysrl]))
                                    chosen_c = ' c=' + str(np.sort(cs)[-1])
                                else:
                                    chosen_c = ''
                                keysrl = [k for k in keysrl if chosen_c in k]
                                
                                # and look at what iterations they have
                                allits = np.sort([parse_hdf5_key(k)['it'] 
                                                for k in keysrl])

                                rlkey = 'rl = {}'.format(rl)
                                if len(allits)>1:
                                    itkey = 'it = np.arange({}, {}, {})'.format(
                                        np.min(allits), np.max(allits), 
                                        np.diff(allits)[0])
                                    its_available[restart][rlkey] = [
                                        np.min(allits), np.max(allits), 
                                        np.diff(allits)[0]]
                                else:
                                    itkey = 'it = {}'.format(allits)
                                    its_available[restart][rlkey] = allits
                                saveprint(it_file, rlkey+' at '+itkey, 
                                        verbose=verbose_file)
            
            # List checkpoints available
            checkpoint_files = glob.glob(
                param['simpath'] + param['simname']
                + '/output-{:04d}/'.format(restart)
                + param['simname'] + '/checkpoint.chkpt.it_*.h5')
            if checkpoint_files != []:
                if 'file_' in checkpoint_files[0]:
                    checkpoint_files = [
                        cf for cf in checkpoint_files if '.file_0.' in cf]
                
            checkpoint_its = []
            for chkfile in checkpoint_files:
                chk_it = int(chkfile.split('checkpoint.chkpt.it_')[1].split('.')[0])
                checkpoint_its += [chk_it]
            checkpoint_its = sorted(list(set(checkpoint_its)))

            if 'its available' not in its_available[restart].keys():
                its_available[restart]['its available'] = [
                    np.min(checkpoint_its), np.max(checkpoint_its)]
                saveprint(it_file, 'it = {} -> {}'.format(
                    np.min(checkpoint_its), np.max(checkpoint_its)), 
                    verbose=verbose_file)

            its_available[restart]['checkpoints'] = checkpoint_its
            saveprint(it_file, 
                      'Checkpoints available at its: {}'.format(
                          list(checkpoint_its)),
                          verbose=verbose_file)
        # Overall iterations
        its_available = collect_overall_iterations(its_available, verbose_file)
        return its_available

def read_iterations(param, **kwargs):
    """Read and parse the iterations.txt file to extract iteration information.

    This is a helper function for `iterations()`.

    This function loads the iterations catalog created by `iterations()` and
    parses it into a structured dictionary format for programmatic access.
    If the iterations.txt file doesn't exist, then the `iterations()` function
    is called instead.

    Parameters
    ----------
    param : dict
        Simulation parameters from `parameters()` function. Must contain:

        - 'simpath': Path to simulation directory
        - 'simname': Simulation name
        
    skip_last : bool, optional
        Skip the last restart directory when creating iterations.txt.
        Default True. This is passed to `iterations()`.
        
    verbose : bool, optional
        Print detailed parsing information. Default False. 
        Also passed to `iterations()`.

    Returns
    -------
    dict
        Dictionary containing iteration information for all restarts::
        
            {
                restart_number: {
                    'var available': [variable names],
                    'its available': [itmin, itmax], 
                    'rl = 0': [itmin, itmax, dit],
                    'rl = 1': [itmin, itmax, dit],
                    ...
                    'checkpoints': [it1, it2, ...]
                },
                ...
            }
    """
    skip_last = kwargs.get('skip_last', True)
    verbose = kwargs.get('verbose', False)

    it_filename = param['simpath']+param['simname']+'/iterations.txt'
    file_existed_before = os.path.isfile(it_filename)

    # if file does not exist, create it
    if not file_existed_before:
        return iterations(param, skip_last=skip_last, verbose=verbose)
    else:
        # read the file
        if verbose:
            print(f'Reading iterations in {it_filename}', flush=True)
        with open(it_filename, "r") as it_file:
            it_file.seek(0)
            contents = it_file.read()
        # Empty file handling
        if contents == '':
            if verbose:
                print('File is empty', flush=True)
            return {}
        else:
            lines = contents.split("\n")
            # each line is a dictionary entry
            its_available = {}
            for l in lines:
                # higher level key is the restart number
                if ' === restart ' in l:
                    restart_nbr = int(l.split(' === restart ')[1])
                    its_available[restart_nbr] = {}
                # lower level keys
                # variables available
                elif '3D variables available' in l:
                    vars = l.split('3D variables available: [')[1].split(', ')
                    vars = [v.split("'")[1] for v in vars]
                    its_available[restart_nbr]['var available'] = vars
                # iterations available (inclusive interval)
                elif '->' in l:
                    its_available[restart_nbr]['its available'] = [
                        int(l.split(' ')[2]), int(l.split(' ')[4])]
                # iterations available for said refinement level
                elif 'rl = ' in l:
                    rl = l.split('rl = ')[1].split(' ')[0]
                    rlkey = 'rl = ' + rl
                    if 'arange' in l:
                        it_list = [int(l.split('(')[1].split(',')[0]), 
                                   int(l.split(', ')[1]), 
                                   int(l.split(', ')[2].split(')')[0])]
                        its_available[restart_nbr][rlkey] = it_list
                    else:
                        it_list = [int(l.split('[')[1].split(']')[0])]
                        its_available[restart_nbr][rlkey] = it_list
                
                elif 'Checkpoints available at its' in l:
                    chk_its = l.split('Checkpoints available at its: ')[1]
                    chk_its = chk_its.replace('[', '').replace(']', '')
                    if chk_its.strip() == '':
                        its_available[restart_nbr]['checkpoints'] = []
                    else:
                        chk_its = [int(it.strip()) for it in chk_its.split(',')]
                        its_available[restart_nbr]['checkpoints'] = chk_its

            return its_available

def collect_overall_iterations(its_available, verbose):
    """Merge iteration data across all restarts into overall summary.

    This is a helper function for `iterations()`.
    
    This function analyzes the iteration information from individual restart
    and merges them into a comprehensive overview of what iterations are 
    available across the entire simulation. It handles multiple refinement 
    levels and various iteration patterns: overlapping iteration ranges, 
    different time stepping patterns, single iterations. These are merged when
    possible, otherwise they are kept as separate entries.

    Parameters
    ----------
    its_available : dict
        Dictionary with iterations available for each restart, typically
        from `read_iterations()` or `iterations()`.
        
    verbose : bool
        If True, print the overall iterations summary to stdout.

    Returns
    -------
    dict
        The input dictionary with an added 'overall' key containing merged
        iteration information across all restarts::
        
            {
                restart_number: {...},  # Original restart data preserved
                'overall': {
                    'rl = 0': [[itmin, itmax, dit], ...],
                    'rl = 1': [[itmin, itmax, dit], ...],
                    ...
                }
            }
    """
    # Find maximum refinement level across all restarts
    rlmax = 0
    for restart in list(its_available.keys()):
        for key in list(its_available[restart].keys()):
            if 'rl = ' in key:
                rl = int(key.split('rl = ')[1])
                if rl > rlmax:
                    rlmax = rl

    # First collect all the iterations, gradually merging them together
    # Start with base level then gradually increment to consider them all
    rl = 0
    its_available['overall'] = {}
    if verbose:
        print(' === Overall iterations', flush=True)
    rl_to_do = True
    while rl_to_do:
        it_situation = []
        rl_to_do = False 
        # If current rl isn't found then while loop is broken
        for restart in list(its_available.keys()):
            rlkey = 'rl = {}'.format(rl)
            if rlkey in list(its_available[restart].keys()):
                # rl found, so the while loop will be continued afterwards
                rl_to_do = True
                rl_it_situation = its_available[restart][rlkey]
                if it_situation == []:
                    # This is the first iteration segment found
                    it_situation += [list(rl_it_situation)]
                else:
                    prev_rl_it_situation = np.copy(it_situation[-1])
                    if len(prev_rl_it_situation)>1:
                        # Previous restart had an array
                        if len(rl_it_situation)>1:
                            # Current restart has an array
                            if prev_rl_it_situation[2] == rl_it_situation[2]:
                                # They have the same dit so just update itmax
                                it_situation[-1][1] = rl_it_situation[1]
                            else:
                                it_situation += [list(rl_it_situation)]
                        else:
                            # Current restart has just one iteration
                            if rl_it_situation[0] in np.linspace(
                                prev_rl_it_situation[0], 
                                prev_rl_it_situation[1], 
                                prev_rl_it_situation[2]):
                                pass
                            elif (abs(rl_it_situation[0]
                                      - prev_rl_it_situation[1])
                                  == prev_rl_it_situation[2]):
                                it_situation[-1] = [prev_rl_it_situation[0], 
                                                    rl_it_situation[0], 
                                                    prev_rl_it_situation[2]]
                            else:
                                it_situation += [list(rl_it_situation)]
                    else:
                        # Previous restart has just one iteration
                        if len(rl_it_situation)>1:
                            # Current restart has an array
                            if prev_rl_it_situation[0] in np.linspace(
                                rl_it_situation[0], 
                                rl_it_situation[1], 
                                rl_it_situation[2]):
                                it_situation[-1] = list(rl_it_situation)
                            elif (abs(rl_it_situation[0] 
                                      - prev_rl_it_situation[0]) 
                                  == rl_it_situation[2]):
                                it_situation[-1] = [prev_rl_it_situation[0], 
                                                    rl_it_situation[1], 
                                                    rl_it_situation[2]]
                            else:
                                it_situation += [list(rl_it_situation)]
                        else:
                            # Current restart has just one iteration
                            if prev_rl_it_situation[0] == rl_it_situation[0]:
                                pass
                            else:
                                it_situation += [list(rl_it_situation)]
        
        if rl_to_do:
            its_available['overall'][rlkey] = it_situation
            if verbose:
                # Create a nice string to present them to the user
                # Start the string
                if len(it_situation) == 1:
                    it_situation_string = ''
                else:
                    it_situation_string = '['
                # Add each iteration segment
                for iit_situation in it_situation:
                    if len(iit_situation)>1:
                        # If it's an array, print np.arange
                        it_situation_string += 'np.arange({}, {}, {})'.format(
                            iit_situation[0], iit_situation[1], 
                            iit_situation[2])
                    else:
                        # If it's just one then give it directly
                        it_situation_string += '{}'.format(iit_situation[0])
                    # End string or prepare for next segment
                    if iit_situation == it_situation[-1]:
                        if len(it_situation) != 1:
                            it_situation_string += ']'
                    else:
                        it_situation_string += ', '
                
                # Print overall iterations of said refinement level
                print(rlkey, 'at it =', it_situation_string, flush=True)
        rl += 1
        if rl > rlmax:
            rl_to_do = False
        else:
            rl_to_do = True
    return its_available

known_groups = {
    'hydrobase-rho': ['rho'],
    'hydrobase-eps': ['eps'],
    'hydrobase-press': ['press'],
    'hydrobase-w_lorentz': ['w_lorentz'],
    'hydrobase-vel': ['vel[0]', 'vel[1]', 'vel[2]'],
    'admbase-lapse': ['alp'],
    'admbase-dtlapse': ['dtalp'],
    'admbase-shift': ['betax', 'betay', 'betaz'],
    'admbase-dtshift': ['dtbetax', 'dtbetay', 'dtbetaz'],
    'admbase-metric': ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'],
    'admbase-curv': ['kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz'],
    'ml_bssn-ml_trace_curv': ['trK'],
    'ml_bssn-ml_ham': ['H'],
    'ml_admconstraints-ml_ham': ['H'],
    'ml_bssn-ml_mom': ['M1', 'M2', 'M3'],
    'ml_admconstraints-ml_mom': ['M1', 'M2', 'M3'],
    'weylscal4-psi4r_group': ['Psi4r'],
    'weylscal4-psi4i_group': ['Psi4i'],
    'cosmolapse-kthreshold': ['Ktransition'],
    'cosmolapse-propertime': ['tau'],
    'carpetreduce-weight': ['weight']
    }
def get_content(param, **kwargs):
    """Analyze Einstein Toolkit output directory and map variables to files.

    This function creates a comprehensive mapping between simulation variables
    and their corresponding HDF5 files, handling both single-variable and
    grouped-variable file organizations. It implements intelligent caching
    to avoid expensive file scanning on repeated calls.

    The function performs several key operations:
    1. Scans all HDF5 files in the directory
    2. Determines file organization (single vs. grouped variables)
    3. Maps each variable to its containing files
    4. Groups variables that share the same file sets
    5. Caches results for faster subsequent access

    Parameters
    ---------- 
    param : dict
        Simulation parameters dictionary containing:

        - 'simpath': Path to simulation root directory
        - 'simname': Simulation name
    restart : int, optional
        Restart number to analyze. Default 0. Used to construct the path to 
        the ET output directory containing .h5 files.
        Example: "/simulations/BHB/output-0000/BHB/"

    overwrite : bool, optional
        Overwrite existing content file. Default False.
        
    verbose : bool, optional
        Print progress information during processing. Default True.
        Shows file scanning progress and caching status.
        
    veryverbose : bool, optional
        Print extra progress information during processing. Default False.

    Returns
    -------
    dict
        Dictionary mapping variable groups to their file lists::

        {
            ('var1', 'var2'): ['path/to/file1.h5', 'path/to/file2.h5'],
            ('var3',): ['path/to/file3.h5', 'path/to/file4.h5'],
            ...
        }

    Notes
    -----
    - Cache file 'content.txt' created in the target directory
    - Variable names follow Einstein Toolkit conventions
    - If variables are grouped in files and are not in the known_groups, 
      the variable names will be retrieved from the keys in the file. 
      Depending on the file size, this may take some time.
    """
    restart = kwargs.get('restart', 0)
    overwrite = kwargs.get('overwrite', False)
    verbose = kwargs.get('verbose', True)
    veryverbose = kwargs.get('veryverbose', False)

    path = (param['simpath']
            + param['simname']
            + '/output-{:04d}/'.format(restart)
            + param['simname'] + '/')
    content_file = path+'content.txt'

    if overwrite:
        if verbose:
            print(f"Removing content file from {content_file}...")
        if os.path.exists(content_file):
            os.remove(content_file)
    
    # Try to load existing content file
    try:
        if verbose:
            print(f"Loading existing content from {content_file}...")
        with open(content_file, 'r') as f:
            content_data = json.load(f)
            # Convert keys back to tuples of variable names
            vars_and_files = {}
            for key_str, files in content_data.items():
                # Convert key string back to tuple of variable names
                key_tuple = tuple(key_str.split(','))
                vars_and_files[key_tuple] = files
        if verbose:
            print(f"Loaded {len(vars_and_files)} variables from cache.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing content file found or invalid format."
              + " Calculating from scratch...")
        
        vars_and_files = {}
        processed_groups = {}  # Track which groups we've already read
        h5files = glob.glob(path+'*.h5')
        # skip checkpoint files
        h5files = [f for f in h5files if 'checkpoint.chkpt' not in f]
        for filepath in h5files:
            file_info = parse_h5file(filepath)

            if file_info is not None:
                if veryverbose:
                    print('Processing: ', filepath, flush=True)

                # If the file is a single variable file
                if not file_info['group_file']:
                    if veryverbose:
                        print('Single variable file', flush=True)
                    files = vars_and_files.setdefault(
                        file_info['variable_or_group'], set())
                    files.add(filepath)
                # Else the file is a group of variables
                else:
                    if veryverbose:
                        print('Grouped variable file', flush=True)
                    base_name = file_info['base_name']
                    if base_name not in processed_groups:
                        # If it's a known group I can skip reading the file
                        if base_name in known_groups:
                            varnames = known_groups[base_name]
                        else:
                            # First time seeing this group 
                            # - read the file to get variable names
                            if verbose:
                                print('Reading variables in file:', 
                                      filepath, flush=True)
                                print(f"Consider adding {base_name} to"
                                      + " known_groups in reading.py for"
                                      + " faster processing.", flush=True)
                            with h5py.File(filepath, "r") as h5f:
                                variables = {parse_hdf5_key(k)['variable'] 
                                             for k in h5f.keys() 
                                             if parse_hdf5_key(k) is not None}
                                varnames = list(variables)
                                processed_groups[base_name] = varnames
                    else:
                        # We've already processed this group 
                        # - reuse the variable names
                        varnames = processed_groups[base_name]
                    
                    if veryverbose:
                        print('Found the variables:', varnames, flush=True)
                    # Add this file to all variables in the group
                    for variable_name in varnames:
                        files = vars_and_files.setdefault(variable_name, set())
                        files.add(filepath)
        # Convert sets to lists
        vars_and_files = {var: list(files) 
                          for var, files in vars_and_files.items()}
        
        # ==== Group variables by shared file paths ====
        # Create a mapping from file paths 
        # (as tuples for hashability) to variables
        files_to_vars = {}
        for var, file_list in vars_and_files.items():
            # Convert list to tuple so it can be used as a dictionary key
            file_tuple = tuple(sorted(file_list))
            
            if file_tuple not in files_to_vars:
                files_to_vars[file_tuple] = []
            files_to_vars[file_tuple].append(var)
        
        # Invert the mapping: group variables by their shared file paths
        vars_and_files = {}
        for file_tuple, var_list in files_to_vars.items():
            # Sort variables for consistent ordering
            var_tuple = tuple(sorted(var_list))
            # Convert file tuple back to list
            file_list = list(file_tuple)
            vars_and_files[var_tuple] = file_list

        # ==== Save results ====
        # Always save the results (whether calculated 
        # or loaded and potentially updated)
        if verbose:
            print(f"Saving content to {content_file}...")
        with open(content_file, 'w') as f:
            # convert keys to strings for JSON serialization
            content_data = {}
            for key, value in vars_and_files.items():
                content_data[','.join(key)] = value
            json.dump(content_data, f, indent=2)
        if verbose:
            print(f"Saved {len(vars_and_files)} variables to content file.")
    return vars_and_files


# =============================================================================
#                             Reading in the data
# =============================================================================

def read_ET_data(param, **kwargs):
    """Read Einstein Toolkit simulation data with intelligent restart handling.

    This is the main Einstein Toolkit data reading function that handles the
    complexity of multi-restart simulations, variable groupings, and chunking.
    It provides a unified interface for accessing ET simulation data
    regardless of the underlying file organization.

    Key Features:

    - **Restart Management**: Automatically finds iterations across restarts
    - **Hybrid Reading**: Combines cached per-iteration files with ET files
    - **Missing Data Handling**: Fills gaps by reading from original ET files
    - **Variable Mapping**: Handles Aurel to Einstein Toolkit name conversions
    - **Caching Strategy**: Saves data in optimized format for future access
    
    Parameters
    ----------
    param : dict
        Simulation parameters from `parameters()` function.
    
    Other Parameters
    ----------------
    it : list of int, optional
        Iterations to read. Default [0].
    
    vars : list of str, optional.
        Variables in Aurel format. Default [] (all available). 
        Examples: ['rho', 'Kdown3'], ['gammadown3', 'alpha']
    
    restart : int, optional
        Specific restart to read from. 
        Default -1 (auto-detect).
    
    split_per_it : bool, optional
        Use cached per-iteration files when available and save read data
        in per-iteration files. Default True.
    
    usecheckpoints : bool, optional
        For ET data: whether to use checkpoint iterations if available. 
        Default False.
    
    verbose : bool, optional
        Print progress information. Default True.
    
    veryverbose : bool, optional
        Print detailed debugging information. Default False.
    
    Returns
    -------
    dict
        Simulation data dictionary with structure::
        
            {
                'it': [iteration_numbers],
                't': [time_values],
                'var1': [data_arrays],
                'var2': [data_arrays],
                ...
            }
    
    Notes
    -----
    Processing Workflow:
    
    1. **Iteration Location**: Determines which restart contains each iteration
    2. **Cache Reading**: Loads available data from per-iteration files  
    3. **Gap Detection**: Identifies missing iterations/variables
    4. **ET Reading**: Fills gaps by reading original Einstein Toolkit files
    5. **Data Integration**: Combines cached and newly-read data
    6. **Cache Update**: Saves newly-read data for future access
    """
    # reading kwargs
    it = sorted(list(set(kwargs.get('it', [0]))))
    var = kwargs.get('vars', [])
    restart = kwargs.get('restart', -1)
    split_per_it = kwargs.get('split_per_it', True)
    usecheckpoints = kwargs.get('usecheckpoints', False)
    verbose = kwargs.get('verbose', True)
    veryverbose = kwargs.get('veryverbose', False)

    # =========================================================================
    # ======== set up iterations to get and which restart it's in
    # Overall information
    kwargs['verbose_file'] = False
    its_available = iterations(param, **kwargs)
    # use user provided restart
    if restart >= 0:
        restarts_available = [restart]
        its_available[restart]['it to do'] = []
        for iit in it:
            if usecheckpoints:
                if 'checkpoints' in list(its_available[restart]):
                    if iit in its_available[restart]['checkpoints']:
                        its_available[restart]['it to do'] += [iit]
            else:
                if 'its available' in list(its_available[restart]):
                    if len(its_available[restart]['its available']) ==1:
                        if iit == its_available[restart]['its available'][0]:
                            its_available[restart]['it to do'] += [iit]
                    else:
                        itmin, itmax = its_available[restart]['its available']
                        if ((itmin <= iit) and (iit <= itmax)):
                            its_available[restart]['it to do'] += [iit]
    # find restart myself
    elif restart == -1:
        restarts_available = list(its_available.keys())

        # create new element containing iterations to do
        for restart in list(its_available.keys()):
                its_available[restart]['it to do'] = []

        # reverse order so that I'm always taking the most recent iteration
        for iit in it[::-1]:
            for restart in list(its_available.keys())[::-1]:
                if usecheckpoints:
                    if 'checkpoints' not in list(its_available[restart]):
                        it_in_restart = False
                    else:
                        # is this iteration a checkpoint within this restart?
                        if iit in its_available[restart]['checkpoints']:
                            it_in_restart = True
                        else:
                            it_in_restart = False
                else:
                    if 'its available' not in list(its_available[restart]):
                        it_in_restart = False
                    else:
                        # is this iteration available within this restart?
                        if len(its_available[restart]['its available']) ==1:
                            it_in_restart = (
                                iit == its_available[restart]['its available'][0])
                        else:
                            itmin, itmax = its_available[restart]['its available']
                            it_in_restart = ((itmin <= iit) and (iit <= itmax))
                if it_in_restart:
                    its_available[restart]['it to do'] += [iit]
                    # I found it, so break to not go through the other restart
                    break

        # sort iterations
        for restart in list(its_available.keys()):
                its_available[restart]['it to do'] = list(np.sort(
                    its_available[restart]['it to do']))
    # can't read restart value
    else:
        raise ValueError(
            "Don't know what to do with restart={}".format(restart))

    # =========================================================================
    # ======== go collect data in each restart
    # big dictionary to save data and to be flattened
    datar = {}
    old_it = it.copy()
    for restart in restarts_available:
        # iterations available in this restart
        it = its_available[restart]['it to do']
        # skip this restart if it doesn't have the it we want
        if it == []:
            pass
        # go collect data
        else:
            # ====== set things up
            kwargs['it'] = it

            # restart
            kwargs['restart'] = restart
            vars_and_files = get_content(
                param, restart=restart, verbose=verbose)
            
            # if no variable is specified, take all available
            if var==[]:
                var = its_available[restart]['var available']
            if verbose:
                print(' =========== Restart {}:'.format(restart), 
                        flush=True)
                print('vars to get {}:'.format(var), flush=True)
            if veryverbose:
                print('its to get {}:'.format(it), flush=True)

            # ====== checkpoints
            if usecheckpoints:
                datar[restart] = read_ET_checkpoints(param, var, **kwargs)
            else:
            
                # ====== directly read and provide the data
                if not split_per_it:
                    datar[restart] = read_ET_variables(
                        param, var, vars_and_files, **kwargs)
                    
                # ====== combination of Einstein Toolkit data and aurel data
                # + save into it files for quicker reading next time
                else:
                    # ====== change variable names to scalar elements
                    avar = transform_vars_tensor_to_scalar(var)
                    kwargs['vars'] = avar

                    # =========================================================
                    # ======= Get variables from split iterations files
                    # Read in data
                    datar[restart] = read_aurel_data(param, **kwargs)
                    if verbose:
                        verbose_cleaned_dict = {
                            k: v for k, v in datar[restart].items() 
                            if (not (isinstance(v, list) 
                                        and all(x is None for x in v))
                                and k!='it')}
                        print('Data read from split iterations:', 
                                list(verbose_cleaned_dict.keys()), flush=True)
                        
                    # Find iterations missing
                    avart = avar + ['t']
                    its_missing = {v: [] for v in avart}
                    for it_idx, iit in enumerate(it):
                        for av in avart:
                            # Include it if it's not present
                            if datar[restart][av][it_idx] is None:
                                its_missing[av] += [iit]
                            # Do a set to not have repeats
                            its_missing[av] = list(set(its_missing[av]))
                    # Need to sort because of the set
                    for av in avart:
                        its_missing[av] = list(np.sort(its_missing[av]))

                    # Verbose update
                    if veryverbose:
                        verbose_clean_its_missing = {
                            k:v for k, v in its_missing.items() if v!= []}
                        print('Iterations missing:', 
                                verbose_clean_its_missing, flush=True)
                                

                    # =========================================================
                    # ======== Collect missing data from ET data
                    ETread_vars = []
                    verbose_saved_vars = []

                    # if variables not grouped, read, join and save them
                    # one by one
                    variables_grouped = False
                    for vgroup in vars_and_files.keys():
                        if len(vgroup) > 1:
                            variables_grouped = True
                            break
                    if not variables_grouped:
                        newvar = []
                        for v in var:
                            av = transform_vars_tensor_to_scalar([v])
                            newvar += av
                        var = newvar.copy()

                    # Now process each variable
                    for v in var:
                        # collect missing iterations
                        avar = transform_vars_tensor_to_scalar([v])
                        its_temp = [item for av in avar 
                                    for item in its_missing[av]]
                        its_temp = list(set(its_temp))
                        # if there are missing iterations
                        if its_temp != []:
                            kwargs['it'] = its_temp
                            # retrieve missing iterations
                            data_temp = read_ET_variables(
                                    param, [v], vars_and_files, **kwargs)
                            # verbose update
                            ETread_vars += list(data_temp.keys())
                            if veryverbose:
                                print('Data read from ET files:', 
                                        v, list(data_temp.keys()), flush=True)
                            # update the data with the missing iterations
                            avart = avar + ['t']
                            for av in avart:
                                if data_temp[av] is not None:
                                    for iidx, iit in enumerate(it):
                                        if iit in its_missing[av]:
                                            iidxtem = np.argmin(np.abs(
                                                data_temp['it'] - iit))
                                            datar[restart][av][iidx] = (
                                                data_temp[av][iidxtem])
                                            # don't save this iteration
                                            if (data_temp[av][iidxtem] is None
                                                or av == 't'):
                                                its_missing[av].remove(iit)
                                                
                                    if its_missing[av] != []:
                                        # save the missing iterations
                                        # and save them individually
                                        kwargs['it'] = its_missing[av]
                                        kwargs['vars'] = [av]
                                        verbose_saved_vars += [av]
                                        if veryverbose:
                                            print(
                                                'Saving data for {}'.format(av)
                                                + ' at it = {}'.format(
                                                    its_missing[av]), 
                                                flush=True)
                                        save_data(
                                            param, data_temp,
                                            **kwargs)
                    if verbose:
                        ETread_vars = list(set(ETread_vars))
                        verbose_saved_vars = list(set(verbose_saved_vars))
                        if ETread_vars != []:
                            print('Variables read from ET files:', 
                                ETread_vars, flush=True)
                        if verbose_saved_vars != []:
                            print('Variables saved to split iterations files:',
                                verbose_saved_vars, flush=True)
    
    # create ultimate data dictionary (flattening datar)
    if veryverbose:
        print(' =========== Joining the data from the different restarts')
    restart = list(datar.keys())[0]
    data = {}
    for key in datar[restart].keys():
        data[key] = []
    
    # copy over everything from datar
    for iit in old_it:
        for restart in datar.keys():
            if iit in datar[restart]['it']:
                it_index = np.argmin(abs(datar[restart]['it'] - iit))
                for key in datar[restart].keys():
                    data[key] += [datar[restart][key][it_index]]
    return data

def read_ET_variables(param, var, vars_and_files, **kwargs):
    """Read the data from Einstein Toolkit simulation output files.
    
    Parameters
    ----------
    param : dict
        The parameters of the simulation.
    var : list
        The variables to read from the simulation output files.

    Other Parameters
    ----------------
    it : list, optional
        The iterations to save from the data.
        The default is [0].
    rl : int, optional
        The refinement level to read from the simulation output files.
        The default is 0.
    restart : int, optional
        The restart number to save the data to.
        The default is 0.
    veryverbose : bool, optional
        If True, print additional information during the joining process.
        The default is False.
        
    Returns
    -------
    dict
        A dictionary containing the data from the simulation output files.
        dict.keys() = ['it', 't', var[0], var[1], ...]
    """
    it = sorted(list(set(kwargs.get('it', [0]))))
    veryverbose = kwargs.get('veryverbose', False)

    if veryverbose:
        print('read_ET_variables was given the vars:', var, flush=True)
    
    # data to be returned
    data = {'it':np.array(it), 't':[]}
    
    # are the chunks together in one file or one file per chunk?
    first_var = list(vars_and_files.keys())[0]
    first_file_of_first_var = vars_and_files[first_var][0]
    if 'file_' in first_file_of_first_var:
        # one file per chunk
        cmax = np.max([parse_h5file(fp)['chunk_number'] 
                       for fp in vars_and_files[first_var]])
    else:
        cmax = 'in file'

    # Translate var names from aurel to ET
    var_ET = transform_vars_aurel_to_ET(var)
    if veryverbose:
        print('In read_ET_variables looking for variables:', 
              var_ET, flush=True)

    # Create a dictionary to hold the wanted variables and their files
    vars_wanted = {}
    for v in var_ET:
        var_found = False
        # have I already copied it over?
        for v_want in vars_wanted.keys():
            if v in v_want:
                var_found = True
                break
        # copy it over if I haven't found it yet
        if not var_found:
            for v_avail in vars_and_files.keys():
                if v in v_avail:
                    vars_wanted[v_avail] = vars_and_files[v_avail]
                    var_found = True
                    break
        # if I still haven't found it, print a warning
        # and don't read it
        if not var_found:
            print('Variable {} not found'.format(v))

    # Get all the data
    for v in vars_wanted.keys():
        data.update(read_ET_group_or_var(v, vars_wanted[v], cmax, **kwargs))

    if veryverbose:
        print('In read_ET_variables we found:', list(data.keys()), flush=True)
    return data

def read_ET_group_or_var(variables, files, cmax, **kwargs):
    """Read variables from Einstein Toolkit simulation output files.
    
    Parameters
    ----------
    variables : list
        The variables to read from the simulation output files.
        Each variable is a string that identifies the variable.
        These should all be found in the same files.
    files : list
        The list of files to read the variables from.
        Each file is a string that identifies the file.
        These should all contain the same variables just at different chunks.
    cmax : int
        The maximum number of chunks to read from the simulation output files.
        If 'in file', it will be extracted from the file.

    Other Parameters
    ----------------
    it : list, optional
        The iterations to save from the data.
        The default is [0].
    rl : int, optional
        The refinement level to read from the simulation output files.
        The default is 0.
        
    Returns
    -------
    dict
        A dictionary containing the data from the simulation output files.

        dict.keys() = ['it', 't', var]
    """

    it = sorted(list(set(kwargs.get('it', [0]))))
    rl = kwargs.get('rl', 0)
    veryextraverbose = kwargs.get('veryextraverbose', False)

    if veryextraverbose:
        print('read_ET_group_or_var(variables, files, cmax) activated with:', 
              variables, files, cmax, flush=True)
    
    var_chunks = {iit:{} for iit in it}
    
    # also collect time while you're at it
    time = []
    collect_time = True

    # go through one file at a time
    for filepath in files:
        # open the file and collect data
        file_present = os.path.exists(filepath)
        if file_present:
            with h5py.File(filepath, 'r') as f:
                if veryextraverbose:
                    print('Reading file: {}'.format(filepath), flush=True)
                # Find all keys that are valid
                file_keys = [k for k in f.keys() 
                             if parse_hdf5_key(k) is not None]
                # Collect all it of that file
                for iit in it:
                    relevant_keys = [k for k in file_keys 
                                     if ((parse_hdf5_key(k)['it'] == iit)
                                     and (parse_hdf5_key(k)['rl'] == rl))]
                    if not relevant_keys:
                        raise ValueError(
                            f"Could not find {variables} for it {iit}"
                            + f" and rl {rl} in file: {filepath}")
                        
                    # find the actual number of chunks
                    if cmax == 'in file':
                        if 'c=' in relevant_keys[0]:
                            actual_cmax = np.max(
                                [parse_hdf5_key(k)['c'] 
                                 for k in relevant_keys])
                        else:
                            actual_cmax = 0
                            relevant_keys_with_c = relevant_keys
                        crange = np.arange(actual_cmax+1)
                    else:
                        actual_cmax = cmax
                        crange = [parse_h5file(filepath)['chunk_number']]
                            
                    # for each chunk
                    for c in crange:
                        if actual_cmax!=0: 
                            relevant_keys_with_c = [k for k in relevant_keys 
                                     if ((parse_hdf5_key(k)['c'] == c))]
                             
                        # For each component
                        for vi, v in enumerate(variables):
                            # the full key to use
                            key = [k for k in relevant_keys_with_c
                                   if (parse_hdf5_key(k)['variable'] == v
                                       or parse_hdf5_key(k)[
                                           'combined variable name'] == v)]
                            # should be only one key
                            if len(key) != 1:
                                other = [k.split(' it=')[1] for k in key]
                                is_rest_same = (sum(
                                    [other[i] == other[i+1] 
                                     for i in range(len(other)-1)]) 
                                     == (len(other)-1))
                                if is_rest_same:
                                    mult_vars = [k.split(' it=')[0] 
                                                 for k in key]
                                    variables[vi] = mult_vars[0]
                                    variables += mult_vars[1:]
                                    v = mult_vars[0]

                                    key = [k for k in relevant_keys_with_c
                                           if v in k]
                                    if len(key) != 1:
                                        raise ValueError(
                                            '{}'.format(len(key))
                                            + ' keys found for variable '
                                            + '{} it={} rl={} c={}'.format(
                                                v, iit, rl, c)
                                            + ' found: {}'.format(key))
                                    else:
                                        key = key[0]
                                else:
                                    raise ValueError(
                                        '{}'.format(len(key))
                                        + ' keys found for variable '
                                        + '{} it={} rl={} c={}'.format(
                                            v, iit, rl, c)
                                        + ' found: {}'.format(key))
                            else:
                                key = key[0]
                            
                            # Read in the variable
                            if veryextraverbose:
                                print('Reading key = {}'.format(key), 
                                      flush=True)
                            var_array = np.array(f[key])
                            # Cut off ghost grid points
                            ghost_x = f[key].attrs['cctk_nghostzones'][0]
                            ghost_y = f[key].attrs['cctk_nghostzones'][1]
                            ghost_z = f[key].attrs['cctk_nghostzones'][2]
                            var_array = var_array[ghost_z:-ghost_z, 
                                                  ghost_y:-ghost_y, 
                                                  ghost_x:-ghost_x]
                            iorigin = tuple(f[key].attrs['iorigin'])
                            var_chunks[iit].setdefault(v, {})[iorigin] = var_array
                            del var_array
                    if collect_time:
                        time += [f[key].attrs['time']]
                collect_time = False # only do this in one file
        else:
            print('File {} not found'.format(filepath), flush=True)
            break

    # per iteration
    # join the chunks together + fix the indexing to be x, y, z
    var = {}
    for iit in it:
        for v in variables:
            if veryextraverbose:
                print('Joining chunks for {} at it = {}'.format(v, iit), 
                      flush=True)
            aurel_v = transform_vars_ET_to_aurel(v)
            varlist = var.setdefault(aurel_v, [])
            varlist += [fixij(join_chunks(
                var_chunks[iit][v], **kwargs))]
        var['t'] = time
        
    return var

def read_ET_checkpoints(param, var, it, restart, rl, **kwargs):
    """Read the data from Einstein Toolkit simulation checkpoint files.
    
    Parameters
    ----------
    param : dict
        The parameters of the simulation.
    var : list
        The variables to read from the simulation checkpoint files.
    it : list
        The iterations to read from the simulation checkpoint files.
    restart : int
        The restart number to read from.
    rl : int
        The refinement level to read from the simulation checkpoint files.

    Other Parameters
    ----------------
    verbose : bool, optional
        Print progress information. Default True.
        
    Returns
    -------
    dict
        A dictionary containing the data from the simulation checkpoint files.
        dict.keys() = ['it', 't', var[0], var[1], ...]
    """
    it = sorted(list(set(it)))
    var = transform_vars_aurel_to_ET(var)
    verbose = kwargs.get('verbose', True)
    veryverbose = kwargs.get('veryverbose', False)
    veryextraverbose = kwargs.get('veryextraverbose', False)

    if verbose:
        print('Using checkpoints for restart {}'.format(restart), flush=True)

    checkpoint_files = glob.glob(
        param['simpath'] + param['simname']
        + '/output-{:04d}/'.format(restart)
        + param['simname'] + '/checkpoint.chkpt.it_*.h5')
    
    # find cmax
    it0_file = []
    i = 0
    while it0_file == []:
        it0 = it[i]
        it0_file = [cf for cf in checkpoint_files 
                    if f"it_{it0}." in cf]
        if len(it0_file) == 0:
            pass
        elif len(it0_file) == 1:
            cmax = 'in file'
        else:
            cmax = np.max([parse_h5file(cf)['chunk_number'] 
                           for cf in it0_file])
        i += 1

    # data to be returned
    data = {'it':np.array(it), 't':[]}
    for iit in it:
        if veryverbose:
            print('Reading checkpoint for it={}'.format(iit), flush=True)
        it_file = [cf for cf in checkpoint_files 
                    if f"it_{iit}." in cf]
        if it_file != []:
            collect_time = True
            var_chunks = {}
            for file in it_file:
                with h5py.File(file, 'r') as f:
                    if veryextraverbose:
                        print('Reading checkpoint file: {}'.format(file), 
                            flush=True)
                        
                    # keys 
                    file_keys = [k for k in f.keys() 
                                 if parse_hdf5_key(k) is not None]
                    relevant_keys = [k for k in file_keys 
                                     if ((parse_hdf5_key(k)['it'] == iit)
                                     and (parse_hdf5_key(k)['rl'] == rl)
                                     and (parse_hdf5_key(k)['tl'] == 0))]
                    
                    # max number of chunks
                    if cmax == 'in file':
                        if 'c=' in relevant_keys[0]:
                            nochunks = False
                            actual_cmax = np.max(
                                [parse_hdf5_key(k)['c'] 
                                 for k in relevant_keys])
                        else:
                            nochunks = True
                            actual_cmax = 0
                        crange = np.arange(actual_cmax+1)
                    else:
                        nochunks = False
                        actual_cmax = cmax
                        crange = [parse_h5file(file)['chunk_number']]

                    for vi, v in enumerate(var):
                        varkeys = [k for k in relevant_keys 
                                   if (parse_hdf5_key(k)['variable'] == v
                                       or parse_hdf5_key(k)[
                                           'combined variable name'] == v)]
                        for c in crange:
                            if nochunks:
                                key = varkeys
                            else:
                                key = [k for k in varkeys 
                                        if parse_hdf5_key(k)['c'] == c]
                                
                            # should be only one key
                            if len(key) != 1:
                                other = [k.split(' it=')[1] for k in key]
                                is_rest_same = (sum(
                                    [other[i] == other[i+1] 
                                     for i in range(len(other)-1)]) 
                                     == (len(other)-1))
                                if is_rest_same:
                                    mult_vars = [k.split(' it=')[0] 
                                                 for k in key]
                                    var[vi] = mult_vars[0]
                                    var += mult_vars[1:]
                                    v = mult_vars[0]

                                    key = [k for k in varkeys
                                           if v in k]
                                    if len(key) != 1:
                                        raise ValueError(
                                            '{}'.format(len(key))
                                            + ' keys found for variable '
                                            + '{} it={} rl={} c={}'.format(
                                                v, iit, rl, c)
                                            + ' found: {}'.format(key))
                                    else:
                                        key = key[0]
                                else:
                                    raise ValueError(
                                        '{}'.format(len(key))
                                        + ' keys found for variable '
                                        + '{} it={} rl={} c={}'.format(
                                            v, iit, rl, c)
                                        + ' found: {}'.format(key))
                            else:
                                key = key[0]
                            
                            # Read in the variable
                            if veryextraverbose:
                                print('Reading key = {}'.format(key), 
                                      flush=True)
                            var_array = np.array(f[key])

                            # Cut off ghost grid points
                            ghost_x = f[key].attrs['cctk_nghostzones'][0]
                            ghost_y = f[key].attrs['cctk_nghostzones'][1]
                            ghost_z = f[key].attrs['cctk_nghostzones'][2]
                            var_array = var_array[ghost_z:-ghost_z, 
                                                  ghost_y:-ghost_y, 
                                                  ghost_x:-ghost_x]
                            iorigin = tuple(f[key].attrs['iorigin'])
                            var_chunks.setdefault(v, {})[iorigin] = var_array
                        
                    if collect_time:
                        data['t'] += [f[key].attrs['time']]
                        collect_time = False
                    
            # Join chunks, fix indexing and save in data dictionary
            for v in var:
                if veryextraverbose:
                    print('Joining chunks for {} at it = {}'.format(v, iit), 
                          flush=True)
                aurel_v = transform_vars_ET_to_aurel(v)
                varlist = data.setdefault(aurel_v, [])
                varlist += [fixij(join_chunks(
                            var_chunks[v], **kwargs))]
        else:
            print('Could not find checkpoint file for'
                    + ' it={}'.format(iit), flush=True)
                           
    return data

def fixij(f):
    """Fix the x-z indexing as you read in the data."""
    return np.transpose(np.array(f), (2, 1, 0))

def join_chunks(cut_data, **kwargs):
    """Join the chunks of data together.
    
    Parameters
    ----------
    cut_data : dict of dict
        A dictionary containing the data from the simulation output files.
        dict.keys() = tuple of the chunks 'iorigin' attribute 
        which maps out how the data is to be joined together.

    Other Parameters
    ----------------
    veryextraverbose : bool, optional
        If True, print additional information during the joining process.
        The default is False.
    
    Returns
    -------
    uncut_data : array_like
        The data now joined together.
    """
    veryextraverbose = kwargs.get('veryextraverbose', False)

    all_keys = list(cut_data.keys())
    cmax = len(all_keys) - 1

    if veryextraverbose:
        print(f'There are {cmax} chunks, here are their keys and shape', 
              flush=True)
        for i, k in enumerate(all_keys):
            print(i, k, np.shape(cut_data[k]))
    # =================
    if cmax == 0:
        uncut_data = cut_data[all_keys[0]]
    elif cmax == 1:
        uncut_data = np.append(
            cut_data[all_keys[0]], cut_data[all_keys[1]], axis=0)
    elif cmax == 2:
        uncut_data = np.append(
            np.append(cut_data[all_keys[0]], cut_data[all_keys[1]], axis=1), 
            cut_data[all_keys[2]], axis=0)
    # =================
    else:
        # --- Append along axis = 2
        if veryextraverbose:
            print('Appending along axis = 2', flush=True)
        # First make groups to be appended together
        ndata_groups = {}
        for k in all_keys:
            if k[1:] in list(ndata_groups.keys()):
                ndata_groups[k[1:]][k[0]] = cut_data[k]
            else:
                ndata_groups[k[1:]] = {k[0]:cut_data[k]}
        if veryextraverbose:
            for k in ndata_groups.keys():
                shapes = {key:np.shape(item) 
                          for key, item in ndata_groups[k].items()}
                print(f"Group {k} has elements with shapes: {shapes}", 
                      flush=True)
        
        # Then append them together
        ndata = {}
        for k12 in ndata_groups.keys():
            all_k0 = np.sort(list(ndata_groups[k12].keys()))
            ndata[k12] = ndata_groups[k12][all_k0[0]]
            for k0 in all_k0[1:]:
                if veryextraverbose:
                    print(f"Appending shape {np.shape(ndata[k12])}"
                          + f" with shape {np.shape(ndata_groups[k12][k0])}", 
                          flush=True)
                ndata[k12] = np.append(
                    ndata[k12], ndata_groups[k12][k0], axis=2)
                
        all_keys = list(ndata.keys())
        del cut_data, ndata_groups

        if veryextraverbose:
            print(' === Key and shape after 1st append', flush=True)
            for i, k in enumerate(all_keys):
                print(i, k, np.shape(ndata[k]), flush=True)

        # --- Append along axis = 1
        if veryextraverbose:
            print('Appending along axis = 1', flush=True)
        # First make groups to be appended together
        nndata_groups = {}
        for k in all_keys:
            if k[1] in list(nndata_groups.keys()):
                nndata_groups[k[1]][k[0]] = ndata[k]
            else:
                nndata_groups[k[1]] = {k[0]:ndata[k]}
        if veryextraverbose:
            for k in nndata_groups.keys():
                shapes = {key:np.shape(item) 
                          for key, item in nndata_groups[k].items()}
                print(f"Group {k} has elements with shapes: {shapes}", 
                      flush=True)
        # Then append them together
        nndata = {}
        for k2 in nndata_groups.keys():
            all_k1 = np.sort(list(nndata_groups[k2].keys()))
            nndata[k2] = nndata_groups[k2][all_k1[0]]
            for k1 in all_k1[1:]:
                if veryextraverbose:
                    print(f"Appending shape {np.shape(nndata[k2])}"
                          + f" with shape {np.shape(nndata_groups[k2][k1])}", 
                          flush=True)
                nndata[k2] = np.append(
                    nndata[k2], nndata_groups[k2][k1], axis=1)
        
        all_keys = list(nndata.keys())
        del ndata, nndata_groups
        
        if veryextraverbose:
            print(' === Key and shape after 2nd append', flush=True)
            for i, k in enumerate(all_keys):
                print(i, k, np.shape(nndata[k]), flush=True)
                
        # --- Append along axis = 0
        if veryextraverbose:
            print('Appending along axis = 0', flush=True)
        all_keys = np.sort(all_keys)
        k = all_keys[0]
        uncut_data = nndata[k]
        for k in all_keys[1:]:
            if veryextraverbose:
                print(f"Appending shape {np.shape(uncut_data)}"
                        + f" with shape {np.shape(nndata[k])}", 
                        flush=True)
            uncut_data = np.append(
                uncut_data, nndata[k], axis=0)
        del nndata
        
        if veryextraverbose:
            print(' === Final shape after 3rd append', flush=True)
            print(np.shape(uncut_data), flush=True)
    return uncut_data