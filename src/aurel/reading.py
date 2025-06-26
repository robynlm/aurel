"""
reading.py

This module contains functions to read/write data, and has specific
functions for using data generated from Einstein Toolkit simulations. 
This includes:

- reading parameters from the simulation
- listing available iterations of the simulation
- reading data from the simulation output files
- joining chunks of data together
- fixing the x-z indexing of the data
- saving data to files

Note
----
Users should first provide the path to their simulations:

**export SIMLOC="/path/to/simulations/"**
"""

import os
import jax.numpy as jnp
import numpy as np
import glob
import h5py
import re
import subprocess
import contextlib
import kuibit

def bash(command):
    """Run a bash command and return the output as a string.

    Parameters
    ----------
    command : str
        The bash command to run.

    Returns
    -------
    str
        The output of the command as a string.
    """
    results = subprocess.check_output(command, shell=True)
    strresults = str(results, 'utf-8').strip()
    return strresults

def parameters(simname):
    """Read the parameters from the simulation.
    
    This also counts the number of restarts, and calculates the
    size and number of grid points in each direction 
    of the simulation box.

    Parameters
    ----------
    simname : str
        The name of the simulation to read the parameters from.
        
    Returns
    -------
    dict
        A dictionary containing the parameters of the simulation.
    
    Note
    ----
    Users should first provide the path to their simulations:

    **export SIMLOC="/path/to/simulations/"**
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
        parampath = (simloc + parameters['simname'] 
                     + '/output-0000/' + parameters['simname'] + '.par')
        if os.path.isfile(parampath):
            founddata = True
            # save paths of files
            parameters['simpath'] = simloc
            parameters['datapath'] = (parameters['simpath'] 
                                      + parameters['simname'] 
                                      + '/output-0000/' 
                                      + parameters['simname'] + '/')
            # number of restarts
            files = bash('ls ' + parameters['simpath'] 
                            + parameters['simname']).split('\n')
            parameters['nbr_restarts'] = len(
                [fl for fl in files 
                 if 'output' in fl and 'active' not in fl])
            break
    if not founddata:
        raise ValueError('Could not find simulation: ' + simname)

    # save all parameters
    lines = bash('cat ' + parampath).split('\n') # read file
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

aurel_tensor_to_scalar_varnames = {
    'gammadown3': ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'],
    'Kdown3': ['kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz'],
    'betaup3': ['betax', 'betay', 'betaz'],
    'dtbetaup3': ['dtbetax', 'dtbetay', 'dtbetaz'],
    'velup3': ['velx', 'vely', 'velz'],
    'Momentumup3': ['Momentumx', 'Momentumy', 'Momentumz'],
    'Weyl_Psi': ['Weyl_Psi4r', 'Weyl_Psi4i'],
}

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
    'velup3' : ['vel'],
    'Hamiltonian' : ['H'],
    'Momentumup3' : ['M1', 'M2', 'M3'],
    'Weyl_Psi' : ['Psi4r', 'Psi4i']
}

ET_to_aurel_varnames = {
    'rho':'rho0', 'alp':'alpha', 'dtalp':'dtalpha',
    'trK':'Ktrace', 'H':'Hamiltonian',
    'vel[0]':'velx', 'vel[1]':'vely', 'vel[2]':'velz',
    'M1':'Momentumx', 'M2':'Momentumy', 'M3':'Momentumz',
    'Psi4r':'Weyl_Psi4r', 'Psi4i':'Weyl_Psi4i'}

def saveprint(it_file, some_str):
    """Save the string to the file and print it to the screen."""
    print(some_str, flush=True)
    with contextlib.redirect_stdout(it_file):
        print(some_str)

def iterations(param, skip_last=True):
    """Lists available iterations of the simulation's 3D output files.

    Prints available iterations on screen and records this 
    in a file called iterations.txt in the simulation directory.
    
    Parameters
    ----------
    param : dict
        The parameters of the simulation.
    skip_last : bool, optional
        If True, skip the last restart. The default is True.
        This is to not mess with things in case they're still running.
    """

    nbr_restarts = param['nbr_restarts']
    if skip_last:
        nbr_restarts -= 1

    it_filename = param['simpath']+param['simname']+'/iterations.txt'
    with open(it_filename, "a+") as it_file:
        # print content of file
        it_file.seek(0)
        contents = it_file.read()
        print(contents, flush=True)

        # restart from which to continue
        lines = contents.split("\n")
        restarts_done = [int(line.split("restart ")[1]) 
                         for line in lines 
                         if 'restart' in line]
        if len(restarts_done) == 0:
            first_restart = 0
        else:
            first_restart = np.max(restarts_done) + 1

        # for each restart
        for restart in range(first_restart, nbr_restarts):
            saveprint(it_file, ' === restart {}'.format(restart))
            # find the data
            datapath = (param['simpath']+param['simname']
                        +'/output-{:04d}/'.format(restart)
                        +param['simname']+'/')
            
            # list available variables
            # get kuibit variables
            sim = kuibit.simdir.SimDir(datapath)
            vars_available = list(sim.gf.xyz.fields.keys())
            
            # error if it didn't find anything
            if vars_available == []:
                raise ValueError('Could not find 3D data in ' + datapath)
            else:
                # transform kuibit variables to aurel variables
                aurel_vars_available = []
                for newv, oldvs in  aurel_to_ET_varnames.items():
                    if np.sum([v in vars_available 
                               for v in oldvs]) == len(oldvs):
                        aurel_vars_available += [newv]
                        for oldv in oldvs:
                            vars_available.remove(oldv)
                # dont consider grid property things
                for v in vars_available:
                    if 'grid_' in v:
                        vars_available.remove(v)
                # everything else
                print('NEED TO UPDATE AUREL TO INFCLUDE:', 
                      vars_available, flush=True)
                aurel_vars_available += vars_available
                saveprint(it_file, '3D variables available: '
                          + str(aurel_vars_available))

            # find a file to read the iterations from
            h5files = glob.glob(datapath+'*.h5')
            found3Dfile = False
            for file in h5files:
                if (('admbase' in file) or ('gxx' in file)):
                    is3D = True
                    for slice in ['.xx.', '.xy.', '.xz.', 
                                  '.yy.', '.yz.', '.zz.']:
                        if slice in file:
                            is3D = False
                            break
                    if is3D:
                        found3Dfile = True
                        break
            if not found3Dfile:
                raise ValueError('Could not find 3D data in ' + datapath)
            
            with h5py.File(file, 'r') as f:
                # only consider one of the variables in this file
                varkey = list(f.keys())[0].split(' ')[0]
                # all the keys of this varible
                fkeys = [k for k in f.keys() if varkey in k]
                # all the iterations
                allits = np.sort([int(k.split('it=')[1].split(' tl')[0]) 
                                  for k in fkeys])
                saveprint(it_file, 'it = {} -> {}'.format(np.min(allits), 
                                                          np.max(allits)))
                # maximum refinement level present
                rlmax = np.max(list(set([int(k.split('rl=')[1].split(' ')[0]) 
                                         for k in fkeys])))
                # for each refinement level
                for rl in range(rlmax+1):
                    # take the corresponding keys
                    keysrl = []
                    for k in fkeys:
                        krl = k.split('rl=')[1].split(' ')[0]
                        if krl == str(rl):
                            keysrl += [k]
                    if keysrl!=[]:
                        if 'c=' in keysrl[0]:
                            cs = [k.split('c=')[1] for k in keysrl]
                            chosen_c = ' c=' + np.sort(list(set(cs)))[-1]
                        else:
                            chosen_c = ''
                        keysrl = [k for k in keysrl if chosen_c in k]
                        
                        # and look at what iterations they have
                        allits = np.sort([int(k.split('=')[1].split(' tl')[0]) 
                                          for k in keysrl])
                        if len(allits)>1:
                            saveprint(
                                it_file, 
                                'rl = {} at it = np.arange({}, {}, {})'.format(
                                rl, np.min(allits), np.max(allits), 
                                np.diff(allits)[0]))
                        else:
                            saveprint(it_file, 'rl = {} at it = {}'.format(
                                rl, allits))
                            
def read_iterations(param):
    """Read the available iterations of the simulation.
    
    This reads in the file the iterations function creates.
    Note that if this file does not exist, it will be created.
    
    Parameters
    ----------
    param : dict
        The parameters of the simulation.
    
    Returns
    -------
    dict :
        A dictionary containing the available iterations of the simulation.
        {restart_nbr: {'var available': [...], 
                       'its available': [itmin, itmax], 
                       'rl = rl_nbr': [itmin, itmax, dit]}}
    """
    it_filename = param['simpath']+param['simname']+'/iterations.txt'
    # if file does not exist, create it
    if not os.path.isfile(it_filename):
        iterations(param)
    
    # read the file
    with open(it_filename, "r") as it_file:
        it_file.seek(0)
        contents = it_file.read()
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
                its_available[restart_nbr]['its available'] = [int(l.split(' ')[2]), int(l.split(' ')[4])]
            # iterations available for said refinement level
            elif 'rl = ' in l:
                rl = l.split('rl = ')[1].split(' ')[0]
                itmin = int(l.split('(')[1].split(',')[0])
                itmax = int(l.split(', ')[1])
                dit = int(l.split(', ')[2].split(')')[0])
                its_available[restart_nbr]['rl = '+rl] = [itmin, itmax, dit]
    return its_available
                            
def read_data(param, **kwargs):
    """Read the data from the simulation output files.

    Parameters
    ----------
    param : dict
        The parameters of the simulation.

    Other Parameters
    ----------------
    it : list, optional
        The iterations to read from the simulation output files.
        The default is [0].
    vars : list, optional
        The variables to read from the simulation output files.
        The default is [], all available variables will be read.
    rl : int, optional
        The refinement level to read from the simulation output files.
        The default is 0.
    restart : int, optional
        The restart number to read and save the data to.
        The default is -1, meaning it'll find the restart of the it you want.
    split_per_it : bool, optional
        Default True, if possible read the data from the split iterations.
        Complete request with ET files and save variables 
        in individual files per iteration.
        Else just read ET files.
    verbose : bool, optional
        If True, print additional information during the reading process.
    veryverbose : bool, optional
        If True, print even more information during the reading process.
    
    Returns
    -------
    dict
        A dictionary containing the data from the simulation output files.
        dict.keys() = ['it', 't', var[0], var[1], ...]
    """
    it = np.sort(kwargs.get('it', [0]))
    split_per_it = kwargs.get('split_per_it', True)
    verbose = kwargs.get('verbose', False)
    veryverbose = kwargs.get('veryverbose', False)
    restart = kwargs.get('restart', -1)
    var = kwargs.get('vars', [])

    if 'simulation' in param.keys():
        its_available = read_iterations(param)
        if restart == -1:
            restarts_available = list(its_available.keys())

            # create new element containing iterations to do
            for restart in list(its_available.keys()):
                    its_available[restart]['it to do'] = []

            # reverse order so that I'm always taking the most recent iteration
            for iit in it[::-1]:
                for restart in list(its_available.keys())[::-1]:
                    itmin, itmax = its_available[restart]['its available']
                    if ((itmin < iit) and (iit < itmax)):
                        its_available[restart]['it to do'] += [iit]
                        break
        else:
            restarts_available = [restart]
            its_available[restart]['it to do'] = it

        for restart in restarts_available:
            # iterations available in this restart
            old_it = it.copy()
            it = its_available[restart]['it to do']
            kwargs['it'] = it
            # if no variable is specified, take all available
            if var==[]:
                var = its_available[restart]['var available']
            # big dictionary to save data and to be flattened
            datar = {}
            
            # combination of variables already split per iteration and new ones
            if split_per_it:
                # change variable names to scalar elements
                avar = var.copy()
                for key, new_vars in aurel_tensor_to_scalar_varnames.items():
                    if key in avar:
                        avar.remove(key)
                        avar += new_vars
                if veryverbose:
                    print('Restart {}:'.format(restart), flush=True)
                    print('vars to get {}:'.format(avar), flush=True)

                # First get variables from split iterations
                kwargs['vars'] = avar
                datar[restart] = read_aurel_data(param, **kwargs)
                if veryverbose:
                    print('keys in data restart {}'.format(restart),
                          ' read from split data:',
                          list(datar[restart].keys()), flush=True)
                    cleaned_dict = {k: v for k, v in datar[restart].items() 
                                    if (not (isinstance(v, list) 
                                            and all(x is None for x in v))
                                        and k!='it')}
                    print('Data read from split iterations:', 
                            list(cleaned_dict.keys()), flush=True)
                # find those that are missing
                its_missing = {v: [] for v in avar}
                avart = avar + ['t']
                for it_idx, iit in enumerate(it):
                    for av in avart:
                        if datar[restart][av][it_idx] is None:
                            its_missing[av] += [iit]
                # if some are missing, read them from the ET data
                ETread_vars = []
                saved_vars = []
                for v in var:
                    # collect missing iterations
                    if v in list(aurel_tensor_to_scalar_varnames.keys()):
                        avar = aurel_tensor_to_scalar_varnames[v]
                        its_temp = []
                        for av in avar:
                            its_temp += its_missing[av]
                        its_temp = list(set(its_temp))
                    else:
                        avar = [v]
                        its_temp = its_missing[v]
                    # if there are missing iterations
                    if its_temp != []:
                        kwargs['it'] = its_temp
                        # retrieve missing iterations
                        data_temp = read_ET_data_kuibit(
                            param, [v], **kwargs)
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
                                        if ((data_temp[av][iidxtem] is None)
                                            or (av == 't')):
                                            its_missing[av].remove(iit)
                                            
                                if its_missing[av] != []:
                                    # save the data for the missing iterations
                                    # and save them individually
                                    kwargs['it'] = its_missing[av]
                                    kwargs['vars'] = [av]
                                    saved_vars += [av]
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
                    saved_vars = list(set(saved_vars))
                    if ETread_vars != []:
                        print('Variables read from ET files:', 
                            ETread_vars, flush=True)
                    if saved_vars != []:
                        print('Variables saved to split iterations files:', 
                            saved_vars, flush=True)
            else:
                if veryverbose:
                    print('Restart {}:'.format(restart), flush=True)
                    print('vars to get {}:'.format(var), flush=True)
                datar[restart] = read_ET_data_kuibit(param, var, **kwargs)
        
        # create ultimate data dictionary (flattening datar)
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
    else:
        data = read_aurel_data(param, **kwargs)
    return data

def read_aurel_data(param, **kwargs):
    """Read the data from AurelCore simulation output files.

    Parameters
    ----------
    param : dict
        The parameters of the simulation.

    Other Parameters
    ----------------
    it : list, optional
        The iterations to read from the simulation output files.
        The default is [0].
    vars : list, optional
        The variables to read from the simulation output files.
        The default is [], all available variables will be read.
    rl : int, optional
        The refinement level to read from the simulation output files.
        The default is 0.
    restart : int, optional
        The restart number to read from the simulation output files.
        The default is 0.
    verbose : bool, optional
        If True, print additional information during the reading process.
        The default is False.

    Returns
    -------
    dict
        A dictionary containing the data from the simulation output files.
        dict.keys() = ['it', 't', var[0], var[1], ...]
    """
    it = np.sort(kwargs.get('it', [0]))
    rl = kwargs.get('rl', 0)
    restart = kwargs.get('restart', 0)
    var = kwargs.get('vars', [])
    verbose = kwargs.get('verbose', False)
    veryverbose = kwargs.get('veryverbose', False)

    if var == []:
        get_them_all= True
    else:
        get_them_all = False

    if veryverbose:
        if get_them_all:
            print('read_aurel_data: reading all variables available')
        else:
            print('read_aurel_data: reading variables {}'.format(var))

    if 'simulation' in param.keys():
        datapath = (param['simpath']
                    + param['simname']
                    + '/output-{:04d}/'.format(restart)
                    + param['simname']
                 + '/all_iterations/')
    else:
        datapath = param['datapath']
    var += ['t']
    data = {'it': it, **{v: [] for v in var}}
    for it_index, iit in enumerate(it):
        fname = '{}it_{}.hdf5'.format(datapath, int(iit))
        if os.path.exists(fname):
            with h5py.File(fname, 'r') as f:
                if get_them_all:
                    # get all variables in the file, 
                    # if it's the right refinement level
                    for key in f.keys():
                        if ' rl={}'.format(rl) in key:
                            var += [key.split(' rl')[0]]
                    var = list(set(var))  # remove duplicates
                for key in var:
                    skey = key + ' rl={}'.format(rl)
                    # include this key into the data dictionary
                    if key not in data.keys():
                        data[key] = [None]*it_index
                    #save the data
                    if skey in list(f.keys()):
                        data[key].append(jnp.array(f[skey]))
                    else:
                        data[key].append(None)
        else:
            for key in var:
                skey = key + ' rl={}'.format(rl)
                data[key].append(None)
    return data

def save_data(param, data, **kwargs):
    """Save the data to a file
    
    Parameters
    ----------
    param : dict
        The parameters of the simulation.
        Needs to contain the key 'datapath', 
        this is where the data will be saved.
    data : dict
        The data to be saved.
        dict.keys() = ['it', 't', var[0], var[1], ...]

    Other Parameters
    ----------------
    vars : list, optional
        The variables to save from the data.
        If not provided, all variables in data will be saved.
        The default is [].
    it : list, optional
        The iterations to save from the data.
        The default is [0].
    rl : int, optional
        The refinement level to save from the data.
        The default is 0.
    restart : int, optional
        The restart number to save the data to.
        The default is 0.
    
    Note
    ----
    The data will be saved in the format:
    datapath/it_<iteration>.hdf5
    where <iteration> is the iteration number.
    The variables will be saved as datasets in the file, 
    with keys '<variable_name> rl=<refinement_level>'
    """
    vars = kwargs.get('vars', [])
    it = np.sort(kwargs.get('it', [0]))
    rl = kwargs.get('rl', 0)
    restart = kwargs.get('restart', 0)

    if vars == []:
        vars = list(data.keys())

    # check paths
    if 'simulation' in param.keys():
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

def read_ET_data_kuibit(param, var, **kwargs):
    """Read Einstein Toolkit simulation data using `kuibit`.
    
    Parameters
    ----------
    param : dict
        The parameters of the simulation.
    var : list
        The variables to read from the simulation output files, in aurel format

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
    reshape : bool, optional
        If True, kuibit will spline interpolate to match whatever 
        data grid you want. The default is False.
    verbose : bool, optional
        If True, print additional information during the reading process.
        The default is False.
    veryverbose : bool, optional
        If True, print even more information during the reading process.
        The default is False.
    Returns
    -------
    dict
        A dictionary containing the data from the simulation output files.
    """

    # input
    it = np.sort(kwargs.get('it', [0]))
    restart = kwargs.get('restart', 0)
    rl = kwargs.get('rl', 0)
    reshape = kwargs.get('reshape', False)
    verbose = kwargs.get('verbose', False)
    veryverbose = kwargs.get('veryverbose', False)

    # aurel to ET variable names
    new_var_list = []
    for v in var:
        if v in list(aurel_to_ET_varnames.keys()):
            if v in ['velup3']:
                new_var_list += [aurel_to_ET_varnames[v][0] + '[0]', 
                                 aurel_to_ET_varnames[v][0] + '[1]', 
                                 aurel_to_ET_varnames[v][0] + '[2]']
            else:
                new_var_list += aurel_to_ET_varnames[v]
        else:
            raise ValueError(
                'Variable {} not recognised'.format(v))
    if verbose:
        print(new_var_list)

    # setup kuibit
    datapath = (param['simpath']+param['simname']
                +'/output-{:04d}/'.format(restart)
                +param['simname']+'/')
    if veryverbose:
        print('Kuibit looking into: ', datapath, flush=True)
    sim = kuibit.simdir.SimDir(datapath)
    grid = kuibit.grid_data.UniformGrid(
        [param['Nx'], param['Ny'], param['Nz']], 
        x0=[param['xmin'], param['ymin'], param['zmin']], 
        dx=[param['dx'], param['dy'], param['dz']])

    # data to be returned
    data = {'it':it, 't':[]}    
    for v in new_var_list:
        # setup the data as OneGridFunctionH5
        if '[' in v:
            vnew = v.split('[')[0]
            index = int(v.split(']')[0].split('[')[1])
            variable = getattr(sim.gf.xyz.fields, vnew)[index]
        else:
            variable = getattr(sim.gf.xyz.fields, v)

        # aurel varnames to save data with
        if v in list(ET_to_aurel_varnames.keys()):
            v = ET_to_aurel_varnames[v]
        data[v] = []

        # for each iteration
        for iit in it:
            # read in the data
            if reshape: # interpolation
                data_at_it = variable.read_on_grid(iit, grid).data_xyz
            else:
                data_at_it = variable[iit].get_level(rl).data_xyz

            # save that to my big dictionary
            data[v] += [data_at_it]
            data['t'] += [variable[iit].time]
    
    return data
