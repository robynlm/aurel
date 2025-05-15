"""
reading.py

This module contains functions to work with data generated from 
Einstein Toolkit simulations, including:

- reading parameters from the simulation
- listing available iterations of the simulation
- reading data from the simulation output files
- joining chunks of data together
- fixing the x-z indexing of the data

Note
----
Users should first provide the path to their simulations:

**export SIMLOC="/path/to/simulations/"**
"""

import os
import numpy as np
import glob
import h5py
import re
import subprocess
import contextlib

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
    parameters = {'simname':simname}

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
        N = int(L/parameters['d'+coord])
        if parameters['boundary_shiftout_'+coord+'_upper'] == 1:
            N += 1
        parameters['L'+coord] = L
        parameters['N'+coord] = N
    if 'max_refinement_levels' not in parameters.keys():
        parameters['max_refinement_levels'] = 1
        
    return parameters

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
            # look at what's going on in one of the files
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
            
            saveprint(it_file, 'Reading '+file)
            with h5py.File(file, 'r') as f:
                print(f.keys())
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
                            
def read_data(param, var, it=[0], rl=0, restart=0):
    """Read the data from the simulation output files.
    
    Parameters
    ----------
    param : dict
        The parameters of the simulation.
    var : list
        The variables to read from the simulation output files.
    it : list, optional
        The iterations to read from the simulation output files.
        The default is [0].
    rl : int, optional
        The refinement level to read from the simulation output files.
    restart : int, optional
        The restart number to read from the simulation output files.
        The default is 0.
        
    Returns
    -------
    dict
        A dictionary containing the data from the simulation output files.
        
        dict.keys() = ['it', 't', var[0], var[1], ...]
    """
    # data to be returned
    it = np.sort(it)
    data = {'it':it, 't':[]}
    
    # data directory path
    data_path = (param['simpath']
                 + param['simname']
                 + '/output-{:04d}/'.format(restart)
                 + param['simname'] + '/')

    h5files = glob.glob(data_path+'*.h5')
    if ((data_path+'gxx.h5' in h5files) 
        or (data_path+'admbase-metric.h5' in h5files)):
        # one file per variable
        cmax = 'in file'
    else:
        # one file per chunk
        cfiles = glob.glob(data_path + 'admbase-metric.file_*.h5')
        cfiles += glob.glob(data_path + 'gxx.file_*.h5')
        cmax = np.max([int(f.split('file_')[1].split('.')[0]) 
                       for f in cfiles])

    individual = np.sum(
        [1 for file in h5files if 'hydrobase-rho.' in file]) == 0
    if individual:
        variables = {
            'rho':['rho.', 'HYDROBASE::rho', 0],
            'eps':['eps.', 'HYDROBASE::eps', 0],
            'w_lorentz':['w_lorentz.', 'HYDROBASE::w_lorentz', 0],
            'press':['press.', 'HYDROBASE::press', 0],
            'vel[0]':['vel[0].', 'HYDROBASE::vel[0]', 0],
            'vel[1]':['vel[1].', 'HYDROBASE::vel[1]', 0],
            'vel[2]':['vel[2].', 'HYDROBASE::vel[2]', 0],
            'alp':['alp.', 'ADMBASE::alp', 0],
            'dtalp':['dtalp.', 'ADMBASE::dtalp', 0],
            'betax':['betax.', 'ADMBASE::betax', 0],
            'betay':['betay.', 'ADMBASE::betay', 0],
            'betaz':['betaz.', 'ADMBASE::betaz', 0],
            'dtbetax':['dtbetax.', 'ADMBASE::dtbetax', 0],
            'dtbetay':['dtbetay.', 'ADMBASE::dtbetay', 0],
            'dtbetaz':['dtbetaz.', 'ADMBASE::dtbetaz', 0],
            'tau':['tau.', 'COSMOLAPSE::tau', 0],
            'gxx':['gxx.', 'ADMBASE::gxx', 0],
            'gxy':['gxy.', 'ADMBASE::gxy', 0],
            'gxz':['gxz.', 'ADMBASE::gxz', 0],
            'gyy':['gyy.', 'ADMBASE::gyy', 0],
            'gyz':['gyz.', 'ADMBASE::gyz', 0],
            'gzz':['gzz.', 'ADMBASE::gzz', 0],
            'kxx':['kxx.', 'ADMBASE::kxx', 0],
            'kxy':['kxy.', 'ADMBASE::kxy', 0],
            'kxz':['kxz.', 'ADMBASE::kxz', 0],
            'kyy':['kyy.', 'ADMBASE::kyy', 0],
            'kyz':['kyz.', 'ADMBASE::kyz', 0],
            'kzz':['kzz.', 'ADMBASE::kzz', 0],
            'trK':['trK.', 'ML_BSSN::trK', 0],
            'H':['H.', 'ML_BSSN::H', 0],
            'M1':['M1.', 'ML_BSSN::M1', 0],
            'M2':['M2.', 'ML_BSSN::M2', 0],
            'M3':['M3.', 'ML_BSSN::M3', 0],
        }
        if 'lapse' in var:
            var.remove('lapse')
            var += ['alp']
        if 'dtlapse' in var:
            var.remove('dtlapse')
            var += ['dtalp']
        if 'gdown' in var:
            var.remove('gdown')
            var += ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
        if 'metric' in var:
            var.remove('metric')
            var += ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
        if 'kdown' in var:
            var.remove('kdown')
            var += ['kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz']
        if 'curv' in var:
            var.remove('curv')
            var += ['kxx', 'kxy', 'kxz', 'kyy', 'kyz', 'kzz']
        if 'betaup' in var:
            var.remove('betaup')
            var += ['betax', 'betay', 'betaz']
        if 'shift' in var:
            var.remove('shift')
            var += ['betax', 'betay', 'betaz']
        if 'dtshift' in var:
            var.remove('dtshift')
            var += ['dtbetax', 'dtbetay', 'dtbetaz']
        if 'vel' in var:
            var.remove('vel')
            var += ['vel[0]', 'vel[1]', 'vel[2]']
        if 'ML_Ham' in var:
            var.remove('ML_Ham')
            var += ['H']
        if 'ML_Mom' in var:
            var.remove('ML_Mom')
            var += ['M1', 'M2', 'M3']
    else:
        variables = {
            'rho':['hydrobase-rho.', 'HYDROBASE::rho', 0],
            'eps':['hydrobase-eps.', 'HYDROBASE::eps', 0],
            'press':['hydrobase-press.', 'HYDROBASE::press', 0],
            'w_lorentz':['hydrobase-w_lorentz.', 'HYDROBASE::w_lorentz', 0],
            'vel':['hydrobase-vel.', 'HYDROBASE::vel', 1],
            'alp':['admbase-lapse.', 'ADMBASE::alp', 0],
            'lapse':['admbase-lapse.', 'ADMBASE::alp', 0],
            'dtalp':['admbase-dtlapse.', 'ADMBASE::dtalp', 0],
            'dtlapse':['admbase-dtlapse.', 'ADMBASE::dtalp', 0],
            'betaup':['admbase-shift.', 'ADMBASE::beta', 1],
            'shift':['admbase-shift.', 'ADMBASE::beta', 1],
            'dtbetaup':['admbase-dtshift.', 'ADMBASE::dtbeta', 1],
            'dtshift':['admbase-dtshift.', 'ADMBASE::dtbeta', 1],
            'tau':['cosmolapse-propertime.', 'COSMOLAPSE::tau', 0],
            'gdown':['admbase-metric.', 'ADMBASE::g', 2],
            'metric':['admbase-metric.', 'ADMBASE::g', 2],
            'kdown':['admbase-curv.', 'ADMBASE::k', 2],
            'curv':['admbase-curv.', 'ADMBASE::k', 2],
            'trK':['ml_bssn-ml_trace_curv.', 'ML_BSSN::trK', 0],
            'ML_Ham':['ml_bssn-ml_ham.', 'ML_BSSN::H', 0],
            'ML_Mom':['ml_bssn-ml_mom.', 'ML_BSSN::M', 1],
        }

    for v in var:
        if v in list(variables.keys()):
            att = variables[v]
            data.update(get_data_tensor(
            data_path, att[0], att[1],
            it, rl, cmax, att[2]))
        else:
            print('Variable {} not found'.format(v))

    return data

def get_data_tensor(path, filename, varkey, it, rl, cmax, rank):
    """Read the data from the simulation output files.
    
    Parameters
    ----------
    path : str
        The path to the simulation output files.
    filename : str
        Identifying part of the variable output file name.
    varkey : str
        Identifying part of the variable's hdf5 key.
    it : list
        The iterations to read from the simulation output files.
    rl : int
        The refinement level to read from the simulation output files.
    cmax : int
        The maximum number of chunks to read from the simulation output files.
        If 'in file', it will be extracted from the file.
    rank : int
        The rank of the variable to read from the simulation output files.
        0 scalar, 1 vector, 2 tensor.
        
    Returns
    -------
    dict
        A dictionary containing the data from the simulation output files.

        dict.keys() = ['it', 't', var]
    """

    # collect data files
    if cmax == 'in file':
        filename = path + filename +'h5'
        files = [filename]
    elif cmax == 0:
        files = [path + filename + 'h5']
    else: # a file per chunk
        files = []
        for c in range(cmax+1):
            files += [path + filename + 'file_{}.h5'.format(c)]
    
    # dictionary to collect the chunked data at each it for each component
    if rank==0:
        all_components = ['']
    if rank==1:
        all_components = ['x', 'y', 'z']
        if 'HYDROBASE::vel' in varkey:
            all_components = ['[0]', '[1]', '[2]']
        elif 'ML_BSSN::M' in varkey:
            all_components = ['1', '2', '3']
    if rank==2:
        all_components = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    var_chunks = {iit:{ij:{} for ij in all_components} for iit in it} 
    
    # also collect time while you're at it
    time = []
    collect_time = True
    
    # go through one file at a time
    for fi, filename in enumerate(files):
        # open the file and collect data
        with h5py.File(filename, 'r') as f:
            # collect all it of that file
            for iit in it:
                # key of data to collect 
                key_end = ' it={} tl=0 rl={}'.format(iit, rl)
    
                # find the actual number of chunks
                if cmax == 'in file':
                    fkeys = [k for k in f.keys() 
                             if varkey+all_components[0]+key_end in k]
                    if 'c=' in fkeys[0]:
                        actual_cmax = np.max(list(set([
                            int(k.split('c=')[1]) 
                            for k in fkeys])))
                    else:
                        actual_cmax = 0
                    crange = np.arange(actual_cmax+1)
                else:
                    actual_cmax = cmax
                    crange = [fi]

                # for each chunk
                for c in crange:
                    if actual_cmax!=0: 
                        key_end_with_c = key_end + ' c={}'.format(c)
                    else:
                        key_end_with_c = key_end

                    # For each component
                    for ij in all_components:
                        # the full key to use
                        key = varkey + ij + key_end_with_c
                        # cut off ghost grid points
                        ghost_x = f[key].attrs['cctk_nghostzones'][0]
                        ghost_y = f[key].attrs['cctk_nghostzones'][1]
                        ghost_z = f[key].attrs['cctk_nghostzones'][2]
                        var = np.array(f[key])[
                            ghost_z:-ghost_z, 
                            ghost_y:-ghost_y, 
                            ghost_x:-ghost_x]
                        iorigin = tuple(f[key].attrs['iorigin'])
                        var_chunks[iit][ij][iorigin] = var
                if collect_time:
                    time += [f[key].attrs['time']]
            collect_time = False # only do this in one file
    

    # rename some variables to work with AurelCore
    varname = varkey.split('::')[1]
    vartorename = {
        'rho':'rho0', 'vel[0]':'velx', 'vel[1]':'vely', 'vel[2]':'velz',
        'alp':'alpha', 'dtalp':'dtalpha', 'trK':'Ktrace', 
        'H':'Hamiltonian', 
        'M1':'Momentumx', 'M2':'Momentumy', 'M3':'Momentumz'}

    # per iteration, join the chunks together
    # fix the indexing to be x, y, z
    var = {}#varname+ij:[] for ij in all_components}
    for iit in it:
        for ij in all_components:
            full_varname = varname+ij
            if full_varname in list(vartorename.keys()):
                full_varname = vartorename[full_varname]
            if iit == it[0]:
                var[full_varname] = [fixij(join_chunks(var_chunks[iit][ij]))]
            else:
                var[full_varname] += [fixij(join_chunks(var_chunks[iit][ij]))]
    var['t'] = time
    return var

def fixij(f):
    """Fix the x-z indexing as you read in the data."""
    return np.transpose(np.array(f), (2, 1, 0))

def unfixij(data):
    """Fix the x-z indexing as you record data."""
    return np.transpose(data, (2, 1, 0))

def join_chunks(cut_data):
    """Join the chunks of data together.
    
    Parameters
    ----------
    cut_data : dict of dict
        A dictionary containing the data from the simulation output files.
        dict.keys() = tuple of the chunks 'iorigin' attribute 
        which maps out how the data is to be joined together.
    
    Returns
    -------
    uncut_data : array_like
        The data now joined together.
    """
    all_keys = list(cut_data.keys())
    cmax = len(all_keys) - 1

    verbose = False
    if verbose:
        print(cmax, all_keys)
        for k in all_keys:
            print(k, np.shape(cut_data[k]))
        print()
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
    elif cmax in [8, 9, 10]: # TO CHECK
        # fix axis = 2
        if cmax == 8: # TO CHECK
            ndata = [np.append(cut_data[0], cut_data[1], axis=2), 
                     cut_data[2], 
                     np.append(cut_data[3], cut_data[4], axis=2), 
                     cut_data[5], 
                     np.append(cut_data[6], cut_data[7], axis=2), 
                     cut_data[8]]
        elif cmax == 9: # TO CHECK
            ndata = [np.append(cut_data[0], cut_data[1], axis=2), 
                     np.append(cut_data[2], cut_data[3], axis=2), 
                     np.append(cut_data[4], cut_data[5], axis=2), 
                     cut_data[6],
                     np.append(cut_data[7], cut_data[8], axis=2), 
                     cut_data[9]]
        elif cmax == 10: # TO CHECK
            ndata = [np.append(cut_data[i], cut_data[i+1], axis=2) 
                     for i in np.arange(5)*2]
            ndata += [cut_data[10]]
        else: # TO CHECK
            ndata = [np.append(cut_data[i], cut_data[i+1], axis=2) 
                     for i in np.arange(6)*2]
        # fix axis = 1 and 0
        nndata = [np.append(ndata[i], ndata[i+1], axis=1) 
                  for i in np.arange(3)*2]
        uncut_data = np.append(
            np.append(nndata[0], nndata[1], axis=0), 
            nndata[2], axis=0)
    # =================
    else:
        # append along axis = 2
        k = all_keys[0]
        current_key = k[1:]
        ndata = {current_key:cut_data[k]}
        for k in all_keys[1:]:
            if current_key == k[1:]:
                ndata[current_key] = np.append(
                    ndata[current_key], cut_data[k], axis=2)
            else:
                current_key = k[1:]
                ndata[current_key] = cut_data[k]
        all_keys = list(ndata.keys())
        
        if verbose:
            for k in all_keys:
                print(k, np.shape(ndata[k]), flush=True)
            print()

        # append along axis = 1
        k = all_keys[0]
        current_key = k[1:]
        nndata = {current_key:ndata[k]}
        for k in all_keys[1:]:
            if current_key == k[1:]:
                nndata[current_key] = np.append(
                    nndata[current_key], ndata[k], axis=1)
            else:
                current_key = k[1:]
                nndata[current_key] = ndata[k]
        all_keys = list(nndata.keys())
        
        if verbose:
            for k in all_keys:
                print(k, np.shape(nndata[k]), flush=True)
            print()
                
        # append along axis = 0
        k = all_keys[0]
        uncut_data = nndata[k]
        for k in all_keys[1:]:
            uncut_data = np.append(
                uncut_data, nndata[k], axis=0)
        
        if verbose:
            print(np.shape(uncut_data), flush=True)
    return uncut_data

