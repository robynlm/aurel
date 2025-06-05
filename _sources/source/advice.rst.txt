Advice
======

Working on an HPC
-----------------

While HPCs won't let you install packages globally, you can still use aurel. 
You just need to create a virtual environment and install aurel there.

Create a virtual environment
++++++++++++++++++++++++++++

You need to install FFTW, typically on an HPC this is one of the modules available, so find its name (replace fftw_name below), load it and check its paths

.. code-block:: bash

   module avail
   module load fftw_name
   module show fftw_name

Now create a virtual environment and activate it.

.. code-block:: bash

   python -m venv ~/myenv
   source ~/myenv/bin/activate

Then you should be able to do:

.. code-block:: bash

   pip install aurel
   
and install any other packages you may use.

Jupyter notebook
++++++++++++++++

If you want to use aurel in a jupyter notebook, add your virtual environment as a Jupyter kernel

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

To have FFTW loaded in your notebook make sure to load the module before starting up your notebook. 

To do this automatically, in the paths listed previously 
(with `module show fftw_name`) you should see: 
`prepend_path("LD_LIBRARY_PATH","/path/to/fftw/lib")`, 
this needs to be added to the kernel configuration file. So,
``vim ~/.local/share/jupyter/kernels/myenv/kernel.json`` and edit this to 
include the path to the FFTW library in the `env` section, 
so it looks like this (change the path below):

.. code-block:: bash
   
   {
   "argv": [lots of things here, keep them as they are],
   "env": {
   "LD_LIBRARY_PATH": "/path/to/fftw/lib:$LD_LIBRARY_PATH"
   },
   "display_name": "Python (myenv)",
   "language": "python",
   "metadata": {
   "debugger": true
   }

In vim, press `i` to enter insert mode, modify/include the `env` section, 
then press `Esc` to exit insert mode, and type `:wq` to save and quit.

Now you can load your jupyter notebook, select the kernel you just created 
`Python (myenv)` and type in your notebook:

.. code-block:: python

   import aurel 

Python script
+++++++++++++

If you want to use aurel in a python script, before running it, activate the environment

.. code-block:: bash

   module load fftw_name
   source ~/myenv/bin/activate
   python myscript.py

then in your python script you can have

.. code-block:: python

   import aurel

Parallelisation
---------------

Aurel uses JAX for vectorisation, JIT compilation and parallelisation.
See the `JAX documentation <https://docs.jax.dev/en/latest/user_guides.html#user-guides>`_ for more information on how to use it.

To accelerate things, make sure anything you pass to aurel is a JAX array.

* In a **jupyter notebook**

.. code-block:: python

   import jax
   jax.config.update('jax_num_cpu_devices', N)
   # Replace N with the number of threads you want to use
   print(jax.devices()) # check the number of devices visible to jax
   import aurel 

All jax configuration options need to be set before importing aurel.

* In a **python script**

You can put the above in your python script, or before calling the script, put in the terminal or your executable:

.. code-block:: bash

   export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=N"

Replace N with the number of threads you want to use.

Convergence
-----------

See `Schwarzschild_check notebook <https://github.com/robynlm/aurel/blob/main/notebooks/Schwarzschild_check.ipynb>`_

* Choose the order of the finite difference scheme you want to use,
  this is done by setting the `fd_order` parameter in the 
  `FiniteDifference` class. Options are: 4 (default), 6, 8.
  ``fd = aurel.FiniteDifference(param, fd_order=6)``

* Increase the grid resolution in your simulation, or reduce the grid spacing 
  for generated data.

* Increase float precision, default for jax is float32, you can increase 
  this by configuring jax before importing aurel:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import aurel

Citation
--------

If you use aurel in your work, please cite it as::

   @misc{aurel2025,
     title     = {Aurel: A Python package for automatic relativistic calculations},
     author    = {Munoz, Robyn L.},
     publisher = {GitHub},
     year      = {2025},
     url       = {https://github.com/robynlm/aurel}}

This code is not published yet, so to add to citation count please also
include::

   @article{R.L.Munoz_M.Bruni_2023,
     title     = {EBWeyl: a Code to Invariantly Characterize Numerical Spacetimes},
     author    = {Munoz, Robyn L. and Bruni, Marco},
     journal   = {Classical and Quantum Gravity},
     volume    = {40},
     number    = {13},
     pages     = {135010},
     year      = {2023},
     month     = {jun},
     doi       = {10.1088/1361-6382/acd6cf},
     archivePrefix = {arXiv},
     eprint    = {gr-qc/2211.08133}}

If you use aurel to calculate $\Psi_{4}^{l,m}$ then you ought to also cite 
the `spinsfast package <https://github.com/moble/spinsfast>`_::

   @software{M.Boyle_Okarin_2024,
     title     = {moble/spinsfast: Release v2022.4.10},
     author    = {Boyle, Mike and Okarin},
     publisher = {Zenodo},
     month     = {dec},
     year      = {2024},
     version   = {v2022.4.10},
     doi       = {10.5281/zenodo.14522969}}

   @article{K.M.Huffenberger_B.D.Wandel_2010,
     title     = {FAST AND EXACT SPIN-s SPHERICAL HARMONIC TRANSFORMS},
     author    = {Huffenberger, Kevin M. and Wandelt, Benjamin D.},
     journal   = {The Astrophysical Journal Supplement Series},
     publisher = {The American Astronomical Society},
     volume    = {189},
     number    = {2},
     pages     = {255},
     month     = {jul},
     year      = {2010},
     doi       = {10.1088/0067-0049/189/2/255},
     archivePrefix = {astro-ph},
     eprint    = {gr-qc/1007.3514}}
