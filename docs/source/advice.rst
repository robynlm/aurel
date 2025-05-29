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

To do this automatically, in the paths listed previously (with `module show fftw_name`) you should see: `prepend_path("LD_LIBRARY_PATH","/path/to/fftw/lib")`, this needs to be added to the kernel configuration file. So,

.. code-block:: bash
   
   vim ~/.local/share/jupyter/kernels/myenv/kernel.json

and edit this to include the path to the FFTW library in the `env` section, so it looks like this (change the path below):

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

In vim, press `i` to enter insert mode, paste the `env` section above, then press `Esc` to exit insert mode, and type `:wq` to save and quit.

Now you can load your jupyter notebook, select the kernel you just created `Python (myenv)` and then in your notebook you can have

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