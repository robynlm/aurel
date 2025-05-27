Advice
======

Working on an HPC
-----------------

While HPCs won't let you install packages globally, you can still use aurel. 
You just need to create a virtual environment and install aurel there.

* Create a virtual environment

.. code-block:: bash

   python -m venv ~/myenv
   source ~/myenv/bin/activate
   pip install aurel
   
and install any other packages you may use.

* If you want to use aurel in a **jupyter notebook**, add your virtual environment as a Jupyter kernel

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

then select `Python (myenv)` for your kernel and then in your notebook you can have

.. code-block:: python

   import aurel 

* If you want to use aurel in a **python script**, before running it, activate the environment

.. code-block:: bash

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