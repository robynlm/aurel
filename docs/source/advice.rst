Working on an HPC
-----------------

While HPCs won't let you install packages globally, you can still use aurel. 
You just need to create a virtual environment and install aurel there.

Create a virtual environment
++++++++++++++++++++++++++++

Create a virtual environment and activate it.

.. code-block:: bash

   python -m venv ~/myenv
   source ~/myenv/bin/activate

Then you should be able to do:

.. code-block:: bash

   pip install aurel
   
or the latest development version.

You should be able to find the package in `myenv/lib/pythonX.X/site-packages/aurel`.
Accompanying required packages will be installed automatically, but at this point you should also install any other python packages you typically use.

Jupyter notebook
++++++++++++++++

If you want to use aurel in a jupyter notebook, while your environment is active, add your virtual environment as a Jupyter kernel

.. code-block:: bash

   python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

Now you can load your jupyter notebook, select the kernel you just created 
`Python (myenv)` and type in your notebook:

.. code-block:: python

   import aurel 

Python script
+++++++++++++

If you want to use aurel in a python script, before running it, activate the environment

.. code-block:: bash

   source ~/myenv/bin/activate
   python myscript.py

then in your python script you can have

.. code-block:: python

   import aurel

Convergence
-----------

See :doc:`../notebooks/Schwarzschild_check`

* Choose the order of the finite difference scheme you want to use,
  this is done by setting the `fd_order` parameter in the 
  :class:`aurel.FiniteDifference` class. Options are: 2, 4 (default), 6, 8.

.. code-block:: python

   fd = aurel.FiniteDifference(param, fd_order=6)

* Increase the grid resolution in your simulation, or reduce the grid spacing 
  for generated data.

Citation
--------

If you use aurel in your work, please cite it as:

.. dropdown:: Munoz, R. L. (2025). Aurel: A Python package for automatic relativistic calculations. *GitHub*. https://github.com/robynlm/aurel

   .. code-block:: bibtex

      @misc{aurel2025,
        title     = {Aurel: A Python package for automatic relativistic calculations},
        author    = {Munoz, Robyn L.},
        publisher = {GitHub},
        year      = {2025},
        url       = {https://github.com/robynlm/aurel}}

This code is not published yet, so to add to citation count please also include:

.. dropdown:: Munoz, R. L., & Bruni, M. (2023). EBWeyl: a Code to Invariantly Characterize Numerical Spacetimes. *Classical and Quantum Gravity*, 40(13), 135010. https://doi.org/10.1088/1361-6382/acd6cf

   .. code-block:: bibtex

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
        eprint    = {2211.08133},
        primaryClass = {gr-qc}}
