.. aurel documentation master file, created by
   sphinx-quickstart on Tue May 13 10:39:45 2025.

Welcome to aurel's documentation! 
=================================

.. raw:: html

   <p>Aurel is an open source Python package for numerical relativity analysis.
   Designed with ease of use in mind, it will 
   <span style="font-weight: bold;">au</span>tomatically calculate 
   <span style="font-weight: bold;">rel</span>ativistic terms. 
   </p>

(As long as it is in the list of available entities, 
see :ref:`descriptions_list`.)

In addition, this package also provides tools for 
advanced numerical computations, including:

- finite difference schemes for 3D grids,
- tools for tensor calculations,
- tools to import data 
  from `Einstein Toolkit <https://einsteintoolkit.org>`_ simulations.

Installation
------------

Install aurel using `pip <https://pip.pypa.io/en/stable/installation/>`_:

.. code-block:: bash

   pip install aurel

all other required packages will be installed automatically. 
Or get the :ref:`dev_version`.

Getting started
---------------

Start your Python session on a jupyter notebook or in a Python script 
and import the AurelCore class:

.. code-block:: python
   
   import aurel
   
   # Define your grid parameters
   param = {
       'Nx': 64, 'Ny': 64, 'Nz': 64,
       'xmin': -1.0, 'ymin': -1.0, 'zmin': -1.0,
       'dx': 0.03125, 'dy': 0.03125, 'dz': 0.03125,
   }
   
   # Initialize the finite difference class
   fd = aurel.FiniteDifference(param)
   
   # Initialize the AurelCore class
   rel = aurel.AurelCore(fd)

At this point you need to provide the spacetime metric, extrinsic curvature
and matter fields, see :ref:`assumed_quantities`, where these would otherwise 
be assumed to correspond to Minkowski vaccum.
These are passed as numpy arrays to aurel in the following way:

.. code-block:: python

   # Define the xx component of the spacetime metric
   rel.data['gxx'] = np.ones((param['Nx'], param['Ny'], param['Nz']))

and so on for the other components and required quantities. 
In this example $g_{xx} = 1$, but you can pass any numpy array;
it can be of numerical relativity simulation data, 
or an array generated from an analytical expression.

With everything defined, you can call any entity listed in the 
:ref:`descriptions_list` list. Just call it as:

.. code-block:: python

   rel["name_of_the_entity"]

**Aurel will automatically do its best to calculate 
any relativistic term you ask for.**

Now if you want calculations over multiple moments in coordinate time, you can 
use the :func:`aurel.over_time` function. First, create a dictionary 
containing the spacetime and matter fields over time:

.. code-block:: python
   
   # Define the spacetime metric gxx over time
   data = {
       'gxx': [np.ones((param['Nx'], param['Ny'], param['Nz'])),
               np.ones((param['Nx'], param['Ny'], param['Nz']))],
       ...
   }

That is a data dictionary with keys the names of the quantities, and values a 
list of numpy arrays, one for each time step. Calculations will be performed 
for each of these time steps as:

.. code-block:: python

   data = aurel.over_time(
      data, fd, 
      vars=['name_of_the_entity', ..., {'custom_quantity': custom_quantity}], 
      estimates=['name_of_estimate', ..., {'custom_estimate': custom_estimate}],
      **kwargs)

where `vars` is a list of entities to calculate at each time step, available 
terms are in :ref:`descriptions_list` but you can also pass custom functions 
to calculate your own quantities.
Likewise `estimates` is a list of estimates to calculate at each time step, 
available functions are in :ref:`estimates_list` but you can also pass custom 
functions to calculate your own estimates. See the examples.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   source/core
   source/coreanalytic
   source/finitedifference
   source/maths
   source/numerical
   source/reading
   source/time
   source/solutions

.. toctree::
   :maxdepth: 1
   :caption: Advice:

   source/advice

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   notebooks/Example
   notebooks/Example_over_time
   notebooks/tov_ET
   notebooks/ICPertFLRW
   notebooks/Schwarzschild_check
   notebooks/Analytic_check
   notebooks/Gravitational_Waves

Links
-----
GitHub page: `<https://github.com/robynlm/aurel>`_

PyPI page: `<https://pypi.org/project/aurel/>`_

Contact the author at : r.l.munoz@sussex.ac.uk