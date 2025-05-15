.. aurel documentation master file, created by
   sphinx-quickstart on Tue May 13 10:39:45 2025.

.. image:: _static/aurel.png
   :alt: aurel logo
   :width: 250px

.. raw:: html

   <div style="margin-top: 20px;"></div>

Welcome to aurel's documentation! 
=================================

.. raw:: html

   <p>Aurel is an open source Python package for numerical relativity analysis.
   Designed with ease of use in mind, it will 
   <span style="font-weight: bold;">au</span>tomatically calculate any 
   <span style="font-weight: bold;">rel</span>ativistic term 
   your heart desires. 
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

In your terminal, run the following command to install aurel:

.. code-block:: bash

   pip install aurel

Getting started
---------------

Start your Python session on a jupyter notebook or in a Python script 
and import the AurelCore class:

.. code-block:: python
   
   import aurel
   
   # Define your grid parameters
   param = {
       'Nx': 64,
       'Ny': 64,
       'Nz': 64,
       'xmin': -1.0,
       'ymin': -1.0,
       'zmin': -1.0,
       'dx': 0.03125,
       'dy': 0.03125,
       'dz': 0.03125,
   }
   
   # Initialize the finite difference class
   fd = aurel.FiniteDifference(param)
   
   # Initialize the AurelCore class
   rel = aurel.AurelCore(fd)

At this point you need to provide the spacetime metric, extrinsic curvature
and matter fields, see :ref:`required_quantities`.
These are passed as numpy arrays to aurel in the following way:

.. code-block:: python

   # Define the xx component of the spacetime metric
   rel.data['gxx'] = np.ones((param['Nx'], param['Ny'], param['Nz']))

and so on for the other components and required quantities. 
In this example $g_{xx} = 1$, but you can pass any numpy array, 
it can be of numerical relativity simulation data, 
or a numpy array generated from an analytical expression.

Assumptions are made for other core quantities, 
if these are not valid they should be overwritten at this point, 
see :ref:`assumed_quantities`.

With everything defined, you can call any entity listed in the aurel.description list, 
see :ref:`descriptions_list`. Just call it as:

.. code-block:: python

   rel["name_of_the_entity"]

**Aurel will automatically do its best to calculate 
any relativistic term you ask for.**

For further examples on how to use aurel see:

- the `Example notebook <https://github.com/robynlm/aurel/blob/main/notebooks/Example.ipynb>`_, 
  for an in depth description with tips and tricks,
- the `tov_ET notebook <https://github.com/robynlm/aurel/blob/main/notebooks/tov_ET.ipynb>`_, 
  for an example of how to load in data from an Einstein Toolkit simulation.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   source/core
   source/coreanalytic
   source/finitedifference
   source/maths
   source/numerical
   source/reading
   source/solutions

You may contact the author at : r.l.munoz@sussex.ac.uk