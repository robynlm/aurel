.. aurel documentation master file, created by
   sphinx-quickstart on Tue May 13 10:39:45 2025.

Welcome to aurel's documentation! 
=================================

.. image:: https://img.shields.io/badge/GitHub-aurel-blue?logo=github
   :target: https://github.com/robynlm/aurel
   :alt: GitHub

.. image:: https://img.shields.io/badge/python-3.11%2B-blue
   :target: https://www.python.org/
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://img.shields.io/pypi/v/aurel?logo=pypi&logoColor=white
   :target: https://pypi.org/project/aurel/
   :alt: PyPI

.. image:: https://github.com/robynlm/aurel/actions/workflows/test-build.yml/badge.svg?branch=main
   :target: https://github.com/robynlm/aurel/actions/workflows/test-build.yml
   :alt: Test build

.. image:: https://img.shields.io/badge/Email-r.l.munoz@sussex.ac.uk-red?logo=gmail&logoColor=white
   :target: mailto:r.l.munoz@sussex.ac.uk
   :alt: Email

.. raw:: html

   <p>Aurel is an open source Python package for numerical relativity analysis.
   Designed with ease of use in mind, it will 
   <span style="font-weight: bold;">au</span>tomatically calculate 
   <span style="font-weight: bold;">rel</span>ativistic terms. 
   </p>

In addition, this package also provides tools for 
advanced numerical computations, including:

- tools for tensor calculations:

  - numerically (:doc:`source/core`, :ref:`getting_started`, :doc:`notebooks/Example`, :doc:`notebooks/Example_over_time`, :doc:`notebooks/Gravitational_Waves`)
  - symbolically (:doc:`source/coresymbolic`, :doc:`notebooks/Example_symbolic`)

- finite difference schemes for 3D grids (:doc:`source/finitedifference`)
- tools to import data from Carpet `Einstein Toolkit <https://einsteintoolkit.org>`_ simulations (:doc:`source/reading`, :doc:`notebooks/tov_ET`)
- analytical solutions (:doc:`source/solutions`, :doc:`notebooks/ICPertFLRW`, :doc:`notebooks/Analytic_check`, :doc:`notebooks/Schwarzschild_check`)

Installation
------------

Install aurel using `pip <https://pip.pypa.io/en/stable/installation/>`_:

.. code-block:: bash

   pip install aurel

all other required packages will be installed automatically. 
Or get the :ref:`dev_version`.

.. _getting_started:

Getting started
---------------

Start your Python session on a jupyter notebook or in a Python script 
and import the :class:`aurel.AurelCore` class:

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

For a more in depth example, see the :doc:`notebooks/Example` for applications. 
Additionally, a symbolic counterpart that works in a very similar way is also available via the 
:class:`aurel.AurelCoreSymbolic` class, see the :doc:`notebooks/Example_symbolic` for details.

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
functions to calculate your own estimates. See the examples in :doc:`notebooks/Example_over_time`.

.. toctree::
   :maxdepth: 1
   :hidden:

   source/advice

.. toctree::
   :maxdepth: 1
   :hidden:

   source/CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   source/core
   source/coresymbolic
   source/finitedifference
   source/maths
   source/numerical
   source/reading
   source/time
   source/solutions

.. toctree::
   :maxdepth: 1
   :caption: Examples:
   :hidden:

   notebooks/Example
   notebooks/Example_symbolic
   notebooks/Example_over_time
   notebooks/tov_ET
   notebooks/ICPertFLRW
   notebooks/Schwarzschild_check
   notebooks/Analytic_check
   notebooks/Gravitational_Waves