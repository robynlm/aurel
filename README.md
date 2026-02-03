<p align="left">
    <a href="https://robynlm.github.io/aurel/" target = "_blank">
    <img src="https://raw.githubusercontent.com/robynlm/aurel/refs/heads/main/docs/_static/aurel.png" width="250px" />
</a>
</p>

# Welcome to aurel’s documentation!

<p align="left">
  <a href="https://robynlm.github.io/aurel/"><img src="https://img.shields.io/badge/docs-available-blue?logo=readthedocs"/></a>
  <a href="https://github.com/robynlm/aurel"><img src="https://img.shields.io/badge/GitHub-aurel-blue?logo=github"/></a>
  <a href="https://arxiv.org/abs/2602.00155"><img src="https://img.shields.io/badge/arXiv-2602.00155-b31b1b.svg?logo=arxiv&logoColor=white"/></a><br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white"/></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg?logo=opensourceinitiative&logoColor=white"/></a>
  <a href="https://pypi.org/project/aurel/"><img src="https://img.shields.io/pypi/v/aurel?logo=pypi&logoColor=white"/></a>
  <a href="https://github.com/robynlm/aurel/actions/workflows/test-build.yml"><img src="https://github.com/robynlm/aurel/actions/workflows/test-build.yml/badge.svg?branch=main"/></a>
  <a href="https://github.com/robynlm/aurel/actions/workflows/test-build.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/robynlm/aurel/main/.github/badges/coverage.json&logo=pytest&logoColor=white"/></a><br>
  <a href="mailto:r.l.munoz@sussex.ac.uk"><img src="https://img.shields.io/badge/Email-r.l.munoz@sussex.ac.uk-red?logo=gmail&logoColor=white"/></a>
  <a href=""><img src=""/></a>
</p>

Aurel is an open source Python package for numerical relativity analysis.
Designed with ease of use in mind, it will **au**tomatically calculate **rel**ativistic terms.

# Key Features

- **Comprehensive tensor calculations**:
  - Numerically (see the following sections and [documentation](https://robynlm.github.io/aurel/source/core.html), [examples](https://robynlm.github.io/aurel/notebooks/Example.html))
  - Symbolically ([documentation](https://robynlm.github.io/aurel/source/coresymbolic.html), [example](https://robynlm.github.io/aurel/notebooks/Example_symbolic.html))
- **Dynamic computation and intelligent caching**: Aurel automatically calculates only what's needed and caches results, significantly improving performance for complex calculations
- **Clear user feedback**: Progress bars and informative messages guide you through computations
- **Time evolution support**: Tools for analyzing data across multiple time steps ([documentation](https://robynlm.github.io/aurel/source/time.html), [example](https://robynlm.github.io/aurel/notebooks/Example_over_time.html))
- **Flexible data input**: Works seamlessly with numerical simulation data, analytical solutions, or custom numpy arrays
- **Einstein Toolkit integration**: Native support for reading Carpet HDF5 outputs from [Einstein Toolkit](https://einsteintoolkit.org) simulations ([documentation](https://robynlm.github.io/aurel/source/reading.html), [example](https://robynlm.github.io/aurel/notebooks/tov_ET.html))
- **Advanced [finite difference schemes](https://robynlm.github.io/aurel/source/finitedifference.html)**: Multiple discretization schemes for 3D spatial grids
- **Built-in [analytical solutions](https://robynlm.github.io/aurel/source/solutions.html)**: Pre-implemented spacetimes for testing and validation ([examples](https://robynlm.github.io/aurel/notebooks/ICPertFLRW.html))

# Installation

**Requirements:** 
- Python 3.11 or higher ([download here](https://www.python.org/downloads/))
- pip (usually included with Python, or [install separately](https://pip.pypa.io/en/stable/installation/))

Install aurel using pip (all other dependencies will be installed automatically):

```bash
pip install aurel
```

To get the latest development version:

```bash
pip install git+https://github.com/robynlm/aurel.git@development
```

# Getting started

Start your Python session on a jupyter notebook or in a Python script 
and import the `aurel.AurelCore` class:

```python
   
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
```

At this point you need to provide the spacetime metric, extrinsic curvature
and matter fields (see [assumed quantities](https://robynlm.github.io/aurel/source/core.html#assumed-quantities)), otherwise these are 
assumed to correspond to Minkowski vacuum.
These are passed as numpy arrays to aurel in the following way:

```python
# Define the xx component of the spacetime metric
rel.data['gxx'] = np.ones((param['Nx'], param['Ny'], param['Nz']))
```

and so on for the other components and required quantities. 
In this example $g_{xx} = 1$, but you can pass any numpy array;
it can be of numerical relativity simulation data, 
or an array generated from an analytical expression.
Take care to run `rel.freeze_data()` so that your input data 
is conserved during the cache cleanup.

With everything defined, you can call any entity listed in the 
[descriptions list](https://robynlm.github.io/aurel/source/core.html#descriptions-of-available-terms). 
Just call it as:

```python
rel["name_of_the_entity"]
```

**Aurel will automatically do its best to calculate any relativistic term you ask for.**

For a more in depth example, see the [example notebook](https://robynlm.github.io/aurel/notebooks/Example.html) for applications. 
Additionally, a symbolic counterpart that works in a very similar way is also available via the 
`aurel.AurelCoreSymbolic` class, see the [symbolic example notebook](https://robynlm.github.io/aurel/notebooks/Example_symbolic.html) for details.

## Calculations over time

If you want calculations over multiple moments in coordinate time, you can 
use the `aurel.over_time` function. First, create a dictionary 
containing the spacetime and matter fields over time:

```python
# Define the spacetime metric gxx over time
data = {
    'gxx': [np.ones((param['Nx'], param['Ny'], param['Nz'])),
            np.ones((param['Nx'], param['Ny'], param['Nz']))],
    ...
}
```

That is a data dictionary with keys the names of the quantities, and values a 
list of numpy arrays, one for each time step. Calculations will be performed 
for each of these time steps as:

```python
data = aurel.over_time(
   data, fd, 
   vars=['name_of_the_entity', ..., {'custom_quantity': custom_quantity}], 
   estimates=['name_of_estimate', ..., {'custom_estimate': custom_estimate}],
   **kwargs)
```

where `vars` is a list of entities to calculate at each time step (available 
terms are in the [descriptions list](https://robynlm.github.io/aurel/source/core.html#descriptions-of-available-terms), but you can also pass custom functions 
to calculate your own quantities).
Likewise `estimates` is a list of estimates to calculate at each time step (available functions are in the [estimates list](https://robynlm.github.io/aurel/source/time.html#available-estimate-functions), but you can also pass custom 
functions to calculate your own estimates). See the [time evolution example](https://robynlm.github.io/aurel/notebooks/Example_over_time.html) for more details.

# Citation

If you use aurel in your work, please cite or reference the arXiv preprint in the following way:

<details>
<summary style="cursor: pointer; padding: 15px; border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 5px; margin: 5px 0; font-weight: bold; font-size: 1.1em;"><b>Munoz, R. L., Byrnes, C. T., & Roper, W. J. (2026). aurel: A Python package for automatic relativistic calculations. <a href="
https://doi.org/10.48550/arXiv.2602.00155">DOI:10.48550/arXiv.2602.00155</a></b></summary>

```bibtex
@misc{aurel2026,
  title         = {aurel: A Python package for automatic relativistic calculations},
  author        = {Robyn L. Munoz and Christian T. Byrnes and Will J. Roper},
  year          = {2026},
  doi           = {10.48550/arXiv.2602.00155},
  eprint        = {2602.00155},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM},
  url           = {https://robynlm.github.io/aurel/}
}
```
</details>

<br>

This code is not published yet, so to add to citation count please also include its predecessor:

<details>
<summary style="cursor: pointer; padding: 15px; border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 5px; margin: 5px 0; font-weight: bold; font-size: 1.1em;"><b>Munoz, R. L., & Bruni, M. (2023). EBWeyl: a Code to Invariantly Characterize Numerical Spacetimes. <i>Classical and Quantum Gravity</i>, 40(13), 135010. <a href="https://doi.org/10.1088/1361-6382/acd6cf">DOI:10.1088/1361-6382/acd6cf</a></b></summary>

```bibtex
@article{R.L.Munoz_M.Bruni_2023,
  title     = {EBWeyl: a Code to Invariantly Characterize Numerical Spacetimes},
  author    = {Robyn L. Munoz and Marco Bruni},
  journal   = {Classical and Quantum Gravity},
  volume    = {40},
  number    = {13},
  pages     = {135010},
  year      = {2023},
  month     = {jun},
  doi       = {10.1088/1361-6382/acd6cf},
  archivePrefix = {arXiv},
  eprint    = {2211.08133},
  primaryClass = {gr-qc}
}
```
</details>
