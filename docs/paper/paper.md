---
title: "aurel: A Python package for automatic relativistic calculations"
tags:
 - Python
 - General Relativity
 - Numerical Relativity
 - tensor calculus
authors:
 - name: Robyn L. Munoz
    orcid: 0000-0003-3345-8520
    affiliation: 1
 - name: Christian T. Byrnes
    orcid: 0000-0003-2583-6536
    affiliation: 1
 - name: Will Roper
    orcid: 0000-0002-3257-8806
    affiliation: 1
affiliations:
 - name: Department of Physics and Astronomy, University of Sussex, Brighton, BN1 9QH, United Kingdom
   index: 1
date: 20 January 2025
bibliography: paper.bib
---

# Summary/Abstract

`aurel` is an open-source Python package designed to *au*tomatically calculate *rel*ativistic quantities. 
It uses an efficient, flexible and user-friendly caching and dependency-tracking system, ideal for managing the highly nonlinear nature of general relativity.
The package supports both symbolic and numerical calculations.
The symbolic part extends `SymPy` with additional tensorial calculations.
The numerical part computes a wide range of tensorial quantities, such as curvature, matter kinematics and much more, directly from any spacetime and matter data arrays using finite-difference methods.
Inputs can be either generated from analytical expressions or imported from numerical relativity simulations. 
For users of `Einstein Toolkit`, `aurel` also provides utilities to load 3D data generated with `Carpet`.
Given the increasing use of numerical relativity, `aurel` offers a timely post-processing tool to support the popularisation of this field.

# Statement of need

General relativity describes matter as moving according to how distances shrink or expand; likewise, the intervals of time and space evolve depending on the distribution of matter. 
Handling this dynamic ``mesh'' of distances and times requires elaborate tensor algebra that, in some cases, can only be managed with symbolic or numerical tools.
Naturally, numerical relativity has become an essential tool in modern astrophysics, cosmology, and gravitational physics, most notably in the modelling of gravitational-wave signals. 

While established computational frameworks focus on solving and evolving Einstein's field equations with specific key diagnostics, they leave calculations of the remaining analysis to the discretion of the researchers.
Newcomers to the field then face a substantial overhead until they develop their own personal post-processing codes.
Established researchers also face the tedious task of handling intermediary variables and indices when calculating new quantities.
The field then suffers from this error-prone, time-consuming process and would benefit from an accessible, open-source, standardised framework to automate these steps.

Therefore, we present `aurel`, an open-source Python package designed to streamline relativistic calculations. 
It is hosted on [GitHub](https://github.com/robynlm/aurel) and is available on [PyPI](https://pypi.org/project/aurel/).
The documentation is available through [GitHub Pages](https://robynlm.github.io/aurel/).

# Features

`aurel` provides an intuitive interface for the automatic calculation of general relativistic quantities, either symbolically (with `AurelCoreSymbolic`, built on `SymPy` [@SymPy2017]) or numerically (with `AurelCore`, which heavily utilises `numpy.einsum` [@NumPy2020] for efficient operations on array data structures).

Both require base quantities such as the spacetime coordinates or the parameters of the Cartesian numerical grid, as well as the spacetime and matter distributions (the Minkowski vacuum is otherwise assumed). 
These inputs can either come from analytical expressions, with a couple of built-in solutions available, or from output data from any numerical relativity simulations. 

Specifically, for simulations run with `Carpet` in the `Einstein Toolkit`, the `reading` module provides helper functions to load data in that specific format.
These can read the parameter file, create summarising files of the iterations and data variables available. Then the `read_data` function can handle data separated into different simulation restarts, join different multiprocessing chunks, and read different refinement levels. Since the raw data can sometimes take a long time to open, `read_data` can also split the data per iteration, instead of per variable, for quicker access in future reads. Additionally, `read_data` also has the capacity to read in data directly from checkpoint files instead.

Then, once input data is provided, users can directly request a wide range of relativistic quantities, including: spacetime; matter (Eulerian, Lagrangian, or conserved); BSSNOK; constraints; fluid covariant kinematics; null ray expansion; 3- and 4-dimensional curvature; gravito-electromagnetism; Weyl scalars and invariants (including gravitational waves). 
To see a full list of available quantities, see: [descriptions](https://robynlm.github.io/aurel/source/core.html#descriptions-of-available-terms).
Tools are also provided for spatial and spacetime covariant derivatives and Lie derivatives.
All spatial derivatives are computed by the `FiniteDifference` class that provides 2nd, 4th, 6th and 8th order schemes, using periodic, symmetric or one-sided boundary conditions.

## Automatic Computational Pathway

The `aurel` automatic process composes a computational pathway at runtime to evaluate the requested quantities.
This is implemented through a lazy-evaluation memoised property pattern, where each quantity is defined as a method of the core class that may depend on other quantities. 

Users request quantities via a dictionary-style access, e.g. `rel["s_RicciS"]`, which triggers the lazy memoised check to see if this is already cached. 
If yes, then the result is directly returned.
If not, then the corresponding method is called, which recursively triggers the calculation of dependencies (e.g. `rel["s_Ricci_down3"]`). 
This continues until the requested quantities can be calculated and so returned. 

To avoid redundant computations, each result is cached, which builds up a cache memory that needs to be efficiently managed. 
So, inspired by Python's garbage collection, `aurel` uses an intelligent eviction policy that tracks memory footprint, evaluation counts, and last-access times. 
When the configurable thresholds are exceeded, the older and heavier cached quantities are removed, while safeguarding protected base quantities. 
Throughout this process, `aurel` keeps the user informed on progress by providing verbose updates on the computation and caching workflow.

## Time dependence

All calculations within the `AurelCore` class are evaluated at a single fixed time, corresponding to one slice in time, so that individual time steps can be treated independently.
For multiple time steps, an `AurelCore` object needs to be created and the requested quantities collected for each.

To streamline this process, `aurel` provides the `over_time` function to do exactly this, and also compute summary statistics over the grid domain (e.g., max/min) at each time step.
By design, it is easily extensible, so `over_time` also accepts custom functions of new relativistic quantities and summary statistics.
This makes `aurel` versatile, supporting an infinite number of ways to view a problem and develop diagnostic tools.

# Related packages

`EBWeyl` [@EBWeyl2023] was a precursor to `aurel` as it provided calculations of gravito-electromagnetic contributions from base spacetime and matter quantities. 
`aurel` has a completely different structure (relying on the automatic dependency resolution), provides calculations of many more terms, over time, and has entirely new features: symbolic calculations, reading `Einstein Toolkit` data, and built-in solutions.
These capabilities, therefore, overlap with many other codes; below, we list only Python packages.

`aurel` provides symbolic calculations, so may overlap with other general relativity symbolic packages, see: `GraviPy` [@GraviPy2014], `SageManifolds` [@SageManifolds2015], `EinsteinPy` [@EinsteinPy2020], `Pytearcat` [@Pytearcat2022], and `OGRePy` [@OGRePy2025].

`aurel` can work with data from any origin, whether analytically generated or from a numerical relativity simulation; they just need to be passed as numpy arrays. But `aurel` also provides tools to read 3D data from `Carpet` `Einstein Toolkit` simulations [@ET2012; @ET2025]. Therefore, there is also an overlap with: `PostCactus` [@PostCactus], `kuibit` [@kuibit2021], `scidata` [@scidata], and `mayawaves` [@mayawaves2025].

Then, several notable Python packages for general relativity focus on different aspects beyond the current targets of `aurel`:
 - code generation: `NRPy` [@NRPy2018]
 - evolution codes: `COFFEE` [@COFFEE2019] and `Engrenage` [@Engrenage]
 - geodesic and raytracing: `PyHole` [@PyHole2017], `EinsteinPy` [@EinsteinPy2020], `GREOPy` [@GREOPy2025], `PyGRO` [@PyGRO2025]
 - marginaly trapped outer surfaces: `distorted-motsfinder` [@distorted-motsfinder2018]

# Acknowledgements

We thank Nat Kemp for being one of the first testers of `aurel`.
We thank Ian Hawke for support and suggestions.

RM and WR are supported by an STFC grant ST/X001040/1.
CB is supported by STFC grants ST/X001040/1 and ST/X000796/1.
