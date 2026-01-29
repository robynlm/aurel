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
 - name: Will J. Roper
   orcid: 0000-0002-3257-8806
   affiliation: 1
affiliations:
 - name: Department of Physics and Astronomy, University of Sussex, Brighton, BN1 9QH, United Kingdom
   index: 1
date: 20 January 2025
bibliography: paper.bib
---

# Summary

`aurel` is an open-source Python package designed to *au*tomatically calculate *rel*ativistic quantities. 
It uses an efficient, flexible and user-friendly caching and dependency-tracking system, ideal for managing the highly nonlinear nature of general relativity.
The package supports both symbolic and numerical calculations.
The symbolic part extends `SymPy` with additional tensorial calculations.
The numerical part computes a wide range of tensorial quantities, such as curvature, matter kinematics and much more, directly from any spacetime and matter data arrays using finite-difference methods.
Inputs can be either generated from analytical expressions or imported from Numerical Relativity (NR) simulations, with helper functions provided to read in data from standard NR codes.
Given the increasing use of NR, `aurel` offers a timely post-processing tool to support the popularisation of this field.

# Statement of need

General relativity describes matter as moving according to how distances shrink or expand; likewise, the intervals of space and time evolve depending on the distribution of matter. 
Handling this dynamic ``mesh" of distances and times requires elaborate tensor algebra that, in some cases, can only be managed with symbolic or numerical tools.
Naturally, NR has become essential for modern astrophysics, cosmology, and gravitational physics, most notably in the modelling of gravitational-wave signals. 

While established computational frameworks focus on solving and evolving Einstein's field equations, with specific key diagnostics, they leave calculations of the remaining analysis to the discretion of the researchers.
Newcomers to the field then face a substantial overhead until they develop their own personal post-processing codes.
Established researchers also face the tedious task of handling intermediary variables and indices when calculating new quantities.
The field then suffers from this error-prone, time-consuming process and would benefit from an accessible, open-source, standardised framework to automate these steps.

We therefore present `aurel`, an open-source Python package designed to streamline relativistic calculations. 
It is hosted on [GitHub](https://github.com/robynlm/aurel) and is available on [PyPI](https://pypi.org/project/aurel/).
The documentation is available through [GitHub Pages](https://robynlm.github.io/aurel/).

# State of the field

When looking for general relativity Python packages, there are a number of tools that provide symbolic calculations [@GraviPy2014; @SageManifolds2015; @PyHole2017; @EinsteinPy2020; @Pytearcat2022; @PyGRO2025; @OGRePy2025; @GREOPy2025].
Or, one may also consider computer algebra systems [@Maple2025; @Mathematica2025; @xAct2025].
However, when non-linearities become too complex for symbolic packages, NR is used instead.

`Einstein Toolkit` [@ET2012; @ET2025] is a large community-driven software whose tools enable the evolution of Einstein's field equations.
Diagnostic and further analysis calculations are typically performed on the fly, during simulations.
To study the outputs, provided by `Carpet`, there are Python reading packages available [@PostCactus; @kuibit2021; @scidata; @mayawaves2025].
These extra calculations can slow down the simulation of the spacetime evolution, and if certain relativistic quantities are not available in `Einstein Toolkit`, or in one of the post-processing packages, then the user needs to code that up themselves. 

There are a number of other well-established NR codes [@METHOD2018; @GRAMSES2019; @GRChombo2021; @ExaGRyPE2024; @MHDuet2025] that also have their own diagnostic tools.
However, these are typically built-in, so going from one code to another, to benchmark or to use their different types of applications, requires learning the ecosystem of each.

To improve the community's versatility and limit the repeated implementation of error-prone calculations, there is a motivation to provide packages for computing relativistic quantities in an NR-code-agnostic way.
Especially in the post-processing sense, where all calculations are done from a given NR spacetime and matter solution.
A couple of notable packages [@distorted-motsfinder2018; @BiGONLight2021] focus on ray tracing, or apparent-horizon finding, which are currently beyond the scope of `aurel`. 
While others have more overlap [@EBWeyl2023; @EinFields2025] in calculating curvature terms, they differ in scope and workflow.

Here, `aurel` innovates in its automatic design, which is easily extendable and provides flexibility and robustness with a large and ever-growing catalogue of relativistic quantities.
A precursor to this package was `EBWeyl` [@EBWeyl2023], as it provided calculations of gravito-electromagnetic contributions from base spacetime and matter quantities. 
`aurel` now has a completely different structure (relying on the automatic dependency resolution), provides calculations of many more terms, over time, and has entirely new features as described in the following section.

# Software Design

`aurel` provides an intuitive interface for the automatic calculation of general relativistic quantities, either symbolically (with `AurelCoreSymbolic`, built on `SymPy` [@SymPy2017]) or numerically (with `AurelCore`, which heavily utilises `numpy.einsum` [@NumPy2020] for efficient operations on array data structures).

Both require base quantities such as the spacetime coordinates or the parameters of the Cartesian numerical grid, as well as the spacetime and matter distributions (the Minkowski vacuum is otherwise assumed). 
These inputs can either come from analytical expressions, with a couple of built-in solutions available, or from output data from any NR simulations; they just need to be passed as numpy arrays.

Specifically, for simulations run with `Carpet` in the `Einstein Toolkit`, the `reading` module provides helper functions to load and organise the 3D data.
These can read the parameter file, summarise available iterations and variables, and handle data separated across restarts, chunks, or refinement levels for normal `Carpet` data files or checkpoint files.
To speed up repeated data reading, `read_data` can also split the data per iteration, instead of per variable. 

Then, once input data is provided, users can directly request a wide range of relativistic quantities, including: spacetime; matter (Eulerian, Lagrangian, or conserved); NR formulations; constraints; fluid covariant kinematics; null ray expansion; 3- and 4-dimensional curvature; gravito-electromagnetism; Weyl scalars and invariants (including gravitational waves). 
To see a full list of available quantities, see: [descriptions](https://robynlm.github.io/aurel/source/core.html#descriptions-of-available-terms).
Tools are also provided for spatial and spacetime covariant derivatives and Lie derivatives.
All spatial derivatives are computed by the `FiniteDifference` class that provides 2nd, 4th, 6th and 8th order schemes, using periodic, symmetric or one-sided boundary conditions.

## Automatic Computational Pathway

The `aurel` automatic process composes a computational pathway at runtime to evaluate the requested quantities.
This is implemented through a lazy-evaluation memoised property pattern, where each quantity is defined as a method of the core class that may depend on other quantities. This design has been chosen for its flexibility and accessibility while remaining robust under future extensions.

Quantities are requested via a user-friendly dictionary-style access, e.g. `rel["s_RicciS"]`, which triggers the lazy memoised check to see if this is already cached. 
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

# Research Impact Statement

`aurel` is a specialist tool for general relativity researchers and streamlines numerical relativists' post-processing workflow.
Through conference interactions and collaborations involving the authors, this package has gradually been disseminated to individual researchers who appreciate the effortless integration, satisfying dependency resolution and substantial reduction to post-processing overhead.
Indeed, in ongoing studies involving NR simulations of primordial black hole formation, `aurel` has increased capacity and redirected repetitive and error-prone development efforts towards exploring a broader range of simulated scenarios.
Additionally, for master students, the straightforward and transparent design has provided an easy gateway for them to analyse NR simulations and so quickly get results within the duration of their projects.
Going forward, awareness of this code will build upon publication, reaching a wider audience and supporting the popularisation of NR.

# AI usage disclosure

GitHub Copilot Claude Sonnet 4 was used for the development and documentation of this package.
Autocompletion suggestions were accepted via the VSCode Copilot plugin, and upon the developer's request, edits and code snippets were generated via the large language model's user interface. 
The most significant AI contributions came in drafting the docstrings and scaffolding the test suite, both of which are essential for the accessibility and robustness of this package.
Each and every suggestion or contribution was meticulously reviewed and adjusted before being included by the authors, who made all core design decisions and innovated the original structural concept.
Finally, this paper was prepared without the use of generative language models, solely with grammar checkers.

# Acknowledgements

We thank Nat Kemp for being one of the first testers of `aurel`.
We thank Ian Hawke for support and suggestions.

RM and WR are supported by an STFC grant ST/X001040/1.
CB is supported by STFC grants ST/X001040/1 and ST/X000796/1.
