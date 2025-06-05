"""
core.py

This module is the main event. It contains:
 - **descriptions**, a dictionary of all the variables used in the aurel 
   project
 - **AurelCore**, the main class responsible for managing the
   **AurelCore.data** dictionary. For an input spacetime and matter 
   distribution, AurelCore can automatically retrieve any relativistic 
   variable listed in the descriptions dictionary.
   This class has many attributes and functions, a
   large part of which are listed in the descriptions dictionary, 
   but also many other tensor calculus functions.
   The descriptions functions are called as:
   
      - **AurelCore.variable_name()** for the function itself
      - **AurelCore.data["variable_name"]** for the dictionary element,
        which is the same as calling the function but then
        saves it in the AurelCore.data dictionary.

"""

import sys
import scipy
import numpy as np
import jax
import jax.numpy as jnp
from collections.abc import Mapping, Sequence
from IPython.display import display, Math, Latex
from . import maths
import spinsfast

# Descriptions for each AurelCore.data entry and function
# assumed variables need to be listed in docs/source/source/generate_rst.py
descriptions = {
    # === Metric quantities
    # Spatial metric
    "gxx": r"$g_{xx}$ Metric with xx indices down (need to input)",
    "gxy": r"$g_{xy}$ Metric with xy indices down (need to input)",
    "gxz": r"$g_{xz}$ Metric with xz indices down (need to input)",
    "gyy": r"$g_{yy}$ Metric with yy indices down (need to input)",
    "gyz": r"$g_{yz}$ Metric with yz indices down (need to input)",
    "gzz": r"$g_{zz}$ Metric with zz indices down (need to input)",
    "gammadown3": r"$\gamma_{ij}$ Spatial metric with spatial indices down",
    "gammaup3": r"$\gamma^{ij}$ Spatial metric with spatial indices up",
    "dtgammaup3": (r"$\partial_t \gamma^{ij}$ Coordinate time derivative of"
                    + r" spatial metric with spatial indices up"),
    "gammadet": r"$\gamma$ Determinant of spatial metric",
    "gammadown4": (r"$\gamma_{\mu\nu}$ Spatial metric with spacetime indices"
                   + r" down"),
    "gammaup4": r"$\gamma^{\mu\nu}$ Spatial metric with spacetime indices up",
    # Extrinsic curvature
    "kxx": (r"$K_{xx}$ Extrinsic curvature with xx indices down"
            + r" (need to input)"),
    "kxy": (r"$K_{xy}$ Extrinsic curvature with xy indices down"
            + r" (need to input)"),
    "kxz": (r"$K_{xz}$ Extrinsic curvature with xz indices down"
            + r" (need to input)"),
    "kyy": (r"$K_{yy}$ Extrinsic curvature with yy indices down"
            + r" (need to input)"),
    "kyz": (r"$K_{yz}$ Extrinsic curvature with yz indices down"
            + r" (need to input)"),
    "kzz": (r"$K_{zz}$ Extrinsic curvature with zz indices down"
            + r" (need to input)"),
    "Kdown3": r"$K_{ij}$ Extrinsic curvature with spatial indices down",
    "Kup3": r"$K^{ij}$ Extrinsic curvature with spatial indices up",
    "Ktrace": r"$K = \gamma^{ij}K_{ij}$ Trace of extrinsic curvature",
    "Adown3": (r"$A_{ij}$ Traceless part of the extrinsic curvature"
               + r" with spatial indices down"),
    "Aup3": (r"$A^{ij}$ Traceless part of the extrinsic curvature"
             + r" with spatial indices up"),
    "A2": r"$A^2$ Magnitude of traceless part of the extrinsic curvature",
    # Lapse
    "alpha": r"$\alpha$ Lapse (need to input or I assume =1)",
    "dtalpha": (r"$\partial_t \alpha$ Coordinate time derivative"
                + r" of the lapse (need to input or I assume =0)"),
    # Shift
    "betaup3": (r"$\beta^{i}$ Shift vector with spatial indices up"
                + r" (need to input or I assume =0)"),
    "dtbetaup3": (r"$\partial_t\beta^{i}$ Coordinate time derivative"
                  + r" of the shift vector with spatial indices up"
                  + r" (need to input or I assume =0)"),
    "betadown3": r"$\beta_{i}$ Shift vector with spatial indices down",
    "betamag": r"$\beta_{i}\beta^{i}$ Magnitude of shift vector",
    # Timelike normal vector
    "nup4": (r"$n^{\mu}$ Timelike vector normal to the spatial metric"
                + r" with spacetime indices up"),
    "ndown4": (r"$n_{\mu}$ Timelike vector normal to the spatial metric"
                + r" with spacetime indices down"),
    # Spacetime metric
    "gdown4": r"$g_{\mu\nu}$ Spacetime metric with spacetime indices down",
    "gup4": r"$g^{\mu\nu}$ Spacetime metric with spacetime indices up",
    "gdet": r"$g$ Determinant of spacetime metric",
    # Null ray expansion
    "null_ray_exp": (r"$\Theta_{out}, \; \Theta_{in}$ List of expansion of"
                     + r" null rays radially going out and in respectively"),

    # === Matter quantities
    # Eulerian observer follows n^mu
    # Lagrangian observer follows u^mu
    "rho0": r"$\rho_0$ Rest mass energy density (need to input)",
    "press": r"$p$ Pressure (need to input or I assume =0)",
    "eps": (r"$\epsilon$ Specific internal energy"
            + r" (need to input or I assume =0)"),
    "rho": r"$\rho$ Energy density",
    "enthalpy": r"$h$ Specific enthalpy of the fluid",
    # Fluid velocity
    "w_lorentz": r"$W$ Lorentz factor (need to input or I assume =1)",
    "velup3": (r"$v^i$ Eulerian fluid three velocity with spatial indices up"
               + r" (need to input or I assume =0)"),
    "uup0": r"$u^t$ Lagrangian fluid four velocity with time indice up",
    "uup3": r"$u^i$ Lagrangian fluid four velocity with spatial indices up",
    "uup4": (r"$u^\mu$ Lagrangian fluid four velocity"
             + r" with spacetime indices up"),
    "udown3": (r"$u_\mu$ Lagrangian fluid four velocity"
               + r" with spatial indices down"),
    "udown4": (r"$u_\mu$ Lagrangian fluid four velocity"
               + r" with spacetime indices down"),
    "hdown4": (r"$h_{\mu\nu}$ Spatial metric orthonomal to fluid flow"
               + r" with spacetime indices down"),
    "hmixed4": (r"${h^{\mu}}_{\nu}$ Spatial metric orthonomal to fluid flow"
                + r" with mixed spacetime indices"),
    "hup4": (r"$h^{\mu\nu}$ Spatial metric orthonomal to fluid flow"
             + r" with spacetime indices up"),
    # Energy-stress tensor
    "Tdown4": r"$T_{\mu\nu}$ Energy-stress tensor with spacetime indices down",
    # Fluid quantities in Eulerian frame
    "rho_n": r"$\rho^{\{n\}}$ Energy density in the $n^\mu$ frame",
    "fluxup3_n": (r"$S^{\{n\}i}$ Energy flux (or momentum density) in the"
                  + r" $n^\mu$ frame with spatial indices up"),
    "fluxdown3_n": (r"$S^{\{n\}}_{i}$ Energy flux (or momentum density) in"
                    + r" the $n^\mu$ frame with spatial indices down"),
    "angmomup3_n": (r"$J^{\{n\}i}$ Angular momentum density"
                    + r" in the $n^\mu$ frame with spatial indices up"),
    "angmomdown3_n": (r"$J^{\{n\}}_{i}$ Angular momentum density"
                    + r" in the $n^\mu$ frame with spatial indices down"),
    "Stressup3_n": (r"$S^{\{n\}ij}$ Stress tensor in the $n^\mu$ frame"
                    + r" with spatial indices up"),
    "Stressdown3_n": (r"$S^{\{n\}}_{ij}$ Stress tensor in the $n^\mu$ frame"
                      + r" with spatial indices down"),
    "Stresstrace_n": (r"$S^{\{n\}}$ Trace of Stress tensor"
                      + r" in the $n^\mu$ frame"),
    "press_n": r"$p^{\{n\}}$ Pressure in the $n^\mu$ frame",
    "anisotropic_press_down3_n": (r"$\pi^{\{n\}_{ij}}$ Anisotropic pressure"
                                  + r" in the $n^\mu$ frame"
                                  + r" with spatial indices down"),
    "rho_n_fromHam": (r"$\rho^{\{n\}}$ Energy density in the $n^\mu$ frame"
                    + r" computed from the Hamiltonian constraint"),
    "fluxup3_n_fromMom": (r"$S^{\{n\}i}$ Energy flux (or momentum density) in"
                          + r" the $n^\mu$ frame with spatial indices up"
                          + r" computed from the Momentum constraint"),
    # Conserved quantities
    "conserved_D": r"$D$ Conserved mass-energy density in Wilson's formalism",
    "conserved_E": (r"$E$ Conserved internal energy density"
                    + r" in Wilson's formalism"),
    "conserved_Sdown4": (r"$S_{\mu}$ Conserved energy flux"
                         + r" (or momentum density) in Wilson's formalism"
                         + r" with spacetime indices down"),
    "conserved_Sdown3": (r"$S_{i}$ Conserved energy flux (or momentum density)"
                         + r" in Wilson's formalism"
                         + r" with spatial indices down"),
    "conserved_Sup4": (r"$S^{\mu}$ Conserved energy flux (or momentum density)"
                       + r" in Wilson's formalism with spacetime indices up"),
    "conserved_Sup3": (r"$S^{i}$ Conserved energy flux (or momentum density)"
                       + r" in Wilson's formalism with spatial indices up"),
    "dtconserved": (r"$\partial_t D, \; \partial_t E, \partial_t S_{i}$"
                    + r" List of coordinate time derivatives of conserved"
                    + r" rest mass-energy density, internal energy density"
                    + r" and energy flux (or momentum density)"
                    + r" with spatial indices down in Wilson's formalism"),
    # Kinematics
    "st_covd_udown4": (r"$\nabla_{\mu} u_{\nu}$ Spacetime covariant derivative"
                       + r" of Lagrangian fluid four velocity"
                       + r" with spacetime indices down"),
    "accelerationdown4": (r"$a_{\mu}$ Acceleration of the fluid"
                          + r" with spacetime indices down"),
    "accelerationup4": (r"$a^{\mu}$ Acceleration of the fluid"
                          + r" with spacetime indices up"),
    "s_covd_udown4": (r"$\mathcal{D}^{\{u\}}_{\mu} u_{\nu}$ Spatial covariant"
                      + r" derivative of Lagrangian fluid four velocity"
                      + r" with spacetime indices down, with respect to"
                      + r" spatial hypersurfaces orthonormal to"
                      + r" the fluid flow"),
    "thetadown4": (r"$\Theta_{\mu\nu}$ Fluid expansion tensor"
                   + r" with spacetime indices down"),
    "theta": r"$\Theta$ Fluid expansion scalar",
    "sheardown4": (r"$\sigma_{\mu\nu}$ Fluid shear tensor"
                   + r" with spacetime indices down"),
    "shear2": r"$\sigma^2$ Magnitude of fluid shear",
    "omegadown4": (r"$\omega_{\mu\nu}$ Fluid vorticity tensor"
                   + r" with spacetime indices down"),
    "omega2": r"$\omega^2$ Magnitude of fluid vorticity",
    "s_RicciS_u": (r"${}^{(3)}R^{\{u\}}$ Ricci scalar of the spatial metric"
                   + r" orthonormal to fluid flow"),

    # === Curvature quantities
    # of spatial metric
    "s_Gamma_udd3": (r"${}^{(3)}{\Gamma^{k}}_{ij}$ Christoffel symbols of"
                     + r" spatial metric with mixed spatial indices"),
    "s_Riemann_uddd3": (r"${}^{(3)}{R^{i}}_{jkl}$ Riemann tensor of"
                        + r" spatial metric with mixed spatial indices"),
    "s_Riemann_down3": (r"${}^{(3)}R_{ijkl}$ Riemann tensor of spatial metric"
                        + r" with all spatial indices down"),
    "s_Ricci_down3": (r"${}^{(3)}R_{ij}$ Ricci tensor of spatial metric"
                        + r" with spatial indices down"),
    "s_RicciS": r"${}^{(3)}R$ Ricci scalar of spatial metric",
    # of spacetime metric
    "st_Gamma_udd4": (r"${}^{(4)}{\Gamma^{\alpha}}_{\mu\nu}$"
                      + r" Christoffel symbols of spacetime metric"
                      + r" with mixed spacetime indices"),
    "st_Riemann_uddd4": (r"${}^{(4)}{R^{\alpha}}_{\beta\mu\nu}$"
                         + r" Riemann tensor of spacetime metric"
                         + r" with mixed spacetime indices"),
    "st_Riemann_down4": (r"${}^{(4)}R_{\alpha\beta\mu\nu}$"
                         + r" Riemann tensor of spacetime metric"
                         + r" with spacetime indices down"),
    "st_Riemann_uudd4": (r"${}^{(4)}{R^{\alpha\beta}}_{\mu\nu}$"
                         + r" Riemann tensor of spacetime metric"
                         + r" with mixed spacetime indices"),
    "st_Ricci_down4": (r"${}^{(4)}R_{\alpha\beta}$ Ricci tensor of spacetime"
                       + r" metric with spacetime indices down"),
    "st_Ricci_down3": (r"${}^{(4)}R_{ij}$ Ricci tensor of spacetime metric"
                       + r" with spatial indices down"),
    "st_RicciS": r"${}^{(4)}R$ Ricci scalar of spacetime metric",
    "Kretschmann": (r"$K={R^{\alpha\beta}}_{\mu\nu}{R_{\alpha\beta}}^{\mu\nu}$"
                    + r" Kretschmann scalar"),

    # Constraints
    "Hamiltonian": r"$\mathcal{H}$ Hamilonian constraint",
    "Hamiltonian_Escale": (r"[$\mathcal{H}$] Hamilonian constraint"
                           + r" energy scale"),
    "Momentumup3": (r"$\mathcal{M}^i$ Momentum constraint"
                    + r" with spatial indices up"),
    "Momentum_Escale": (r"[$\mathcal{M}$] Momentum constraint"
                        + r" energy scale"),

    # === Gravito-electromagnetism quantities
    "st_Weyl_down4": (r"$C_{\alpha\beta\mu\nu}$ Weyl tensor of spacetime"
                      + r" metric with spacetime indices down"),
    "Weyl_Psi": (r"$\Psi_0, \; \Psi_1, \; \Psi_2, \; \Psi_3, \; \Psi_4$"
                 + r" List of Weyl scalars for an null vector base defined"
                 + r" with AurelCore.tetrad_to_use"),
    "Psi4_lm": (r"$\Psi_4^{l,m}$ Dictionary of spin weighted spherical"
                + r" harmonic decomposition of the 4th Weyl scalar,"
                + r" with AurelCore.Psi4_lm_radius and AurelCore.Psi4_lm_lmax."
                + r" ``spinsfast`` is used for the decomposition."),
    "Weyl_invariants": (r"$I, \; J, \; L, \; K, \; N$"
                        + r" Dictionary of Weyl invariants"),
    "eweyl_u_down4": (r"$E^{\{u\}}_{\alpha\beta}$ Electric part of the Weyl"
                      + r" tensor on the hypersurface orthogonal to $u^{\mu}$"
                      + r" with spacetime indices down"),
    "eweyl_n_down3": (r"$E^{\{n\}}_{ij}$ Electric part of the Weyl"
                      + r" tensor on the hypersurface orthogonal to $n^{\mu}$"
                      + r" with spatial indices down"),
    "bweyl_u_down4": (r"$B^{\{u\}}_{\alpha\beta}$ Magnetic part of the Weyl"
                      + r" tensor on the hypersurface orthogonal to $u^{\mu}$"
                      + r" with spacetime indices down"),
    "bweyl_n_down3": (r"$B^{\{n\}}_{ij}$ Magnetic part of the Weyl"
                      + r" tensor on the hypersurface orthogonal to $n^{\mu}$"
                      + r" with spatial indices down"),
}
    
###############################################################################
# Core class for aurel
###############################################################################

class AurelCore():
    r"""Class able to calculate any variable in aurel.descriptions.
    
    Parameters
    ----------
    fd : class
        aurel.finitedifference.FiniteDifference
    verbose : bool
        If True, display messages about the calculations.

    Attributes
    ----------
    param : dict
        (*dict*) - Dictionary containing the data grid parameters, from fd. 
    data : dict
        (*dict*) - Dictionary where all the variables are stored.
    data_shape : tuple
        (*tuple*) - Shape of the data arrays: (Nx, Ny, Nz)
    required_vars : set
        (*set*) - Set of required variables that must be defined by the user.

        "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",

        "kxx", "kxy", "kxz", "kyy", "kyz", "kzz",
        
        "rho0"
    fancy_print : bool
        (*bool*) - If True, display messages in a fancy ipython format, 
        else normal print is used. Default is True.
    Lambda : float
        (*float*) - Cosmological constant. Default is 0.0.
    tetrad_to_use : str
        (*str*) - Tetrad to use for calculations. 
        Default is "quasi-Kinnersley".
    Psi4_lm_lmax : int
        (*int*) - Maximum ell value used, increase this to improve convergence.
        Default is 8.
    Psi4_lm_radius : float
        (*float*) - Radius for the Psi4_lm calculations. 
        Default is 0.9 * min(Lx, Ly, Lz).
    kappa : float
        (*float*) - Einstein's constant with G = c = 1. Default is 8 * pi.
    calculation_count : int
        (*int*) - Number of calculations performed.
    clear_cache_every_nbr_calc : int
        (*int*) - Number of calculations before clearing the cache.
    memory_threshold_inGB : int
        (*int*) - Memory threshold in GB for clearing the cache.
    last_accessed : dict
        (*dict*) - Dictionary to keep track of when each variable was 
        last accessed.
    var_importance : dict
        (*dict*) - Dictionary to keep track of the importance of each variable
        for cache cleanup. To never delete a variable, set its importance to 0.
    """
    def __init__(self, fd, verbose=True):
        """Initialize the AurelCore class."""

        self.param = fd.param
        self.fd = fd
        self.data_shape = (self.param['Nx'], 
                           self.param['Ny'], 
                           self.param['Nz'])
        self.verbose = verbose
        self.fancy_print = True

        # Physics variables
        self.kappa = 8*jnp.pi  # Einstein's constant with G = c = 1
        self.myprint(f"Setting Cosmological constant "
                     + r'$\Lambda$'
                     + f" to 0.0, if not then redefine AurelCore.Lambda")
        self.Lambda = 0.0
        self.tetrad_to_use = "quasi-Kinnersley"
        self.Psi4_lm_lmax = 8
        self.Psi4_lm_radius = 0.9 * min(
            [self.param['Nx']*self.param['dx'] * 0.5, 
             self.param['Ny']*self.param['dy'] * 0.5, 
             self.param['Nz']*self.param['dz'] * 0.5])

        # data dictionary where everything is stored
        self.data = {}
        
        # List of required parameters 
        # (the user must define these in the data dictionary)
        # these will never be removed from the cache
        self.required_vars = {
            "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",
            "kxx", "kxy", "kxz", "kyy", "kyz", "kzz",
            "rho0", 
        }

        # To clean up cache
        self.calculation_count = 0
        self.clear_cache_every_nbr_calc = 20
        self.memory_threshold_inGB = 4 # GB
        self.last_accessed = {}

        # Importance of each variable for cache cleanup
        self.var_importance = {key:1.0 for key in descriptions.keys()}
        for key in list(self.required_vars):
            self.var_importance[key] = 0.0
        self.var_importance["s_Gamma_udd3"] = 0.002
        self.var_importance["s_RicciS"] = 0.1

    ###########################################################################
    # Functions to manage the data dictionary
    ###########################################################################

    def myprint(self, message):
        """Print a message with a fancy format."""
        if self.verbose:
            if self.fancy_print:
                display(Latex(message))
            else:
                print(message, flush=True)

    def assumption(self, keyname, assumption):
        """Print assumption message."""
        if self.verbose:
            message = ("I assume " + assumption + ", "
                    + "if not then please define "
                    + "AurelCore.data['"+keyname+"'] = ... ")
            self.myprint(message)

    def __getitem__(self, key):
        """Get data[key] or compute it if not present."""
        # First check if the key is already cached
        if key in self.data:
            # Update the last accessed time
            self.last_accessed[key] = self.calculation_count
            return self.data[key]

        # If the key is not in the data dictionary, check if it is required
        if key in self.required_vars:
            # Raise an error asking the user to define the variable
            self.myprint(f"I need {key}: {descriptions[key]}")
            raise ValueError(
                f"'{key}' is not defined. "
                + f"Please define AurelCore.data['{key}'] = ..."
                + f" in the data dictionary. ")
        
        # Dynamically get the function by name
        func = getattr(self, key)

        # Call the function if it takes no additional arguments
        if func.__code__.co_argcount == 1:
            self.data[key] = block_all(func())
            # Print the calculation description if available
            self.myprint(f"Calculated {key}: " + descriptions[key])
            self.calculation_count += 1 # Increment the calculation count
            self.last_accessed[key] = self.calculation_count # Update
            self.cleanup_cache()  # Clean the cache after each new calculation
            return self.data[key]

        # Return the function itself if it requires arguments
        return func
    
    def cleanup_cache(self):
        """Remove old entries from the cache based on age and size."""
        regular_cleanup = (
            self.calculation_count % self.clear_cache_every_nbr_calc == 0)

        total_cache_size = sum(sys.getsizeof(value) 
                               for value in self.data.values())
        memory_threshold = self.memory_threshold_inGB * 1024 * 1024 * 1024
        memory_limit_exceeded = total_cache_size >= memory_threshold
        if regular_cleanup or memory_limit_exceeded:
            self.myprint('CLEAN-UP: '+
                f"Cleaning up cache after {self.calculation_count}"
                + f" calculations...")
            self.myprint('CLEAN-UP: '+
                f"data size before cleanup: "
                + f"{total_cache_size / 1_048_576:.2f} MB")
            
            scalar_size = (
                self.param['Nx'] * self.param['Ny'] * self.param['Nz'] * 8)
            strain_tolerance = (self.clear_cache_every_nbr_calc * scalar_size)
            
            key_to_remove = []
            for key, last_time in self.last_accessed.items():
                # Calculate how many calculations ago 
                # the data entry was last accessed
                time_since_last_access = self.calculation_count - last_time
                # Get the size of the entry
                data_size = sys.getsizeof(self.data[key])
                if time_since_last_access > 1:
                    importance = self.var_importance.get(key, 1.0)
                    strain = (time_since_last_access * data_size 
                              * importance)
                else:
                    strain = 0

                # Consider old entries and large entries for removal
                if strain > strain_tolerance:
                    self.myprint('CLEAN-UP: '+
                        f"Removing cached value for '{key}'"
                        + f" used {time_since_last_access}"
                        + f" calculations ago (size: "
                        + f"{data_size / 1_048_576:.2f} MB).")
                    key_to_remove += [key]
            
            for key in key_to_remove:
                del self.data[key], self.last_accessed[key]
            nbr_keys_removed = len(key_to_remove)

            # if it's still too bog, then remove the most strain
            total_cache_size = sum(sys.getsizeof(value) 
                                   for value in self.data.values())
            while total_cache_size >= memory_threshold:
                maxstrain = 0
                for key, last_time in self.last_accessed.items():
                    time_since_last_access = self.calculation_count - last_time
                    if time_since_last_access > 1:
                        importance = self.var_importance.get(key, 1.0)
                        strain = (time_since_last_access 
                                * sys.getsizeof(self.data[key]) 
                                * importance)
                        if strain > maxstrain:
                            maxstrain = strain
                            key_to_remove = key
                if maxstrain == 0:
                    self.myprint('CLEAN-UP: '+
                        f"Current cache size "
                        + f"{total_cache_size / 1_048_576:.2f} MB, "
                        + f"max memory "
                        + f"{memory_threshold / 1_048_576:.2f} MB")
                    self.myprint('CLEAN-UP: '+
                        "Max memory too small,"
                        + "no more unimportant cache to remove.")
                    self.myprint('CLEAN-UP: '+
                                 "Current variables: ", self.data.keys())
                    break
                else:
                    # Remove the key with the maximum strain
                    if self.verbose:
                        calc_age = (self.calculation_count 
                                    - self.last_accessed[key_to_remove])
                        varsize = sys.getsizeof(self.data[key_to_remove])
                        self.myprint('CLEAN-UP: '+
                            f"Removing cached value for '{key_to_remove}' "
                            + f"used {calc_age} "
                            + f"calculations ago (size: "
                            + f"{varsize / 1_048_576:.2f} MB).")
                    del self.data[key_to_remove]
                    del self.last_accessed[key_to_remove]
                    nbr_keys_removed += 1
                    total_cache_size = sum(sys.getsizeof(value) 
                                           for value in self.data.values())

            self.myprint('CLEAN-UP: '+
                         f"Removed {nbr_keys_removed} items")
            self.myprint('CLEAN-UP: '+
                         f"data size after cleanup: "
                         + f"{total_cache_size / 1_048_576:.2f} MB")

    def load_data(self, sim_data, iteration):
        """Load simulation data into this classe's data dictionary, and freeze.

        Parameters
        ----------
        sim_data : dict
            Dictionary containing the simulation data, 
            each key has a list of values for each iteration.
        iteration : int
            The iteration number to load data from.
        """
        for key, values in sim_data.items():
            self.data[key] = values[iteration]
        self.freeze_data()

    def freeze_data(self):
        """Freeze the data dictionary to prevent cache removal."""
        for k in self.data.keys():
            self.var_importance[k] = 0

    ###########################################################################
    # Functions of the data dictionary
    ###########################################################################
        
    # === Metric quantities
    # Spatial metric
    def gammadown3(self):
        return maths.format_rank2_3(
            [self["gxx"], self["gxy"], self["gxz"],
             self["gyy"], self["gyz"], self["gzz"]])

    def gammaup3(self):
        return maths.inverse3(self["gammadown3"])
    
    def dtgammaup3(self):
        dbetaup = self.fd.d3_rank1tensor(self["betaup3"])
        Lbgup = (jnp.einsum(
            's..., sij... -> ij...', 
            self["betaup3"], self.fd.d3_rank2tensor(self["gammaup3"]))
            - jnp.einsum('si..., sj... -> ij...', dbetaup, self["gammaup3"])
            - jnp.einsum('sj..., is... -> ij...', dbetaup, self["gammaup3"]))
        return Lbgup - 2 * self["alpha"] * self["Kup3"]

    def gammadet(self):
        return maths.determinant3(self["gammadown3"])
    
    def gammadown4(self):
        return jnp.array(
                [[self["betamag"], self["betadown3"][0], 
                  self["betadown3"][1], self["betadown3"][2]], 
                 [self["betadown3"][0], self["gammadown3"][0,0], 
                  self["gammadown3"][0,1], self["gammadown3"][0,2]], 
                 [self["betadown3"][1], self["gammadown3"][1,0], 
                  self["gammadown3"][1,1], self["gammadown3"][1,2]], 
                 [self["betadown3"][2], self["gammadown3"][2,0], 
                  self["gammadown3"][2,1], self["gammadown3"][2,2]]])
    
    def gammaup4(self):
        zero = jnp.zeros(self.data_shape)
        return jnp.array(
                [[zero, zero, zero, zero], 
                 [zero, self["gammaup3"][0,0], 
                  self["gammaup3"][0,1], self["gammaup3"][0,2]], 
                 [zero, self["gammaup3"][1,0], 
                  self["gammaup3"][1,1], self["gammaup3"][1,2]], 
                 [zero, self["gammaup3"][2,0], 
                  self["gammaup3"][2,1], self["gammaup3"][2,2]]])
    
    # Extrinsic curvature
    def Kdown3(self):
        return maths.format_rank2_3(
            [self["kxx"], self["kxy"], self["kxz"],
             self["kyy"], self["kyz"], self["kzz"]])
    
    def Kup3(self):
        return jnp.einsum(
            'ia..., jb..., ij... -> ab...', 
            self["gammaup3"], self["gammaup3"],  self["Kdown3"])
    
    def Ktrace(self):
        return self.trace3(self["Kdown3"])
    
    def Adown3(self):
        return self["Kdown3"] - (1/3)*self["gammadown3"]*self["Ktrace"]
    
    def Aup3(self):
        return jnp.einsum('ia..., jb..., ab... -> ij...',
                         self["gammaup3"], self["gammaup3"], self["Adown3"])
    
    def A2(self):
        return self.magnitude3(self["Adown3"])
    
    # Lapse
    def alpha(self):
        self.assumption('alpha', r"$\alpha=1$")
        return jnp.ones(self.data_shape)
    
    def dtalpha(self):
        self.assumption('dtalpha', r"$\partial_t \alpha=0$")
        return jnp.zeros(self.data_shape)
    
    # Shift
    def betaup3(self):
        if "betax" in self.data:
            return jnp.array([
                self["betax"], self["betay"], self["betaz"]])
        else:
            self.assumption('betaup3', r"$\beta^i=0$")
            return jnp.zeros(
                (3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
    
    def dtbetaup3(self):
        if "dtbetax" in self.data:
            return jnp.array([
                self["dtbetax"], self["dtbetay"], self["dtbetaz"]])
        else:
            self.assumption('dtbetaup3', r"$\partial_t \beta^i=0$")
            return jnp.zeros(
                (3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
    
    def betadown3(self):
        return jnp.einsum(
                'i..., ij... -> j...',
                self["betaup3"], self["gammadown3"])
        
    def betamag(self):
        return jnp.einsum(
                'i..., i... -> ...',
                self["betaup3"], self["betadown3"])
    
    # Timelike normal vector
    def nup4(self):
        return maths.safe_division(jnp.array(
            [jnp.ones(self.data_shape), 
             -self["betaup3"][0],
             -self["betaup3"][1], 
             -self["betaup3"][2]]),
             self["alpha"])

    def ndown4(self):
        return jnp.array(
                [-self["alpha"], 
                 jnp.zeros(self.data_shape),
                 jnp.zeros(self.data_shape),
                 jnp.zeros(self.data_shape)])
    
    # Spacetime metric
    def gdown4(self):
        g00 = (-self["alpha"]**2 
               + jnp.einsum('i..., j..., ij... -> ...', 
                           self["betaup3"], self["betaup3"], 
                           self["gammadown3"]))
        return jnp.array(
                [[g00, self["betadown3"][0], 
                  self["betadown3"][1], self["betadown3"][2]],
                 [self["betadown3"][0], self["gammadown3"][0,0], 
                  self["gammadown3"][0,1], self["gammadown3"][0,2]],
                 [self["betadown3"][1], self["gammadown3"][1,0], 
                  self["gammadown3"][1,1], self["gammadown3"][1,2]],
                 [self["betadown3"][2], self["gammadown3"][2,0], 
                  self["gammadown3"][2,1], self["gammadown3"][2,2]]])
    
    def gup4(self):
        return maths.inverse4(self["gdown4"])

    def gdet(self):
        return maths.determinant4(self["gdown4"])
    
    # Null ray expansion
    def null_ray_exp(self):
        # outward pointing unit spatial vector
        r, phi, theta = self.fd.spherical_coords
        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        xynorm = (cosphi**2 * self["gammadown3"][0,0] 
              + 2 * cosphi * sinphi * self["gammadown3"][0,1]
              + sinphi**2 * self["gammadown3"][1,1])
        xyznorm = (
            sintheta**2 * xynorm 
            + 2 * costheta * sintheta * (
                cosphi * self["gammadown3"][0,2]
                + sinphi * self["gammadown3"][1,2])
            + costheta**2 * self["gammadown3"][2,2])
        nfac = maths.safe_division(1, jnp.sqrt(xyznorm))
        Sx = cosphi * sintheta * nfac
        Sy = sinphi * sintheta * nfac
        Sz = costheta * nfac
        sup = jnp.array([Sx, Sy, Sz])
        
        # expansion
        Disi = jnp.einsum('aa... -> ...', self.s_covd(sup, 'u'))
        Kss = jnp.einsum('ij..., i..., j... -> ...', self["Kdown3"], sup, sup)
        Theta_out = (Disi + Kss - self["Ktrace"])
        Theta_in = ( - Disi + Kss - self["Ktrace"])
        return Theta_out,  Theta_in
    
    # === Matter quantities
    # Eulerian observer follows n^mu
    # Lagrangian observer follows u^mu
    def press(self):
        self.assumption('press', r"$p=0$")
        return jnp.zeros(self.data_shape)
    
    def eps(self):
        self.assumption('eps', r"$\epsilon=0$")
        return jnp.zeros(self.data_shape)
    
    def rho(self):
        return self["rho0"] * (1 + self["eps"])
    
    def enthalpy(self):
        return (
            1 + self["eps"] 
            + maths.safe_division(self["press"], self["rho0"]))
    
    # Fluid velocity
    def w_lorentz(self):
        self.assumption('w_lorentz', r"$W=1$")
        return jnp.ones(self.data_shape)
    
    def velup3(self):
        if "velx" in self.data:
            return jnp.array([
                self["velx"], self["vely"], self["velz"]])
        else:
            self.assumption('velup3', r"$v^i=0$")
            return jnp.zeros(
                (3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
    
    def uup0(self):
        return maths.safe_division(self["w_lorentz"], self["alpha"])
    
    def uup3(self):
        return self["w_lorentz"] * (
            self["velup3"] 
            - maths.safe_division(self["betaup3"], self["alpha"]))
    
    def uup4(self):
        return jnp.array(
            [self["uup0"], self["uup3"][0], self["uup3"][1], self["uup3"][2]])
    
    def udown4(self):
        return jnp.einsum('ab..., b... -> a...', self["gdown4"], self["uup4"])
    
    def udown3(self):
        return self["udown4"][1:]
    
    def hdown4(self):
        return self["gdown4"] + jnp.einsum('a..., b... -> ab...', 
                                          self["udown4"], self["udown4"])
    
    def hmixed4(self):
        return jnp.einsum('ac...,cb...->ab...', 
                         self["gup4"], self["hdown4"])
    
    def hup4(self):
        return self["gup4"] + jnp.einsum('a..., b... -> ab...', 
                                        self["uup4"], self["uup4"])
    
    # Energy-stress tensor
    def Tdown4(self):
        return (self["rho"] * jnp.einsum('a..., b... -> ab...', 
                                        self["uup4"], self["uup4"])
                + self["press"] * self["hdown4"])
    
    # Fluid quantities in Eulerian frame
    def rho_n(self):
        return jnp.einsum('ab..., a..., b... -> ...',
                         self["Tdown4"], self["nup4"], self["nup4"])
    
    def fluxup3_n(self):
        return - jnp.einsum(
            'ab..., bc..., c... -> a...', 
            self["gammaup4"], self["Tdown4"], self["nup4"])[1:]
    
    def fluxdown3_n(self):
        return jnp.einsum(
            'a..., ab... -> b...', 
            self["fluxup3_n"], self["gammadown3"])
    
    def angmomup3_n(self):
        return jnp.einsum(
            'ij..., j... -> i...', 
            self["gammaup3"], self["angmomdown3_n"])
    
    def angmomdown3_n(self):
        return jnp.einsum(
            'ijk..., j..., k... -> i...', 
            self.levicivita_down3(), 
            self.fd.cartesian_coords, 
            self["fluxup3_n"])
    
    def Stressup3_n(self):
        return jnp.einsum(
            'ac..., bd..., ab... -> cd...', 
            self["gammaup3"], self["gammaup3"], self["Tdown4"][1:,1:])
    
    def Stressdown3_n(self):
        return jnp.einsum(
            'ac..., bd..., ab... -> cd...', 
            self["gammadown3"], self["gammadown3"], self["Stressup3_n"])
    
    def Stresstrace_n(self):
        return self.trace3(self["Tdown4"][1:,1:])
    
    def press_n(self):
        return jnp.einsum('ab..., ab... -> ...', 
                        self["gammaup3"], self["Tdown4"][1:,1:]) / 3
    
    def anisotropic_press_down3_n(self):
        return self["Stressdown3_n"] - self["gammadown3"] * self["press_n"]
    
    def rho_n_fromHam(self):
        return (self["s_RicciS"] 
                + self["Ktrace"]**2 
                - jnp.einsum('ij..., ij... -> ...', 
                            self["Kdown3"], self["Kup3"])
                - 2 * self.Lambda) / (2 * self.kappa)
    
    def fluxup3_n_fromMom(self):
        CovD_term = self.s_covd(
                self["Kup3"] - self["gammaup3"] * self["Ktrace"], 'uu')
        return jnp.einsum('bab... -> a...', CovD_term) / self.kappa
    
    # Conserved quantities
    def conserved_D(self):
        return self["rho0"] * self["w_lorentz"] * jnp.sqrt(self["gammadet"])
    
    def conserved_E(self):
        return self["conserved_D"] * self["eps"]
    
    def conserved_Sdown4(self):
        return self["conserved_D"] * self["enthalpy"] * self["udown4"]
    
    def conserved_Sdown3(self):
        return self["conserved_Sdown4"][1:]
    
    def conserved_Sup4(self):
        return jnp.einsum('im...,m...->i...', 
                         self["gup4"], self["conserved_Sdown4"])
    
    def conserved_Sup3(self):
        return self["conserved_Sup4"][1:]

    def dtconserved(self):
        self.myprint('WARNING: dtconserved only works for constant press/rho')
        V = maths.safe_division(self["uup4"][1:], self["uup4"][0])
        sgdet = jnp.sqrt(self["gammadet"]) 

        divD = jnp.einsum('ii...->...', 
                         self.fd.d3_rank1tensor(self["conserved_D"] * V))
        divE = jnp.einsum('ii... -> ...', 
                         self.fd.d3_rank1tensor(self["conserved_E"] * V))
        divW = jnp.einsum('ii... -> ...', 
                         self.fd.d3_rank1tensor(sgdet * self["w_lorentz"] * V))
        divSdown3 = jnp.einsum(
                    'jij...->i...', 
                    self.fd.d3_rank2tensor(
                        jnp.einsum('i...,j...->ij...', 
                                  self["conserved_Sdown3"], V)))

        dtD = - divD
        SSdg = ((self["conserved_Sdown4"][0] 
                 * self["conserved_Sdown4"][0] 
                 * self.fd.d3_scalar(self["gdown4"][0,0]))
                + (self["conserved_Sdown4"][0] 
                   * jnp.einsum('j..., ij... -> i...', 
                               self["conserved_Sup3"], 
                               self.fd.d3_rank1tensor(self["betadown3"])))
                + (jnp.einsum('j..., k..., ijk... -> i...', 
                             self["conserved_Sup3"],
                             self["conserved_Sup3"], 
                             self.fd.d3_rank2tensor(self["gammadown3"]))))
        dtSdown3 = (- divSdown3 
               + maths.safe_division( SSdg, 2 * self["conserved_Sdown4"][0])
               - self["alpha"] * sgdet * self.fd.d3_scalar(self["press"]))
        # dtw_lorentz
        CovDbeta = self.s_covd(self["betaup3"], 'u')
        dtsgdet = sgdet * ( - self["alpha"] * self["Ktrace"]
                            + jnp.einsum('ii... -> ...', CovDbeta))
        W2m1 = self["w_lorentz"]**2 - 1
        W2m1oDpE = maths.safe_division(
            W2m1, (self["conserved_D"] + self["conserved_E"]))
        fac1 = maths.safe_division(
            W2m1, 2 * self["w_lorentz"] 
            * jnp.einsum('ij..., i..., j... -> ...', 
                self["gammaup3"], 
                self["conserved_Sdown3"], 
                self["conserved_Sdown3"]))
        par1 = (jnp.einsum('ij..., i..., j... -> ...',
                          self["dtgammaup3"], 
                          self["conserved_Sdown3"], 
                          self["conserved_Sdown3"])
                + 2 * jnp.einsum('ij..., i..., j... -> ...',
                                self["gammaup3"], 
                                self["conserved_Sdown3"], 
                                dtSdown3))
        fac2 = - maths.safe_division(W2m1oDpE, self["w_lorentz"])
        par2 = (dtD
                 - divE
                 - self["press"] * divW)
        fac3 = self["press"] * W2m1oDpE * dtsgdet
        dtW = maths.safe_division(
            (fac1 * par1 + fac2 * par2 + fac3),
            ( 1 - maths.safe_division(self["press"] * sgdet * W2m1oDpE,
                                      self["w_lorentz"] )))
        dtsgdetW = dtsgdet * self["w_lorentz"] + sgdet * dtW
        dtE = (- divE - self["press"] * (dtsgdetW + divW))
        return dtD, dtE, dtSdown3
    
    # Kinematics
    def st_covd_udown4(self):
        dtD, dtE, dtSdown3 = self["dtconserved"]
    
        # dtu
        dtudown3 = self["udown3"] * (
            maths.safe_division(dtSdown3, self["conserved_Sdown3"])
            - maths.safe_division(dtD + dtE, 
                                  self["conserved_D"] + self["conserved_E"]))
        dtudown0 = (
            maths.safe_division(
                (jnp.einsum('ij...,i...,j...->...',
                        self["dtgammaup3"], self["udown3"], self["udown3"])
                + 2*jnp.einsum('i...,i...->...', self["uup3"], dtudown3)),
                (2 * self["alpha"] * self["w_lorentz"])) 
            - maths.safe_division(self["uup0"] * self["dtalpha"], 
                                  self["alpha"]))
        dtudown4 = jnp.array(
                    [dtudown0, 
                     dtudown3[0], dtudown3[1], dtudown3[2]])
        # spacetime covariant derivative
        return self.st_covd(self["udown4"], dtudown4, 'u')
    
    def accelerationdown4(self):
        return jnp.einsum(
            'a..., ab... -> b...',
            self["uup4"], self["st_covd_udown4"])
    
    def accelerationup4(self):
        return jnp.einsum(
            'ab..., b... -> a...',
            self["gup4"], self["accelerationdown4"])
    
    def s_covd_udown4(self):
        return jnp.einsum(
            'ab...,ac...->bc...', self["hmixed4"], self["st_covd_udown4"])
    
    def thetadown4(self):
        return maths.symmetrise_tensor(self["s_covd_udown4"])
    
    def theta(self):
        return jnp.einsum('ab..., ab... -> ...', 
                         self["hup4"], self["thetadown4"])
    
    def sheardown4(self):
        return self["thetadown4"] - (1/3) * self["theta"] * self["hdown4"]
    
    def shear2(self):
        return 0.5 * jnp.einsum(
            'ai..., bj..., ab..., ij... -> ...', 
            self["hup4"], self["hup4"], self["sheardown4"], self["sheardown4"])
    
    def omegadown4(self):
        return maths.antisymmetrise_tensor(self["s_covd_udown4"])
    
    def omega2(self):
        return 0.5 * jnp.einsum(
            'ai..., bj..., ab..., ij... -> ...', 
            self["hup4"], self["hup4"], self["omegadown4"], self["omegadown4"])
    
    def s_RicciS_u(self):
        return 2 * (
            self["shear2"]
            - (1/3) * self["theta"]**2
            + self.Lambda
            + self.kappa * self["rho"]
        )

    # === Curvature quantities
    # of spatial metric
    def s_Gamma_udd3(self):
        # First the spatial derivatives of the metric derivative are computed.
        dgammaxx = self.fd.d3_scalar(self["gammadown3"][0, 0]) 
        #        = [dxgxx, dygxx, dzgxx]
        dgammaxy = self.fd.d3_scalar(self["gammadown3"][0, 1])
        dgammaxz = self.fd.d3_scalar(self["gammadown3"][0, 2])
        dgammayy = self.fd.d3_scalar(self["gammadown3"][1, 1])
        dgammayz = self.fd.d3_scalar(self["gammadown3"][1, 2])
        dgammazz = self.fd.d3_scalar(self["gammadown3"][2, 2])
            
        # Spatial Christoffel symbols with all indices down: Gamma_{jkl}.
        Gxyz = dgammaxz[1] + dgammaxy[2] - dgammayz[0]
        Gx = 0.5 * jnp.array(
            [[dgammaxx[0], dgammaxx[1], dgammaxx[2]],
             [dgammaxx[1], 2*dgammaxy[1]-dgammayy[0], Gxyz],
             [dgammaxx[2], Gxyz, 2*dgammaxz[2]-dgammazz[0]]])
            
        Gyxz = dgammayz[0] + dgammaxy[2] - dgammaxz[1]
        Gy = 0.5 * jnp.array(
            [[2*dgammaxy[0]-dgammaxx[1], dgammayy[0], Gyxz],
             [dgammayy[0], dgammayy[1], dgammayy[2]],
             [Gyxz, dgammayy[2], 2*dgammayz[2]-dgammazz[1]]])
            
        Gzxy = dgammayz[0] + dgammaxz[1] - dgammaxy[2]
        Gz = 0.5 * jnp.array(
            [[2*dgammaxz[0]-dgammaxx[2], Gzxy, dgammazz[0]],
             [Gzxy, 2*dgammayz[1]-dgammayy[2], dgammazz[1]],
             [dgammazz[0], dgammazz[1], dgammazz[2]]])
        Gddd = jnp.array([Gx,Gy,Gz])
            
        # Spatial Christoffel symbols with indices: Gamma^{i}_{kl}.
        return jnp.einsum('ij..., jkl... -> ikl...', self["gammaup3"], Gddd)
    
    def s_Riemann_uddd3(self):
        dGudd3 = jnp.array([
            [self.fd.d3x_rank2tensor(self["s_Gamma_udd3"][j]) 
             for j in range(3)],
            [self.fd.d3y_rank2tensor(self["s_Gamma_udd3"][j]) 
             for j in range(3)],
            [self.fd.d3z_rank2tensor(self["s_Gamma_udd3"][j]) 
             for j in range(3)]])
        Rterm0 = jnp.einsum('cabd... -> abcd...', dGudd3)
        Rterm1 = jnp.einsum('dabc... -> abcd...', dGudd3)
        Rterm2 = jnp.einsum('apc..., pbd... -> abcd...', 
                        self["s_Gamma_udd3"], self["s_Gamma_udd3"])
        Rterm3 = jnp.einsum('apd..., pbc... -> abcd...', 
                            self["s_Gamma_udd3"], self["s_Gamma_udd3"])
        return Rterm0 - Rterm1 + Rterm2 - Rterm3
    
    def s_Riemann_down3(self):
        return jnp.einsum('abcd..., ai... -> ibcd...', 
                         self["s_Riemann_uddd3"], self["gammadown3"])
    
    def s_Ricci_down3(self):
        if "s_Riemann_down3" in self.data.keys():
            return jnp.einsum('abcd..., ac... -> bd...', 
                             self["s_Riemann_down3"], self["gammaup3"])
        else:
            dGudd3 = jnp.array([
                [self.fd.d3x_rank2tensor(self["s_Gamma_udd3"][j]) 
                    for j in range(3)],
                [self.fd.d3y_rank2tensor(self["s_Gamma_udd3"][j]) 
                    for j in range(3)],
                [self.fd.d3z_rank2tensor(self["s_Gamma_udd3"][j]) 
                    for j in range(3)]])
            Rterm0 = jnp.einsum('aabd... -> bd...', dGudd3)
            Rterm1 = jnp.einsum('daba... -> bd...', dGudd3)
            Rterm2 = jnp.einsum('apa..., pbd... -> bd...', 
                            self["s_Gamma_udd3"], self["s_Gamma_udd3"])
            Rterm3 = jnp.einsum('apd..., pba... -> bd...', 
                                self["s_Gamma_udd3"], self["s_Gamma_udd3"])
            return Rterm0 - Rterm1 + Rterm2 - Rterm3

    def s_RicciS(self):
        return self.trace3(self["s_Ricci_down3"])
        
    # of spacetime metric
    def st_Gamma_udd4(self):
        # Repeated calculations
        dalpha = self.fd.d3_scalar(self["alpha"])
        betadalpha = jnp.einsum('m..., m... -> ...', self["betaup3"], dalpha)
        betaK = jnp.einsum('m..., mn... -> n...', 
                          self["betaup3"], self["Kdown3"])
        betabetaK = jnp.einsum('m..., n... -> ...', self["betaup3"], betaK)
        dbeta = self.s_covd(self["betaup3"], 'u')

        # time part of index up
        Gttt = maths.safe_division(
            self["dtalpha"] + betadalpha - betabetaK, 
            self["alpha"])
        Gtti = maths.safe_division(dalpha - betaK, self["alpha"])
        Gtij = maths.safe_division(- self["Kdown3"], self["alpha"])
        Gt = jnp.array(
            [[Gttt, Gtti[0], Gtti[1], Gtti[2]],
            [Gtti[0], Gtij[0,0], Gtij[0,1], Gtij[0,2]],
            [Gtti[1], Gtij[1,0], Gtij[1,1], Gtij[1,2]],
            [Gtti[2], Gtij[2,0], Gtij[2,1], Gtij[2,2]]])

        # space part of index up
        Gltt = (
            jnp.einsum('lm..., m... -> l...', 
                    self["gammaup3"], 
                    self["alpha"] * dalpha - 2 * self["alpha"]*betaK)
            - self["betaup3"] * Gttt 
            + self["dtbetaup3"]
            + jnp.einsum('m..., ml...-> l...', self["betaup3"], dbeta))
        Glmt = (
            jnp.einsum('l..., m...-> lm...', -self["betaup3"], Gtti)
            - self["alpha"] * jnp.einsum('ln..., nm... -> lm...', 
                                    self["gammaup3"], 
                                    self["Kdown3"])
            + dbeta)
        Glij = (
            self["s_Gamma_udd3"] 
            + maths.safe_division(
                jnp.einsum(
                    'l..., ij... -> lij...', self["betaup3"], self["Kdown3"]), 
                    self["alpha"]))
        Gl = jnp.array(
            [[Gltt, Glmt[:,0], Glmt[:,1], Glmt[:,2]], 
            [Glmt[:,0], Glij[:,0,0], Glij[:,0,1], Glij[:,0,2]],
            [Glmt[:,1], Glij[:,1,0], Glij[:,1,1], Glij[:,1,2]],
            [Glmt[:,2], Glij[:,2,0], Glij[:,2,1], Glij[:,2,2]]])
        return jnp.array([Gt, Gl[:,:,0], Gl[:,:,1], Gl[:,:,2]])
    
    def st_Riemann_uddd4(self):
        return jnp.einsum('abcd..., ai... -> ibcd...',
                         self["st_Riemann_down4"], self["gup4"])
    
    def st_Riemann_down4(self):
        # Riemann_ssss : Gauss equation, eq 2.38 in Shibata
        Riemann_ssss = (self["s_Riemann_down3"]
                        + jnp.einsum('ac..., bd... -> abcd...', 
                                    self["Kdown3"], self["Kdown3"])
                        - jnp.einsum('ad..., bc... -> abcd...', 
                                    self["Kdown3"], self["Kdown3"]))
        
        # Riemann_ssst : Codazzi equation, eq 2.41 in Shibata
        dKdown = self.s_covd(self["Kdown3"], 'dd')
        Riemann_ssst = (
            jnp.einsum('ijkl..., l... -> ijk...', 
                       Riemann_ssss, self["betaup3"])
            + self["alpha"] * (
                jnp.einsum('jik... -> ijk...', dKdown) 
                - dKdown))
            
        # Riemann_stst: the Mainardi equation, eq 2.56 in Shibata
        Kdown4 = self.s_to_st(self["Kdown3"])
        Riemann_stst = (
            jnp.einsum('jki..., k... -> ij...', Riemann_ssst, self["betaup3"])
            + jnp.einsum('ikj..., k... -> ij...', 
                         Riemann_ssst, self["betaup3"])
            + jnp.einsum('ikjl..., k..., l... -> ij...', 
                        Riemann_ssss, self["betaup3"], self["betaup3"])
        + self["alpha"]**2 * (
            self["s_Ricci_down3"]
            - self["st_Ricci_down3"]
            - jnp.einsum('ib..., ja..., ab... -> ij...', 
                        Kdown4, Kdown4, self["gup4"])[1:,1:]
            + self["Kdown3"] * self["Ktrace"]))
            
        # put it all together
        R = maths.populate_4Riemann(
            Riemann_ssss, Riemann_ssst, Riemann_stst)
        return R
    
    def st_Riemann_uudd4(self):
        return jnp.einsum('abcd..., ae..., bf... -> efcd...', 
                         self["st_Riemann_down4"], self["gup4"], self["gup4"])
    
    def st_Ricci_down4(self):
        if "Tdown4" in self.data.keys():
            return (
            self.Lambda * self["gdown4"]
            + self.kappa * (
               self["Tdown4"]
                - 0.5 * self.trace4(self["Tdown4"]) * self["gdown4"]))
        else:
            return jnp.einsum('abad... -> bd...', self["st_Riemann_uddd4"])
    
    def st_Ricci_down3(self):
        if "st_Ricci_down4" in self.data.keys():
            return self["st_Ricci_down4"][1:,1:]
        else:
            return (
                self.Lambda * self["gammadown3"]
                + self.kappa * (
                    self["Tdown4"][1:,1:]
                    - 0.5 * self.trace4(self["Tdown4"]) * self["gammadown3"]))
    
    def st_RicciS(self):
        return self.trace4(self["st_Ricci_down4"])
    
    def Kretschmann(self):
        return jnp.einsum(
            'abcd..., cdab... -> ...', 
            self["st_Riemann_uudd4"], self["st_Riemann_uudd4"])
    
    # Constraints
    def Hamiltonian(self):
        return (self["s_RicciS"] 
                + self["Ktrace"]**2 
                - jnp.einsum('ij..., ij... -> ...', 
                            self["Kdown3"], self["Kup3"])
                - 2 * self.kappa * self["rho_n"] 
                - 2 * self.Lambda)
    
    def Hamiltonian_Escale(self):
        return jnp.sqrt(abs(
            self["s_RicciS"]**2 
            + self["Ktrace"]**4 
            + jnp.einsum('ij..., ij... -> ...', 
                        self["Kdown3"], self["Kup3"])**2 
            + (2 * self.kappa * self["rho_n"])**2 
            + (2 * self.Lambda)**2))
    
    def Momentumup3(self):
        if "Momentumx" in self.data.keys():
            return jnp.array(
                [self["Momentumx"], self["Momentumy"], self["Momentumz"]]) 
        else:
            CovD_term = self.s_covd(
                self["Kup3"] - self["gammaup3"] * self["Ktrace"], 'uu')
            return (jnp.einsum('bab... -> a...', CovD_term) 
                    - self.kappa * self["fluxup3_n"])
    
    def Momentum_Escale(self):
        DdKdd = self.s_covd(self["Kdown3"], 'dd')

        DKd = jnp.einsum('ab..., abc... -> c...', self["gammaup3"], DdKdd)
        DdK = jnp.einsum('bc..., abc... -> a...', self["gammaup3"], DdKdd)
        DKd2 = jnp.einsum('a..., ad..., d... -> ...', 
                         DKd, self["gammaup3"], DKd)
        DdK2 = jnp.einsum('a..., ad..., d... -> ...', 
                         DdK, self["gammaup3"], DdK)
        Eflux2 = ((self.kappa**2)
                  * jnp.einsum('a..., a... -> ...', 
                              self["fluxup3_n"], self["fluxdown3_n"]))
        return jnp.sqrt(abs(DKd2 + DdK2 + Eflux2))
    
    # === Gravito-electromagnetism quantities
    def st_Weyl_down4(self): 
        if "st_Riemann_down4" in self.data.keys():
            # TODO: accelerate this
            Cdown = jnp.zeros(jnp.shape(self["st_Riemann_down4"]))
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        for d in range(4):
                            Cdown = Cdown.at[a,b,c,d].set(
            self["st_Riemann_down4"][a,b,c,d]
            - 0.5 * (self["gdown4"][a,c] * self["st_Ricci_down4"][d,b]
                       - self["gdown4"][a,d] * self["st_Ricci_down4"][c,b]
                       + self["gdown4"][b,c] * self["st_Ricci_down4"][d,a]
                       - self["gdown4"][b,d] * self["st_Ricci_down4"][c,a])
            + (1/6) * self["st_RicciS"] * (
                self["gdown4"][a,c] * self["gdown4"][d,b]
                - self["gdown4"][a,d] * self["gdown4"][c,b]))
            return Cdown
        else:
            Endown4 = self.s_to_st(self["eweyl_n_down3"])
            Bndown4 = self.s_to_st(self["bweyl_n_down3"])
            ldown4 = (self["gdown4"] 
                      + 2.0 * jnp.einsum('a..., b... -> ab...',
                                        self["ndown4"], self["ndown4"]))
            LCudd4 = jnp.einsum('ec..., d..., dcab... -> eab...', self["gup4"], 
                            self["nup4"], self.levicivita_down4())
                
            Cdown4 = (jnp.einsum('ac..., db... -> abcd...', ldown4, Endown4)
                    - jnp.einsum('ad..., cb... -> abcd...', ldown4, Endown4))
            Cdown4 -= (jnp.einsum('bc..., da... -> abcd...', ldown4, Endown4)
                    - jnp.einsum('bd..., ca... -> abcd...', ldown4, Endown4))
            Cdown4 -= jnp.einsum('cde..., eab... -> abcd...',
                                (jnp.einsum('c..., de... -> cde...',
                                        self["ndown4"], Bndown4)
                                - jnp.einsum('d..., ce... -> cde...',
                                            self["ndown4"], Bndown4)), LCudd4)
            Cdown4 -= jnp.einsum('abe..., ecd... -> abcd...', 
                                (jnp.einsum('a..., be... -> abe...', 
                                        self["ndown4"], Bndown4) 
                                - jnp.einsum('b..., ae... -> abe...', 
                                            self["ndown4"], Bndown4)), LCudd4)
            return Cdown4
                   
    def Weyl_Psi(self):
        if "Weyl_Psi4r" in self.data.keys():
            return [None, None, None, None, 
                    self["Weyl_Psi4r"] + 1j * self["Weyl_Psi4i"]]
        else:
            lup4, kup4, mup4, mbup4 = self.null_vector_base()
            psi0 = jnp.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], kup4, mup4, kup4, mup4)
            psi1 = jnp.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], lup4, kup4, mup4, kup4)
            psi2 = jnp.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], kup4, mup4, mbup4, lup4)
            psi3 = jnp.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], kup4, lup4, mbup4, lup4)
            psi4 = jnp.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], lup4, mbup4, lup4, mbup4)
            return [psi0, psi1, psi2, psi3, psi4]
        
    def Psi4_lm(self):
        Ntheta = 2 * self.Psi4_lm_lmax + 1
        Nphi = 2 * self.Psi4_lm_lmax + 1
        # spinsfast assumes band-limited functions

        theta = jnp.linspace(0, jnp.pi, Ntheta)
        phi = jnp.linspace(0, 2*jnp.pi, Nphi, endpoint=False)
        theta_sphere, phi_sphere = jnp.meshgrid(theta, phi, indexing='ij')

        x_sphere = (self.Psi4_lm_radius 
                    * jnp.sin(theta_sphere) * jnp.cos(phi_sphere))
        y_sphere = (self.Psi4_lm_radius 
                    * jnp.sin(theta_sphere) * jnp.sin(phi_sphere))
        z_sphere = (self.Psi4_lm_radius 
                    * jnp.cos(theta_sphere))
        points_sphere = jnp.stack(
            (x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()), 
            axis=-1)

        # Psi4 on a sphere
        # Reconstruct full box
        # sphere around the origin
        # TODO: make origin location an option
        coord, Psi4r = self.fd.reconstruct(jnp.real(self["Weyl_Psi"][4]))
        coord, Psi4i = self.fd.reconstruct(jnp.imag(self["Weyl_Psi"][4]))

        # Interpolation functions
        interp_real = scipy.interpolate.RegularGridInterpolator(coord, Psi4r)
        interp_imag = scipy.interpolate.RegularGridInterpolator(coord, Psi4i)

        # Interpolate on sphere locations
        # This will error if point not in domain
        psi4_sphere = (interp_real(points_sphere) 
                           + 1j * interp_imag(points_sphere))
        # reshape for spinsfast
        psi4_sphere = psi4_sphere.reshape(Ntheta, Nphi)

        # lm mode of Psi4
        self.myprint("WARNING: using ``spinsfast``, outputs from this code "
                     + "may differ from others as different codes use "
                     + "different normalisations, integration schemes, "
                     + "conventions...")
        alm = spinsfast.map2salm(psi4_sphere, -2, self.Psi4_lm_lmax)

        # change to a format I like better
        lm_dict = {}
        for l in range(self.Psi4_lm_lmax + 1):
            for m in range(-l, l + 1):
                lm_dict[l, m] = alm[l**2 + m + l]
        return lm_dict
    
    def Weyl_invariants(self):
        Psis = self["Weyl_Psi"]
        I_inv = Psis[0]*Psis[4] - 4*Psis[1]*Psis[3] + 3*Psis[2]*Psis[2]
        J_inv = maths.determinant3(
            jnp.array([[Psis[4], Psis[3], Psis[2]], 
                       [Psis[3], Psis[2], Psis[1]], 
                       [Psis[2], Psis[1], Psis[0]]]))
        
        self.myprint("WARNING: I'm not switching Psi0 and Psi4 here, "
                     + "so the invariants are not correct if Psi4 = 0."
                     + "Same for Psi1 and Psi3.")
        L_inv = Psis[2]*Psis[4] - (Psis[3]**2)
        K_inv = (Psis[1]*(Psis[4]**2) 
                 - 3*Psis[4]*Psis[3]*Psis[2] 
                 + 2*(Psis[3]**3))
        N_inv = 12*(L_inv**2) - (Psis[4]**2)*I_inv
        return {'I': I_inv, 'J': J_inv, 'L': L_inv, 'K': K_inv, 'N': N_inv}
    
    def eweyl_u_down4(self):
        return jnp.einsum(
            'b..., d..., abcd... -> ac...', 
            self["uup4"], self["uup4"], self["st_Weyl_down4"])
    
    def eweyl_n_down3(self):
        # 1st compute K terms
        Kmixed3 = jnp.einsum('ij..., jk... -> ik...', 
                            self["gammaup3"], self["Kdown3"])
        KKterm = jnp.einsum('im..., mj... -> ij...', self["Kdown3"], Kmixed3)
        KKtermH = jnp.einsum('ij..., ji... -> ...', Kmixed3, Kmixed3)
        del Kmixed3
            
        # 2nd compute S terrms
        # TODO: make this more efficient
        gmixed4 = jnp.einsum('ab..., bc... -> ac...', 
                            self["gup4"], self["gdown4"])
        gammamixed4 = gmixed4 + jnp.einsum('a..., c... -> ac...',
                                        self["ndown4"], self["nup4"])
        Sdown3 = jnp.einsum('ca..., db..., cd... -> ab...', 
                        gammamixed4, gammamixed4, self["Tdown4"])[1:,1:]
        del gmixed4, gammamixed4
            
        # Now find E
        return (
            self["s_Ricci_down3"] 
            + self["Ktrace"]*self["Kdown3"] 
            - KKterm 
            - (1/3) * self["gammadown3"] * (
                self["s_RicciS"] 
                + self["Ktrace"] * self["Ktrace"] 
                - KKtermH) 
            - 0.5 * self.kappa * (
                Sdown3 
                - (1/3) * self["gammadown3"] * self["Stresstrace_n"]))
    
    def bweyl_u_down4(self):
        LCuudd4 = jnp.einsum(
            'ac..., bd..., abef... -> cdef...', 
            self["gup4"], self["gup4"], self.levicivita_down4())
        return 0.5 * jnp.einsum(
            'b..., f..., abcd..., cdef... -> ae...', 
            self["uup4"], self["uup4"], self["st_Weyl_down4"], LCuudd4)
    
    def bweyl_n_down3(self):
        LCuud3 = jnp.einsum('ae..., bf..., d..., defc... -> abc...', 
                        self["gup4"], self["gup4"], self["nup4"], 
                        self.levicivita_down4())[1:, 1:, 1:]
            
        dKdown = self.s_covd(self["Kdown3"], 'dd')
        Bterm1 = jnp.einsum('cdb..., cda... -> ab...', LCuud3, dKdown)

        Kmixed3 = jnp.einsum('ij..., jk... -> ik...', 
                            self["gammaup3"], self["Kdown3"])
        Bterm2K = (self.s_covd(self["Ktrace"], '') 
                - jnp.einsum('ccb... -> b...', 
                            self.s_covd(Kmixed3, 'ud')))
        Bterm2 = 0.5 * jnp.einsum('cdb..., ac..., d... -> ab...', LCuud3, 
                        self["gammadown3"], Bterm2K)
            
        return Bterm1 + Bterm2
    
    ###########################################################################
    # Tetrads
    ###########################################################################

    def null_vector_base(self):
        """Return an arbitrary null vector base.
        
        Returns
        -------
        lup4, kup4, mup4, mbup4: list
            Each is (4, Nx, Ny, Nz) array_like complex
        
        Note
        ----
        See 'Introduction to 3+1 Numerical Relativity' 2008
        by M. Alcubierre page 295
        """
        e0up4, e1up4, e2up4, e3up4 = self.tetrad_base()
        inverse_sqrt_2 = 1 / jnp.sqrt(2)
        kup4 = (e0up4 + e1up4) * inverse_sqrt_2
        lup4 = (e0up4 - e1up4) * inverse_sqrt_2
        mup4 = (e2up4 + 1j*e3up4) * inverse_sqrt_2
        mbup4 = (e2up4 - 1j*e3up4) * inverse_sqrt_2
        return lup4, kup4, mup4, mbup4

    def tetrad_base(self):
        r"""Return an quasi-Kinnersley or arbitrary tetrad base.
        
        if AurelCore.tetrad_to_use == "quasi-Kinnersley":
            Which is the default, because the tetrad_to_use is set to
            "quasi-Kinnersley" in the init.
            Then quasi-Kinnersley tetrad is returned where the first tetrad
            is the normal to the hypersurface $n^\mu$.
        else:
            Then an arbitrary orthonormal tetrad is returned where 
            the first tetrad is the fluid 4-velocity $u^\mu$.
    
        Returns
        -------
        e0up4, e1up4, e2up4, e3up4: list
            Each is (4, Nx, Ny, Nz) array_like
                   
        Note
        ----
        - for Kinnersly tetrad see https://doi.org/10.1103/PhysRevD.65.044001 
          also see: 
          Cactus/arrangements/EinsteinAnalysis/WeylScal4/m/WeylScal4.m 
          Like the WeylScal4 thorn: tetrad given in wave zone (lapse = 1, 
          shift = 0) and we do not perform the final rotation.
          Unlike the WeylScal4 thorn: I calc Weyl tensor in full (no wave zone 
          assumption), but this gives the same result.
        - for Gram-Schmidt scheme see Chapter 7 of 
          'Linear Algebra, Theory and applications' by W.Cheney and D.Kincaid

        """
        if self.tetrad_to_use == "quasi-Kinnersley":
            nup4 = jnp.array(
                [jnp.ones(self.data_shape), 
                 jnp.zeros(self.data_shape),
                 jnp.zeros(self.data_shape),
                 jnp.zeros(self.data_shape)])
            # not self["nup4"] because wave zone
            v1 = jnp.array([-self.fd.y, self.fd.x, 
                           jnp.zeros(jnp.shape(self.fd.x))])
            v2 = jnp.array([self.fd.x, self.fd.y, self.fd.z])
            LC = jnp.einsum('a..., abcd... -> bcd...', 
                        nup4, self.levicivita_down4())[1:,1:,1:]
            v3 = jnp.sqrt(self["gammadet"]) * jnp.einsum(
                'ad..., dbc..., b..., c... -> a...',
                self["gammaup3"], LC, v1, v2)

            # Gram-Schmidt orthonormalization
            v1 = maths.safe_division(v1, self.norm3(v1))

            v2 = v2 - self.vector_inner_product3(v1, v2) * v1
            v2 = maths.safe_division(v2, self.norm3(v2))

            v3 = (v3 - self.vector_inner_product3(v1, v3) * v1 
                - self.vector_inner_product3(v2, v3) * v2)
            v3 = maths.safe_division(v3, self.norm3(v3))

            
            e0up4 = nup4
            e1up4 = jnp.array(
                [jnp.zeros(self.data_shape), v2[0], v2[1], v2[2]])
            # typo in ref paper
            e2up4 = jnp.array(
                [jnp.zeros(self.data_shape), v3[0], v3[1], v3[2]])
            e3up4 = jnp.array(
                [jnp.zeros(self.data_shape), v1[0], v1[1], v1[2]])
        else:
            # this is just an arbitrary orthonormal tetrad base
            zeros = jnp.zeros(self.data_shape)
            v1 = jnp.array([zeros, maths.safe_division(
                1.0, jnp.sqrt(self["gdown4"][1,1])), zeros, zeros])
            v2 = jnp.array([zeros, zeros, maths.safe_division(
                1.0, jnp.sqrt(self["gdown4"][2,2])), zeros])
            v3 = jnp.array([zeros, zeros, zeros, maths.safe_division(
                1.0, jnp.sqrt(self["gdown4"][3,3]))])

            e0up4 = self["uup4"]
            
            u1 = v1 + self.vector_inner_product4(e0up4, v1) * e0up4
            e1up4 = maths.safe_division(u1, self.norm4(u1))
            
            u2 = (v2 + self.vector_inner_product4(e0up4, v2) * e0up4 
                - self.vector_inner_product4(e1up4, v2) * e1up4)
            e2up4 = maths.safe_division(u2, self.norm4(u2))
            
            u3 = (v3 + self.vector_inner_product4(e0up4, v3) * e0up4 
                - self.vector_inner_product4(e1up4, v3) * e1up4
                - self.vector_inner_product4(e2up4, v3) * e2up4)
            e3up4 = maths.safe_division(u3, self.norm4(u3))

        return e0up4, e1up4, e2up4, e3up4
    
    ###########################################################################
    # Covariant derivatives
    ###########################################################################
        
    def s_covd(self, f, indexing):
        """Spatial covariant derivative of a 3D tensor of rank 0, 1 or 2.
        
        Covariant derivative with respects to the spatial metric.
        
        Parameters
        ---------- 
        f : (..., Nx, Ny, Nz) array_like
            Tensor of rank 0, 1 or 2. The first indices are the spatial ones.
        indexing : str
            '' for scalar,
            'u' for upper index, 'd' for down index,
            'uu' for two upper indices, 'dd' for two down indices,
            'ud' for one upper and one down index
        
        Returns
        -------
        (3, ..., Nx, Ny, Nz) array_like
        """
        rank = len(indexing)
        if rank == 0:
            covd = self.fd.d3_scalar(f)
        elif rank == 1:
            df = self.fd.d3_rank1tensor(f)
            if indexing == 'u':
                G1 = jnp.einsum('abc..., b... -> ca...', 
                               self["s_Gamma_udd3"], f)
            elif indexing == 'd':
                G1 = - jnp.einsum('abc..., a... -> bc...', 
                                 self["s_Gamma_udd3"], f)
            else:
                raise ValueError(f"Field if of rank {rank} so indexing"
                                 + f" must be 'u' or 'd'")
            covd = df + G1
        elif rank ==2:
            df = self.fd.d3_rank2tensor(f)
            if indexing == 'uu':
                G1 = jnp.einsum('acd..., db... -> cab...', 
                               self["s_Gamma_udd3"], f)
                G2 = jnp.einsum('bcd..., ad... -> cab...', 
                               self["s_Gamma_udd3"], f)
            elif indexing == 'dd':
                G1 = - jnp.einsum('dca..., db... -> cab...', 
                                 self["s_Gamma_udd3"], f)
                G2 = - jnp.einsum('dcb..., ad... -> cab...', 
                                 self["s_Gamma_udd3"], f)
            elif indexing == 'ud':
                G1 = jnp.einsum('acd..., db... -> cab...', 
                               self["s_Gamma_udd3"], f)
                G2 = - jnp.einsum('dcb..., ad... -> cab...', 
                                 self["s_Gamma_udd3"], f)
            elif indexing == 'du':
                G1 = - jnp.einsum('dca..., db... -> cab...', 
                                 self["s_Gamma_udd3"], f)
                G2 = jnp.einsum('bcd..., ad... -> cab...', 
                               self["s_Gamma_udd3"], f)
            else:
                raise ValueError(f"Field if of rank {rank} so indexing"
                                 + f" must be 'uu', 'ud', 'du' or 'dd'")
            covd = df + G1 + G2
        else:
            raise ValueError(f"Don't know how to compute spatial covariant"
                             + f" derivative of rank {rank}")
        return covd
    
    def st_covd(self, f, dtf, indexing):
        """Spacetime covariant derivative of a 4D tensor of rank 0 or 1.
        
        Covariant derivative with respects to the spacetime metric.
        
        Parameters
        ---------- 
        f : (..., Nx, Ny, Nz) array_like
            Tensor of rank 0, or 1. The first indices are the spacetime ones.
        dtf : (..., Nx, Ny, Nz) array_like
            Time derivative of the tensor. This array must have the same shape.
        indexing : str
            '' for scalar
            'u' for upper index, 'd' for down index
        
        Returns
        -------
        (4, ...., Nx, Ny, Nz) array_like
        """
        rank = len(indexing)
        if rank == 0:
            covd = jnp.append(jnp.array([dtf]), self.fd.d3_scalar(f), axis = 0)
        elif rank == 1:
            df = jnp.append(jnp.array([dtf]), self.fd.d3_rank1tensor(f), 
                           axis = 0)
            if indexing == 'd':
                G1 = jnp.einsum('abc..., b... -> ca...', 
                               self["st_Gamma_udd4"], f)
            elif indexing == 'u':
                G1 = - jnp.einsum('abc..., a... -> bc...', 
                                 self["st_Gamma_udd4"], f)
            else:
                raise ValueError(f"Field if of rank {rank} so indexing"+
                                 + f" must be 'u' or 'd'")
            covd = df + G1
        else:
            raise ValueError(f"Don't know how to compute spacetime covariant"+
                             + f" derivative of rank {rank}")
        return covd
    
    def s_div(self, f, indexing):
        """Compute divergence along n of a 3D tensor of rank 1 or 2.
        
        Parameters
        ---------- 
        f : 
            if rank 1: (3, Nx, Ny, Nz) array_like,
            if rank 2: (3, 3, Nx, Ny, Nz) array_like
        indexing : str
            'u' for upper index, 'd' for down index
        
        Returns
        -------
        if rank 1 (Nx, Ny, Nz) array_like
        if rank 2 (3, Nx, Ny, Nz) array_like
        """
        covd = self.s_covd(f, indexing)
        if indexing == 'u':
            return jnp.einsum('aa... -> ...', covd)
        elif indexing == 'd':
            return jnp.einsum('ab..., ab... -> ...', self["gammaup3"], covd)
        elif indexing == 'uu' or indexing == 'ud':
            return jnp.einsum('aab... -> b...', covd)
        elif indexing == 'du':
            return jnp.einsum('aba... -> b...', covd)
        elif indexing == 'dd':
            return jnp.einsum('ab..., abc... -> c...', self["gammaup3"], covd)
        else:
            raise ValueError(f"Don't know how to compute divergence of tensor"+
                             + f" with indices {indexing}")
    
    def s_curl(self, fdown3, indexing):
        """Compute curl along n of a 3D rank 2 covariant tensor.
        
        Parameters
        ---------- 
        fdown3 : (3, 3, Nx, Ny, Nz) array_like
            Rank 2 spatial tensor with both indices down
        indexing : str
            only accepts 'dd' for two down indices
        
        Returns
        -------
        (3, 3, Nx, Ny, Nz) array_like
        """
        covd = self.s_covd(fdown3, indexing)
        if indexing == 'dd':
            LCuud3 = jnp.einsum('ae..., bf..., d..., defc... -> abc...', 
                            self["gup4"], self["gup4"], self["nup4"], 
                            self.levicivita_down4())[1:, 1:, 1:]
            curl = maths.symmetrise_tensor(
                jnp.einsum('cda..., cbd... -> ab...', 
                          LCuud3, covd))
        else:
            raise ValueError(f"Don't know how to compute curl of tensor"+
                             + f" with indices {indexing}")
        return curl
    
    ###########################################################################
    # Tensorial operations
    ###########################################################################

    def s_to_st(self, fdown3):
        r"""Compute spacetime tensor from the spatial tensor.
        
        Parameters
        ----------  
        fdown3 : (3, 3, Nx, Ny, Nz) array_like
            Rank 2 spatial tensor with both indices down
        
        Returns
        -------
        fdown4 : (4, 4, Nx, Ny, Nz) array_like
            Rank 2 spacetime tensor with both indices down
            
        Note
        ----
        By definition the electric part of the Weyl tensor in the $n^\mu$
        frame with indices up, ${}^{(n)}E^{\alpha \beta}$, 
        only has spatial components, same for the magnetic part, 
        ${}^{(n)}B^{\alpha \beta}$, and the extrinsic curvature, 
        $K^{\alpha\beta}$. 
        So this function only applied to those three tensors.
        """
        f00 = jnp.einsum('i..., j..., ij... -> ...', 
                        self["betaup3"], self["betaup3"], fdown3)
        f0k = jnp.einsum('i..., ik... -> k...', self["betaup3"], fdown3)
        fdown4 = jnp.array([[f00, f0k[0], f0k[1], f0k[2]],
                           [f0k[0], fdown3[0, 0], fdown3[0, 1], fdown3[0, 2]],
                           [f0k[1], fdown3[1, 0], fdown3[1, 1], fdown3[1, 2]],
                           [f0k[2], fdown3[2, 0], fdown3[2, 1], fdown3[2, 2]]])
        return fdown4
    
    def vector_inner_product4(self, a, b):
        """Inner product of rank 1 4D tensors with indices up."""
        return jnp.einsum('a..., b..., ab... -> ...', 
                         a, b, self["gdown4"])
    
    def vector_inner_product3(self, a, b):
        """Inner product of rank 1 3D tensors with indices up."""
        return jnp.einsum('a..., b..., ab... -> ...', 
                         a, b, self["gammadown3"])

    def trace4(self, fdown4):
        """Compute trace of a 4D rank 2 tensor."""
        return jnp.einsum('jk..., jk... -> ...', 
                         self["gup4"], fdown4)

    def trace3(self, fdown3):
        """Compute trace of a 3D rank 2 tensor."""
        return jnp.einsum('jk..., jk... -> ...', 
                         self["gammaup3"], fdown3)
    
    def magnitude4(self, fdown):
        """Compute magnitude of a 4D rank 2 tensor."""
        return 0.5 * jnp.einsum(
            'ab..., ij..., ai..., bj... -> ...', 
            fdown, fdown,
            self["gup4"], self["gup4"])
    
    def magnitude3(self, fdown):
        """Compute magnitude of a 3D rank 2 tensor."""
        return 0.5 * jnp.einsum(
            'ab..., ij..., ai..., bj... -> ...', 
            fdown, fdown,
            self["gammaup3"], self["gammaup3"])
    
    def norm4(self, a): 
        """Compute norm of a 4D rank 1 tensor."""
        return jnp.sqrt(abs(self.vector_inner_product4(a, a)))
    
    def norm3(self, a): 
        """Compute norm of a 3D rank 1 tensor."""
        return jnp.sqrt(abs(self.vector_inner_product3(a, a)))
    
    def kronecker_delta4(self):
        """Compute Kronecker delta with 4 4D indices."""
        kronecker = jnp.zeros(
            (4, 4, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        for i in range(4):
            kronecker = kronecker.at[i, i].set(1.0)
        return kronecker
    
    def kronecker_delta3(self):
        """Compute Kronecker delta with 3 3D indices."""
        kronecker = jnp.zeros(
            (3, 3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        for i in range(3):
            kronecker = kronecker.at[i, i].set(1.0)
        return kronecker

    def levicivita_down4(self):
        """Compute Levi-Civita tensor with 4 4D indices down."""
        return (self.levicivita_symbol_down4() 
                * jnp.sqrt(-self["gdet"]))
    
    def levicivita_down3(self):
        """Compute Levi-Civita tensor with 3 3D indices down."""
        return (self.levicivita_symbol_down3() 
                * jnp.sqrt(self["gammadet"]))

    def levicivita_symbol_down4(self): 
        """Compute Levi-Civita symbol with 4 4D indices down."""
        LC = jnp.zeros(
            (4, 4, 4, 4, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        allindices = [0, 1, 2, 3]
        for i0 in allindices:
            for i1 in np.delete(allindices, i0):
                for i2 in np.delete(allindices, [i0, i1]):
                    for i3 in np.delete(allindices, [i0, i1, i2]):
                        top = ((i1-i0) * (i2-i0) * (i3-i0) 
                               * (i2-i1) * (i3-i1) * (i3-i2))
                        bot = (abs(i1-i0) * abs(i2-i0) * abs(i3-i0)
                               * abs(i2-i1) * abs(i3-i1) * abs(i3-i2))
                        LC = LC.at[i0, i1, i2, i3, :, :, :].set(float(top/bot))
        return LC

    def levicivita_symbol_down3(self):
        """Compute Levi-Civita symbol with 3 3D indices down."""
        LC = jnp.zeros(
            (3, 3, 3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        allindices = [1, 2, 3]
        for i1 in allindices:
            for i2 in np.delete(allindices, i1-1):
                for i3 in np.delete(allindices, [i1-1, i2-1]):
                    top = ((i2-i1) * (i3-i1) * (i3-i2))
                    bot = (abs(i2-i1) * abs(i3-i1) * abs(i3-i2))
                    LC = LC.at[i1-1, i2-1, i3-1, :, :, :].set(float(top/bot))
        return LC
    
def block_all(x):
    """Block all JAX arrays until they are ready.
    This is useful to ensure that all computations are completed before
    returning results, especially in a JAX-based environment.

    Parameters
    ----------
    x : any
        The input can be a JAX array, a dictionary, or a sequence 
        (like a list or tuple).
    
    Returns
    -------
    any
        The input with all JAX arrays blocked until ready.
    """
    if isinstance(x, jax.Array):
        return x.block_until_ready()
    elif isinstance(x, Mapping):
        return {k: block_all(v) for k, v in x.items()}
    elif isinstance(x, Sequence) and not isinstance(x, str):
        return type(x)(block_all(v) for v in x)
    else:
        return x

# Update __doc__ of the functions listed in descriptions
for func_name, doc in descriptions.items():
    func = getattr(AurelCore, func_name, None)
    if func is not None:
        func.__doc__ = doc