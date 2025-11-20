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
import numpy as np
from IPython.display import display, Latex
from . import maths
from . import numerical
is_notebook = 'ipykernel' in sys.modules

# Descriptions for each AurelCore.data entry and function
# assumed variables need to be listed in docs/source/source/generate_rst.py
def assumption(keyname, assumption):
    """Print assumption message."""
    message = (" I assume " + assumption + ", "
               + "if not then please define "
               + "AurelCore.data['"+keyname+"'] = ... ")
    return message
descriptions = {
    # === Metric quantities
    # Spatial metric
    "gxx": (r"$g_{xx}$ Metric with xx indices down." 
        + assumption('gxx', r"$g_{xx}=1$")),
    "gxy": (r"$g_{xy}$ Metric with xy indices down." 
        + assumption('gxy', r"$g_{xy}=0$")),
    "gxz": (r"$g_{xz}$ Metric with xz indices down." 
        + assumption('gxz', r"$g_{xz}=0$")),
    "gyy": (r"$g_{yy}$ Metric with yy indices down." 
        + assumption('gyy', r"$g_{yy}=1$")),
    "gyz": (r"$g_{yz}$ Metric with yz indices down." 
        + assumption('gyz', r"$g_{yz}=0$")),
    "gzz": (r"$g_{zz}$ Metric with zz indices down." 
        + assumption('gzz', r"$g_{zz}=1$")),
    "gammadown3": r"$\gamma_{ij}$ Spatial metric with spatial indices down",
    "gammaup3": r"$\gamma^{ij}$ Spatial metric with spatial indices up",
    "dtgammaup3": (r"$\partial_t \gamma^{ij}$ Coordinate time derivative of"
        + r" spatial metric with spatial indices up"),
    "gammadet": r"$\gamma$ Determinant of spatial metric",
    "gammadown4": (r"$\gamma_{\mu\nu}$ Spatial metric with spacetime indices"
        + r" down"),
    "gammaup4": r"$\gamma^{\mu\nu}$ Spatial metric with spacetime indices up",
    # BSSNOK metric
    "psi_bssnok": r"$\psi = \gamma^{1/12}$ BSSNOK conformal factor",
    "phi_bssnok": r"$\phi = \ln(\gamma^{1/12})$ BSSNOK conformal factor",
    "dtphi_bssnok": (r"$\partial_t \phi$ Coordinate time derivative of BSSNOK"
        + r" $\phi$ conformal factor"),
    "gammadown3_bssnok": (r"$\tilde{\gamma}_{ij}$ Conformal spatial metric"
        + r" with spatial indices down"),
    "gammaup3_bssnok": (r"$\tilde{\gamma}^{ij}$ Conformal spatial metric"
        + r" with spatial indices up"),
    "dtgammadown3_bssnok": (r"$\partial_t \tilde{\gamma}_{ij}$ Coordinate"
        + r" time derivative of conformal spatial metric with spatial"
        + r" indices down"),
    # Extrinsic curvature
    "kxx": (r"$K_{xx}$ Extrinsic curvature with xx indices down." 
        + assumption('kxx', r"$K_{xx}=0$")),
    "kxy": (r"$K_{xy}$ Extrinsic curvature with xy indices down." 
        + assumption('kxy', r"$K_{xy}=0$")),
    "kxz": (r"$K_{xz}$ Extrinsic curvature with xz indices down." 
        + assumption('kxz', r"$K_{xz}=0$")),
    "kyy": (r"$K_{yy}$ Extrinsic curvature with yy indices down." 
        + assumption('kyy', r"$K_{yy}=0$")),
    "kyz": (r"$K_{yz}$ Extrinsic curvature with yz indices down." 
        + assumption('kyz', r"$K_{yz}=0$")),
    "kzz": (r"$K_{zz}$ Extrinsic curvature with zz indices down." 
        + assumption('kzz', r"$K_{zz}=0$")),
    "Kdown3": r"$K_{ij}$ Extrinsic curvature with spatial indices down",
    "Kup3": r"$K^{ij}$ Extrinsic curvature with spatial indices up",
    "Ktrace": r"$K = \gamma^{ij}K_{ij}$ Trace of extrinsic curvature",
    "dtKtrace": (r"$\partial_t K$ Coordinate time derivative of the"
        + r" trace of extrinsic curvature"),
    "Adown3": (r"$A_{ij}$ Traceless part of the extrinsic curvature"
        + r" with spatial indices down"),
    "Aup3": (r"$A^{ij}$ Traceless part of the extrinsic curvature"
        + r" with spatial indices up"),
    "A2": r"$A^2$ Magnitude of traceless part of the extrinsic curvature",
    # BSSN extrinsic curvature
    "Adown3_bssnok": (r"$\tilde{A}_{ij}$ Conformal traceless part of the"
        + r" extrinsic curvature with spatial indices down"),
    "Aup3_bssnok": (r"$\tilde{A}^{ij}$ Conformal traceless part of the"
        + r" extrinsic curvature with spatial indices up"),
    "A2_bssnok": (r"$\tilde{A}^2$ Magnitude of conformal traceless"
        + r" part of the extrinsic curvature"),
    "dtAdown3_bssnok": (r"$\partial_t \tilde{A}_{ij}$ Coordinate time"
        + r" derivative of conformal traceless part of the extrinsic"
        + r" curvature with spatial indices down"),
    # Lapse
    "alpha": r"$\alpha$ Lapse." + assumption('alpha', r"$\alpha=1$"),
    "dtalpha": (r"$\partial_t \alpha$ Coordinate time derivative"
        + r" of the lapse." + assumption('dtalpha', r"$\partial_t \alpha=0$")),
    "DDalpha": (r"D_iD_j\alpha$ Spatial covariant second"
        + r" derivative of the lapse with spatial indices down"),
    # Shift
    "betax": (r"$\beta^{x}$ x component of the shift vector with indices up." 
        + assumption('betax', r"$\beta^{x}=0$")),
    "betay": (r"$\beta^{y}$ y component of the shift vector with indices up." 
        + assumption('betay', r"$\beta^{y}=0$")),
    "betaz": (r"$\beta^{z}$ z component of the shift vector with indices up." 
        + assumption('betaz', r"$\beta^{z}=0$")),
    "betaup3": r"$\beta^{i}$ Shift vector with spatial indices up",
    "dtbetax": (r"$\partial_t\beta^{x}$ Coordinate time derivative of the"
        + r" x component of the shift vector with indices up." 
        + assumption('dtbetax', r"$\partial_t\beta^{x}=0$")),
    "dtbetay": (r"$\partial_t\beta^{y}$ Coordinate time derivative of the"
        + r" y component of the shift vector with indices up." 
        + assumption('dtbetay', r"$\partial_t\beta^{y}=0$")),
    "dtbetaz": (r"$\partial_t\beta^{z}$ Coordinate time derivative of the"
        + r" z component of the shift vector with indices up." 
        + assumption('dtbetaz', r"$\partial_t\beta^{z}=0$")),
    "dtbetaup3": (r"$\partial_t\beta^{i}$ Coordinate time derivative"
        + r" of the shift vector with spatial indices up"),
    "betadown3": r"$\beta_{i}$ Shift vector with spatial indices down",
    "betamag": r"$\beta_{i}\beta^{i}$ Magnitude of shift vector",
    # Proper time
    "dttau": r"$\partial_t \tau$ Coordinate time derivative of proper time",
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
    "null_ray_exp_out": (r"$\Theta_{out}$ List of expansion of"
        + r" null rays radially going out"),
    "null_ray_exp_in": (r"$\Theta_{in}$ List of expansion of"
        + r" null rays radially going in"),

    # === Matter quantities
    # Eulerian observer follows n^mu
    # Lagrangian observer follows u^mu
    "rho0": (r"$\rho_0$ Rest mass energy density."
        + assumption('rho0', r"$\rho_0=0$")),
    "press": (r"$p$ Pressure." + assumption('press', r"$p=0$")),
    "eps": (r"$\epsilon$ Specific internal energy."
        + assumption('eps', r"$\epsilon=0$")),
    "rho": r"$\rho$ Energy density",
    "enthalpy": r"$h$ Specific enthalpy of the fluid",
    # Fluid velocity
    "w_lorentz": (r"$W$ Lorentz factor." + assumption('w_lorentz', r"$W=1$")),
    "velx": (r"$v^x$ x component of Eulerian fluid three velocity"
        + r" with indice up."+ assumption('velx', r"$v^x=0$")),
    "vely": (r"$v^y$ y component of Eulerian fluid three velocity"
        + r" with indice up."+ assumption('vely', r"$v^y=0$")),
    "velz": (r"$v^z$ z component of Eulerian fluid three velocity"
        + r" with indice up."+ assumption('velz', r"$v^z=0$")),
    "velup3": r"$v^i$ Eulerian fluid three velocity with spatial indices up.",
    "velup4": (r"$v^\mu$ Eulerian fluid three velocity"
        + r" with spacetime indices up."),
    "veldown3": (r"$v_i$ Eulerian fluid three velocity"
        + r" with spatial indices down"),
    "veldown4": (r"$v_\mu$ Eulerian fluid three velocity"
        + r" with spacetime indices down"),
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
    "hdet": (r"$h$ Determinant of spatial part of spatial metric orthonormal"
        + r" to fluid flow"),
    "hmixed4": (r"${h^{\mu}}_{\nu}$ Spatial metric orthonomal to fluid flow"
        + r" with mixed spacetime indices"),
    "hup4": (r"$h^{\mu\nu}$ Spatial metric orthonomal to fluid flow"
        + r" with spacetime indices up"),
    # Energy-stress tensor
    "Tdown4": r"$T_{\mu\nu}$ Energy-stress tensor with spacetime indices down",
    "Tup4": r"$T^{\mu\nu}$ Energy-stress tensor with spacetime indices up",
    "Ttrace": r"$T$ Trace of the energy-stress tensor",
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
        + r" in the $n^\mu$ frame with spatial indices down"),
    "rho_n_fromHam": (r"$\rho^{\{n\}}$ Energy density in the $n^\mu$ frame"
        + r" computed from the Hamiltonian constraint"),
    "fluxup3_n_fromMom": (r"$S^{\{n\}i}$ Energy flux (or momentum density) in"
        + r" the $n^\mu$ frame with spatial indices up computed from"
        + r" the Momentum constraint"),
    # Conserved quantities
    "conserved_D": r"$D$ Conserved mass-energy density in Wilson's formalism",
    "conserved_E": (r"$E$ Conserved internal energy density"
        + r" in Wilson's formalism"),
    "conserved_Sdown4": (r"$S_{\mu}$ Conserved energy flux"
        + r" (or momentum density) in Wilson's formalism"
        + r" with spacetime indices down"),
    "conserved_Sdown3": (r"$S_{i}$ Conserved energy flux (or momentum density)"
        + r" in Wilson's formalism with spatial indices down"),
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
        + r" of Lagrangian fluid four velocity with spacetime indices down"),
    "accelerationdown4": (r"$a_{\mu}$ Acceleration of the fluid"
        + r" with spacetime indices down"),
    "accelerationup4": (r"$a^{\mu}$ Acceleration of the fluid"
        + r" with spacetime indices up"),
    "s_covd_udown4": (r"$\mathcal{D}^{\{u\}}_{\mu} u_{\nu}$ Spatial covariant"
        + r" derivative of Lagrangian fluid four velocity with spacetime"
        + r" indices down, with respect to spatial hypersurfaces orthonormal"
        + r" to the fluid flow"),
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
    "st_Gamma_udd4": (r"${}^{(4)}{\Gamma^{\alpha}}_{\mu\nu}$ Christoffel"
        + r" symbols of spacetime metric with mixed spacetime indices"),
    "st_Riemann_uddd4": (r"${}^{(4)}{R^{\alpha}}_{\beta\mu\nu}$ Riemann"
        + r" tensor of spacetime metric with mixed spacetime indices"),
    "st_Riemann_down4": (r"${}^{(4)}R_{\alpha\beta\mu\nu}$ Riemann tensor"
        + r" of spacetime metric with spacetime indices down"),
    "st_Riemann_uudd4": (r"${}^{(4)}{R^{\alpha\beta}}_{\mu\nu}$ Riemann"
        + r" tensor of spacetime metric with mixed spacetime indices"),
    "st_Ricci_down4": (r"${}^{(4)}R_{\alpha\beta}$ Ricci tensor of spacetime"
        + r" metric with spacetime indices down"),
    "st_Ricci_down3": (r"${}^{(4)}R_{ij}$ Ricci tensor of spacetime metric"
        + r" with spatial indices down"),
    "st_RicciS": r"${}^{(4)}R$ Ricci scalar of spacetime metric",
    "Einsteindown4": (r"$G_{\alpha\beta}$ Einstein tensor with spacetime"
        + r" indices down"),
    "Kretschmann": (r"$K={R^{\alpha\beta}}_{\mu\nu}{R_{\alpha\beta}}^{\mu\nu}$"
        + r" Kretschmann scalar"),
    # in BSSNOK form
    "s_Gamma_udd3_bssnok": (r"${}^{(3)}{\tilde{\Gamma}^{k}}_{ij}$ Christoffel"
        + r" symbols of conformal spatial metric with mixed spatial indices"),
    "s_Gamma_bssnok": (r"${}^{(3)}\tilde{\Gamma}^i$ Conformal connection"
        + r" functions with spatial indice up"),
    "dts_Gamma_bssnok": (r"$\partial_t {}^{(3)}\tilde{\Gamma}^i$"
        + r" Coordinate time derivative of conformal connection functions"
        + r" with spatial indice up"),
    "s_Ricci_down3_bssnok": (r"${}^{(3)}\tilde{R}_{ij}$ Ricci tensor of"
        + r" conformal spatial metric with spatial indices down"),
    "s_RicciS_bssnok": (r"${}^{(3)}\tilde{R}$ Ricci scalar of conformal"
        + r" spatial metric"),
    "s_Ricci_down3_phi": (r"${}^{(3)}R^{\phi}_{ij}$ Ricci terms that depend on the"
        + r" conformal function $\phi$"),
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
        + r" with AurelCore.tetrad"),
    "Psi4_lm": (r"$\Psi_4^{l,m}$ List of dictionaries of spin weighted"
        + r" spherical harmonic decomposition of the 4th Weyl scalar."
        + r" Control with AurelCore.lmax, center, extract_radii, and"
        + r" interp_method."),
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
    fd : FiniteDifference
        aurel.finitedifference.FiniteDifference
    verbose : bool
        If True, display messages about the calculations.
    fancy_print : bool
        If True, display messages in a fancy ipython format, 
        else normal print is used. Default is True.
    clear_cache_every_nbr_calc : int
        Number of calculations before clearing the cache. Default is 20.
    memory_threshold_inGB : int
        Memory threshold in GB for clearing the cache. Default is 4 GB.
    Lambda : float, optional, also attribute
        Cosmological constant. Default is 0.0.
    vaccum : bool, optional, also attribute
        If True, assume vacuum spacetime (no matter). Default is False.
    tetrad : str, optional, also attribute
        Tetrad choice for Weyl scalar calculations.
        Default is "quasi-Kinnersley". Any other value provides ...
    lmax : int, optional, also attribute
        Maximum l value for spherical harmonic decompositions. Default is 8.
    center : tuple, optional, also attribute
        Center of the Psi4_lm sphere.
        Default is (0.0, 0.0, 0.0).
    extract_radii : list, optional, also attribute
        List of extraction radii for spherical harmonic decompositions.
        Default is [0.9 * min radius in the grid].
    interp_method : str, optional, also attribute
        Interpolation method for 
        `scipy.interpolate.RegularGridInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_.
        Default is 'linear'.

    Attributes
    ----------
    param : dict
        (*dict*) - Dictionary containing the data grid parameters, from fd. 
    data : dict
        (*dict*) - Dictionary where all the variables are stored.
    data_shape : tuple
        (*tuple*) - Shape of the data arrays: (Nx, Ny, Nz)
    kappa : float
        (*float*) - Einstein's constant with G = c = 1. Default is 8 * pi.
    calculation_count : int
        (*int*) - Number of calculations performed.
    last_accessed : dict
        (*dict*) - Dictionary to keep track of when each variable was 
        last accessed.
    var_importance : dict
        (*dict*) - Dictionary to keep track of the importance of each variable
        for cache cleanup. To never delete a variable, set its importance to 0.
    """
    
    def __init__(self, fd, **kwargs):
        """Initialize the AurelCore class."""

        self.param = fd.param
        self.fd = fd
        self.data_shape = (self.param['Nx'], 
                           self.param['Ny'], 
                           self.param['Nz'])

        # Kwargs or use defaults
        self.verbose = kwargs.get('verbose', True)
        self.fancy_print = kwargs.get('fancy_print', True)
        self.clear_cache_every_nbr_calc = kwargs.get(
            'clear_cache_every_nbr_calc', 20)
        self.memory_threshold_inGB = kwargs.get(
            'memory_threshold_inGB', 4)
        self.Lambda = kwargs.get('Lambda', 0.0)
        self.vaccum = kwargs.get('vaccum', False)
        self.tetrad = kwargs.get('tetrad', "quasi-Kinnersley")
        self.lmax = kwargs.get('lmax', 8)
        self.center = kwargs.get('center', (0.0, 0.0, 0.0))
        min_box_radius = np.min(abs(np.array([
            self.fd.xmin - self.center[0], self.fd.xmax - self.center[0], 
            self.fd.ymin - self.center[1], self.fd.ymax - self.center[1], 
            self.fd.zmin - self.center[2], self.fd.zmax - self.center[2]])))
        self.extract_radii = kwargs.get('extract_radii', 
                                        [min_box_radius * 0.9])
        # Convert extract_radii to list if it's a single number
        if isinstance(self.extract_radii, (int, float)):
            self.extract_radii = [self.extract_radii]
        
        self.interp_method = kwargs.get('interp_method', 'linear')
        
        # Save any additional kwargs as attributes
        handled_kwargs = {'Lambda', 'tetrad', 'lmax', 'extract_radii',
                          'interp_method'}
        for key, value in kwargs.items():
            if key not in handled_kwargs:
                setattr(self, key, value)
        
        # Physics variables
        self.kappa = 8*np.pi  # Einstein's constant with G = c = 1
        self.myprint(f"Cosmological constant set to AurelCore.Lambda"
                     + f" = {self.Lambda:.2e}")

        # data dictionary where everything is stored
        self.data = {}

        # To clean up cache
        self.calculation_count = 0
        self.last_accessed = {}

        # Importance of each variable for cache cleanup
        self.var_importance = {key:1.0 for key in descriptions.keys()}
        self.var_importance["s_Gamma_udd3"] = 0.002
        self.var_importance["s_RicciS"] = 0.1

    ###########################################################################
    # Functions to manage the data dictionary
    ###########################################################################

    def myprint(self, message):
        """Print a message with a fancy format."""
        if self.verbose:
            if is_notebook:
                if self.fancy_print:
                    display(Latex(message))
                else:
                    print(message, flush=True)
            else:
                print(message, flush=True)

    def __getitem__(self, key):
        """Get data[key] or compute it if not present."""
        # First check if the key is already cached
        if key in self.data:
            # Update the last accessed time
            self.last_accessed[key] = self.calculation_count
            return self.data[key]
        
        # Dynamically get the function by name
        func = getattr(self, key)

        # Call the function if it takes no additional arguments
        if func.__code__.co_argcount == 1:
            self.data[key] = func()
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
                    self.myprint('CLEAN-UP: '
                        + f"Removing cached value for '{key}'"
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
                    self.myprint('CLEAN-UP: '
                        + f"Current cache size "
                        + f"{total_cache_size / 1_048_576:.2f} MB, "
                        + f"max memory "
                        + f"{memory_threshold / 1_048_576:.2f} MB")
                    self.myprint('CLEAN-UP: '
                        + "Max memory too small,"
                        + "no more unimportant cache to remove.")
                    self.myprint('CLEAN-UP: '
                        + "Current variables: " + str(self.data.keys()))
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
    def gxx(self):
        if 'gammadown3' in self.data:
            return self["gammadown3"][0,0]
        else:
            return np.ones(self.data_shape)
    
    def gxy(self):
        if 'gammadown3' in self.data:
            return self["gammadown3"][0,1]
        else:
            return np.zeros(self.data_shape)
    
    def gxz(self):
        if 'gammadown3' in self.data:
            return self["gammadown3"][0,2]
        else:
            return np.zeros(self.data_shape)
    
    def gyy(self):
        if 'gammadown3' in self.data:
            return self["gammadown3"][1,1]
        else:
            return np.ones(self.data_shape)
    
    def gyz(self):
        if 'gammadown3' in self.data:
            return self["gammadown3"][1,2]
        else:
            return np.zeros(self.data_shape)
    
    def gzz(self):
        if 'gammadown3' in self.data:
            return self["gammadown3"][2,2]
        else:
            return np.ones(self.data_shape)

    def gammadown3(self):
        return maths.format_rank2_3(
            [self["gxx"], self["gxy"], self["gxz"],
             self["gyy"], self["gyz"], self["gzz"]])

    def gammaup3(self):
        return maths.inverse3(self["gammadown3"])
    
    def dtgammaup3(self):
        return (self.Lie_beta(self["gammaup3"], 's_uu') 
                - 2 * self["alpha"] * self["Kup3"])

    def gammadet(self):
        return maths.determinant3(self["gammadown3"])
    
    def gammadown4(self):
        return np.array(
                [[self["betamag"], self["betadown3"][0], 
                  self["betadown3"][1], self["betadown3"][2]], 
                 [self["betadown3"][0], self["gammadown3"][0,0], 
                  self["gammadown3"][0,1], self["gammadown3"][0,2]], 
                 [self["betadown3"][1], self["gammadown3"][1,0], 
                  self["gammadown3"][1,1], self["gammadown3"][1,2]], 
                 [self["betadown3"][2], self["gammadown3"][2,0], 
                  self["gammadown3"][2,1], self["gammadown3"][2,2]]])
    
    def gammaup4(self):
        zero = np.zeros(self.data_shape)
        return np.array(
                [[zero, zero, zero, zero], 
                 [zero, self["gammaup3"][0,0], 
                  self["gammaup3"][0,1], self["gammaup3"][0,2]], 
                 [zero, self["gammaup3"][1,0], 
                  self["gammaup3"][1,1], self["gammaup3"][1,2]], 
                 [zero, self["gammaup3"][2,0], 
                  self["gammaup3"][2,1], self["gammaup3"][2,2]]])
    
    def psi_bssnok(self):
        return self["gammadet"]**(1/12)
    
    def phi_bssnok(self):
        return np.log(self["psi_bssnok"])
    
    def dtphi_bssnok(self):
        return (self.Lie_beta(self["phi_bssnok"], '', weight=1/6)
                - (1/6) * self["alpha"] * self["Ktrace"])
    
    def gammaup3_bssnok(self):
        return self["psi_bssnok"]**(4) * self["gammaup3"]
    
    def gammadown3_bssnok(self):
        return self["psi_bssnok"]**(-4) * self["gammadown3"]
    
    def dtgammadown3_bssnok(self):
        return (self.Lie_beta(self["gammadown3_bssnok"], 's_dd', weight=-2/3) 
                -2 * self["alpha"] * self["Adown3_bssnok"])
    
    # Extrinsic curvature
    def kxx(self):
        if 'Kdown3' in self.data:
            return self["Kdown3"][0,0]
        else:
            return np.zeros(self.data_shape)
    
    def kxy(self):
        if 'Kdown3' in self.data:
            return self["Kdown3"][0,1]
        else:
            return np.zeros(self.data_shape)
    
    def kxz(self):
        if 'Kdown3' in self.data:
            return self["Kdown3"][0,2]
        else:
            return np.zeros(self.data_shape)
    
    def kyy(self):
        if 'Kdown3' in self.data:
            return self["Kdown3"][1,1]
        else:
            return np.zeros(self.data_shape)
    
    def kyz(self):
        if 'Kdown3' in self.data:
            return self["Kdown3"][1,2]
        else:
            return np.zeros(self.data_shape)
    
    def kzz(self):
        if 'Kdown3' in self.data:
            return self["Kdown3"][2,2]
        else:
            return np.zeros(self.data_shape)
    
    def Kdown3(self):
        return maths.format_rank2_3(
            [self["kxx"], self["kxy"], self["kxz"],
             self["kyy"], self["kyz"], self["kzz"]])
    
    def Kup3(self):
        return np.einsum(
            'ia..., jb..., ij... -> ab...', 
            self["gammaup3"], self["gammaup3"],  self["Kdown3"])
    
    def Ktrace(self):
        return self.trace3(self["Kdown3"])
    
    def dtKtrace(self):
        dtK = (
            self.Lie_beta(self["Ktrace"], '')
            - np.einsum('ij..., ij... -> ...', 
                         self["gammaup3"], self["DDalpha"])
            + self["alpha"] * (
                self["A2_bssnok"]
                + (1/3) * self["Ktrace"]**2))
        if self.vaccum:
            self.myprint('Using vacuum shortcut for dtKtrace')
            return dtK
        else:
            return dtK + 0.5 * self.kappa * self["alpha"] * (
                self["rho_n"] + self["Stresstrace_n"] 
                - 2 * (self.Lambda / self.kappa))
    
    def Adown3(self):
        return self.tracefree3(self["Kdown3"])
    
    def Aup3(self):
        return np.einsum('ia..., jb..., ab... -> ij...',
                         self["gammaup3"], self["gammaup3"], self["Adown3"])
    
    def A2(self):
        return self.magnitude3(self["Adown3"])
    
    def Adown3_bssnok(self):
        return self["psi_bssnok"]**(-4) * self["Adown3"]
    
    def dtAdown3_bssnok(self):
        # Eq 11.38 Baumgarte & Shapiro
        Ricci = self["s_Ricci_down3_bssnok"] + self["s_Ricci_down3_phi"]
        if self.vaccum:
            self.myprint('Using vacuum shortcut for dtAdown3_bssnok')
            innerterm = self.tracefree3(
                - self["DDalpha"]
                + self["alpha"] * Ricci)
        else:
            innerterm = self.tracefree3(
                - self["DDalpha"]
                + self["alpha"] * Ricci
                - self["alpha"] * self.kappa * self["Stressdown3_n"])
        AAterm = np.einsum(
            'ia..., bj..., ab... -> ij...',
            self["Adown3_bssnok"], self["Adown3_bssnok"], 
            self["gammaup3_bssnok"])
        return (self.Lie_beta(self["Adown3_bssnok"], 's_dd', weight=-2/3)
                + np.exp(-4 * self["phi_bssnok"]) * innerterm
                + self["alpha"] * (
                    self["Ktrace"] * self["Adown3_bssnok"]
                    - 2 * AAterm))
    
    def Aup3_bssnok(self):
        return self["psi_bssnok"]**(4) * self["Aup3"]
    
    def A2_bssnok(self):
        return np.einsum('ij..., ij... -> ...', 
                          self["Adown3_bssnok"], self["Aup3_bssnok"])
    
    # Lapse
    def alpha(self):
        return np.ones(self.data_shape)
    
    def dtalpha(self):
        return np.zeros(self.data_shape)
    
    def DDalpha(self):
        return self.s_covd(self.s_covd(self["alpha"], ''), 'd')
    
    # Shift
    def betax(self):
        if 'betaup3' in self.data:
            return self["betaup3"][0]
        else:
            return np.zeros(self.data_shape)
    
    def betay(self):
        if 'betaup3' in self.data:
            return self["betaup3"][1]
        else:
            return np.zeros(self.data_shape)
    
    def betaz(self):
        if 'betaup3' in self.data:
            return self["betaup3"][2]
        else:
            return np.zeros(self.data_shape)

    def betaup3(self):
        return np.array([self["betax"], self["betay"], self["betaz"]])
        
    def dtbetax(self):
        if 'dtbetaup3' in self.data:
            return self["dtbetaup3"][0]
        else:
            return np.zeros(self.data_shape)
    
    def dtbetay(self):
        if 'dtbetaup3' in self.data:
            return self["dtbetaup3"][1]
        else:
            return np.zeros(self.data_shape)
    
    def dtbetaz(self):
        if 'dtbetaup3' in self.data:
            return self["dtbetaup3"][2]
        else:
            return np.zeros(self.data_shape)
    
    def dtbetaup3(self):
        return np.array([self["dtbetax"], self["dtbetay"], self["dtbetaz"]])
    
    def betadown3(self):
        return np.einsum(
                'i..., ij... -> j...',
                self["betaup3"], self["gammadown3"])
        
    def betamag(self):
        return np.einsum(
                'i..., i... -> ...',
                self["betaup3"], self["betadown3"])
    
    def dttau(self):
        return np.sqrt(abs(self["alpha"]**2 - self["betamag"]))
    
    # Timelike normal vector
    def nup4(self):
        return maths.safe_division(np.array(
            [np.ones(self.data_shape), 
             -self["betaup3"][0],
             -self["betaup3"][1], 
             -self["betaup3"][2]]),
             self["alpha"])

    def ndown4(self):
        return np.array(
                [-self["alpha"], 
                 np.zeros(self.data_shape),
                 np.zeros(self.data_shape),
                 np.zeros(self.data_shape)])
    
    # Spacetime metric
    def gdown4(self):
        g00 = (-self["alpha"]**2 
               + np.einsum('i..., j..., ij... -> ...', 
                           self["betaup3"], self["betaup3"], 
                           self["gammadown3"]))
        return np.array(
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
        if 'gdown4' in self.data:
            return maths.determinant4(self["gdown4"])
        else:
            return - self["alpha"]**2 * self["gammadet"]
    
    # Null ray expansion
    def null_ray_exp_out(self):
        self.myprint(f"Center of extraction sphere set to"
                     + f" AurelCore.center = {self.center}")
        r = self.fd.cartesian_to_spherical(
            self.fd.x - self.center[0], 
            self.fd.y - self.center[1], 
            self.fd.z - self.center[2]
        )[0]
        return self.null_ray_expansion(r, direction='out')
    
    # Null ray expansion
    def null_ray_exp_in(self):
        self.myprint(f"Center of extraction sphere set to"
                     + f" AurelCore.center = {self.center}")
        r = self.fd.cartesian_to_spherical(
            self.fd.x - self.center[0], 
            self.fd.y - self.center[1], 
            self.fd.z - self.center[2]
        )[0]
        return self.null_ray_expansion(r, direction='in')
    
    # === Matter quantities
    # Eulerian observer follows n^mu
    # Lagrangian observer follows u^mu

    def rho0(self):
        if 'rho' in self.data:
            return maths.safe_division(self.data['rho'], (1 + self['eps']))
        else:
            return np.zeros(self.data_shape)
    
    def press(self):
        return np.zeros(self.data_shape)
    
    def eps(self):
        if (('rho' in self.data) and ('rho0' in self.data)):
            return maths.safe_division(self.data['rho'], self.data['rho0']) - 1
        else:
            return np.zeros(self.data_shape)
    
    def rho(self):
        return self["rho0"] * (1 + self["eps"])
    
    def enthalpy(self):
        return (
            1 + self["eps"] 
            + maths.safe_division(self["press"], self["rho0"]))
    
    # Fluid velocity
    def w_lorentz(self):
        return np.ones(self.data_shape)
    
    def velx(self):
        return np.zeros(self.data_shape)
    
    def vely(self):
        return np.zeros(self.data_shape)
    
    def velz(self):
        return np.zeros(self.data_shape)
    
    def velup3(self):
        return np.array([self["velx"], self["vely"], self["velz"]])
    
    def velup4(self):
        return np.array([np.zeros(self.data_shape), 
                          self["velx"], self["vely"], self["velz"]])
    
    def veldown3(self):
        return self["veldown4"][1:]
    
    def veldown4(self):
        return np.einsum(
            'i..., ij... -> j...', 
            self["velup4"], self["gammadown4"])
    
    def uup0(self):
        return maths.safe_division(self["w_lorentz"], self["alpha"])
    
    def uup3(self):
        return self["w_lorentz"] * (
            self["velup3"] 
            - maths.safe_division(self["betaup3"], self["alpha"]))
    
    def uup4(self):
        return np.array(
            [self["uup0"], self["uup3"][0], self["uup3"][1], self["uup3"][2]])
    
    def udown4(self):
        return np.einsum('ab..., b... -> a...', self["gdown4"], self["uup4"])
    
    def udown3(self):
        return self["udown4"][1:]
    
    def hdown4(self):
        return self["gdown4"] + np.einsum('a..., b... -> ab...', 
                                          self["udown4"], self["udown4"])
    
    def hdet(self):
        return maths.determinant3(self["hdown4"][1:,1:])
    
    def hmixed4(self):
        return np.einsum('ac..., cb... -> ab...', 
                         self["gup4"], self["hdown4"])
    
    def hup4(self):
        return self["gup4"] + np.einsum('a..., b... -> ab...', 
                                        self["uup4"], self["uup4"])
    
    # Energy-stress tensor
    def Tdown4(self):
        return (self["rho"] * np.einsum('a..., b... -> ab...', 
                                        self["uup4"], self["uup4"])
                + self["press"] * self["hdown4"])
    
    def Tup4(self):
        return np.einsum('ac..., bd..., ab... -> cd...', 
                          self["gup4"], self["gup4"], self["Tdown4"])
    
    def Ttrace(self):
        if "Tdown4" not in self.data:
            return 3 * self["press_n"] - self["rho_n"]
        else:
            return self.trace4(self["Tdown4"])
    
    # Fluid quantities in Eulerian frame
    def rho_n(self):
        return np.einsum('ab..., a..., b... -> ...',
                         self["Tdown4"], self["nup4"], self["nup4"])
    
    def fluxup3_n(self):
        return - np.einsum(
            'ab..., bc..., c... -> a...', 
            self["gammaup4"], self["Tdown4"], self["nup4"])[1:]
    
    def fluxdown3_n(self):
        return np.einsum(
            'a..., ab... -> b...', 
            self["fluxup3_n"], self["gammadown3"])
    
    def angmomup3_n(self):
        return np.einsum(
            'ij..., j... -> i...', 
            self["gammaup3"], self["angmomdown3_n"])
    
    def angmomdown3_n(self):
        return np.einsum(
            'ijk..., j..., k... -> i...', 
            self.levicivita_down3(), 
            self.fd.cartesian_coords, 
            self["fluxup3_n"])
    
    def Stressup3_n(self):
        return np.einsum(
            'ac..., bd..., ab... -> cd...', 
            self["gammaup3"], self["gammaup3"], self["Tdown4"][1:,1:])
    
    def Stressdown3_n(self):
        return np.einsum(
            'ac..., bd..., ab... -> cd...', 
            self["gammadown3"], self["gammadown3"], self["Stressup3_n"])
    
    def Stresstrace_n(self):
        return self.trace3(self["Tdown4"][1:,1:])
    
    def press_n(self):
        return np.einsum('ab..., ab... -> ...', 
                        self["gammaup3"], self["Tdown4"][1:,1:]) / 3
    
    def anisotropic_press_down3_n(self):
        return self["Stressdown3_n"] - self["gammadown3"] * self["press_n"]
    
    def rho_n_fromHam(self):
        return (self["s_RicciS"] 
                + self["Ktrace"]**2 
                - np.einsum('ij..., ij... -> ...', 
                            self["Kdown3"], self["Kup3"])
                - 2 * self.Lambda) / (2 * self.kappa)
    
    def fluxup3_n_fromMom(self):
        CovD_term = self.s_covd(
                self["Kup3"] - self["gammaup3"] * self["Ktrace"], 'uu')
        return np.einsum('bab... -> a...', CovD_term) / self.kappa
    
    # Conserved quantities
    def conserved_D(self):
        return self["rho0"] * self["w_lorentz"] * np.sqrt(self["gammadet"])
    
    def conserved_E(self):
        return self["conserved_D"] * self["eps"]
    
    def conserved_Sdown4(self):
        return self["conserved_D"] * self["enthalpy"] * self["udown4"]
    
    def conserved_Sdown3(self):
        return self["conserved_Sdown4"][1:]
    
    def conserved_Sup4(self):
        return np.einsum('im...,m...->i...', 
                         self["gup4"], self["conserved_Sdown4"])
    
    def conserved_Sup3(self):
        return self["conserved_Sup4"][1:]

    def dtconserved(self):
        self.myprint('WARNING: dtconserved only works for constant press/rho')
        V = maths.safe_division(self["uup4"][1:], self["uup4"][0])
        sgdet = np.sqrt(self["gammadet"]) 

        divD = np.einsum('ii...->...', 
                         self.fd.d3_rank1tensor(self["conserved_D"] * V))
        divE = np.einsum('ii... -> ...', 
                         self.fd.d3_rank1tensor(self["conserved_E"] * V))
        divW = np.einsum('ii... -> ...', 
                         self.fd.d3_rank1tensor(sgdet * self["w_lorentz"] * V))
        divSdown3 = np.einsum(
                    'jij...->i...', 
                    self.fd.d3_rank2tensor(
                        np.einsum('i...,j...->ij...', 
                                  self["conserved_Sdown3"], V)))

        dtD = - divD
        SSdg = ((self["conserved_Sdown4"][0] 
                 * self["conserved_Sdown4"][0] 
                 * self.fd.d3_scalar(self["gdown4"][0,0]))
                + (self["conserved_Sdown4"][0] 
                   * np.einsum('j..., ij... -> i...', 
                               self["conserved_Sup3"], 
                               self.fd.d3_rank1tensor(self["betadown3"])))
                + (np.einsum('j..., k..., ijk... -> i...', 
                             self["conserved_Sup3"],
                             self["conserved_Sup3"], 
                             self.fd.d3_rank2tensor(self["gammadown3"]))))
        dtSdown3 = (- divSdown3 
               + maths.safe_division( SSdg, 2 * self["conserved_Sdown4"][0])
               - self["alpha"] * sgdet * self.fd.d3_scalar(self["press"]))
        # dtw_lorentz
        CovDbeta = self.s_covd(self["betaup3"], 'u')
        dtsgdet = sgdet * ( - self["alpha"] * self["Ktrace"]
                            + np.einsum('ii... -> ...', CovDbeta))
        W2m1 = self["w_lorentz"]**2 - 1
        W2m1oDpE = maths.safe_division(
            W2m1, (self["conserved_D"] + self["conserved_E"]))
        fac1 = maths.safe_division(
            W2m1, 2 * self["w_lorentz"] 
            * np.einsum('ij..., i..., j... -> ...', 
                self["gammaup3"], 
                self["conserved_Sdown3"], 
                self["conserved_Sdown3"]))
        par1 = (np.einsum('ij..., i..., j... -> ...',
                          self["dtgammaup3"], 
                          self["conserved_Sdown3"], 
                          self["conserved_Sdown3"])
                + 2 * np.einsum('ij..., i..., j... -> ...',
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
                (np.einsum('ij...,i...,j...->...',
                        self["dtgammaup3"], self["udown3"], self["udown3"])
                + 2*np.einsum('i...,i...->...', self["uup3"], dtudown3)),
                (2 * self["alpha"] * self["w_lorentz"])) 
            - maths.safe_division(self["uup0"] * self["dtalpha"], 
                                  self["alpha"]))
        dtudown4 = np.array(
                    [dtudown0, 
                     dtudown3[0], dtudown3[1], dtudown3[2]])
        # spacetime covariant derivative
        return self.st_covd(self["udown4"], dtudown4, 'u')
    
    def accelerationdown4(self):
        return np.einsum(
            'a..., ab... -> b...',
            self["uup4"], self["st_covd_udown4"])
    
    def accelerationup4(self):
        return np.einsum(
            'ab..., b... -> a...',
            self["gup4"], self["accelerationdown4"])
    
    def s_covd_udown4(self):
        return np.einsum(
            'ab...,ac...->bc...', self["hmixed4"], self["st_covd_udown4"])
    
    def thetadown4(self):
        return maths.symmetrise_tensor(self["s_covd_udown4"])
    
    def theta(self):
        return np.einsum('ab..., ab... -> ...', 
                         self["hup4"], self["thetadown4"])
    
    def sheardown4(self):
        return self["thetadown4"] - (1/3) * self["theta"] * self["hdown4"]
    
    def shear2(self):
        return 0.5 * np.einsum(
            'ai..., bj..., ab..., ij... -> ...', 
            self["hup4"], self["hup4"], self["sheardown4"], self["sheardown4"])
    
    def omegadown4(self):
        return maths.antisymmetrise_tensor(self["s_covd_udown4"])
    
    def omega2(self):
        return 0.5 * np.einsum(
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
        Gx = 0.5 * np.array(
            [[dgammaxx[0], dgammaxx[1], dgammaxx[2]],
             [dgammaxx[1], 2*dgammaxy[1]-dgammayy[0], Gxyz],
             [dgammaxx[2], Gxyz, 2*dgammaxz[2]-dgammazz[0]]])
            
        Gyxz = dgammayz[0] + dgammaxy[2] - dgammaxz[1]
        Gy = 0.5 * np.array(
            [[2*dgammaxy[0]-dgammaxx[1], dgammayy[0], Gyxz],
             [dgammayy[0], dgammayy[1], dgammayy[2]],
             [Gyxz, dgammayy[2], 2*dgammayz[2]-dgammazz[1]]])
            
        Gzxy = dgammayz[0] + dgammaxz[1] - dgammaxy[2]
        Gz = 0.5 * np.array(
            [[2*dgammaxz[0]-dgammaxx[2], Gzxy, dgammazz[0]],
             [Gzxy, 2*dgammayz[1]-dgammayy[2], dgammazz[1]],
             [dgammazz[0], dgammazz[1], dgammazz[2]]])
        Gddd = np.array([Gx,Gy,Gz])
            
        # Spatial Christoffel symbols with indices: Gamma^{i}_{kl}.
        return np.einsum('ij..., jkl... -> ikl...', self["gammaup3"], Gddd)
    
    def s_Riemann_uddd3(self):
        dGudd3 = np.array([
            [self.fd.d3x_rank2tensor(self["s_Gamma_udd3"][j]) 
             for j in range(3)],
            [self.fd.d3y_rank2tensor(self["s_Gamma_udd3"][j]) 
             for j in range(3)],
            [self.fd.d3z_rank2tensor(self["s_Gamma_udd3"][j]) 
             for j in range(3)]])
        Rterm0 = np.einsum('cabd... -> abcd...', dGudd3)
        Rterm1 = np.einsum('dabc... -> abcd...', dGudd3)
        Rterm2 = np.einsum('apc..., pbd... -> abcd...', 
                        self["s_Gamma_udd3"], self["s_Gamma_udd3"])
        Rterm3 = np.einsum('apd..., pbc... -> abcd...', 
                            self["s_Gamma_udd3"], self["s_Gamma_udd3"])
        return Rterm0 - Rterm1 + Rterm2 - Rterm3
    
    def s_Riemann_down3(self):
        return np.einsum('abcd..., ai... -> ibcd...', 
                         self["s_Riemann_uddd3"], self["gammadown3"])
    
    def s_Ricci_down3(self):
        if "s_Riemann_down3" in self.data.keys():
            return np.einsum('abcd..., ac... -> bd...', 
                             self["s_Riemann_down3"], self["gammaup3"])
        else:
            dGudd3 = np.array([
                [self.fd.d3x_rank2tensor(self["s_Gamma_udd3"][j]) 
                    for j in range(3)],
                [self.fd.d3y_rank2tensor(self["s_Gamma_udd3"][j]) 
                    for j in range(3)],
                [self.fd.d3z_rank2tensor(self["s_Gamma_udd3"][j]) 
                    for j in range(3)]])
            Rterm0 = np.einsum('aabd... -> bd...', dGudd3)
            Rterm1 = np.einsum('daba... -> bd...', dGudd3)
            Rterm2 = np.einsum('apa..., pbd... -> bd...', 
                            self["s_Gamma_udd3"], self["s_Gamma_udd3"])
            Rterm3 = np.einsum('apd..., pba... -> bd...', 
                                self["s_Gamma_udd3"], self["s_Gamma_udd3"])
            return Rterm0 - Rterm1 + Rterm2 - Rterm3

    def s_RicciS(self):
        return self.trace3(self["s_Ricci_down3"])
        
    # of spacetime metric
    def st_Gamma_udd4(self):
        # Repeated calculations
        dalpha = self.fd.d3_scalar(self["alpha"])
        betadalpha = np.einsum('m..., m... -> ...', self["betaup3"], dalpha)
        betaK = np.einsum('m..., mn... -> n...', 
                          self["betaup3"], self["Kdown3"])
        betabetaK = np.einsum('m..., n... -> ...', self["betaup3"], betaK)
        dbeta = self.s_covd(self["betaup3"], 'u')

        # time part of index up
        Gttt = maths.safe_division(
            self["dtalpha"] + betadalpha - betabetaK, 
            self["alpha"])
        Gtti = maths.safe_division(dalpha - betaK, self["alpha"])
        Gtij = maths.safe_division(- self["Kdown3"], self["alpha"])
        Gt = np.array(
            [[Gttt, Gtti[0], Gtti[1], Gtti[2]],
            [Gtti[0], Gtij[0,0], Gtij[0,1], Gtij[0,2]],
            [Gtti[1], Gtij[1,0], Gtij[1,1], Gtij[1,2]],
            [Gtti[2], Gtij[2,0], Gtij[2,1], Gtij[2,2]]])

        # space part of index up
        Gltt = (
            np.einsum('lm..., m... -> l...', 
                    self["gammaup3"], 
                    self["alpha"] * dalpha - 2 * self["alpha"]*betaK)
            - self["betaup3"] * Gttt 
            + self["dtbetaup3"]
            + np.einsum('m..., ml...-> l...', self["betaup3"], dbeta))
        Glmt = (
            np.einsum('l..., m...-> lm...', -self["betaup3"], Gtti)
            - self["alpha"] * np.einsum('ln..., nm... -> lm...', 
                                    self["gammaup3"], 
                                    self["Kdown3"])
            + dbeta)
        Glij = (
            self["s_Gamma_udd3"] 
            + maths.safe_division(
                np.einsum(
                    'l..., ij... -> lij...', self["betaup3"], self["Kdown3"]), 
                    self["alpha"]))
        Gl = np.array(
            [[Gltt, Glmt[:,0], Glmt[:,1], Glmt[:,2]], 
            [Glmt[:,0], Glij[:,0,0], Glij[:,0,1], Glij[:,0,2]],
            [Glmt[:,1], Glij[:,1,0], Glij[:,1,1], Glij[:,1,2]],
            [Glmt[:,2], Glij[:,2,0], Glij[:,2,1], Glij[:,2,2]]])
        return np.array([Gt, Gl[:,:,0], Gl[:,:,1], Gl[:,:,2]])
    
    def st_Riemann_uddd4(self):
        return np.einsum('abcd..., ai... -> ibcd...',
                         self["st_Riemann_down4"], self["gup4"])
    
    def st_Riemann_down4(self):
        # Riemann_ssss : Gauss equation, eq 2.38 in Shibata
        Riemann_ssss = (self["s_Riemann_down3"]
                        + np.einsum('ac..., bd... -> abcd...', 
                                    self["Kdown3"], self["Kdown3"])
                        - np.einsum('ad..., bc... -> abcd...', 
                                    self["Kdown3"], self["Kdown3"]))
        
        # Riemann_ssst : Codazzi equation, eq 2.41 in Shibata
        dKdown = self.s_covd(self["Kdown3"], 'dd')
        Riemann_ssst = (
            np.einsum('ijkl..., l... -> ijk...', 
                       Riemann_ssss, self["betaup3"])
            + self["alpha"] * (
                np.einsum('jik... -> ijk...', dKdown) 
                - dKdown))
            
        # Riemann_stst: the Mainardi equation, eq 2.56 in Shibata
        Kdown4 = self.s_to_st(self["Kdown3"])
        Riemann_stst = (
            np.einsum('jki..., k... -> ij...', Riemann_ssst, self["betaup3"])
            + np.einsum('ikj..., k... -> ij...', 
                         Riemann_ssst, self["betaup3"])
            + np.einsum('ikjl..., k..., l... -> ij...', 
                        Riemann_ssss, self["betaup3"], self["betaup3"])
        + self["alpha"]**2 * (
            self["s_Ricci_down3"]
            - np.einsum('ib..., ja..., ab... -> ij...', 
                        Kdown4, Kdown4, self["gup4"])[1:,1:]
            + self["Kdown3"] * self["Ktrace"]))
        if self.vaccum:
            self.myprint('Using vacuum shortcut for st_Riemann_down4')
        else:
            Riemann_stst += - self["alpha"]**2 * self["st_Ricci_down3"]
            
        # put it all together
        R = maths.populate_4Riemann(
            Riemann_ssss, Riemann_ssst, Riemann_stst)
        return R
    
    def st_Riemann_uudd4(self):
        return np.einsum('abcd..., ae..., bf... -> efcd...', 
                         self["st_Riemann_down4"], self["gup4"], self["gup4"])
    
    def st_Ricci_down4(self):
        if "Tdown4" in self.data.keys():
            return (
            self.Lambda * self["gdown4"]
            + self.kappa * (
               self["Tdown4"]
                - 0.5 * self["Ttrace"] * self["gdown4"]))
        else:
            return np.einsum('abad... -> bd...', self["st_Riemann_uddd4"])
    
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
    
    def Einsteindown4(self):
        return (self["st_Ricci_down4"] 
                - 0.5 * self["st_RicciS"] * self["gdown4"])
    
    def Kretschmann(self):
        return np.einsum(
            'abcd..., cdab... -> ...', 
            self["st_Riemann_uudd4"], self["st_Riemann_uudd4"])
    
    # In BSSNOK form
    def s_Gamma_udd3_bssnok(self):
        # Alcubierre 2.8.14
        dphi = self.fd.d3_scalar(self["phi_bssnok"])
        return self["s_Gamma_udd3"] - 2 * (
            np.einsum('ki..., j... -> kij...', self.kronecker_delta3(), dphi)
            + np.einsum('kj..., i... -> kij...', self.kronecker_delta3(), dphi)
            - np.einsum('ij..., kl..., l... -> kij...', 
                        self["gammadown3"], self["gammaup3"], dphi))
    
    def s_Gamma_bssnok(self):
        return - np.einsum('jij... -> i...', 
                           self.fd.d3_rank2tensor(self["gammaup3_bssnok"]))
    
    def dts_Gamma_bssnok(self):
        # Eq 2.8.25 of Alcubierre
        ddbeta = self.fd.d3_rank2tensor(
            self.fd.d3_rank1tensor(self["betaup3"]))
        Gamma = self.Lie_beta(self["s_Gamma_bssnok"], 's_u', weight=2/3)
        Gamma += np.einsum('jk..., jki... -> i...', 
                          self["gammaup3_bssnok"], ddbeta)
        Gamma += (1/3) * np.einsum('ij..., jkk... -> i...',
                                   self["gammaup3_bssnok"], ddbeta)
        Gamma += - 2 * np.einsum('ij..., j... -> i...', 
                                 self["Aup3_bssnok"], 
                                 self.fd.d3_scalar(self["alpha"]))
        Gamma += 2 * self["alpha"] * np.einsum('ijk..., jk... -> i...', 
                                               self["s_Gamma_udd3_bssnok"], 
                                               self["Aup3_bssnok"])
        Gamma += 12 * self["alpha"] * np.einsum('ij..., j... -> i...', 
                                                self["Aup3_bssnok"], 
                                                self.fd.d3_scalar(
                                                    self["phi_bssnok"]))
        Gamma += - (4/3) * self["alpha"] * np.einsum('ij..., j... -> i...',
                                                    self["gammaup3_bssnok"], 
                                                    self.fd.d3_scalar(
                                                        self["Ktrace"]))
        if self.vaccum:
            self.myprint('Using vacuum shortcut for dts_Gamma_bssnok')
        else:
            Gamma += (- 2 * self.kappa * self["alpha"] 
                    * np.exp(4 * self["phi_bssnok"]) 
                    * self["fluxup3_n"])
        return Gamma
    
    def s_Ricci_down3_bssnok(self):
        # Eq 2.8.18 of Alcubierre
        Ricci = - 0.5 * np.einsum('lm..., lmij... -> ij...', 
                                  self["gammaup3_bssnok"],
                                  self.fd.d3_rank3tensor(
                                      self.fd.d3_rank2tensor(
                                          self["gammadown3_bssnok"])))
        tosymmetrise = np.einsum('ki..., jk... -> ij...',
                           self["gammadown3_bssnok"],
                           self.fd.d3_rank1tensor(self["s_Gamma_bssnok"]))
        Gddd = np.einsum('oi..., ojk... -> ijk...', 
                         self["gammadown3_bssnok"],
                         self["s_Gamma_udd3_bssnok"])
        tosymmetrise += np.einsum('k..., ijk... -> ij...',
                           self["s_Gamma_bssnok"],
                           Gddd)
        tosymmetrise += 2 * np.einsum('lm..., kli..., jkm... -> ij...',
                            self["gammaup3_bssnok"],
                            self["s_Gamma_udd3_bssnok"],
                            Gddd)
        Ricci += maths.symmetrise_tensor(tosymmetrise)
        Ricci += np.einsum('lm..., kim..., klj... -> ij...',
                                  self["gammaup3_bssnok"],
                                  self["s_Gamma_udd3_bssnok"],
                                  Gddd)
        return Ricci
    
    def s_RicciS_bssnok(self):
        return np.einsum('ij..., ij... -> ...', 
                         self["gammaup3_bssnok"], 
                         self["s_Ricci_down3_bssnok"])
    
    def s_Ricci_down3_phi(self):
        # Alcubierre 2.8.18
        dphi = self.fd.d3_scalar(self["phi_bssnok"])
        ddphi = (self.fd.d3_rank1tensor(dphi) 
                 - np.einsum('kij..., k... -> ij...', 
                             self["s_Gamma_udd3_bssnok"], dphi))
        return (
            - 2 * ddphi 
            - 2 * self["gammadown3_bssnok"] * np.einsum(
                'ij..., ij... -> ...', self["gammaup3_bssnok"], ddphi)
            + 4 * np.einsum('i..., j... -> ij...', dphi, dphi)
            - 4 * self["gammadown3_bssnok"] * np.einsum(
                'ij..., i..., j... -> ...', self["gammaup3_bssnok"], dphi, dphi)
            )
    
    # Constraints
    def Hamiltonian(self):
        Ham = (self["s_RicciS"] 
                + self["Ktrace"]**2 
                - np.einsum('ij..., ij... -> ...', 
                            self["Kdown3"], self["Kup3"]))
        if self.vaccum:
            self.myprint('Using vacuum shortcut for Hamiltonian')
            return Ham
        else:
            return (Ham
                - 2 * self.kappa * self["rho_n"] 
                - 2 * self.Lambda)
    
    def Hamiltonian_Escale(self):
        HamE = (
            self["s_RicciS"]**2 
            + self["Ktrace"]**4 
            + np.einsum('ij..., ij... -> ...', 
                        self["Kdown3"], self["Kup3"])**2 )
        if self.vaccum:
            self.myprint('Using vacuum shortcut for Hamiltonian_Escale')
            return np.sqrt(abs(HamE))
        else:
            return np.sqrt(abs(
                HamE 
                + (2 * self.kappa * self["rho_n"])**2 
                + (2 * self.Lambda)**2))
    
    def Momentumup3(self):
        if "Momentumx" in self.data.keys():
            return np.array(
                [self["Momentumx"], self["Momentumy"], self["Momentumz"]]) 
        else:
            CovD_term = self.s_covd(
                self["Kup3"] - self["gammaup3"] * self["Ktrace"], 'uu')
            Mom = np.einsum('bab... -> a...', CovD_term)
            if self.vaccum:
                self.myprint('Using vacuum shortcut for Momentumup3')
                return Mom
            else:
                return (Mom - self.kappa * self["fluxup3_n"])
    
    def Momentum_Escale(self):
        DdKdd = self.s_covd(self["Kdown3"], 'dd')

        DKd = np.einsum('ab..., abc... -> c...', self["gammaup3"], DdKdd)
        DdK = np.einsum('bc..., abc... -> a...', self["gammaup3"], DdKdd)
        DKd2 = np.einsum('a..., ad..., d... -> ...', 
                         DKd, self["gammaup3"], DKd)
        DdK2 = np.einsum('a..., ad..., d... -> ...', 
                         DdK, self["gammaup3"], DdK)
        if self.vaccum:
            self.myprint('Using vacuum shortcut for Momentum_Escale')
            return np.sqrt(abs(DKd2 + DdK2))
        else:
            Eflux2 = ((self.kappa**2)
                  * np.einsum('a..., a... -> ...', 
                              self["fluxup3_n"], self["fluxdown3_n"]))
            return np.sqrt(abs(DKd2 + DdK2 + Eflux2))
    
    # === Gravito-electromagnetism quantities
    def st_Weyl_down4(self): 
        if "st_Riemann_down4" in self.data.keys():            
            Cdown = self["st_Riemann_down4"]
            if self.vaccum:
                self.myprint('Using vacuum shortcut for st_Weyl_down4')
                return Cdown
            else:
                for a in range(4):
                    for b in range(4):
                        for c in range(4):
                            for d in range(4):
                                Cdown[a,b,c,d] += (
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
                      + 2.0 * np.einsum('a..., b... -> ab...',
                                        self["ndown4"], self["ndown4"]))
            LCudd4 = np.einsum('ec..., d..., dcab... -> eab...', self["gup4"], 
                            self["nup4"], self.levicivita_down4())
                
            Cdown4 = (np.einsum('ac..., db... -> abcd...', ldown4, Endown4)
                    - np.einsum('ad..., cb... -> abcd...', ldown4, Endown4))
            Cdown4 -= (np.einsum('bc..., da... -> abcd...', ldown4, Endown4)
                    - np.einsum('bd..., ca... -> abcd...', ldown4, Endown4))
            Cdown4 -= np.einsum('cde..., eab... -> abcd...',
                                (np.einsum('c..., de... -> cde...',
                                        self["ndown4"], Bndown4)
                                - np.einsum('d..., ce... -> cde...',
                                            self["ndown4"], Bndown4)), LCudd4)
            Cdown4 -= np.einsum('abe..., ecd... -> abcd...', 
                                (np.einsum('a..., be... -> abe...', 
                                        self["ndown4"], Bndown4) 
                                - np.einsum('b..., ae... -> abe...', 
                                            self["ndown4"], Bndown4)), LCudd4)
            return Cdown4
                   
    def Weyl_Psi(self):
        if "Weyl_Psi4r" in self.data.keys():
            return [None, None, None, None, 
                    self["Weyl_Psi4r"] + 1j * self["Weyl_Psi4i"]]
        else:
            lup4, kup4, mup4, mbup4 = self.null_vector_base()
            psi0 = np.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], kup4, mup4, kup4, mup4)
            psi1 = np.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], lup4, kup4, mup4, kup4)
            psi2 = np.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], kup4, mup4, mbup4, lup4)
            psi3 = np.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], kup4, lup4, mbup4, lup4)
            psi4 = np.einsum('abcd..., a..., b..., c..., d... -> ...',
                            self["st_Weyl_down4"], lup4, mbup4, lup4, mbup4)
            return [psi0, psi1, psi2, psi3, psi4]
        
    def Psi4_lm(self):
        self.myprint(f"Maximum l of spherical decomposition is set to"
                     + f" AurelCore.lmax = {self.lmax}")
        self.myprint(f"Extraction radii set to AurelCore.extract_radii"
                     + f" = {self.extract_radii}")
        self.myprint(f"Center of extraction sphere set to"
                     + f" AurelCore.center = {self.center}")
        self.myprint(f"Scipy interpolation method is set to"
                     + f" AurelCore.interp_method = {self.interp_method}")
        
        # Uniform sampling of spherical points
        # Inclination points
        Ntheta = np.max([np.min([self.fd.Nx, self.fd.Ny, self.fd.Nz]), 
                         self.lmax + 1])
        theta2_array = np.pi * np.arange(0.5, Ntheta+1.5, 1) / (Ntheta+1)
        dtheta = np.diff(theta2_array)[0]
        
        # Azimuthal points
        Nphi = 2 * Ntheta
        phi2_array = 2 * np.pi * np.arange(0.5, Nphi+1.5, 1) / (Nphi+1)
        dphi = np.diff(phi2_array)[0]
        
        # 2D spherical grid
        theta2, phi2 = np.meshgrid(theta2_array, phi2_array, indexing='ij')
        
        # Shift grid around sphere center
        shifted_grid = (
            self.fd.xarray - self.center[0],
            self.fd.yarray - self.center[1],
            self.fd.zarray - self.center[2])
        
        # For each extraction radius, interpolate Psi4 and decompose in
        psi4lm = {}
        for radius in self.extract_radii:
            
            # Points on sphere
            points_sphere = self.fd.spherical_to_cartesian(
                radius, theta2, phi2)
            
            # Interpolate Psi4 on sphere
            psi4_sphere = (
                numerical.interpolate(
                    np.real(self["Weyl_Psi"][4]), 
                    shifted_grid,
                    points_sphere, 
                    method=self.interp_method)
                + 1j * numerical.interpolate(
                    np.imag(self["Weyl_Psi"][4]), 
                    shifted_grid,
                    points_sphere, 
                    method=self.interp_method)
                )
            
            # Spherical harmonic decomposition
            psi4lm[radius] = maths.sYlm_coefficients(
                -2, self.lmax, psi4_sphere, 
                theta2, phi2, np.sin(theta2) * dtheta, dphi)
        
        return psi4lm
    
    def Weyl_invariants(self):
        Psis = self["Weyl_Psi"]
        I_inv = Psis[0]*Psis[4] - 4*Psis[1]*Psis[3] + 3*Psis[2]*Psis[2]
        J_inv = maths.determinant3(
            np.array([[Psis[4], Psis[3], Psis[2]], 
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
        return np.einsum(
            'b..., d..., abcd... -> ac...', 
            self["uup4"], self["uup4"], self["st_Weyl_down4"])
    
    def eweyl_n_down3(self):
        KKterm = np.einsum(
            'ia..., bj..., ab... -> ij...', 
            self["Kdown3"], self["Kdown3"], self["gammaup3"])
        Edown = self.tracefree3(self["s_Ricci_down3"]
                                + self["Ktrace"]*self["Kdown3"]
                                - KKterm)
        if self.vaccum:
            self.myprint('Using vacuum shortcut for eweyl_n_down3')
            return Edown
        else:
            return (Edown
                    - 0.5 * self.kappa * self.tracefree3(self["Stressdown3_n"])
                    )
    
    def bweyl_u_down4(self):
        LCuudd4 = np.einsum(
            'ac..., bd..., abef... -> cdef...', 
            self["gup4"], self["gup4"], self.levicivita_down4())
        return 0.5 * np.einsum(
            'b..., f..., abcd..., cdef... -> ae...', 
            self["uup4"], self["uup4"], self["st_Weyl_down4"], LCuudd4)
    
    def bweyl_n_down3(self):
        LCuud3 = np.einsum('ae..., bf..., efc... -> abc...', 
                        self["gammaup3"], self["gammaup3"], 
                        self.levicivita_down3())
        
        dKdown = self.s_covd(self["Kdown3"], 'dd')
        Bterm1 = np.einsum('cdb..., cda... -> ab...', LCuud3, dKdown)

        Kmixed3 = np.einsum('ij..., jk... -> ik...', 
                            self["gammaup3"], self["Kdown3"])
        Bterm2K = (self.s_covd(self["Ktrace"], '') 
                - np.einsum('ccb... -> b...', 
                            self.s_covd(Kmixed3, 'ud')))
        Bterm2 = 0.5 * np.einsum('cdb..., ac..., d... -> ab...', LCuud3, 
                        self["gammadown3"], Bterm2K)
            
        return Bterm1 + Bterm2
    
    ###########################################################################

    def null_ray_expansion(self, F, direction='out'):
        """Compute the expansion of null rays normal to a surface discribed F"""
        dF = self.fd.d3_scalar(F)
        dFmag = np.sqrt(np.einsum('ij..., i..., j... -> ...', 
                                        self["gammaup3"], dF, dF))
        sup = maths.safe_division(
            np.einsum('ij..., j... -> i...', self["gammaup3"], dF),
            dFmag)
        Disi = self.s_div(sup, 'u')
        Kss = np.einsum('ij..., i..., j... -> ...', 
                        self["Kdown3"], sup, sup)
        if direction == 'out':
            return Disi + Kss - self["Ktrace"]
        elif direction == 'in':
            return - Disi + Kss - self["Ktrace"]
        else:
            raise ValueError("direction must be 'out' or 'in'")
    
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
        inverse_sqrt_2 = 1 / np.sqrt(2)
        kup4 = (e0up4 + e1up4) * inverse_sqrt_2
        lup4 = (e0up4 - e1up4) * inverse_sqrt_2
        mup4 = (e2up4 + 1j*e3up4) * inverse_sqrt_2
        mbup4 = (e2up4 - 1j*e3up4) * inverse_sqrt_2
        return lup4, kup4, mup4, mbup4

    def tetrad_base(self):
        r"""Return an quasi-Kinnersley or arbitrary tetrad base.
        
        if AurelCore.tetrad == "quasi-Kinnersley":
            Which is the default, because the tetrad is set to
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

        self.myprint(f"Tetrad is set to AurelCore.tetrad"
                     + f" = {self.tetrad}")
        if self.tetrad == "quasi-Kinnersley":
            nup4 = np.array(
                [np.ones(self.data_shape), 
                 np.zeros(self.data_shape),
                 np.zeros(self.data_shape),
                 np.zeros(self.data_shape)])
            # not self["nup4"] because wave zone
            v1 = np.array([-self.fd.y, self.fd.x, 
                           np.zeros(np.shape(self.fd.x))])
            v2 = np.array([self.fd.x, self.fd.y, self.fd.z])
            LC = np.einsum('a..., abcd... -> bcd...', 
                        nup4, self.levicivita_down4())[1:,1:,1:]
            v3 = np.sqrt(self["gammadet"]) * np.einsum(
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
            e1up4 = np.array(
                [np.zeros(self.data_shape), v2[0], v2[1], v2[2]])
            # typo in ref paper
            e2up4 = np.array(
                [np.zeros(self.data_shape), v3[0], v3[1], v3[2]])
            e3up4 = np.array(
                [np.zeros(self.data_shape), v1[0], v1[1], v1[2]])
        else:
            # this is just an arbitrary orthonormal tetrad base
            zeros = np.zeros(self.data_shape)
            v1 = np.array([zeros, maths.safe_division(
                1.0, np.sqrt(self["gdown4"][1,1])), zeros, zeros])
            v2 = np.array([zeros, zeros, maths.safe_division(
                1.0, np.sqrt(self["gdown4"][2,2])), zeros])
            v3 = np.array([zeros, zeros, zeros, maths.safe_division(
                1.0, np.sqrt(self["gdown4"][3,3]))])

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
    # Derivatives
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
                G1 = np.einsum('abc..., b... -> ca...', 
                               self["s_Gamma_udd3"], f)
            elif indexing == 'd':
                G1 = - np.einsum('abc..., a... -> bc...', 
                                 self["s_Gamma_udd3"], f)
            else:
                raise ValueError(f"Field if of rank {rank} so indexing"
                                 + f" must be 'u' or 'd'")
            covd = df + G1
        elif rank ==2:
            df = self.fd.d3_rank2tensor(f)
            if indexing == 'uu':
                G1 = np.einsum('acd..., db... -> cab...', 
                               self["s_Gamma_udd3"], f)
                G2 = np.einsum('bcd..., ad... -> cab...', 
                               self["s_Gamma_udd3"], f)
            elif indexing == 'dd':
                G1 = - np.einsum('dca..., db... -> cab...', 
                                 self["s_Gamma_udd3"], f)
                G2 = - np.einsum('dcb..., ad... -> cab...', 
                                 self["s_Gamma_udd3"], f)
            elif indexing == 'ud':
                G1 = np.einsum('acd..., db... -> cab...', 
                               self["s_Gamma_udd3"], f)
                G2 = - np.einsum('dcb..., ad... -> cab...', 
                                 self["s_Gamma_udd3"], f)
            elif indexing == 'du':
                G1 = - np.einsum('dca..., db... -> cab...', 
                                 self["s_Gamma_udd3"], f)
                G2 = np.einsum('bcd..., ad... -> cab...', 
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
            covd = np.append(np.array([dtf]), self.fd.d3_scalar(f), axis = 0)
        elif rank == 1:
            df = np.append(np.array([dtf]), self.fd.d3_rank1tensor(f), 
                           axis = 0)
            if indexing == 'd':
                G1 = np.einsum('abc..., b... -> ca...', 
                               self["st_Gamma_udd4"], f)
            elif indexing == 'u':
                G1 = - np.einsum('abc..., a... -> bc...', 
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
            return np.einsum('aa... -> ...', covd)
        elif indexing == 'd':
            return np.einsum('ab..., ab... -> ...', self["gammaup3"], covd)
        elif indexing == 'uu' or indexing == 'ud':
            return np.einsum('aab... -> b...', covd)
        elif indexing == 'du':
            return np.einsum('aba... -> b...', covd)
        elif indexing == 'dd':
            return np.einsum('ab..., abc... -> c...', self["gammaup3"], covd)
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
            LCuud3 = np.einsum('ae..., bf..., d..., defc... -> abc...', 
                            self["gup4"], self["gup4"], self["nup4"], 
                            self.levicivita_down4())[1:, 1:, 1:]
            curl = maths.symmetrise_tensor(
                np.einsum('cda..., cbd... -> ab...', 
                          LCuud3, covd))
        else:
            raise ValueError(f"Don't know how to compute curl of tensor"+
                             + f" with indices {indexing}")
        return curl
    
    def Lie_beta(self, f, indexing, weight=0):
        """Compute the Lie derivative along the shift vector.

        Parameters
        ----------
        f : array_like
            The field to differentiate. Can be a scalar, vector, or rank-2 
            tensor, with shape corresponding to `indexing`.
        indexing : str
            String describing the type and index structure of `f`. 
            Must be one of::
                
                - ''     : scalar
                - 's_u'  : spatial vector, upper index
                - 'st_u' : spacetime vector, upper index
                - 's_d'  : spatial vector, lower index
                - 'st_d' : spacetime vector, lower index
                - 's_uu' : spatial rank-2 tensor, both indices up
                - 's_ud' : spatial rank-2 tensor, first up, second down
                - 's_du' : spatial rank-2 tensor, first down, second up
                - 's_dd' : spatial rank-2 tensor, both indices down

        weight : int or float, optional
            Weight for the density correction term (default: 0). 
            If nonzero, adds `weight * div(beta) * f` to the result, 
            where `div(beta)` is the divergence of the shift vector.

        Returns
        -------
        Lie : array_like
            The Lie derivative of `f` along β^i, with the same shape as `f`.

        """

        # Input checks
        if indexing == '':
            rank = 0
        else:
            if ((('s_' in indexing) or ('st_' in indexing)) 
                and (indexing.count('_') == 1)):
                s_or_st = indexing.split('_')[0]
                indices = indexing.split('_')[1]
                rank = len(indices)
                if rank>2:
                    raise NotImplementedError(
                        f"Haven't implemented Lie derivative along beta"
                        + f" for tensor of rank {rank}")
                for i in indices:
                    if i not in ['u', 'd']:
                        raise ValueError(
                            f"indexing must be 'u' or 'd' not {i}")
                if rank ==2 and s_or_st == 'st':
                    raise NotImplementedError(
                        "Haven't implemented Lie derivative along beta"
                        + " for rank 2 spacetime tensor")
            else:
                raise ValueError(
                    f"indexing must be '' or contain 's_' or 'st_'")
            
            # Check shape of f
            dim = {'s':3, 'st':4}
            for i in range(rank):
                if np.shape(f)[i] != dim[s_or_st]:
                    raise ValueError(
                        f"In Lie derivative along beta, you say f is"
                        + f" {indexing}, so rank {rank} with dim"
                        + f" {dim[s_or_st]}, but f is of shape {np.shape(f)}")
        
        if rank == 0:
            Lie = np.einsum('a..., a... -> ...', 
                            self["betaup3"], self.fd.d3_scalar(f))
        elif rank == 1:
            betaupdf = np.einsum('i..., ij... -> j...', 
                                  self["betaup3"], self.fd.d3_rank1tensor(f))
            dbetaup = self.fd.d3_rank1tensor(self["betaup3"])
            if indexing == 's_u':
                Lie = betaupdf - np.einsum('ij..., i... -> j...', dbetaup, f)
            elif indexing == 'st_u':
                Lie_t = (betaupdf[0])  # Bcs beta^t = 0
                Lie_s = (betaupdf[1:] - self["dtbetaup3"] * f[0]
                          - np.einsum('ij..., i... -> j...', dbetaup, f[1:]))
                Lie = np.array([Lie_t, Lie_s[0], Lie_s[1], Lie_s[2]])
            elif indexing == 's_d':
                Lie = betaupdf + np.einsum('ij..., j... -> i...', dbetaup, f)
            else: # 'st_d'
                Lie_t = (betaupdf[0] + np.einsum(
                    'j..., j... -> ...', self["dtbetaup3"], f[1:]))
                Lie_s = (betaupdf[1:] + np.einsum(
                    'ij..., j... -> i...', dbetaup, f[1:]))
                Lie = np.array([Lie_t, Lie_s[0], Lie_s[1], Lie_s[2]])
        else:
            betaupdf = np.einsum('s..., sjk... -> jk...', 
                             self["betaup3"], self.fd.d3_rank2tensor(f))
            dbetaup = self.fd.d3_rank1tensor(self["betaup3"])
            if indexing == 's_uu':
                Lie = (betaupdf 
                       - np.einsum('sj..., sk... -> jk...', dbetaup, f)
                       - np.einsum('sk..., js... -> jk...', dbetaup, f))
            #elif indexing == 'st_uu':
            #    Lie = (betaupdf - ... - ...)
            elif indexing == 's_ud':
                Lie = (betaupdf 
                       - np.einsum('sj..., sk... -> jk...', dbetaup, f)
                       + np.einsum('ks..., js... -> jk...', dbetaup, f))
            #elif indexing == 'st_ud':
            #    Lie = (betaupdf - ... + ...)
            elif indexing == 's_du':
                Lie = (betaupdf 
                       + np.einsum('js..., sk... -> jk...', dbetaup, f)
                       - np.einsum('sk..., js... -> jk...', dbetaup, f))
            #elif indexing == 'st_du':
            #    Lie = (betaupdf + ... - ...)
            elif indexing == 's_dd':
                Lie = (betaupdf 
                       + np.einsum('js..., sk... -> jk...', dbetaup, f)
                       + np.einsum('ks..., js... -> jk...', dbetaup, f))
            #else: # 'st_dd'
            #    Lie = (betaupdf + ... + ...)

        if weight != 0:
            divb = (self.fd.d3x(self["betaup3"][0])
                    + self.fd.d3y(self["betaup3"][1])
                    + self.fd.d3z(self["betaup3"][2]))
            Lie += weight * divb * f
        
        return Lie
    
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
        if ('betaup3' not in self.data) and ('betax' not in self.data):
            self.myprint(r"Using $\beta^i = 0$ shortcut for s_to_st")
            f00 = np.zeros(self.data_shape)
            f0k = np.zeros(
                (3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        else:
            f00 = np.einsum('i..., j..., ij... -> ...', 
                            self["betaup3"], self["betaup3"], fdown3)
            f0k = np.einsum('i..., ik... -> k...', self["betaup3"], fdown3)
        fdown4 = np.array([[f00, f0k[0], f0k[1], f0k[2]],
                           [f0k[0], fdown3[0, 0], fdown3[0, 1], fdown3[0, 2]],
                           [f0k[1], fdown3[1, 0], fdown3[1, 1], fdown3[1, 2]],
                           [f0k[2], fdown3[2, 0], fdown3[2, 1], fdown3[2, 2]]])
        return fdown4
    
    def vector_inner_product4(self, a, b):
        """Inner product of rank 1 4D tensors with indices up."""
        return np.einsum('a..., b..., ab... -> ...', 
                         a, b, self["gdown4"])
    
    def vector_inner_product3(self, a, b):
        """Inner product of rank 1 3D tensors with indices up."""
        return np.einsum('a..., b..., ab... -> ...', 
                         a, b, self["gammadown3"])

    def trace4(self, fdown4):
        """Compute trace of a 4D rank 2 tensor."""
        return np.einsum('jk..., jk... -> ...', 
                         self["gup4"], fdown4)

    def trace3(self, fdown3):
        """Compute trace of a 3D rank 2 tensor."""
        return np.einsum('jk..., jk... -> ...', 
                         self["gammaup3"], fdown3)
    
    def tracefree3(self, fdown3):
        """Compute tracefree part of a 3D rank 2 tensor."""
        return fdown3 - (1/3) * self["gammadown3"] * self.trace3(fdown3)
    
    def magnitude4(self, fdown):
        """Compute magnitude of a 4D rank 2 tensor."""
        return 0.5 * np.einsum(
            'ab..., ij..., ai..., bj... -> ...', 
            fdown, fdown,
            self["gup4"], self["gup4"])
    
    def magnitude3(self, fdown):
        """Compute magnitude of a 3D rank 2 tensor."""
        return 0.5 * np.einsum(
            'ab..., ij..., ai..., bj... -> ...', 
            fdown, fdown,
            self["gammaup3"], self["gammaup3"])
    
    def norm4(self, a): 
        """Compute norm of a 4D rank 1 tensor."""
        return np.sqrt(abs(self.vector_inner_product4(a, a)))
    
    def norm3(self, a): 
        """Compute norm of a 3D rank 1 tensor."""
        return np.sqrt(abs(self.vector_inner_product3(a, a)))
    
    def kronecker_delta4(self):
        """Compute Kronecker delta with 4 4D indices."""
        kronecker = np.zeros(
            (4, 4, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        for i in range(4):
            kronecker[i, i] = 1.0
        return kronecker
    
    def kronecker_delta3(self):
        """Compute Kronecker delta with 3 3D indices."""
        kronecker = np.zeros(
            (3, 3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        for i in range(3):
            kronecker[i, i] = 1.0
        return kronecker

    def levicivita_down4(self):
        """Compute Levi-Civita tensor with 4 4D indices down."""
        return (self.levicivita_symbol_down4() 
                * np.sqrt(-self["gdet"]))
    
    def levicivita_down3(self):
        """Compute Levi-Civita tensor with 3 3D indices down."""
        return (self.levicivita_symbol_down3() 
                * np.sqrt(self["gammadet"]))

    def levicivita_symbol_down4(self): 
        """Compute Levi-Civita symbol with 4 4D indices down."""
        LC = np.zeros(
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
                        LC[i0, i1, i2, i3, :, :, :] = float(top/bot)
        return LC

    def levicivita_symbol_down3(self):
        """Compute Levi-Civita symbol with 3 3D indices down."""
        LC = np.zeros(
            (3, 3, 3, self.param['Nx'], self.param['Ny'], self.param['Nz']))
        allindices = [1, 2, 3]
        for i1 in allindices:
            for i2 in np.delete(allindices, i1-1):
                for i3 in np.delete(allindices, [i1-1, i2-1]):
                    top = ((i2-i1) * (i3-i1) * (i3-i2))
                    bot = (abs(i2-i1) * abs(i3-i1) * abs(i3-i2))
                    LC[i1-1, i2-1, i3-1, :, :, :] = float(top/bot)
        return LC

# Update __doc__ of the functions listed in descriptions
for func_name, doc in descriptions.items():
    func = getattr(AurelCore, func_name, None)
    if func is not None:
        func.__doc__ = doc