aurel.core
##########

.. automodule:: aurel.core
   :noindex:

.. _descriptions_list:

descriptions
************

.. _assumed_quantities:

Assumed quantities
==================

If not defined, vacuum Minkowski is assumed for the definition of the following quantities:

$\Lambda = 0$, the Cosmological constant, to change this do **AurelCore.Lambda = ...** before running calculations

**alpha**: $\alpha = 1$, the lapse, to change this do **AurelCore.data["alpha"] = ...** before running calculations

**dtalpha**: $\partial_t \alpha = 0$, the time derivative of the lapse

**betax, betay, betaz**: $\beta^i = 0$, the shift vector with spatial indices up

**dtbetax, dtbetay, dtbetaz**: $\partial_t \beta^i = 0$, the time derivative of the shift vector with spatial indices up

**gxx, gxy, gxz, gyy, gyz, gzz**: $g_{ij} = \delta_{ij}$, the spatial components of the spacetime metric with indices down

**kxx, kxy, kxz, kyy, kyz, kzz**: $K_{ij} = 0$, the spatial components of the extrinsic curvature with indices down

**rho0**: $\rho_0 = 0$, the rest-mass energy density

**press**: $p = 0$, the fluid pressure

**eps**: $\epsilon = 0$, the fluid specific internal energy

**w_lorentz**: $W = 1$, the Lorentz factor

**velup3**: $v^i = 0$, the Eulerian fluid three velocity with spatial indices up

Metric quantities
=================

Spatial metric
--------------

**gxx**: $g_{xx}$ Metric with xx indices down. I assume $g_{xx}=1$, if not then please define AurelCore.data['gxx'] = ... 

**gxy**: $g_{xy}$ Metric with xy indices down. I assume $g_{xy}=0$, if not then please define AurelCore.data['gxy'] = ... 

**gxz**: $g_{xz}$ Metric with xz indices down. I assume $g_{xz}=0$, if not then please define AurelCore.data['gxz'] = ... 

**gyy**: $g_{yy}$ Metric with yy indices down. I assume $g_{yy}=1$, if not then please define AurelCore.data['gyy'] = ... 

**gyz**: $g_{yz}$ Metric with yz indices down. I assume $g_{yz}=0$, if not then please define AurelCore.data['gyz'] = ... 

**gzz**: $g_{zz}$ Metric with zz indices down. I assume $g_{zz}=1$, if not then please define AurelCore.data['gzz'] = ... 

**gammadown3**: $\gamma_{ij}$ Spatial metric with spatial indices down

**gammaup3**: $\gamma^{ij}$ Spatial metric with spatial indices up

**dtgammaup3**: $\partial_t \gamma^{ij}$ Coordinate time derivative of spatial metric with spatial indices up

**gammadet**: $\gamma$ Determinant of spatial metric

**gammadown4**: $\gamma_{\mu\nu}$ Spatial metric with spacetime indices down

**gammaup4**: $\gamma^{\mu\nu}$ Spatial metric with spacetime indices up

Extrinsic curvature
-------------------

**kxx**: $K_{xx}$ Extrinsic curvature with xx indices down. I assume $K_{xx}=0$, if not then please define AurelCore.data['kxx'] = ... 

**kxy**: $K_{xy}$ Extrinsic curvature with xy indices down. I assume $K_{xy}=0$, if not then please define AurelCore.data['kxy'] = ... 

**kxz**: $K_{xz}$ Extrinsic curvature with xz indices down. I assume $K_{xz}=0$, if not then please define AurelCore.data['kxz'] = ... 

**kyy**: $K_{yy}$ Extrinsic curvature with yy indices down. I assume $K_{yy}=0$, if not then please define AurelCore.data['kyy'] = ... 

**kyz**: $K_{yz}$ Extrinsic curvature with yz indices down. I assume $K_{yz}=0$, if not then please define AurelCore.data['kyz'] = ... 

**kzz**: $K_{zz}$ Extrinsic curvature with zz indices down. I assume $K_{zz}=0$, if not then please define AurelCore.data['kzz'] = ... 

**Kdown3**: $K_{ij}$ Extrinsic curvature with spatial indices down

**Kup3**: $K^{ij}$ Extrinsic curvature with spatial indices up

**Ktrace**: $K = \gamma^{ij}K_{ij}$ Trace of extrinsic curvature

**Adown3**: $A_{ij}$ Traceless part of the extrinsic curvature with spatial indices down

**Aup3**: $A^{ij}$ Traceless part of the extrinsic curvature with spatial indices up

**A2**: $A^2$ Magnitude of traceless part of the extrinsic curvature

Lapse
-----

**alpha**: $\alpha$ Lapse. I assume $\alpha=1$, if not then please define AurelCore.data['alpha'] = ... 

**dtalpha**: $\partial_t \alpha$ Coordinate time derivative of the lapse. I assume $\partial_t \alpha=0$, if not then please define AurelCore.data['dtalpha'] = ... 

Shift
-----

**betax**: $\beta^{x}$ x component of the shift vector with indices up. I assume $\beta^{x}=0$, if not then please define AurelCore.data['betax'] = ... 

**betay**: $\beta^{y}$ y component of the shift vector with indices up. I assume $\beta^{y}=0$, if not then please define AurelCore.data['betay'] = ... 

**betaz**: $\beta^{z}$ z component of the shift vector with indices up. I assume $\beta^{z}=0$, if not then please define AurelCore.data['betaz'] = ... 

**betaup3**: $\beta^{i}$ Shift vector with spatial indices up

**dtbetax**: $\partial_t\beta^{x}$ Coordinate time derivative of the x component of the shift vector with indices up. I assume $\partial_t\beta^{x}=0$, if not then please define AurelCore.data['dtbetax'] = ... 

**dtbetay**: $\partial_t\beta^{y}$ Coordinate time derivative of the y component of the shift vector with indices up. I assume $\partial_t\beta^{y}=0$, if not then please define AurelCore.data['dtbetay'] = ... 

**dtbetaz**: $\partial_t\beta^{z}$ Coordinate time derivative of the z component of the shift vector with indices up. I assume $\partial_t\beta^{z}=0$, if not then please define AurelCore.data['dtbetaz'] = ... 

**dtbetaup3**: $\partial_t\beta^{i}$ Coordinate time derivative of the shift vector with spatial indices up

**betadown3**: $\beta_{i}$ Shift vector with spatial indices down

**betamag**: $\beta_{i}\beta^{i}$ Magnitude of shift vector

Timelike normal vector
----------------------

**nup4**: $n^{\mu}$ Timelike vector normal to the spatial metric with spacetime indices up

**ndown4**: $n_{\mu}$ Timelike vector normal to the spatial metric with spacetime indices down

Spacetime metric
----------------

**gdown4**: $g_{\mu\nu}$ Spacetime metric with spacetime indices down

**gup4**: $g^{\mu\nu}$ Spacetime metric with spacetime indices up

**gdet**: $g$ Determinant of spacetime metric

Matter quantities
=================

Eulerian observer follows $n^\mu$

Lagrangian observer follows $u^\mu$

Lagrangian matter variables
---------------------------

**rho0**: $\rho_0$ Rest mass energy density. I assume $\rho_0=0$, if not then please define AurelCore.data['rho0'] = ... 

**press**: $p$ Pressure. I assume $p=0$, if not then please define AurelCore.data['press'] = ... 

**eps**: $\epsilon$ Specific internal energy. I assume $\epsilon=0$, if not then please define AurelCore.data['eps'] = ... 

**rho**: $\rho$ Energy density

**enthalpy**: $h$ Specific enthalpy of the fluid

Fluid velocity
--------------

**w_lorentz**: $W$ Lorentz factor. I assume $W=1$, if not then please define AurelCore.data['w_lorentz'] = ... 

**velx**: $v^x$ x component of Eulerian fluid three velocity with indice up. I assume $v^x=0$, if not then please define AurelCore.data['velx'] = ... 

**vely**: $v^y$ y component of Eulerian fluid three velocity with indice up. I assume $v^y=0$, if not then please define AurelCore.data['vely'] = ... 

**velz**: $v^z$ z component of Eulerian fluid three velocity with indice up. I assume $v^z=0$, if not then please define AurelCore.data['velz'] = ... 

**velup3**: $v^i$ Eulerian fluid three velocity with spatial indices up.

**uup0**: $u^t$ Lagrangian fluid four velocity with time indice up

**uup3**: $u^i$ Lagrangian fluid four velocity with spatial indices up

**uup4**: $u^\mu$ Lagrangian fluid four velocity with spacetime indices up

**udown3**: $u_\mu$ Lagrangian fluid four velocity with spatial indices down

**udown4**: $u_\mu$ Lagrangian fluid four velocity with spacetime indices down

**hdown4**: $h_{\mu\nu}$ Spatial metric orthonomal to fluid flow with spacetime indices down

**hmixed4**: ${h^{\mu}}_{\nu}$ Spatial metric orthonomal to fluid flow with mixed spacetime indices

**hup4**: $h^{\mu\nu}$ Spatial metric orthonomal to fluid flow with spacetime indices up

Energy-stress tensor
--------------------

**Tdown4**: $T_{\mu\nu}$ Energy-stress tensor with spacetime indices down

**Tup4**: $T^{\mu\nu}$ Energy-stress tensor with spacetime indices up

**Ttrace**: $T$ Trace of the energy-stress tensor

Eulerian matter variables
-------------------------

**rho_n**: $\rho^{\{n\}}$ Energy density in the $n^\mu$ frame

**fluxup3_n**: $S^{\{n\}i}$ Energy flux (or momentum density) in the $n^\mu$ frame with spatial indices up

**fluxdown3_n**: $S^{\{n\}}_{i}$ Energy flux (or momentum density) in the $n^\mu$ frame with spatial indices down

**angmomup3_n**: $J^{\{n\}i}$ Angular momentum density in the $n^\mu$ frame with spatial indices up

**angmomdown3_n**: $J^{\{n\}}_{i}$ Angular momentum density in the $n^\mu$ frame with spatial indices down

**Stressup3_n**: $S^{\{n\}ij}$ Stress tensor in the $n^\mu$ frame with spatial indices up

**Stressdown3_n**: $S^{\{n\}}_{ij}$ Stress tensor in the $n^\mu$ frame with spatial indices down

**Stresstrace_n**: $S^{\{n\}}$ Trace of Stress tensor in the $n^\mu$ frame

**press_n**: $p^{\{n\}}$ Pressure in the $n^\mu$ frame

**anisotropic_press_down3_n**: $\pi^{\{n\}_{ij}}$ Anisotropic pressure in the $n^\mu$ frame with spatial indices down

**rho_n_fromHam**: $\rho^{\{n\}}$ Energy density in the $n^\mu$ frame computed from the Hamiltonian constraint

**fluxup3_n_fromMom**: $S^{\{n\}i}$ Energy flux (or momentum density) in the $n^\mu$ frame with spatial indices up computed from the Momentum constraint

Conserved variables
-------------------

**conserved_D**: $D$ Conserved mass-energy density in Wilson's formalism

**conserved_E**: $E$ Conserved internal energy density in Wilson's formalism

**conserved_Sdown4**: $S_{\mu}$ Conserved energy flux (or momentum density) in Wilson's formalism with spacetime indices down

**conserved_Sdown3**: $S_{i}$ Conserved energy flux (or momentum density) in Wilson's formalism with spatial indices down

**conserved_Sup4**: $S^{\mu}$ Conserved energy flux (or momentum density) in Wilson's formalism with spacetime indices up

**conserved_Sup3**: $S^{i}$ Conserved energy flux (or momentum density) in Wilson's formalism with spatial indices up

**dtconserved**: $\partial_t D, \; \partial_t E, \partial_t S_{i}$ List of coordinate time derivatives of conserved rest mass-energy density, internal energy density and energy flux (or momentum density) with spatial indices down in Wilson's formalism

Kinematic variables
-------------------

**st_covd_udown4**: $\nabla_{\mu} u_{\nu}$ Spacetime covariant derivative of Lagrangian fluid four velocity with spacetime indices down

**accelerationdown4**: $a_{\mu}$ Acceleration of the fluid with spacetime indices down

**accelerationup4**: $a^{\mu}$ Acceleration of the fluid with spacetime indices up

**s_covd_udown4**: $\mathcal{D}^{\{u\}}_{\mu} u_{\nu}$ Spatial covariant derivative of Lagrangian fluid four velocity with spacetime indices down, with respect to spatial hypersurfaces orthonormal to the fluid flow

**thetadown4**: $\Theta_{\mu\nu}$ Fluid expansion tensor with spacetime indices down

**theta**: $\Theta$ Fluid expansion scalar

**sheardown4**: $\sigma_{\mu\nu}$ Fluid shear tensor with spacetime indices down

**shear2**: $\sigma^2$ Magnitude of fluid shear

**omegadown4**: $\omega_{\mu\nu}$ Fluid vorticity tensor with spacetime indices down

**omega2**: $\omega^2$ Magnitude of fluid vorticity

Curvature quantities
====================

Spatial curvature
-----------------

**s_RicciS_u**: ${}^{(3)}R^{\{u\}}$ Ricci scalar of the spatial metric orthonormal to fluid flow

**s_Gamma_udd3**: ${}^{(3)}{\Gamma^{k}}_{ij}$ Christoffel symbols of spatial metric with mixed spatial indices

**s_Riemann_uddd3**: ${}^{(3)}{R^{i}}_{jkl}$ Riemann tensor of spatial metric with mixed spatial indices

**s_Riemann_down3**: ${}^{(3)}R_{ijkl}$ Riemann tensor of spatial metric with all spatial indices down

**s_Ricci_down3**: ${}^{(3)}R_{ij}$ Ricci tensor of spatial metric with spatial indices down

**s_RicciS**: ${}^{(3)}R$ Ricci scalar of spatial metric

Spacetime curvature
-------------------

**st_Gamma_udd4**: ${}^{(4)}{\Gamma^{\alpha}}_{\mu\nu}$ Christoffel symbols of spacetime metric with mixed spacetime indices

**st_Riemann_uddd4**: ${}^{(4)}{R^{\alpha}}_{\beta\mu\nu}$ Riemann tensor of spacetime metric with mixed spacetime indices

**st_Riemann_down4**: ${}^{(4)}R_{\alpha\beta\mu\nu}$ Riemann tensor of spacetime metric with spacetime indices down

**st_Riemann_uudd4**: ${}^{(4)}{R^{\alpha\beta}}_{\mu\nu}$ Riemann tensor of spacetime metric with mixed spacetime indices

**st_Ricci_down4**: ${}^{(4)}R_{\alpha\beta}$ Ricci tensor of spacetime metric with spacetime indices down

**st_Ricci_down3**: ${}^{(4)}R_{ij}$ Ricci tensor of spacetime metric with spatial indices down

**st_RicciS**: ${}^{(4)}R$ Ricci scalar of spacetime metric

**Einsteindown4**: $G_{\alpha\beta}$ Einstein tensor with spacetime indices down

**Kretschmann**: $K={R^{\alpha\beta}}_{\mu\nu}{R_{\alpha\beta}}^{\mu\nu}$ Kretschmann scalar

Weyl decomposition
------------------

**st_Weyl_down4**: $C_{\alpha\beta\mu\nu}$ Weyl tensor of spacetime metric with spacetime indices down

**Weyl_Psi**: $\Psi_0, \; \Psi_1, \; \Psi_2, \; \Psi_3, \; \Psi_4$ List of Weyl scalars for an null vector base defined with AurelCore.tetrad_to_use

**Psi4_lm**: $\Psi_4^{l,m}$ Dictionary of spin weighted spherical harmonic decomposition of the 4th Weyl scalar, with AurelCore.Psi4_lm_radius and AurelCore.Psi4_lm_lmax. ``spinsfast`` is used for the decomposition.

**Weyl_invariants**: $I, \; J, \; L, \; K, \; N$ Dictionary of Weyl invariants

**eweyl_u_down4**: $E^{\{u\}}_{\alpha\beta}$ Electric part of the Weyl tensor on the hypersurface orthogonal to $u^{\mu}$ with spacetime indices down

**eweyl_n_down3**: $E^{\{n\}}_{ij}$ Electric part of the Weyl tensor on the hypersurface orthogonal to $n^{\mu}$ with spatial indices down

**bweyl_u_down4**: $B^{\{u\}}_{\alpha\beta}$ Magnetic part of the Weyl tensor on the hypersurface orthogonal to $u^{\mu}$ with spacetime indices down

**bweyl_n_down3**: $B^{\{n\}}_{ij}$ Magnetic part of the Weyl tensor on the hypersurface orthogonal to $n^{\mu}$ with spatial indices down

Null ray expansion
==================

**null_ray_exp**: $\Theta_{out}, \; \Theta_{in}$ List of expansion of null rays radially going out and in respectively

Constraints
===========

**Hamiltonian**: $\mathcal{H}$ Hamilonian constraint

**Hamiltonian_Escale**: [$\mathcal{H}$] Hamilonian constraint energy scale

**Momentumup3**: $\mathcal{M}^i$ Momentum constraint with spatial indices up

**Momentum_Escale**: [$\mathcal{M}$] Momentum constraint energy scale

AurelCore
*********

.. autoclass:: aurel.core.AurelCore
   :show-inheritance:
   :members:
