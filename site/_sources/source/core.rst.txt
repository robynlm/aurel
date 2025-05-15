aurel.core
##########

.. automodule:: aurel.core
   :noindex:

.. _descriptions_list:

descriptions
************

.. _required_quantities:

Required quantities
===================

**gxx**: $g_{xx}$ Metric with xx indices down (need to input)

**gxy**: $g_{xy}$ Metric with xy indices down (need to input)

**gxz**: $g_{xz}$ Metric with xz indices down (need to input)

**gyy**: $g_{yy}$ Metric with yy indices down (need to input)

**gyz**: $g_{yz}$ Metric with yz indices down (need to input)

**gzz**: $g_{zz}$ Metric with zz indices down (need to input)

**kxx**: $K_{xx}$ Extrinsic curvature with xx indices down (need to input)

**kxy**: $K_{xy}$ Extrinsic curvature with xy indices down (need to input)

**kxz**: $K_{xz}$ Extrinsic curvature with xz indices down (need to input)

**kyy**: $K_{yy}$ Extrinsic curvature with yy indices down (need to input)

**kyz**: $K_{yz}$ Extrinsic curvature with yz indices down (need to input)

**kzz**: $K_{zz}$ Extrinsic curvature with zz indices down (need to input)

**rho0**: $\rho_0$ Rest mass energy density (need to input)

.. _assumed_quantities:

Assumed quantities
==================

$\Lambda = 0$, the Cosmological constant, to change this do **AurelCore.Lambda = ...** before running calculations

**alpha**: $\alpha = 1$, the lapse, to change this do **AurelCore.data["alpha"] = ...** before running calculations

**dtalpha**: $\partial_t \alpha = 0$, the time derivative of the lapse

**betaup3**: $\beta^i = 0$, the shift vector with spatial indices up

**dtbetaup3**: $\partial_t \beta^i = 0$, the time derivative of the shift vector with spatial indices up

**press**: $p = 0$, the fluid pressure

**eps**: $\epsilon = 0$, the fluid specific internal energy

**w_lorentz**: $W = 1$, the Lorentz factor

**velup3**: $v^i = 0$, the Eulerian fluid three velocity with spatial indices up

Metric quantities
=================

Spatial metric
--------------

**gammadown3**: $\gamma_{ij}$ Spatial metric with spatial indices down

**gammaup3**: $\gamma^{ij}$ Spatial metric with spatial indices up

**dtgammaup3**: $\partial_t \gamma^{ij}$ Coordinate time derivative of spatial metric with spatial indices up

**gammadet**: $\gamma$ Determinant of spatial metric

**gammadown4**: $\gamma_{\mu\nu}$ Spatial metric with spacetime indices down

**gammaup4**: $\gamma^{\mu\nu}$ Spatial metric with spacetime indices up

Extrinsic curvature
-------------------

**Kdown3**: $K_{ij}$ Extrinsic curvature with spatial indices down

**Kup3**: $K^{ij}$ Extrinsic curvature with spatial indices up

**Ktrace**: $K = \gamma^{ij}K_{ij}$ Trace of extrinsic curvature

**Adown3**: $A_{ij}$ Traceless part of the extrinsic curvature with spatial indices down

**Aup3**: $A^{ij}$ Traceless part of the extrinsic curvature with spatial indices up

**A2**: $A^2$ Magnitude of traceless part of the extrinsic curvature

Lapse
-----

**alpha**: $\alpha$ Lapse (need to input or I assume =1)

**dtalpha**: $\partial_t \alpha$ Coordinate time derivative of the lapse (need to input or I assume =0)

Shift
-----

**betaup3**: $\beta^{i}$ Shift vector with spatial indices up (need to input or I assume =0)

**dtbetaup3**: $\partial_t\beta^{i}$ Coordinate time derivative of the shift vector with spatial indices up (need to input or I assume =0)

**betadown3**: $\beta_{i}$ Shift vector with spatial indices down

**betamag**: $\beta_{i}\beta^{i}$ Magnitude of shift vector

Timeline normal vector
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

**press**: $p$ Pressure (need to input or I assume =0)

**eps**: $\epsilon$ Specific internal energy (need to input or I assume =0)

**rho**: $\rho$ Energy density

**rho_fromHam**: $\rho$ Energy density computed from the Hamiltonian constraint

**enthalpy**: $h$ Specific enthalpy of the fluid

Fluid velocity
--------------

**w_lorentz**: $W$ Lorentz factor (need to input or I assume =1)

**velup3**: $v^i$ Eulerian fluid three velocity with spatial indices up (need to input or I assume =0)

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

**thetadown4**: $\Theta_{\mu\nu}$ Fluid expansion tensor with spacetime indices down

**theta**: $\Theta$ Fluid expansion scalar

**sheardown4**: $\sigma_{\mu\nu}$ Fluid shear tensor with spacetime indices down

**shear2**: $\sigma^2$ Magnitude of fluid shear

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

**Kretschmann**: $K={R^{\alpha\beta}}_{\mu\nu}{R_{\alpha\beta}}^{\mu\nu}$ Kretschmann scalar

Weyl decomposition
------------------

**st_Weyl_down4**: $C_{\alpha\beta\mu\nu}$ Weyl tensor of spacetime metric with spacetime indices down

**Weyl_Psi**: $\Psi_0, \; \Psi_1, \; \Psi_2, \; \Psi_3, \; \Psi_4$ List of Weyl scalars for an null vector base defined with AurelCore.tetrad_to_use

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
