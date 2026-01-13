aurel.core
##########

.. automodule:: aurel.core

.. raw:: html

   <div style='display: none;' aria-hidden='true'>

.. autoclass:: aurel.core.AurelCore
   :members: alpha, dtalpha, DDalpha, betax, betay, betaz, betaup3, dtbetax, dtbetay, dtbetaz, dtbetaup3, betadown3, betamag, nup4, ndown4, gxx, gxy, gxz, gyy, gyz, gzz, gammadown3, gammaup3, dtgammaup3, gammadet, gammadown4, gammaup4, gtt, gtx, gty, gtz, gdown4, gup4, gdet, psi_bssnok, phi_bssnok, dtphi_bssnok, gammadown3_bssnok, gammaup3_bssnok, dtgammadown3_bssnok, kxx, kxy, kxz, kyy, kyz, kzz, Kdown3, Kup3, Ktrace, dtKtrace, Adown3, Aup3, A2, Adown3_bssnok, Aup3_bssnok, A2_bssnok, dtAdown3_bssnok, dttau, rho0, press, eps, rho, enthalpy, w_lorentz, velx, vely, velz, velup3, velup4, veldown3, veldown4, uup0, uup3, uup4, udown3, udown4, hdown4, hdet, hmixed4, hup4, Tdown4, Tup4, Ttrace, rho_n, fluxup3_n, fluxdown3_n, angmomup3_n, angmomdown3_n, Stressup3_n, Stressdown3_n, Stresstrace_n, press_n, anisotropic_press_down3_n, rho_n_fromHam, fluxup3_n_fromMom, conserved_D, conserved_E, conserved_Sdown4, conserved_Sdown3, conserved_Sup4, conserved_Sup3, dtconserved, st_covd_udown4, accelerationdown4, accelerationup4, s_covd_udown4, thetadown4, theta, sheardown4, shear2, omegadown4, omega2, s_RicciS_u, s_Gamma_udd3, s_Riemann_uddd3, s_Riemann_down3, s_Ricci_down3, s_RicciS, st_Gamma_udd4, st_Riemann_uddd4, st_Riemann_down4, st_Riemann_uudd4, st_Ricci_down4, st_Ricci_down3, st_RicciS, Einsteindown4, Kretschmann, s_Gamma_udd3_bssnok, s_Gamma_bssnok, dts_Gamma_bssnok, s_Ricci_down3_bssnok, s_RicciS_bssnok, s_Ricci_down3_phi, st_Weyl_down4, Weyl_Psi, Psi4_lm, Weyl_invariants, eweyl_u_down4, eweyl_n_down3, bweyl_u_down4, bweyl_n_down3, null_ray_exp_out, null_ray_exp_in, Hamiltonian, Hamiltonian_Escale, Hamiltonian_norm, Momentumx, Momentumy, Momentumz, Momentumdownx, Momentumdowny, Momentumdownz, Momentumup3, Momentumdown3, Momentum_Escale, Momentumx_norm, Momentumy_norm, Momentumz_norm, Momentumdownx_norm, Momentumdowny_norm, Momentumdownz_norm
   :noindex:

.. raw:: html

   </div>

.. _descriptions_list:

descriptions
************

.. _assumed_quantities:

Assumed quantities
==================

If not defined, vacuum Minkowski is assumed for the definition of the following quantities:

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

**velx, vely, velz**: $v^i = 0$, the Eulerian fluid three velocity with spatial indices up

Metric quantities
=================

Lapse
-----

.. _aurel.core.AurelCore.alpha:

`alpha <../_modules/aurel/core.html#AurelCore.alpha>`_: $\alpha$ Lapse. I assume $\alpha=1$, if not then please define AurelCore.data['alpha'] = ... 

.. _aurel.core.AurelCore.dtalpha:

`dtalpha <../_modules/aurel/core.html#AurelCore.dtalpha>`_: $\partial_t \alpha$ Coordinate time derivative of the lapse. I assume $\partial_t \alpha=0$, if not then please define AurelCore.data['dtalpha'] = ... 

.. _aurel.core.AurelCore.DDalpha:

`DDalpha <../_modules/aurel/core.html#AurelCore.DDalpha>`_: D_iD_j\alpha$ Spatial covariant second derivative of the lapse with spatial indices down

Shift
-----

.. _aurel.core.AurelCore.betax:

`betax <../_modules/aurel/core.html#AurelCore.betax>`_: $\beta^{x}$ x component of the shift vector with indices up. I assume $\beta^{x}=0$, if not then please define AurelCore.data['betax'] = ... 

.. _aurel.core.AurelCore.betay:

`betay <../_modules/aurel/core.html#AurelCore.betay>`_: $\beta^{y}$ y component of the shift vector with indices up. I assume $\beta^{y}=0$, if not then please define AurelCore.data['betay'] = ... 

.. _aurel.core.AurelCore.betaz:

`betaz <../_modules/aurel/core.html#AurelCore.betaz>`_: $\beta^{z}$ z component of the shift vector with indices up. I assume $\beta^{z}=0$, if not then please define AurelCore.data['betaz'] = ... 

.. _aurel.core.AurelCore.betaup3:

`betaup3 <../_modules/aurel/core.html#AurelCore.betaup3>`_: $\beta^{i}$ Shift vector with spatial indices up

.. _aurel.core.AurelCore.dtbetax:

`dtbetax <../_modules/aurel/core.html#AurelCore.dtbetax>`_: $\partial_t\beta^{x}$ Coordinate time derivative of the x component of the shift vector with indices up. I assume $\partial_t\beta^{x}=0$, if not then please define AurelCore.data['dtbetax'] = ... 

.. _aurel.core.AurelCore.dtbetay:

`dtbetay <../_modules/aurel/core.html#AurelCore.dtbetay>`_: $\partial_t\beta^{y}$ Coordinate time derivative of the y component of the shift vector with indices up. I assume $\partial_t\beta^{y}=0$, if not then please define AurelCore.data['dtbetay'] = ... 

.. _aurel.core.AurelCore.dtbetaz:

`dtbetaz <../_modules/aurel/core.html#AurelCore.dtbetaz>`_: $\partial_t\beta^{z}$ Coordinate time derivative of the z component of the shift vector with indices up. I assume $\partial_t\beta^{z}=0$, if not then please define AurelCore.data['dtbetaz'] = ... 

.. _aurel.core.AurelCore.dtbetaup3:

`dtbetaup3 <../_modules/aurel/core.html#AurelCore.dtbetaup3>`_: $\partial_t\beta^{i}$ Coordinate time derivative of the shift vector with spatial indices up

.. _aurel.core.AurelCore.betadown3:

`betadown3 <../_modules/aurel/core.html#AurelCore.betadown3>`_: $\beta_{i}$ Shift vector with spatial indices down

.. _aurel.core.AurelCore.betamag:

`betamag <../_modules/aurel/core.html#AurelCore.betamag>`_: $\beta_{i}\beta^{i}$ Magnitude of shift vector

Timelike normal vector
----------------------

.. _aurel.core.AurelCore.nup4:

`nup4 <../_modules/aurel/core.html#AurelCore.nup4>`_: $n^{\mu}$ Timelike vector normal to the spatial metric with spacetime indices up

.. _aurel.core.AurelCore.ndown4:

`ndown4 <../_modules/aurel/core.html#AurelCore.ndown4>`_: $n_{\mu}$ Timelike vector normal to the spatial metric with spacetime indices down

Spatial metric
--------------

.. _aurel.core.AurelCore.gxx:

`gxx <../_modules/aurel/core.html#AurelCore.gxx>`_: $g_{xx}$ Metric with xx indices down. I assume $g_{xx}=1$, if not then please define AurelCore.data['gxx'] = ... 

.. _aurel.core.AurelCore.gxy:

`gxy <../_modules/aurel/core.html#AurelCore.gxy>`_: $g_{xy}$ Metric with xy indices down. I assume $g_{xy}=0$, if not then please define AurelCore.data['gxy'] = ... 

.. _aurel.core.AurelCore.gxz:

`gxz <../_modules/aurel/core.html#AurelCore.gxz>`_: $g_{xz}$ Metric with xz indices down. I assume $g_{xz}=0$, if not then please define AurelCore.data['gxz'] = ... 

.. _aurel.core.AurelCore.gyy:

`gyy <../_modules/aurel/core.html#AurelCore.gyy>`_: $g_{yy}$ Metric with yy indices down. I assume $g_{yy}=1$, if not then please define AurelCore.data['gyy'] = ... 

.. _aurel.core.AurelCore.gyz:

`gyz <../_modules/aurel/core.html#AurelCore.gyz>`_: $g_{yz}$ Metric with yz indices down. I assume $g_{yz}=0$, if not then please define AurelCore.data['gyz'] = ... 

.. _aurel.core.AurelCore.gzz:

`gzz <../_modules/aurel/core.html#AurelCore.gzz>`_: $g_{zz}$ Metric with zz indices down. I assume $g_{zz}=1$, if not then please define AurelCore.data['gzz'] = ... 

.. _aurel.core.AurelCore.gammadown3:

`gammadown3 <../_modules/aurel/core.html#AurelCore.gammadown3>`_: $\gamma_{ij}$ Spatial metric with spatial indices down

.. _aurel.core.AurelCore.gammaup3:

`gammaup3 <../_modules/aurel/core.html#AurelCore.gammaup3>`_: $\gamma^{ij}$ Spatial metric with spatial indices up

.. _aurel.core.AurelCore.dtgammaup3:

`dtgammaup3 <../_modules/aurel/core.html#AurelCore.dtgammaup3>`_: $\partial_t \gamma^{ij}$ Coordinate time derivative of spatial metric with spatial indices up

.. _aurel.core.AurelCore.gammadet:

`gammadet <../_modules/aurel/core.html#AurelCore.gammadet>`_: $\gamma$ Determinant of spatial metric

.. _aurel.core.AurelCore.gammadown4:

`gammadown4 <../_modules/aurel/core.html#AurelCore.gammadown4>`_: $\gamma_{\mu\nu}$ Spatial metric with spacetime indices down

.. _aurel.core.AurelCore.gammaup4:

`gammaup4 <../_modules/aurel/core.html#AurelCore.gammaup4>`_: $\gamma^{\mu\nu}$ Spatial metric with spacetime indices up

Spacetime metric
----------------

.. _aurel.core.AurelCore.gtt:

`gtt <../_modules/aurel/core.html#AurelCore.gtt>`_: $g_{tt}$ Metric with tt indices down.

.. _aurel.core.AurelCore.gtx:

`gtx <../_modules/aurel/core.html#AurelCore.gtx>`_: $g_{tx}$ Metric with tx indices down.

.. _aurel.core.AurelCore.gty:

`gty <../_modules/aurel/core.html#AurelCore.gty>`_: $g_{ty}$ Metric with ty indices down.

.. _aurel.core.AurelCore.gtz:

`gtz <../_modules/aurel/core.html#AurelCore.gtz>`_: $g_{tz}$ Metric with tz indices down.

.. _aurel.core.AurelCore.gdown4:

`gdown4 <../_modules/aurel/core.html#AurelCore.gdown4>`_: $g_{\mu\nu}$ Spacetime metric with spacetime indices down

.. _aurel.core.AurelCore.gup4:

`gup4 <../_modules/aurel/core.html#AurelCore.gup4>`_: $g^{\mu\nu}$ Spacetime metric with spacetime indices up

.. _aurel.core.AurelCore.gdet:

`gdet <../_modules/aurel/core.html#AurelCore.gdet>`_: $g$ Determinant of spacetime metric

BSSNOK metric
-------------

.. _aurel.core.AurelCore.psi_bssnok:

`psi_bssnok <../_modules/aurel/core.html#AurelCore.psi_bssnok>`_: $\psi = \gamma^{1/12}$ BSSNOK conformal factor

.. _aurel.core.AurelCore.phi_bssnok:

`phi_bssnok <../_modules/aurel/core.html#AurelCore.phi_bssnok>`_: $\phi = \ln(\gamma^{1/12})$ BSSNOK conformal factor

.. _aurel.core.AurelCore.dtphi_bssnok:

`dtphi_bssnok <../_modules/aurel/core.html#AurelCore.dtphi_bssnok>`_: $\partial_t \phi$ Coordinate time derivative of BSSNOK $\phi$ conformal factor

.. _aurel.core.AurelCore.gammadown3_bssnok:

`gammadown3_bssnok <../_modules/aurel/core.html#AurelCore.gammadown3_bssnok>`_: $\tilde{\gamma}_{ij}$ Conformal spatial metric with spatial indices down

.. _aurel.core.AurelCore.gammaup3_bssnok:

`gammaup3_bssnok <../_modules/aurel/core.html#AurelCore.gammaup3_bssnok>`_: $\tilde{\gamma}^{ij}$ Conformal spatial metric with spatial indices up

.. _aurel.core.AurelCore.dtgammadown3_bssnok:

`dtgammadown3_bssnok <../_modules/aurel/core.html#AurelCore.dtgammadown3_bssnok>`_: $\partial_t \tilde{\gamma}_{ij}$ Coordinate time derivative of conformal spatial metric with spatial indices down

Extrinsic curvature
-------------------

.. _aurel.core.AurelCore.kxx:

`kxx <../_modules/aurel/core.html#AurelCore.kxx>`_: $K_{xx}$ Extrinsic curvature with xx indices down. I assume $K_{xx}=0$, if not then please define AurelCore.data['kxx'] = ... 

.. _aurel.core.AurelCore.kxy:

`kxy <../_modules/aurel/core.html#AurelCore.kxy>`_: $K_{xy}$ Extrinsic curvature with xy indices down. I assume $K_{xy}=0$, if not then please define AurelCore.data['kxy'] = ... 

.. _aurel.core.AurelCore.kxz:

`kxz <../_modules/aurel/core.html#AurelCore.kxz>`_: $K_{xz}$ Extrinsic curvature with xz indices down. I assume $K_{xz}=0$, if not then please define AurelCore.data['kxz'] = ... 

.. _aurel.core.AurelCore.kyy:

`kyy <../_modules/aurel/core.html#AurelCore.kyy>`_: $K_{yy}$ Extrinsic curvature with yy indices down. I assume $K_{yy}=0$, if not then please define AurelCore.data['kyy'] = ... 

.. _aurel.core.AurelCore.kyz:

`kyz <../_modules/aurel/core.html#AurelCore.kyz>`_: $K_{yz}$ Extrinsic curvature with yz indices down. I assume $K_{yz}=0$, if not then please define AurelCore.data['kyz'] = ... 

.. _aurel.core.AurelCore.kzz:

`kzz <../_modules/aurel/core.html#AurelCore.kzz>`_: $K_{zz}$ Extrinsic curvature with zz indices down. I assume $K_{zz}=0$, if not then please define AurelCore.data['kzz'] = ... 

.. _aurel.core.AurelCore.Kdown3:

`Kdown3 <../_modules/aurel/core.html#AurelCore.Kdown3>`_: $K_{ij}$ Extrinsic curvature with spatial indices down

.. _aurel.core.AurelCore.Kup3:

`Kup3 <../_modules/aurel/core.html#AurelCore.Kup3>`_: $K^{ij}$ Extrinsic curvature with spatial indices up

.. _aurel.core.AurelCore.Ktrace:

`Ktrace <../_modules/aurel/core.html#AurelCore.Ktrace>`_: $K = \gamma^{ij}K_{ij}$ Trace of extrinsic curvature

.. _aurel.core.AurelCore.dtKtrace:

`dtKtrace <../_modules/aurel/core.html#AurelCore.dtKtrace>`_: $\partial_t K$ Coordinate time derivative of the trace of extrinsic curvature

.. _aurel.core.AurelCore.Adown3:

`Adown3 <../_modules/aurel/core.html#AurelCore.Adown3>`_: $A_{ij}$ Traceless part of the extrinsic curvature with spatial indices down

.. _aurel.core.AurelCore.Aup3:

`Aup3 <../_modules/aurel/core.html#AurelCore.Aup3>`_: $A^{ij}$ Traceless part of the extrinsic curvature with spatial indices up

.. _aurel.core.AurelCore.A2:

`A2 <../_modules/aurel/core.html#AurelCore.A2>`_: $A^2$ Magnitude of traceless part of the extrinsic curvature

BSSNOK extrinsic curvature
--------------------------

.. _aurel.core.AurelCore.Adown3_bssnok:

`Adown3_bssnok <../_modules/aurel/core.html#AurelCore.Adown3_bssnok>`_: $\tilde{A}_{ij}$ Conformal traceless part of the extrinsic curvature with spatial indices down

.. _aurel.core.AurelCore.Aup3_bssnok:

`Aup3_bssnok <../_modules/aurel/core.html#AurelCore.Aup3_bssnok>`_: $\tilde{A}^{ij}$ Conformal traceless part of the extrinsic curvature with spatial indices up

.. _aurel.core.AurelCore.A2_bssnok:

`A2_bssnok <../_modules/aurel/core.html#AurelCore.A2_bssnok>`_: $\tilde{A}^2$ Magnitude of conformal traceless part of the extrinsic curvature

.. _aurel.core.AurelCore.dtAdown3_bssnok:

`dtAdown3_bssnok <../_modules/aurel/core.html#AurelCore.dtAdown3_bssnok>`_: $\partial_t \tilde{A}_{ij}$ Coordinate time derivative of conformal traceless part of the extrinsic curvature with spatial indices down

Proper time
-----------

.. _aurel.core.AurelCore.dttau:

`dttau <../_modules/aurel/core.html#AurelCore.dttau>`_: $\partial_t \tau$ Coordinate time derivative of proper time

Matter quantities
=================

Eulerian observers follow $n^\mu$, Lagrangian observers follow $u^\mu$

Lagrangian matter variables
---------------------------

.. _aurel.core.AurelCore.rho0:

`rho0 <../_modules/aurel/core.html#AurelCore.rho0>`_: $\rho_0$ Rest mass energy density. I assume $\rho_0=0$, if not then please define AurelCore.data['rho0'] = ... 

.. _aurel.core.AurelCore.press:

`press <../_modules/aurel/core.html#AurelCore.press>`_: $p$ Pressure. I assume $p=0$, if not then please define AurelCore.data['press'] = ... 

.. _aurel.core.AurelCore.eps:

`eps <../_modules/aurel/core.html#AurelCore.eps>`_: $\epsilon$ Specific internal energy. I assume $\epsilon=0$, if not then please define AurelCore.data['eps'] = ... 

.. _aurel.core.AurelCore.rho:

`rho <../_modules/aurel/core.html#AurelCore.rho>`_: $\rho$ Energy density

.. _aurel.core.AurelCore.enthalpy:

`enthalpy <../_modules/aurel/core.html#AurelCore.enthalpy>`_: $h$ Specific enthalpy of the fluid

Fluid velocity
--------------

.. _aurel.core.AurelCore.w_lorentz:

`w_lorentz <../_modules/aurel/core.html#AurelCore.w_lorentz>`_: $W$ Lorentz factor. I assume $W=1$, if not then please define AurelCore.data['w_lorentz'] = ... 

.. _aurel.core.AurelCore.velx:

`velx <../_modules/aurel/core.html#AurelCore.velx>`_: $v^x$ x component of Eulerian fluid three velocity with indice up. I assume $v^x=0$, if not then please define AurelCore.data['velx'] = ... 

.. _aurel.core.AurelCore.vely:

`vely <../_modules/aurel/core.html#AurelCore.vely>`_: $v^y$ y component of Eulerian fluid three velocity with indice up. I assume $v^y=0$, if not then please define AurelCore.data['vely'] = ... 

.. _aurel.core.AurelCore.velz:

`velz <../_modules/aurel/core.html#AurelCore.velz>`_: $v^z$ z component of Eulerian fluid three velocity with indice up. I assume $v^z=0$, if not then please define AurelCore.data['velz'] = ... 

.. _aurel.core.AurelCore.velup3:

`velup3 <../_modules/aurel/core.html#AurelCore.velup3>`_: $v^i$ Eulerian fluid three velocity with spatial indices up.

.. _aurel.core.AurelCore.velup4:

`velup4 <../_modules/aurel/core.html#AurelCore.velup4>`_: $v^\mu$ Eulerian fluid three velocity with spacetime indices up.

.. _aurel.core.AurelCore.veldown3:

`veldown3 <../_modules/aurel/core.html#AurelCore.veldown3>`_: $v_i$ Eulerian fluid three velocity with spatial indices down

.. _aurel.core.AurelCore.veldown4:

`veldown4 <../_modules/aurel/core.html#AurelCore.veldown4>`_: $v_\mu$ Eulerian fluid three velocity with spacetime indices down

.. _aurel.core.AurelCore.uup0:

`uup0 <../_modules/aurel/core.html#AurelCore.uup0>`_: $u^t$ Lagrangian fluid four velocity with time indice up

.. _aurel.core.AurelCore.uup3:

`uup3 <../_modules/aurel/core.html#AurelCore.uup3>`_: $u^i$ Lagrangian fluid four velocity with spatial indices up

.. _aurel.core.AurelCore.uup4:

`uup4 <../_modules/aurel/core.html#AurelCore.uup4>`_: $u^\mu$ Lagrangian fluid four velocity with spacetime indices up

.. _aurel.core.AurelCore.udown3:

`udown3 <../_modules/aurel/core.html#AurelCore.udown3>`_: $u_\mu$ Lagrangian fluid four velocity with spatial indices down

.. _aurel.core.AurelCore.udown4:

`udown4 <../_modules/aurel/core.html#AurelCore.udown4>`_: $u_\mu$ Lagrangian fluid four velocity with spacetime indices down

.. _aurel.core.AurelCore.hdown4:

`hdown4 <../_modules/aurel/core.html#AurelCore.hdown4>`_: $h_{\mu\nu}$ Spatial metric orthonomal to fluid flow with spacetime indices down

.. _aurel.core.AurelCore.hdet:

`hdet <../_modules/aurel/core.html#AurelCore.hdet>`_: $h$ Determinant of spatial part of spatial metric orthonormal to fluid flow

.. _aurel.core.AurelCore.hmixed4:

`hmixed4 <../_modules/aurel/core.html#AurelCore.hmixed4>`_: ${h^{\mu}}_{\nu}$ Spatial metric orthonomal to fluid flow with mixed spacetime indices

.. _aurel.core.AurelCore.hup4:

`hup4 <../_modules/aurel/core.html#AurelCore.hup4>`_: $h^{\mu\nu}$ Spatial metric orthonomal to fluid flow with spacetime indices up

Energy-stress tensor
--------------------

.. _aurel.core.AurelCore.Tdown4:

`Tdown4 <../_modules/aurel/core.html#AurelCore.Tdown4>`_: $T_{\mu\nu}$ Energy-stress tensor with spacetime indices down

.. _aurel.core.AurelCore.Tup4:

`Tup4 <../_modules/aurel/core.html#AurelCore.Tup4>`_: $T^{\mu\nu}$ Energy-stress tensor with spacetime indices up

.. _aurel.core.AurelCore.Ttrace:

`Ttrace <../_modules/aurel/core.html#AurelCore.Ttrace>`_: $T$ Trace of the energy-stress tensor

Eulerian matter variables
-------------------------

.. _aurel.core.AurelCore.rho_n:

`rho_n <../_modules/aurel/core.html#AurelCore.rho_n>`_: $\rho^{\{n\}}$ Energy density in the $n^\mu$ frame

.. _aurel.core.AurelCore.fluxup3_n:

`fluxup3_n <../_modules/aurel/core.html#AurelCore.fluxup3_n>`_: $S^{\{n\}i}$ Energy flux (or momentum density) in the $n^\mu$ frame with spatial indices up

.. _aurel.core.AurelCore.fluxdown3_n:

`fluxdown3_n <../_modules/aurel/core.html#AurelCore.fluxdown3_n>`_: $S^{\{n\}}_{i}$ Energy flux (or momentum density) in the $n^\mu$ frame with spatial indices down

.. _aurel.core.AurelCore.angmomup3_n:

`angmomup3_n <../_modules/aurel/core.html#AurelCore.angmomup3_n>`_: $J^{\{n\}i}$ Angular momentum density in the $n^\mu$ frame with spatial indices up

.. _aurel.core.AurelCore.angmomdown3_n:

`angmomdown3_n <../_modules/aurel/core.html#AurelCore.angmomdown3_n>`_: $J^{\{n\}}_{i}$ Angular momentum density in the $n^\mu$ frame with spatial indices down

.. _aurel.core.AurelCore.Stressup3_n:

`Stressup3_n <../_modules/aurel/core.html#AurelCore.Stressup3_n>`_: $S^{\{n\}ij}$ Stress tensor in the $n^\mu$ frame with spatial indices up

.. _aurel.core.AurelCore.Stressdown3_n:

`Stressdown3_n <../_modules/aurel/core.html#AurelCore.Stressdown3_n>`_: $S^{\{n\}}_{ij}$ Stress tensor in the $n^\mu$ frame with spatial indices down

.. _aurel.core.AurelCore.Stresstrace_n:

`Stresstrace_n <../_modules/aurel/core.html#AurelCore.Stresstrace_n>`_: $S^{\{n\}}$ Trace of Stress tensor in the $n^\mu$ frame

.. _aurel.core.AurelCore.press_n:

`press_n <../_modules/aurel/core.html#AurelCore.press_n>`_: $p^{\{n\}}$ Pressure in the $n^\mu$ frame

.. _aurel.core.AurelCore.anisotropic_press_down3_n:

`anisotropic_press_down3_n <../_modules/aurel/core.html#AurelCore.anisotropic_press_down3_n>`_: $\pi^{\{n\}_{ij}}$ Anisotropic pressure in the $n^\mu$ frame with spatial indices down

.. _aurel.core.AurelCore.rho_n_fromHam:

`rho_n_fromHam <../_modules/aurel/core.html#AurelCore.rho_n_fromHam>`_: $\rho^{\{n\}}$ Energy density in the $n^\mu$ frame computed from the Hamiltonian constraint

.. _aurel.core.AurelCore.fluxup3_n_fromMom:

`fluxup3_n_fromMom <../_modules/aurel/core.html#AurelCore.fluxup3_n_fromMom>`_: $S^{\{n\}i}$ Energy flux (or momentum density) in the $n^\mu$ frame with spatial indices up computed from the Momentum constraint

Conserved variables
-------------------

.. _aurel.core.AurelCore.conserved_D:

`conserved_D <../_modules/aurel/core.html#AurelCore.conserved_D>`_: $D$ Conserved mass-energy density in Wilson's formalism

.. _aurel.core.AurelCore.conserved_E:

`conserved_E <../_modules/aurel/core.html#AurelCore.conserved_E>`_: $E$ Conserved internal energy density in Wilson's formalism

.. _aurel.core.AurelCore.conserved_Sdown4:

`conserved_Sdown4 <../_modules/aurel/core.html#AurelCore.conserved_Sdown4>`_: $S_{\mu}$ Conserved energy flux (or momentum density) in Wilson's formalism with spacetime indices down

.. _aurel.core.AurelCore.conserved_Sdown3:

`conserved_Sdown3 <../_modules/aurel/core.html#AurelCore.conserved_Sdown3>`_: $S_{i}$ Conserved energy flux (or momentum density) in Wilson's formalism with spatial indices down

.. _aurel.core.AurelCore.conserved_Sup4:

`conserved_Sup4 <../_modules/aurel/core.html#AurelCore.conserved_Sup4>`_: $S^{\mu}$ Conserved energy flux (or momentum density) in Wilson's formalism with spacetime indices up

.. _aurel.core.AurelCore.conserved_Sup3:

`conserved_Sup3 <../_modules/aurel/core.html#AurelCore.conserved_Sup3>`_: $S^{i}$ Conserved energy flux (or momentum density) in Wilson's formalism with spatial indices up

.. _aurel.core.AurelCore.dtconserved:

`dtconserved <../_modules/aurel/core.html#AurelCore.dtconserved>`_: $\partial_t D, \; \partial_t E, \partial_t S_{i}$ List of coordinate time derivatives of conserved rest mass-energy density, internal energy density and energy flux (or momentum density) with spatial indices down in Wilson's formalism

Kinematic variables
-------------------

.. _aurel.core.AurelCore.st_covd_udown4:

`st_covd_udown4 <../_modules/aurel/core.html#AurelCore.st_covd_udown4>`_: $\nabla_{\mu} u_{\nu}$ Spacetime covariant derivative of Lagrangian fluid four velocity with spacetime indices down

.. _aurel.core.AurelCore.accelerationdown4:

`accelerationdown4 <../_modules/aurel/core.html#AurelCore.accelerationdown4>`_: $a_{\mu}$ Acceleration of the fluid with spacetime indices down

.. _aurel.core.AurelCore.accelerationup4:

`accelerationup4 <../_modules/aurel/core.html#AurelCore.accelerationup4>`_: $a^{\mu}$ Acceleration of the fluid with spacetime indices up

.. _aurel.core.AurelCore.s_covd_udown4:

`s_covd_udown4 <../_modules/aurel/core.html#AurelCore.s_covd_udown4>`_: $\mathcal{D}^{\{u\}}_{\mu} u_{\nu}$ Spatial covariant derivative of Lagrangian fluid four velocity with spacetime indices down, with respect to spatial hypersurfaces orthonormal to the fluid flow

.. _aurel.core.AurelCore.thetadown4:

`thetadown4 <../_modules/aurel/core.html#AurelCore.thetadown4>`_: $\Theta_{\mu\nu}$ Fluid expansion tensor with spacetime indices down

.. _aurel.core.AurelCore.theta:

`theta <../_modules/aurel/core.html#AurelCore.theta>`_: $\Theta$ Fluid expansion scalar

.. _aurel.core.AurelCore.sheardown4:

`sheardown4 <../_modules/aurel/core.html#AurelCore.sheardown4>`_: $\sigma_{\mu\nu}$ Fluid shear tensor with spacetime indices down

.. _aurel.core.AurelCore.shear2:

`shear2 <../_modules/aurel/core.html#AurelCore.shear2>`_: $\sigma^2$ Magnitude of fluid shear

.. _aurel.core.AurelCore.omegadown4:

`omegadown4 <../_modules/aurel/core.html#AurelCore.omegadown4>`_: $\omega_{\mu\nu}$ Fluid vorticity tensor with spacetime indices down

.. _aurel.core.AurelCore.omega2:

`omega2 <../_modules/aurel/core.html#AurelCore.omega2>`_: $\omega^2$ Magnitude of fluid vorticity

.. _aurel.core.AurelCore.s_RicciS_u:

`s_RicciS_u <../_modules/aurel/core.html#AurelCore.s_RicciS_u>`_: ${}^{(3)}R^{\{u\}}$ Ricci scalar of the spatial metric orthonormal to fluid flow

Curvature quantities
====================

Spatial curvature
-----------------

.. _aurel.core.AurelCore.s_Gamma_udd3:

`s_Gamma_udd3 <../_modules/aurel/core.html#AurelCore.s_Gamma_udd3>`_: ${}^{(3)}{\Gamma^{k}}_{ij}$ Christoffel symbols of spatial metric with mixed spatial indices

.. _aurel.core.AurelCore.s_Riemann_uddd3:

`s_Riemann_uddd3 <../_modules/aurel/core.html#AurelCore.s_Riemann_uddd3>`_: ${}^{(3)}{R^{i}}_{jkl}$ Riemann tensor of spatial metric with mixed spatial indices

.. _aurel.core.AurelCore.s_Riemann_down3:

`s_Riemann_down3 <../_modules/aurel/core.html#AurelCore.s_Riemann_down3>`_: ${}^{(3)}R_{ijkl}$ Riemann tensor of spatial metric with all spatial indices down

.. _aurel.core.AurelCore.s_Ricci_down3:

`s_Ricci_down3 <../_modules/aurel/core.html#AurelCore.s_Ricci_down3>`_: ${}^{(3)}R_{ij}$ Ricci tensor of spatial metric with spatial indices down

.. _aurel.core.AurelCore.s_RicciS:

`s_RicciS <../_modules/aurel/core.html#AurelCore.s_RicciS>`_: ${}^{(3)}R$ Ricci scalar of spatial metric

Spacetime curvature
-------------------

.. _aurel.core.AurelCore.st_Gamma_udd4:

`st_Gamma_udd4 <../_modules/aurel/core.html#AurelCore.st_Gamma_udd4>`_: ${}^{(4)}{\Gamma^{\alpha}}_{\mu\nu}$ Christoffel symbols of spacetime metric with mixed spacetime indices

.. _aurel.core.AurelCore.st_Riemann_uddd4:

`st_Riemann_uddd4 <../_modules/aurel/core.html#AurelCore.st_Riemann_uddd4>`_: ${}^{(4)}{R^{\alpha}}_{\beta\mu\nu}$ Riemann tensor of spacetime metric with mixed spacetime indices

.. _aurel.core.AurelCore.st_Riemann_down4:

`st_Riemann_down4 <../_modules/aurel/core.html#AurelCore.st_Riemann_down4>`_: ${}^{(4)}R_{\alpha\beta\mu\nu}$ Riemann tensor of spacetime metric with spacetime indices down

.. _aurel.core.AurelCore.st_Riemann_uudd4:

`st_Riemann_uudd4 <../_modules/aurel/core.html#AurelCore.st_Riemann_uudd4>`_: ${}^{(4)}{R^{\alpha\beta}}_{\mu\nu}$ Riemann tensor of spacetime metric with mixed spacetime indices

.. _aurel.core.AurelCore.st_Ricci_down4:

`st_Ricci_down4 <../_modules/aurel/core.html#AurelCore.st_Ricci_down4>`_: ${}^{(4)}R_{\alpha\beta}$ Ricci tensor of spacetime metric with spacetime indices down

.. _aurel.core.AurelCore.st_Ricci_down3:

`st_Ricci_down3 <../_modules/aurel/core.html#AurelCore.st_Ricci_down3>`_: ${}^{(4)}R_{ij}$ Ricci tensor of spacetime metric with spatial indices down

.. _aurel.core.AurelCore.st_RicciS:

`st_RicciS <../_modules/aurel/core.html#AurelCore.st_RicciS>`_: ${}^{(4)}R$ Ricci scalar of spacetime metric

.. _aurel.core.AurelCore.Einsteindown4:

`Einsteindown4 <../_modules/aurel/core.html#AurelCore.Einsteindown4>`_: $G_{\alpha\beta}$ Einstein tensor with spacetime indices down

.. _aurel.core.AurelCore.Kretschmann:

`Kretschmann <../_modules/aurel/core.html#AurelCore.Kretschmann>`_: $K={R^{\alpha\beta}}_{\mu\nu}{R_{\alpha\beta}}^{\mu\nu}$ Kretschmann scalar

BSSNOK curvature
----------------

.. _aurel.core.AurelCore.s_Gamma_udd3_bssnok:

`s_Gamma_udd3_bssnok <../_modules/aurel/core.html#AurelCore.s_Gamma_udd3_bssnok>`_: ${}^{(3)}{\tilde{\Gamma}^{k}}_{ij}$ Christoffel symbols of conformal spatial metric with mixed spatial indices

.. _aurel.core.AurelCore.s_Gamma_bssnok:

`s_Gamma_bssnok <../_modules/aurel/core.html#AurelCore.s_Gamma_bssnok>`_: ${}^{(3)}\tilde{\Gamma}^i$ Conformal connection functions with spatial indice up

.. _aurel.core.AurelCore.dts_Gamma_bssnok:

`dts_Gamma_bssnok <../_modules/aurel/core.html#AurelCore.dts_Gamma_bssnok>`_: $\partial_t {}^{(3)}\tilde{\Gamma}^i$ Coordinate time derivative of conformal connection functions with spatial indice up

.. _aurel.core.AurelCore.s_Ricci_down3_bssnok:

`s_Ricci_down3_bssnok <../_modules/aurel/core.html#AurelCore.s_Ricci_down3_bssnok>`_: ${}^{(3)}\tilde{R}_{ij}$ Ricci tensor of conformal spatial metric with spatial indices down

.. _aurel.core.AurelCore.s_RicciS_bssnok:

`s_RicciS_bssnok <../_modules/aurel/core.html#AurelCore.s_RicciS_bssnok>`_: ${}^{(3)}\tilde{R}$ Ricci scalar of conformal spatial metric

.. _aurel.core.AurelCore.s_Ricci_down3_phi:

`s_Ricci_down3_phi <../_modules/aurel/core.html#AurelCore.s_Ricci_down3_phi>`_: ${}^{(3)}R^{\phi}_{ij}$ Ricci terms that depend on the conformal function $\phi$

Weyl decomposition
------------------

.. _aurel.core.AurelCore.st_Weyl_down4:

`st_Weyl_down4 <../_modules/aurel/core.html#AurelCore.st_Weyl_down4>`_: $C_{\alpha\beta\mu\nu}$ Weyl tensor of spacetime metric with spacetime indices down

.. _aurel.core.AurelCore.Weyl_Psi:

`Weyl_Psi <../_modules/aurel/core.html#AurelCore.Weyl_Psi>`_: $\Psi_0, \; \Psi_1, \; \Psi_2, \; \Psi_3, \; \Psi_4$ List of Weyl scalars for an null vector base defined with AurelCore.tetrad

.. _aurel.core.AurelCore.Psi4_lm:

`Psi4_lm <../_modules/aurel/core.html#AurelCore.Psi4_lm>`_: $\Psi_4^{l,m}$ List of dictionaries of spin weighted spherical harmonic decomposition of the 4th Weyl scalar. Control with AurelCore.lmax, center, extract_radii, and interp_method.

.. _aurel.core.AurelCore.Weyl_invariants:

`Weyl_invariants <../_modules/aurel/core.html#AurelCore.Weyl_invariants>`_: $I, \; J, \; L, \; K, \; N$ Dictionary of Weyl invariants

.. _aurel.core.AurelCore.eweyl_u_down4:

`eweyl_u_down4 <../_modules/aurel/core.html#AurelCore.eweyl_u_down4>`_: $E^{\{u\}}_{\alpha\beta}$ Electric part of the Weyl tensor on the hypersurface orthogonal to $u^{\mu}$ with spacetime indices down

.. _aurel.core.AurelCore.eweyl_n_down3:

`eweyl_n_down3 <../_modules/aurel/core.html#AurelCore.eweyl_n_down3>`_: $E^{\{n\}}_{ij}$ Electric part of the Weyl tensor on the hypersurface orthogonal to $n^{\mu}$ with spatial indices down

.. _aurel.core.AurelCore.bweyl_u_down4:

`bweyl_u_down4 <../_modules/aurel/core.html#AurelCore.bweyl_u_down4>`_: $B^{\{u\}}_{\alpha\beta}$ Magnetic part of the Weyl tensor on the hypersurface orthogonal to $u^{\mu}$ with spacetime indices down

.. _aurel.core.AurelCore.bweyl_n_down3:

`bweyl_n_down3 <../_modules/aurel/core.html#AurelCore.bweyl_n_down3>`_: $B^{\{n\}}_{ij}$ Magnetic part of the Weyl tensor on the hypersurface orthogonal to $n^{\mu}$ with spatial indices down

Null ray expansion
==================

.. _aurel.core.AurelCore.null_ray_exp_out:

`null_ray_exp_out <../_modules/aurel/core.html#AurelCore.null_ray_exp_out>`_: $\Theta_{out}$ List of expansion of null rays radially going out

.. _aurel.core.AurelCore.null_ray_exp_in:

`null_ray_exp_in <../_modules/aurel/core.html#AurelCore.null_ray_exp_in>`_: $\Theta_{in}$ List of expansion of null rays radially going in

Constraints
===========

.. _aurel.core.AurelCore.Hamiltonian:

`Hamiltonian <../_modules/aurel/core.html#AurelCore.Hamiltonian>`_: $\mathcal{H}$ Hamilonian constraint

.. _aurel.core.AurelCore.Hamiltonian_Escale:

`Hamiltonian_Escale <../_modules/aurel/core.html#AurelCore.Hamiltonian_Escale>`_: [$\mathcal{H}$] Hamilonian constraint energy scale

.. _aurel.core.AurelCore.Hamiltonian_norm:

`Hamiltonian_norm <../_modules/aurel/core.html#AurelCore.Hamiltonian_norm>`_: $\mathcal{H}/[\mathcal{H}]$ Normalized Hamilonian constraint

.. _aurel.core.AurelCore.Momentumx:

`Momentumx <../_modules/aurel/core.html#AurelCore.Momentumx>`_: $\mathcal{M}^x$ Momentum constraint with x spatial indices up

.. _aurel.core.AurelCore.Momentumy:

`Momentumy <../_modules/aurel/core.html#AurelCore.Momentumy>`_: $\mathcal{M}^y$ Momentum constraint with y spatial indices up

.. _aurel.core.AurelCore.Momentumz:

`Momentumz <../_modules/aurel/core.html#AurelCore.Momentumz>`_: $\mathcal{M}^z$ Momentum constraint with z spatial indices up

.. _aurel.core.AurelCore.Momentumdownx:

`Momentumdownx <../_modules/aurel/core.html#AurelCore.Momentumdownx>`_: $\mathcal{M}_x$ Momentum constraint with x spatial indices down

.. _aurel.core.AurelCore.Momentumdowny:

`Momentumdowny <../_modules/aurel/core.html#AurelCore.Momentumdowny>`_: $\mathcal{M}_y$ Momentum constraint with y spatial indices down

.. _aurel.core.AurelCore.Momentumdownz:

`Momentumdownz <../_modules/aurel/core.html#AurelCore.Momentumdownz>`_: $\mathcal{M}_z$ Momentum constraint with z spatial indices down

.. _aurel.core.AurelCore.Momentumup3:

`Momentumup3 <../_modules/aurel/core.html#AurelCore.Momentumup3>`_: $\mathcal{M}^i$ Momentum constraint with spatial indices up

.. _aurel.core.AurelCore.Momentumdown3:

`Momentumdown3 <../_modules/aurel/core.html#AurelCore.Momentumdown3>`_: $\mathcal{M}_i$ Momentum constraint with spatial indices down

.. _aurel.core.AurelCore.Momentum_Escale:

`Momentum_Escale <../_modules/aurel/core.html#AurelCore.Momentum_Escale>`_: [$\mathcal{M}$] Momentum constraint energy scale

.. _aurel.core.AurelCore.Momentumx_norm:

`Momentumx_norm <../_modules/aurel/core.html#AurelCore.Momentumx_norm>`_: $\mathcal{M}^x/[\mathcal{M}]$ Normalized Momentum constraint with x spatial indices up

.. _aurel.core.AurelCore.Momentumy_norm:

`Momentumy_norm <../_modules/aurel/core.html#AurelCore.Momentumy_norm>`_: $\mathcal{M}^y/[\mathcal{M}]$ Normalized Momentum constraint with y spatial indices up

.. _aurel.core.AurelCore.Momentumz_norm:

`Momentumz_norm <../_modules/aurel/core.html#AurelCore.Momentumz_norm>`_: $\mathcal{M}^z/[\mathcal{M}]$ Normalized Momentum constraint with z spatial indices up

.. _aurel.core.AurelCore.Momentumdownx_norm:

`Momentumdownx_norm <../_modules/aurel/core.html#AurelCore.Momentumdownx_norm>`_: $\mathcal{M}_x/[\mathcal{M}]$ Normalized Momentum constraint with x spatial indices down

.. _aurel.core.AurelCore.Momentumdowny_norm:

`Momentumdowny_norm <../_modules/aurel/core.html#AurelCore.Momentumdowny_norm>`_: $\mathcal{M}_y/[\mathcal{M}]$ Normalized Momentum constraint with y spatial indices down

.. _aurel.core.AurelCore.Momentumdownz_norm:

`Momentumdownz_norm <../_modules/aurel/core.html#AurelCore.Momentumdownz_norm>`_: $\mathcal{M}_z/[\mathcal{M}]$ Normalized Momentum constraint with z spatial indices down

AurelCore Methods
*****************

.. autoclass:: aurel.core.AurelCore
   :show-inheritance:
   :members:
   :exclude-members: alpha, dtalpha, DDalpha, betax, betay, betaz, betaup3, dtbetax, dtbetay, dtbetaz, dtbetaup3, betadown3, betamag, nup4, ndown4, gxx, gxy, gxz, gyy, gyz, gzz, gammadown3, gammaup3, dtgammaup3, gammadet, gammadown4, gammaup4, gtt, gtx, gty, gtz, gdown4, gup4, gdet, psi_bssnok, phi_bssnok, dtphi_bssnok, gammadown3_bssnok, gammaup3_bssnok, dtgammadown3_bssnok, kxx, kxy, kxz, kyy, kyz, kzz, Kdown3, Kup3, Ktrace, dtKtrace, Adown3, Aup3, A2, Adown3_bssnok, Aup3_bssnok, A2_bssnok, dtAdown3_bssnok, dttau, rho0, press, eps, rho, enthalpy, w_lorentz, velx, vely, velz, velup3, velup4, veldown3, veldown4, uup0, uup3, uup4, udown3, udown4, hdown4, hdet, hmixed4, hup4, Tdown4, Tup4, Ttrace, rho_n, fluxup3_n, fluxdown3_n, angmomup3_n, angmomdown3_n, Stressup3_n, Stressdown3_n, Stresstrace_n, press_n, anisotropic_press_down3_n, rho_n_fromHam, fluxup3_n_fromMom, conserved_D, conserved_E, conserved_Sdown4, conserved_Sdown3, conserved_Sup4, conserved_Sup3, dtconserved, st_covd_udown4, accelerationdown4, accelerationup4, s_covd_udown4, thetadown4, theta, sheardown4, shear2, omegadown4, omega2, s_RicciS_u, s_Gamma_udd3, s_Riemann_uddd3, s_Riemann_down3, s_Ricci_down3, s_RicciS, st_Gamma_udd4, st_Riemann_uddd4, st_Riemann_down4, st_Riemann_uudd4, st_Ricci_down4, st_Ricci_down3, st_RicciS, Einsteindown4, Kretschmann, s_Gamma_udd3_bssnok, s_Gamma_bssnok, dts_Gamma_bssnok, s_Ricci_down3_bssnok, s_RicciS_bssnok, s_Ricci_down3_phi, st_Weyl_down4, Weyl_Psi, Psi4_lm, Weyl_invariants, eweyl_u_down4, eweyl_n_down3, bweyl_u_down4, bweyl_n_down3, null_ray_exp_out, null_ray_exp_in, Hamiltonian, Hamiltonian_Escale, Hamiltonian_norm, Momentumx, Momentumy, Momentumz, Momentumdownx, Momentumdowny, Momentumdownz, Momentumup3, Momentumdown3, Momentum_Escale, Momentumx_norm, Momentumy_norm, Momentumz_norm, Momentumdownx_norm, Momentumdowny_norm, Momentumdownz_norm

