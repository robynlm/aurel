"""See: https://arxiv.org/pdf/2302.09033
 and: https://arxiv.org/pdf/1307.1478"""

import numpy as np
import sympy as sp

def Rc_func(x, y, z, amp, lamb):
    """Comoving curvature perturbation"""
    Ax, Ay, Az = amp
    Lx, Ly, Lz = lamb
    return (
        Ax * np.sin( 2 * np.pi * x / Lx)
        + Ay * np.sin( 2 * np.pi * y / Ly)
        + Az * np.sin( 2 * np.pi * z / Lz))

def gammadown3(sol, fd, t, Rc):
    """Spatial metric with 1st order perturbations"""
    a2 = sol.a(t)**2
    F = sol.fL(t) + (3/2) * sol.Omega_m(t)
    m2iFH2 = - 2 / ( F *sol.Hprop(t)**2 )
    gxx = a2 * ( 1 - 2 * Rc ) + m2iFH2 * fd.d3x(fd.d3x(Rc))
    gxy = m2iFH2 * fd.d3x(fd.d3y(Rc))
    gxz = m2iFH2 * fd.d3x(fd.d3z(Rc))
    gyy = a2 * ( 1 - 2 * Rc ) + m2iFH2 * fd.d3y(fd.d3y(Rc))
    gyz = m2iFH2 * fd.d3y(fd.d3z(Rc))
    gzz = a2 * ( 1 - 2 * Rc ) + m2iFH2 * fd.d3z(fd.d3z(Rc))
    return np.array([
            [gxx, gxy, gxz],
            [gxy, gyy, gyz],
            [gxz, gyz, gzz]])

def Kdown3(sol, fd, t, Rc):
    """Extrinsic curvature, nonlinear from gammmadown3"""
    a2 = sol.a(t)**2
    F = sol.fL(t) + (3/2) * sol.Omega_m(t)
    iFH = 1 / ( F * sol.Hprop(t) )
    kxx = (- a2 * sol.Hprop(t) * ( 1 - 2 * Rc ) 
           + ( 2 + sol.fL(t) ) * fd.d3x(fd.d3x(Rc)) * iFH)
    kxy = ( 2 + sol.fL(t) ) * fd.d3x(fd.d3y(Rc)) * iFH
    kxz = ( 2 + sol.fL(t) ) * fd.d3x(fd.d3z(Rc)) * iFH
    kyy = (- a2 * sol.Hprop(t) * ( 1 - 2 * Rc ) 
           + ( 2 + sol.fL(t) ) * fd.d3y(fd.d3y(Rc)) * iFH)
    kyz = ( 2 + sol.fL(t) ) * fd.d3y(fd.d3z(Rc)) * iFH
    kzz = (- a2 * sol.Hprop(t) * ( 1 - 2 * Rc ) 
           + ( 2 + sol.fL(t) ) * fd.d3z(fd.d3z(Rc)) * iFH)

    return np.array([
            [kxx, kxy, kxz],
            [kxy, kyy, kyz],
            [kxz, kyz, kzz]])

def delta1(sol, fd, t, Rc):
    """Linear density contrast"""
    F = sol.fL(t) + (3/2) * sol.Omega_m(t)
    return (
        (fd.d3x(fd.d3x(Rc)) + fd.d3y(fd.d3y(Rc)) + fd.d3z(fd.d3z(Rc))) 
        / (sol.a(t)**2 * F * sol.Hprop(t)**2))