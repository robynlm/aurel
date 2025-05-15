r"""This is a $\Lambda$ - Szekeres solution
which is perturbed solution of the flat dust FLRW + LCDM spacetime.
See section 3.1 of 2211.08133"""

import numpy as np
import sympy as sp
import scipy.special as sc
from . import LCDM

Amp = 1000 # Amplitude of the perturbation
L = 10 # Wavelength of the perturbation
k = 2*np.pi/L # Frequency of the perturbation

tauC = np.sqrt(3*LCDM.Lambda/4)
B = (3/4)*(LCDM.Hprop_today**2)*(LCDM.Omega_l_today
                                 *(LCDM.Omega_m_today**2))**(1/3)

def Z_terms(t, x, y, z, analytical=False):
    """Returns F, Z, dtZ functions"""
    if analytical:
        pac = sp
    else:
        pac = np
    betaP = Amp*(1-pac.sin(k*z))
    betaM = 0
    A = 1+B*betaP*(x**2+y**2)
    tau = tauC*t
    fM = pac.cosh(tau)/pac.sinh(tau)
    if analytical:
        hyperthing = sp.hyper([5/6, 3/2], [11/6], -pac.sinh(tau)**2)
    else:
        hyperthing = sc.hyp2f1(5/6, 3/2, 11/6, -pac.sinh(tau)**2) 
    integrated_part = ((3/5) * pac.sqrt(pac.cosh(tau)**2)
                       * hyperthing * (pac.sinh(tau)**(5/3)) / pac.cosh(tau))
    fP = fM*integrated_part
    
    dtaufM = -1/pac.sinh(tau)**2
    dtfM = tauC*dtaufM
    part_to_integrate = (pac.sinh(tau)**(2/3))/pac.cosh(tau)**2
    dtaufP = dtaufM * integrated_part + fM * part_to_integrate
    dtfP = tauC * dtaufP

    F = betaM*fM + betaP*fP
    Z = F + A
    dtZ = betaM*dtfM + betaP*dtfP
    return F, Z, dtZ

def rho(t, x, y, z):
    """Returns the energy density"""
    F, Z, dtZ = Z_terms(t, x, y, z)
    delta =  -F/Z
    return  LCDM.rho(t)*(1+delta)

def press(t, x, y, z):
    """Returns the pressure"""
    return np.zeros(np.shape(x))

def gammadown3(t, x, y, z, analytical=False):
    """Returns the spatial metric"""
    gd3 = LCDM.gammadown3(t, x, y, z, analytical=analytical)
    F, Z, dtZ = Z_terms(t, x, y, z, analytical=analytical)
    gd3[2, 2] *= Z**2
    return gd3

def Kdown3(t, x, y, z):
    """Returns the extrinsic curvature"""
    # Define the Kdown matrix
    Kd3 = LCDM.gammadown3(t, x, y, z)
    F, Z, dtZ = Z_terms(t, x, y, z)
    Kd3[2, 2] += - (LCDM.a(t)**2) * (dtZ / Z)
    Kd3[2, 2] *= Z**2
    return Kd3

def alpha(t, x, y, z):
    """Returns the lapse function"""
    return np.ones(np.shape(x))

def betaup3(t, x, y, z):
    """Returns the shift vector"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((3, Nx, Ny, Nz))

def gdown4(t, x, y, z, analytical=False):
    """Returns the spacetime metric"""
    gij = gammadown3(t, x, y, z, analytical=analytical)
    if analytical:
        return sp.Matrix([
            [-1,    0,      0,            0],
            [ 0, gij[0,0], gij[0,1], gij[0,2]],
            [ 0, gij[1,0], gij[1,1], gij[1,2]],
            [ 0, gij[2,0], gij[2,1], gij[2,2]]
        ])
    else:
        ones = np.ones(np.shape(x))
        zeros = np.zeros(np.shape(x))
        return np.array([
            [-ones,      zeros,  zeros,        zeros],
            [zeros, gij[0,0], gij[0,1], gij[0,2]],
            [zeros, gij[1,0], gij[1,1], gij[1,2]],
            [zeros, gij[2,0], gij[2,1], gij[2,2]]
        ])

def data(t, x, y, z):
    """Returns dictionary of Szekeres data"""
    return {'gammadown3': gammadown3(t, x, y, z),
            'rho': rho(t, x, y, z),
            'Kdown3': Kdown3(t, x, y, z)}