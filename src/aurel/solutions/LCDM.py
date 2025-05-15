r"""FLRW spacetime with a $\Lambda$CDM model"""
import numpy as np
import sympy as sp

h = 0.6737 # dimensionless Hubble constant
c = 1.0 # speed of light so Length = Time
G = 1.0 # gravitational constant so Mass = Length = Time
a_today = 1.0 # scale factor today
Omega_m_today = 0.3147 # matter density parameter today

kappa = 8 * np.pi # Einstein's gravitational constant
Omega_l_today = 1 - Omega_m_today # dark energy density parameter today
Hprop_today = (h * c) / 2997.9 # Hubble constant
t_today_EdS = 2 / (3 * Hprop_today) # time today in EdS universe
Lambda = (Omega_l_today * 3 * (Hprop_today ** 2) 
                       / (c ** 2)) # cosmological constant

def a(t, analytical=False):
    """Scale factor"""
    if analytical:
        return (
            a_today 
            * (Omega_m_today / Omega_l_today) ** (1 / 3) 
            * sp.sinh(np.sqrt(Omega_l_today) 
                    * t / t_today_EdS) ** (2 / 3))
    else:
        return (
            a_today 
            * (Omega_m_today / Omega_l_today) ** (1 / 3) 
            * np.sinh(np.sqrt(Omega_l_today) 
                    * t / t_today_EdS) ** (2 / 3))

def Hprop(t):
    """Proper Hubble function"""
    return (
        Hprop_today 
        * np.sqrt( Omega_m_today / (an_today(t) ** 3) 
                  + Omega_l_today ))

def Omega_m(t):
    """Matter density parameter"""
    return (Omega_m_today 
                / ( Omega_m_today 
                   + Omega_l_today * (an_today(t) ** 3) ))

def an_today(t):
    """Scale factor normalised by a(z=0)"""
    return a(t) / a_today

def redshift(t):
    """Redshift"""
    return -1 + ( a_today / a(t) )

def Hconf(t):
    """Conformal Hubble function"""
    return a(t) * Hprop(t)

def fL(t):
    """Growth index = d ln (delta) / d ln (a)"""
    return Omega_m(t) ** (6/11)

def rho(t):
    """Energy density"""
    return (3 * Omega_m(t) * Hprop(t)**2) / kappa

def alpha(t, x, y, z):
    """Lapse"""
    return np.ones(np.shape(x))

def betaup3(t, x, y, z):
    """Shift"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((3, Nx, Ny, Nz))

def gammadown3(t, x, y, z, analytical=False):
    """Spatial metric"""
    a2 = (a(t, analytical=analytical)**2)
    if analytical:
        return sp.Matrix([
            [a2, 0, 0],
            [0, a2, 0],
            [0, 0, a2]])
    else:
        a2 *= np.ones(np.shape(x))
        zeros = np.zeros(np.shape(x))
        return np.array([
            [   a2, zeros, zeros],
            [zeros,    a2, zeros],
            [zeros, zeros,    a2]])

def Kdown3(t, x, y, z):
    """Extrinsic curvature"""
    return - gammadown3(t, x, y, z) * Hprop(t)