""" FLRW spacetime with an Einstein-de Sitter model"""

import numpy as np

h = 0.6737 # dimensionless Hubble constant
c = 1.0 # speed of light so Length = Time
G = 1.0 # gravitational constant so Mass = Length = Time
a_today = 1.0 # scale factor today
Omega_m_EdS = 1.0 # matter density parameter today

w = 0.0 # for equation of state: w = p / rho
        # 0 for dust or 1/3 for radiation

kappa = 8 * np.pi # Einstein's gravitational constant
Hprop_today = (h * c) / 2997.9 # Hubble constant
t_today = 2.0 / (3.0 * Hprop_today * (1.0 + w)) # time today in EdS universe
Lambda = 0.0 # cosmological constant
Omega_l_today = 0.0 # dark energy density parameter today

def a(t):
    """Scale factor"""
    return a_today * ((t / t_today) ** (2 / (3 * (1.0 + w))))
    
def t_func_a(a):
    """Proper time from scale factor"""
    return t_today * ((a / a_today) ** ((3 * (1.0 + w)) / 2))

def Hprop(t):
    """Proper Hubble function"""
    return Hprop_today * t_today / t
    
def t_func_Hprop(Hprop):
    """Proper time from Hubble"""
    return Hprop_today * t_today / Hprop

def Omega_m(t):
    """Matter density parameter"""
    return Omega_m_EdS

def an_today(t):
    """Scale factor normalised by a(z=0)"""
    return a(t) / a_today

def redshift(t):
    """Redshift"""
    return -1 + ( a_today / a(t) )
    
def a_func_z(z):
    """Scale factor from redshift"""
    return a_today / (1 + z)
    
def t_func_z(z):
    """Proper time from redshift"""
    return t_func_a(a_func_z(z))

def Hconf(t):
    """Conformal Hubble function"""
    return a(t) * Hprop(t)

def fL(t):
    """Growth index = d ln (delta) / d ln (a)"""
    return Omega_m(t) ** (6 / 11)

def rho(t):
    """Energy density"""
    return (3 * Omega_m(t) * Hprop(t)**2) / kappa

def press(t):
    """Pressure"""
    return w * rho(t)

def alpha(t, x, y, z):
    """Lapse"""
    return np.ones(np.shape(x))

def betaup3(t, x, y, z):
    """Shift"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((3, Nx, Ny, Nz))

def gammadown3(t, x, y, z):
    """Spatial metric"""
    a2 = (a(t)**2) * np.ones(np.shape(x))
    zeros = np.zeros(np.shape(x))
    return np.array([
        [   a2, zeros, zeros],
        [zeros,    a2, zeros],
        [zeros, zeros,    a2]])

def Kdown3(t, x, y, z):
    """Extrinsic curvature"""
    return - gammadown3(t, x, y, z) * Hprop(t)

