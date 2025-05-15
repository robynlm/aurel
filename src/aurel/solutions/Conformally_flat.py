import numpy as np
import sympy as sp

eps = 2 # arbitraty positive constant
kappa = 8 * np.pi # Einstein's gravitational constant

def Omega(x):
    """Returns the conformal factor"""
    return 1 + eps*(x**2)
# or use Om = sp.Function('Om')(x) for analytic

def dxOmega(x):
    """Returns the 1st derivative of the conformal factor"""
    return 2*eps*x

def dxdxOmega(x):
    """Returns the 2nd derivatire of the conformal factor"""
    return 2*eps

def alpha(t, x, y, z):
    """Lapse"""
    return Omega(x)

def gdown4(t, x, y, z, analytical=False):
    """Spacetime metric"""
    Om2 = Omega(x)**2
    if analytical:
        zeros = 0
        return sp.Matrix([
        [-Om2, zeros, zeros, zeros],
        [zeros, Om2, zeros, zeros],
        [zeros, zeros, Om2, zeros],
        [zeros, zeros, zeros, Om2]
    ])
    else:
        zeros = np.zeros(np.shape(x))
        return np.array([
        [-Om2, zeros, zeros, zeros],
        [zeros, Om2, zeros, zeros],
        [zeros, zeros, Om2, zeros],
        [zeros, zeros, zeros, Om2]
    ])

def gammadown3(t, x, y, z, analytical=False):
    """Spatial metric"""
    Om2 = Omega(x)**2
    if analytical:
        zeros = 0
        return sp.Matrix([
        [Om2, zeros, zeros],
        [zeros, Om2, zeros],
        [zeros, zeros, Om2]
    ])
    else:
        zeros = np.zeros(np.shape(x))
        return np.array([
        [Om2, zeros, zeros],
        [zeros, Om2, zeros],
        [zeros, zeros, Om2]
    ])

def Kdown3(t, x, y, z):
    """Extrinsic curvature"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((3, 3, Nx, Ny, Nz))

def st_RicciS(x):
    """Spacetime Ricci scalar"""
    return -6 * dxdxOmega(x) / (Omega(x)**3)

def Tdown4(t, x, y, z):
    """Stress-energy tensor, from Einstein's field equations"""
    zeros = np.zeros(np.shape(x))
    Om = Omega(x)
    dxOm = dxOmega(x)
    dxdxOm = dxdxOmega(x)
    Rxx = 3.0 * (-Om * dxdxOm + dxOm**2) / Om**2
    Ryy = 1.0 * (-Om * dxdxOm - dxOm**2) / Om**2
    Ricci_down4 = np.array([
        [-Ryy, zeros, zeros, zeros], 
        [zeros, Rxx, zeros, zeros], 
        [zeros, zeros, Ryy, zeros], 
        [zeros, zeros, zeros, Ryy]])
    Gdown4 = Ricci_down4 - (1/2) * st_RicciS(x) * gdown4(t, x, y, z)
    return Gdown4 / kappa

def data(t, x, y, z):
    """Returns dictionary of Collins Stewart data"""
    return {'gammadown3': gammadown3(t, x, y, z),
            'alpha': alpha(t, x, y, z),
            'Kdown3': Kdown3(t, x, y, z),
            'Tdown4': Tdown4(t, x, y, z)}