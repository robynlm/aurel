"""
This is the Schwarzschild solution in in isotropic coordinates
with maximal slicing.
See https://arxiv.org/pdf/0904.4184
"""

import numpy as np
import sympy as sp
from .. import maths

kappa = 8 * np.pi # Einstein's gravitational constant
M = 1.0 # Mass of the black hole
    
def alpha(t, x, y, z, analytical=False):
    """Returns the lapse function"""
    if analytical:
        r = sp.sqrt(x**2 + y**2 + z**2)
    else:
        r = np.sqrt(x**2 + y**2 + z**2)
    alpha = ((2 * r) - M) / ((2 * r) + M)
    return alpha
    
def betaup3(t, x, y, z):
    """Returns the shift vector"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((3, Nx, Ny, Nz))

def gammadown3(t, x, y, z, analytical=False):
    """Returns the spatial metric"""
    if analytical:
        r = sp.sqrt(x**2 + y**2 + z**2)
        A = (1 + M / (2 * r))**4
        return sp.Matrix([[A, 0, 0],
                          [0, A, 0],
                          [0, 0, A]])

    else:
        r = np.sqrt(x**2 + y**2 + z**2)
        A = (1 + maths.safe_division(M, 2 * r))**4
        zero = np.zeros(np.shape(x))
        return np.array([[A, zero, zero],
                        [zero, A, zero],
                        [zero, zero, A]])

def gdown4(t, x, y, z, analytical=False):
    """Returns the spacetime metric"""
    a2 = alpha(t, x, y, z, analytical=analytical)**2
    gij = gammadown3(t, x, y, z, analytical=analytical)
    if analytical:
        return sp.Matrix([
            [-a2,    0,      0,            0],
            [ 0, gij[0,0], gij[0,1], gij[0,2]],
            [ 0, gij[1,0], gij[1,1], gij[1,2]],
            [ 0, gij[2,0], gij[2,1], gij[2,2]]
        ])
    else:
        ones = np.ones(np.shape(x))
        zeros = np.zeros(np.shape(x))
        return np.array([
            [-a2,      zeros,  zeros,        zeros],
            [zeros, gij[0,0], gij[0,1], gij[0,2]],
            [zeros, gij[1,0], gij[1,1], gij[1,2]],
            [zeros, gij[2,0], gij[2,1], gij[2,2]]
        ])

def Kdown3(t, x, y, z):
    """Returns the extrinsic curvature"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((3, 3, Nx, Ny, Nz))

def Tdown4(t, x, y, z):
    """Returns the energy-stress tensor"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((4, 4, Nx, Ny, Nz))

def data(t, x, y, z):
    """Returns dictionary of Schwarzschild data"""
    return {'alpha': alpha(t, x, y, z),
            'gammadown3': gammadown3(t, x, y, z),
            'Kdown3': Kdown3(t, x, y, z),
            'Tdown4': Tdown4(t, x, y, z)}

def Kretschmann(t, x, y, z):
    """Kretschmann scalar"""
    r = np.sqrt(x**2 + y**2 + z**2)
    rn = r * (1 + maths.safe_division(M, 2 * r))**2
    return maths.safe_division(12 * (2*M)**2, rn**6)

def null_ray_exp_out(t, x, y, z):
    """Outward null ray expansion"""
    r = np.sqrt(x**2 + y**2 + z**2)
    return 8 * (-M + 2 * r)*r/(M + 2 * r)**3
