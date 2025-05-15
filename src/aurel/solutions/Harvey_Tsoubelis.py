"""This is a A.Harvey and T.Tsoubelis solution that describes 
a vacuum Bianchi IV plane wave homogeneous spacetime 
page 191 of 'Dynamical Systems in Cosmology' by J.Wainwright and G.F.R.Ellis
See section 3.5 of 2211.08133"""

import numpy as np
import sympy as sp

def alpha(t, x, y, z):
    """Returns the lapse function"""
    return np.ones(np.shape(x))

def rho(t, x, y, z):
    """Returns the energy density"""
    return np.zeros(np.shape(x))

def press(t, x, y, z):
    """Returns the pressure"""
    return np.zeros(np.shape(x))

def betaup3(t, x, y, z):
    """Returns the shift vector"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((3, Nx, Ny, Nz))

def uup4(t, x, y, z):
    """Fluid 4 velocity"""
    Nx, Ny, Nz = np.shape(x)
    return np.array([np.ones((Nx, Ny, Nz)), 
                     np.zeros((Nx, Ny, Nz)), 
                     np.zeros((Nx, Ny, Nz)), 
                     np.zeros((Nx, Ny, Nz))])

def Tdown4(t, x, y, z):
    """Energy stress tensor"""
    Nx, Ny, Nz = np.shape(x)
    return np.zeros((4, 4, Nx, Ny, Nz))

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

def gammadown3(t, x, y, z, analytical=False):
    """Returns the spatial metric"""
    if analytical:
        ex = sp.exp(x)
        B = x + sp.log(t)
        return sp.Matrix([
            [ t**2,      0,            0],
            [    0,   t*ex,       t*ex*B],
            [    0, t*ex*B, t*ex*(B*B+1)]
        ])
    else:
        B = (x+np.log(t))
        ex = np.exp(x)
        ones = np.ones(np.shape(x))
        zeros = np.zeros(np.shape(x))
        return np.array([
            [(t*t)*ones,  zeros,        zeros],
            [     zeros,   t*ex,       t*ex*B],
            [     zeros, t*ex*B, t*ex*(B*B+1)]])

def Kdown3(t, x, y, z):
    """Returns the extrinsic curvature"""
    B = (x+np.log(t))
    ex = np.exp(x)
    ones = np.ones(np.shape(x))
    zeros = np.zeros(np.shape(x))
    dtB = 1/t  # Time derivative of B function
    # Time derivative of metric
    dtgammadown3 = np.array([
        [2*t*ones,           zeros,                       zeros],
        [   zeros,              ex,             ex*B + t*ex*dtB],
        [   zeros, ex*B + t*ex*dtB, ex*(B*B+1) + t*ex*(2*dtB*B)]])
    return (-1/2)*dtgammadown3

def data(t, x, y, z):
    """Returns dictionary of Harvey Tsoubelis data"""
    return {'gammadown3': gammadown3(t, x, y, z),
            'Kdown3': Kdown3(t, x, y, z),
            'Tdown4': Tdown4(t, x, y, z)}