"""This is a Collins and Stewart 1971 solution that describes
a Bianchi II γ-law perfect fluid homogeneous solution.
See section 3.3 of 2211.08133"""

import numpy as np
import sympy as sp

kappa = 8 * np.pi # Einstein's constant
gamma = 4/3  #dust: 1, radiation: 4/3
p1 = (2-gamma)/(2*gamma)
p2 = (2+gamma)/(4*gamma)
s = np.sqrt((2 - gamma)*(3*gamma-2))

def rho(t, x, y, z):
    """Energy density"""
    return (6 - gamma) / (4 * kappa * (t**2) * (gamma**2))

def press(t, x, y, z):
    """Pressure"""
    return (gamma - 1) * rho(t, x, y, z)

def gdown4(t, x, y, z, analytical=False):
    """Spacetime metric"""
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
    """Spatial metric"""
    if analytical:
        ones = 1
        zeros = 0
    else:
        ones = np.ones(np.shape(x))
        zeros = np.zeros(np.shape(x))
    t2p1 = (t**(2*p1)) * ones
    t2p2 = (t**(2*p2)) * ones
    sz2g = s * z / (2 * gamma)
    if analytical:
        return sp.Matrix([
        [       t2p1,         t2p1 * sz2g, zeros],
        [t2p1 * sz2g, t2p2+t2p1*(sz2g**2), zeros],
        [      zeros,               zeros,  t2p2]
    ])
    else:
        return np.array([
        [       t2p1,         t2p1 * sz2g, zeros],
        [t2p1 * sz2g, t2p2+t2p1*(sz2g**2), zeros],
        [      zeros,               zeros,  t2p2]
    ])

def Kdown3(t, x, y, z):
    """Extrinsic curvature"""
    ones = np.ones(np.shape(x))
    zeros = np.zeros(np.shape(x))
    dtt2p1 = 2 * p1 * (t**(2*p1 - 1)) * ones
    dtt2p2 = 2 * p2 * (t**(2*p2 - 1)) * ones
    sz2g = s * z / (2 * gamma)
    return (-1/2) * np.array([
        [     dtt2p1,             dtt2p1*sz2g,  zeros],
        [dtt2p1*sz2g, dtt2p2+dtt2p1*(sz2g**2),  zeros],
        [      zeros,                   zeros, dtt2p2]])

def data(t, x, y, z):
    """Returns dictionary of Collins Stewart data"""
    return {'gammadown3': gammadown3(t, x, y, z),
            'rho': rho(t, x, y, z),
            'press': press(t, x, y, z),
            'Kdown3': Kdown3(t, x, y, z)}