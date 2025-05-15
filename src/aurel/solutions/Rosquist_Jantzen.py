"""This is a Rosquist and Jantzen solution that describes 
a Bianchi VI tilted γ ̃-law perfect fluid homogeneous solution with vorticity
See section 3.4 of 2211.08133"""

import numpy as np
import sympy as sp

kappa = 8 * np.pi # Einstein's gravitational constant

gamma = 1.22 # needs to be between 6/5 and 1.7169
sign = 1
s = (2 - gamma)/(2*gamma)
q = ((6-5*gamma)*(2-gamma+2*sign*np.sqrt((9*gamma-1)*(gamma-1)))
     /(2*gamma*(35*gamma-36)))
k = np.sqrt(-(3*s+3*q-1)/((s+3*q-1)*(3*s*s+(6*q-1)*s-q*q-q)))
m = np.sqrt(-32*q*q*s/((s-q-1)*(s-q-1)*(3*s+3*q-1)))


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
        zeros = 0
        A = k*k*t*t*(1+m*m)
        B = m*k*(t**(1+s-q))*sp.exp(x)
        C = (t**(2*(s-q)))*sp.exp(2*x)
        D = (t**(2*(s+q)))*sp.exp(-2*x)
    else:
        ones = np.ones(np.shape(x))
        zeros = np.zeros(np.shape(x))
        A = k*k*t*t*(1+m*m)*ones
        B = m*k*(t**(1+s-q))*np.exp(x)
        C = (t**(2*(s-q)))*np.exp(2*x)
        D = (t**(2*(s+q)))*np.exp(-2*x)
    if analytical:
        return sp.Matrix([
            [    A,     B, zeros],
            [    B,     C, zeros],
            [zeros, zeros,     D]
    ])
    else:
        return np.array([
            [    A,     B, zeros],
            [    B,     C, zeros],
            [zeros, zeros,     D]])
    
def Kdown3(t, x, y, z):
    """Returns the extrinsic curvature"""
    ones = np.ones(np.shape(x))
    zeros = np.zeros(np.shape(x))
    B = (1+s-q)*m*k*(t**(s-q))*np.exp(x)
    return (-1/2)*np.array([
        [2*k*k*t*(1+m*m)*ones, B, zeros],
        [B, (2*(s-q))*(t**(2*(s-q)-1))*np.exp(2*x), zeros],
        [zeros, zeros, (2*(s+q))*(t**(2*(s+q)-1))*np.exp(-2*x)]])

def Tdown4(t, x, y, z):
    """Returns the energy-stress tensor"""
    Box_zero = np.zeros(np.shape(x))
    Box_ones = np.ones(np.shape(x))
    udown = np.array([-Box_ones, Box_zero, Box_zero, Box_zero])
    u_axu_b = np.einsum('a..., b... -> ab...', udown, udown)
    hdown = gdown4(t, x, y, z) + u_axu_b
    rho = -(4 + k*k*(4*q*q + m*m*((1 + q - s)**2) 
                     - 4*s*(2 + s)))/(4*k*k*kappa*t*t)
    p = (4 - k*k*(12*q*q + 3*m*m*((1 + q - s)**2) 
                  + 4*s*(-2 + 5*s)))/(12*k*k*kappa*t*t)
    qx = Box_ones*(-4*q+m*m*(-1-q+s))/(2*kappa*t)
    qy = -(np.exp(x)*m*(1+q-s)*(t**(-2-q+s)))/(2*k*kappa)
    qdown = np.array([Box_zero, qx, qy, Box_zero])
    pixx = Box_ones*(-8 - 3*k*k*m*m*m*m*((1 + q - s)**2) - 8*k*k*(-1 + s)*s 
                     + m*m*(4 - k*k*(3 + 3*q*q - 6*q*(-1 + s) 
                                     - 14*s + 11*s*s)))/(6*kappa)
    pixy = -(np.exp(x)*m*(
        -4 + k*k*(3 + 6*q + 3*q*q + 3*m*m*((1 + q - s)**2) - 8*s + 5*s*s))
        *(t**(-1 - q + s)))/(6*k*kappa)
    piyy = (np.exp(2*x)
            *(4 - k*k*(3*m*m*((1 + q - s)**2) + 4*(1 + 3*q - s)*s))
            *(t**(-2 - 2*q + 2*s))/(6*k*k*kappa))
    pizz = (np.exp(-2*x)*(2 + 2*k*k*s*(-1 + 3*q + s))
            *(t**(2*(-1 + q + s)))/(3*k*k*kappa))
    pidown = np.array([[Box_zero, Box_zero, Box_zero, Box_zero],
                    [Box_zero, pixx, pixy, Box_zero],
                    [Box_zero, pixy, piyy, Box_zero],
                    [Box_zero, Box_zero, Box_zero, pizz]])
    return (
        rho * u_axu_b
        + p * hdown
        + np.einsum('a..., b... -> ab...', qdown, udown)
        + np.einsum('b..., a... -> ab...', qdown, udown)
        + pidown)

def data(t, x, y, z):
    """Returns dictionary of Roquist Jantzen data"""
    return {'gammadown3': gammadown3(t, x, y, z),
            'Kdown3': Kdown3(t, x, y, z),
            'Tdown4': Tdown4(t, x, y, z)}