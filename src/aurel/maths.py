"""
maths.py 

This module contains functions for manipulating rank 2 tensors, including:
 - extracting components or formatting components into matrices
 - computing determinants and inverses
 - symmetrizing or antisymmetrizing tensors
 - safe division 
 - spin weighted spherical harmonics

"""

import numpy as np
import scipy.special as sc

def getcomponents3(f):
    """Extract components of a rank 2 tensor with 3D indices.
    
    This assumes this tensor is symmetric.
    
    Parameters
    ----------
    f : (3, 3, ...) array_like or list of 6 components [xx, xy, xz, yy, yz, zz]
    
    Returns
    -------
    [xx, xy, xz, yy, yz, zz]: list
        Each element is (...) array_like
    """
    if isinstance(f, list):
        return f
    elif isinstance(f, np.ndarray) or isinstance(f, np.ndarray):
        return [f[0, 0], f[0, 1], f[0, 2], 
                f[1, 1], f[1, 2], f[2, 2]]

def getcomponents4(f):
    """Extract components of a rank 2 tensor with 4D indices.
    
    This assumes this tensor is symmetric.
    
    Parameters
    ----------
    f : (4, 4, ...) array_like or list of 10 components [tt, tx, ty, tz, xx, xy, xz, yy, yz, zz]
    
    Returns
    -------
    [tt, tx, ty, tz, xx, xy, xz, yy, yz, zz]: list
            Each element is (...) array_like
    """
    if isinstance(f, list):
        return f
    elif isinstance(f, np.ndarray) or isinstance(f, np.ndarray):
        return [f[0, 0], f[0, 1], f[0, 2], f[0, 3], 
                f[1, 1], f[1, 2], f[1, 3], 
                f[2, 2], f[2, 3], f[3, 3]]

def format_rank2_3(f):
    """Format a rank 2 tensor with 3D indices into a 3x3 array."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    farray = np.array([[xx, xy, xz], 
                        [xy, yy, yz],
                        [xz, yz, zz]])
    return farray

def format_rank2_4(f):
    """Format a rank 2 tensor with 4D indices into a 4x4 array."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    farray = np.array([[tt, tx, ty, tz],
                        [tx, xx, xy, xz], 
                        [ty, xy, yy, yz],
                        [tz, xz, yz, zz]])
    return farray

def determinant3(f):
    """Determinant 3x3 matrice in every position of the data grid."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    return -xz*xz*yy + 2*xy*xz*yz - xx*yz*yz - xy*xy*zz + xx*yy*zz

def determinant4(f):
    """Determinant of a 4x4 matrice in every position of the data grid."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    return (tz*tz*xy*xy - 2*ty*tz*xy*xz + ty*ty*xz*xz 
            - tz*tz*xx*yy + 2*tx*tz*xz*yy - tt*xz*xz*yy 
            + 2*ty*tz*xx*yz - 2*tx*tz*xy*yz - 2*tx*ty*xz*yz 
            + 2*tt*xy*xz*yz + tx*tx*yz*yz - tt*xx*yz*yz 
            - ty*ty*xx*zz + 2*tx*ty*xy*zz - tt*xy*xy*zz 
            - tx*tx*yy*zz + tt*xx*yy*zz)

def inverse3(f):
    """Inverse of a 3x3 matrice in every position of the data grid."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    fup = np.array([[yy*zz - yz*yz, -(xy*zz - yz*xz), xy*yz - yy*xz], 
                     [-(xy*zz - xz*yz), xx*zz - xz*xz, -(xx*yz - xy*xz)],
                     [xy*yz - xz*yy, -(xx*yz - xz*xy), xx*yy - xy*xy]])
    return safe_division(fup, determinant3(f))

def inverse4(f):
    """Inverse of a 4x4 matrice in every position of the data grid."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    fup = np.array([
        [-xz*xz*yy + 2*xy*xz*yz - xx*yz*yz - xy*xy*zz + xx*yy*zz, 
         tz*xz*yy - tz*xy*yz - ty*xz*yz + tx*yz*yz + ty*xy*zz - tx*yy*zz, 
         -tz*xy*xz + ty*xz*xz + tz*xx*yz - tx*xz*yz - ty*xx*zz + tx*xy*zz, 
         tz*xy*xy - ty*xy*xz - tz*xx*yy + tx*xz*yy + ty*xx*yz - tx*xy*yz], 
        [tz*xz*yy - tz*xy*yz - ty*xz*yz + tx*yz*yz + ty*xy*zz - tx*yy*zz, 
         -tz*tz*yy + 2*ty*tz*yz - tt*yz*yz - ty*ty*zz + tt*yy*zz,
         tz*tz*xy - ty*tz*xz - tx*tz*yz + tt*xz*yz + tx*ty*zz - tt*xy*zz, 
         -ty*tz*xy + ty*ty*xz + tx*tz*yy - tt*xz*yy - tx*ty*yz + tt*xy*yz], 
        [-tz*xy*xz + ty*xz*xz + tz*xx*yz - tx*xz*yz - ty*xx*zz + tx*xy*zz, 
         tz*tz*xy - ty*tz*xz - tx*tz*yz + tt*xz*yz + tx*ty*zz - tt*xy*zz, 
         -tz*tz*xx + 2*tx*tz*xz - tt*xz*xz - tx*tx*zz + tt*xx*zz, 
         ty*tz*xx - tx*tz*xy - tx*ty*xz + tt*xy*xz + tx*tx*yz - tt*xx*yz], 
        [tz*xy*xy - ty*xy*xz - tz*xx*yy + tx*xz*yy + ty*xx*yz - tx*xy*yz, 
         -ty*tz*xy + ty*ty*xz + tx*tz*yy - tt*xz*yy - tx*ty*yz + tt*xy*yz, 
         ty*tz*xx - tx*tz*xy - tx*ty*xz + tt*xy*xz + tx*tx*yz - tt*xx*yz, 
         -ty*ty*xx + 2*tx*ty*xy - tt*xy*xy - tx*tx*yy + tt*xx*yy]])
    return safe_division(fup, determinant4(f))

def symmetrise_tensor(fdown):
    """Symmetrise a rank 2 tensor."""
    return (fdown + np.einsum('ab... -> ba...', fdown)) * 0.5

def antisymmetrise_tensor(fdown):
    """Antisymmetrise a rank 2 tensor."""
    return (fdown - np.einsum('ab... -> ba...', fdown)) * 0.5

def safe_division(a, b):
    """Safe division to avoid division by zero, so x/0 = 0."""
    # make sure I'm only working with floats
    if isinstance(a, int):
        a = a*1.0
    elif isinstance(a, np.ndarray):
        if a.dtype == np.int32:
            a = a.astype(np.float32)
        elif a.dtype == np.int64:
            a = a.astype(np.float64)
    if isinstance(b, int):
        b = b*1.0
    elif isinstance(b, np.ndarray):
        if b.dtype == np.int32:
            b = b.astype(np.float32)
        elif b.dtype == np.int64:
            b = b.astype(np.float64)

    # now divide
    if isinstance(b, float):
        if isinstance(a, float):
            c = 0.0 if b==0 else a/b
        else:
            c = np.zeros_like(a) if b==0 else a/b
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.where(b != 0, a / b, 0.0)
    return c

def populate_4Riemann(Riemann_ssss, Riemann_ssst, Riemann_stst):
    """Populate the 4Riemann tensor with R_ssss, R_ssst, and R_stst"""
    Nx, Ny, Nz = np.shape(Riemann_ssss[0,0,0,0])
    R = np.zeros((4, 4, 4, 4, Nx, Ny, Nz))

    # Riemann_ssss part
    R[1:4, 1:4, 1:4, 1:4] = Riemann_ssss

    # Riemann_ssst part
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                # ijk0
                R[i, j, k, 0] =  Riemann_ssst[i-1, j-1, k-1]
                R[i, j, 0, k] = -Riemann_ssst[i-1, j-1, k-1]
                R[k, 0, i, j] =  Riemann_ssst[i-1, j-1, k-1]
                R[0, k, i, j] = -Riemann_ssst[i-1, j-1, k-1]

                # ikj0
                R[i, k, j, 0] =  Riemann_ssst[i-1, k-1, j-1]
                R[i, k, 0, j] = -Riemann_ssst[i-1, k-1, j-1]
                R[j, 0, i, k] =  Riemann_ssst[i-1, k-1, j-1]
                R[0, j, i, k] = -Riemann_ssst[i-1, k-1, j-1]

                # kji0
                R[k, j, i, 0] =  Riemann_ssst[k-1, j-1, i-1]
                R[k, j, 0, i] = -Riemann_ssst[k-1, j-1, i-1]
                R[i, 0, k, j] =  Riemann_ssst[k-1, j-1, i-1]
                R[0, i, k, j] = -Riemann_ssst[k-1, j-1, i-1]

    # Riemann_stst part
    for i in range(1, 4):
        for j in range(1, 4):
            R[i, 0, j, 0] =  Riemann_stst[i-1, j-1]
            R[i, 0, 0, j] = -Riemann_stst[i-1, j-1]
            R[0, i, 0, j] =  Riemann_stst[i-1, j-1]
            R[0, i, j, 0] = -Riemann_stst[i-1, j-1]
    
    # remaining terms are all zero
    return R

def factorial(n):
    """Returns the factorial of n"""
    if n<=1:
        return 1
    else:
        return sc.factorial(n)
    
def sYlm(s, l, m, theta, phi):
    """Spin-weighted spherical harmonics ${}_sY_{lm}$, Eq 3.1 of https://doi.org/10.1063/1.1705135
    
    Parameters
    ----------
    s : int
        Spin weight.
    l : int
        Degree.
    m : int
        Order.
    theta : ndarray
        Inclination/polar angle grid.
    phi : ndarray
        Azimuthal angle grid.
    
    Returns
    -------
    sYlm : ndarray
        Spin-weighted spherical harmonics on the (theta, phi) grid.
    """
    fac = np.sqrt(factorial(l + m) * factorial(l - m) * (2*l + 1)
                  /(factorial(l + s) * factorial(l - s) * 4 * np.pi))
    sumY = 0
    costh2 = np.cos(theta/2)
    sinth2 = np.sin(theta/2)
    for r in range(max(m - s, 0), min(l + m, l - s) + 1):
        cos = costh2**(2*r + s - m)
        sin = sinth2**(2*l - 2*r - s + m)
        sumY += (sc.binom(l-s, r) * sc.binom(l+s, r+s-m) 
            * ((-1)**(l - r - s)) * np.exp(1j * m * phi) * cos * sin)
    return fac * sumY
    
def sYlm_coefficients(s, lmax, f, theta, phi, dtheta_weight, dphi):
    """Coefficients of spin-weighted spherical harmonics decomposition of f
    
    Parameters
    ----------
    s : int
        Spin weight.
    lmax : int
        Maximum l in the expansion.
    f : ndarray
        Spin-weighted function on the (theta, phi) grid.
    theta, phi : ndarray
        Angular grids.
    dtheta_weight : ndarray
        Weights for integration over theta, e.g. including sin(theta), 
        depending on sampling scheme.
    dphi : float
        Step size in phi direction.
    
    Returns
    -------
    alm : dict[l,m]
        Harmonic coefficients.
    """
    alm = {}
    for l in range(lmax+1):
        for m in range(-l, l+1):
            alm[l,m] = np.sum(
                np.conj(sYlm(s, l, m, theta, phi)) 
                * f
                * dtheta_weight * dphi)
    return alm

def sYlm_reconstruct(s, lmax, alm, theta, phi):
    """Reconstruct a spin-weighted function from its harmonic coefficients.
    
    Parameters
    ----------
    s : int
        Spin weight.
    lmax : int
        Maximum l in the expansion.
    alm : dict[l,m]
        Harmonic coefficients.
    theta, phi : ndarray
        Angular grids.
    
    Returns
    -------
    f : ndarray
        Reconstructed spin-weighted function on the (theta, phi) grid.
    """
    f = np.zeros_like(theta, dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            f += alm[l, m] * sYlm(s, l, m, theta, phi)
    return f