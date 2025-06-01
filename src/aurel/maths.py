"""
maths.py 

This module contains functions for manipulating rank 2 tensors, including:
 - extracting components or formatting components into matrices, 
 - computing determinants and inverses,
 - symmetrizing or antisymmetrizing tensors.
 - safe division 

"""

import numpy as np
import jax
import jax.numpy as jnp
           
@jax.jit
def getcomponents3(f):
    """Extract components of a rank 2 tensor with 3D indices.
    
    This assumes this tensor is symmetric.
    
    Parameters
    ----------
    f : (3, 3, Nx, Ny, Nz) array_like or list of 6 components [xx, xy, xz, yy, yz, zz]
    
    Returns
    -------
    [xx, xy, xz, yy, yz, zz]: list
        Each element is (Nx, Ny, Nz) array_like
    """
    if isinstance(f, list):
        return f
    elif isinstance(f, np.ndarray) or isinstance(f, jnp.ndarray):
        return [f[0, 0], f[0, 1], f[0, 2], 
                f[1, 1], f[1, 2], f[2, 2]]

@jax.jit
def getcomponents4(f):
    """Extract components of a rank 2 tensor with 4D indices.
    
    This assumes this tensor is symmetric.
    
    Parameters
    ----------
    f : (4, 4, Nx, Ny, Nz) array_like or list of 10 components [tt, tx, ty, tz, xx, xy, xz, yy, yz, zz]
    
    Returns
    -------
    [tt, tx, ty, tz, xx, xy, xz, yy, yz, zz]: list
            Each element is (Nx, Ny, Nz) array_like
    """
    if isinstance(f, list):
        return f
    elif isinstance(f, np.ndarray) or isinstance(f, jnp.ndarray):
        return [f[0, 0], f[0, 1], f[0, 2], f[0, 3], 
                f[1, 1], f[1, 2], f[1, 3], 
                f[2, 2], f[2, 3], f[3, 3]]

@jax.jit
def format_rank2_3(f):
    """Format a rank 2 tensor with 3D indices into a 3x3 array."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    farray = jnp.array([[xx, xy, xz], 
                        [xy, yy, yz],
                        [xz, yz, zz]])
    return farray

@jax.jit
def format_rank2_4(f):
    """Format a rank 2 tensor with 4D indices into a 4x4 array."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    farray = jnp.array([[tt, tx, ty, tz],
                        [tx, xx, xy, xz], 
                        [ty, xy, yy, yz],
                        [tz, xz, yz, zz]])
    return farray

@jax.jit
def determinant3(f):
    """Determinant 3x3 matrice in every position of the data grid."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    return -xz*xz*yy + 2*xy*xz*yz - xx*yz*yz - xy*xy*zz + xx*yy*zz       

@jax.jit
def determinant4(f):
    """Determinant of a 4x4 matrice in every position of the data grid."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    return (tz*tz*xy*xy - 2*ty*tz*xy*xz + ty*ty*xz*xz 
            - tz*tz*xx*yy + 2*tx*tz*xz*yy - tt*xz*xz*yy 
            + 2*ty*tz*xx*yz - 2*tx*tz*xy*yz - 2*tx*ty*xz*yz 
            + 2*tt*xy*xz*yz + tx*tx*yz*yz - tt*xx*yz*yz 
            - ty*ty*xx*zz + 2*tx*ty*xy*zz - tt*xy*xy*zz 
            - tx*tx*yy*zz + tt*xx*yy*zz)

@jax.jit
def inverse3(f):
    """Inverse of a 3x3 matrice in every position of the data grid."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    fup = jnp.array([[yy*zz - yz*yz, -(xy*zz - yz*xz), xy*yz - yy*xz], 
                     [-(xy*zz - xz*yz), xx*zz - xz*xz, -(xx*yz - xy*xz)],
                     [xy*yz - xz*yy, -(xx*yz - xz*xy), xx*yy - xy*xy]])
    return safe_division(fup, determinant3(f))

@jax.jit
def inverse4(f):
    """Inverse of a 4x4 matrice in every position of the data grid."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    fup = jnp.array([
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

@jax.jit
def symmetrise_tensor(fdown):
    """Symmetrise a rank 2 tensor."""
    return (fdown + jnp.einsum('ab... -> ba...', fdown)) * 0.5

@jax.jit
def antisymmetrise_tensor(fdown):
    """Antisymmetrise a rank 2 tensor."""
    return (fdown - jnp.einsum('ab... -> ba...', fdown)) * 0.5

@jax.jit
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
    elif isinstance(a, jnp.ndarray):
        if a.dtype == jnp.int32:
            a = a.astype(jnp.float32)
        elif a.dtype == jnp.int64:
            a = a.astype(jnp.float64)
    if isinstance(b, int):
        b = b*1.0
    elif isinstance(b, np.ndarray):
        if b.dtype == np.int32:
            b = b.astype(np.float32)
        elif b.dtype == np.int64:
            b = b.astype(np.float64)
    elif isinstance(b, jnp.ndarray):
        if b.dtype == jnp.int32:
            b = b.astype(jnp.float32)
        elif b.dtype == jnp.int64:
            b = b.astype(jnp.float64)

    if isinstance(a, jnp.ndarray) or isinstance(b, jnp.ndarray):
        a = jnp.asarray(a)
        b = jnp.asarray(b)

    # now divide
    if isinstance(b, float):
        if isinstance(a, float):
            c = 0.0 if b==0 else a/b
        elif isinstance(a, np.ndarray):
            c = np.zeros_like(a) if b==0 else a/b
        else:
            c = jnp.zeros_like(a) if b==0 else a/b
    elif isinstance(b, np.ndarray):
        c = np.where(b != 0, a / b, 0.0)
    else:
        c = jnp.where(b != 0, a / b, 0.0)
    return c

@jax.jit
def populate_4Riemann(Riemann_ssss, Riemann_ssst, Riemann_stst):
    """Populate the 4Riemann tensor with R_ssss, R_ssst, and R_stst"""
    Nx, Ny, Nz = np.shape(Riemann_ssss[0,0,0,0])
    R = jnp.zeros((4, 4, 4, 4, Nx, Ny, Nz))
    
    # Riemann_ssss part
    R = R.at[1:4, 1:4, 1:4, 1:4].set(Riemann_ssss)

    # Riemann_ssst part
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                # ijk0
                R = R.at[i, j, k, 0].set( Riemann_ssst[i-1, j-1, k-1])
                R = R.at[i, j, 0, k].set(-Riemann_ssst[i-1, j-1, k-1])
                R = R.at[j, i, 0, k].set( Riemann_ssst[i-1, j-1, k-1])
                R = R.at[j, i, k, 0].set(-Riemann_ssst[i-1, j-1, k-1])

                R = R.at[k, 0, i, j].set( Riemann_ssst[i-1, j-1, k-1])
                R = R.at[k, 0, j, i].set(-Riemann_ssst[i-1, j-1, k-1])
                R = R.at[0, k, j, i].set( Riemann_ssst[i-1, j-1, k-1])
                R = R.at[0, k, i, j].set(-Riemann_ssst[i-1, j-1, k-1])

                # ikj0
                R = R.at[i, k, j, 0].set( Riemann_ssst[i-1, k-1, j-1])
                R = R.at[i, k, 0, j].set(-Riemann_ssst[i-1, k-1, j-1])
                R = R.at[k, i, 0, j].set( Riemann_ssst[i-1, k-1, j-1])
                R = R.at[k, i, j, 0].set(-Riemann_ssst[i-1, k-1, j-1])

                R = R.at[j, 0, i, k].set( Riemann_ssst[i-1, k-1, j-1])
                R = R.at[j, 0, k, i].set(-Riemann_ssst[i-1, k-1, j-1])
                R = R.at[0, j, k, i].set( Riemann_ssst[i-1, k-1, j-1])
                R = R.at[0, j, i, k].set(-Riemann_ssst[i-1, k-1, j-1])

                # kji0
                R = R.at[k, j, i, 0].set( Riemann_ssst[k-1, j-1, i-1])
                R = R.at[k, j, 0, i].set(-Riemann_ssst[k-1, j-1, i-1])
                R = R.at[j, k, 0, i].set( Riemann_ssst[k-1, j-1, i-1])
                R = R.at[j, k, i, 0].set(-Riemann_ssst[k-1, j-1, i-1])

                R = R.at[i, 0, k, j].set( Riemann_ssst[k-1, j-1, i-1])
                R = R.at[i, 0, j, k].set(-Riemann_ssst[k-1, j-1, i-1])
                R = R.at[0, i, j, k].set( Riemann_ssst[k-1, j-1, i-1])
                R = R.at[0, i, k, j].set(-Riemann_ssst[k-1, j-1, i-1])
                
    # Riemann_stst part
    for i in range(1, 4):
        for j in range(1, 4):
            R = R.at[i, 0, j, 0].set( Riemann_stst[i-1, j-1])
            R = R.at[i, 0, 0, j].set(-Riemann_stst[i-1, j-1])
            R = R.at[0, i, 0, j].set( Riemann_stst[i-1, j-1])
            R = R.at[0, i, j, 0].set(-Riemann_stst[i-1, j-1])
            
            R = R.at[j, 0, i, 0].set( Riemann_stst[i-1, j-1])
            R = R.at[j, 0, 0, i].set(-Riemann_stst[i-1, j-1])
            R = R.at[0, j, 0, i].set( Riemann_stst[i-1, j-1])
            R = R.at[0, j, i, 0].set(-Riemann_stst[i-1, j-1])
    
    # remaining terms are all zero
    return R