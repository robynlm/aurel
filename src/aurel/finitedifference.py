"""
finitedifference.py

This module provides finite difference schemes for computing spatial
derivatives of scalar or tensor fields on 3D grids.

It contains the following classes and functions:

- 4th, 6th, and 8th order finite difference schemes for backward,
  centered, and forward differences.

- FiniteDifference: A class that applies finite difference schemes to
  3D data grids.

  - Provides Cartesian and Spherical coordinates.
  - Functions for computing the spatial derivatives of rank 0, 1, and 2
    tensors along the x, y, and z axes.
  - Function for removing points affected by the boundary condition.
"""

import numpy as np
import jax
import jax.numpy as jnp
from . import maths

###############################################################################
# Finite differencing schemes.
###############################################################################

# 4th order
@jax.jit
def fd4_backward(f, i, inverse_dx):
    """4th order backward finite difference scheme."""
    return ((25/12) * f[i]
            + (-4) * f[i-1]
            + (3) * f[i-2]
            + (-4/3) * f[i-3]
            + (1/4) * f[i-4]) * inverse_dx

@jax.jit
def fd4_centered(f, i, inverse_dx):
    """4th order centered finite difference scheme."""
    return ((1/12) * f[i-2]
            + (-2/3) * f[i-1]
            + (2/3) * f[i+1]
            + (-1/12) * f[i+2]) * inverse_dx

@jax.jit
def fd4_forward(f, i, inverse_dx):
    """4th order forward finite difference scheme."""
    return ((-25/12) * f[i]
            + (4) * f[i+1]
            + (-3) * f[i+2]
            + (4/3) * f[i+3]
            + (-1/4) * f[i+4]) * inverse_dx
    
# 6th order
@jax.jit
def fd6_backward(f, i, inverse_dx):
    """6th order backward finite difference scheme."""
    return ((49/20) * f[i]
            + (-6) * f[i-1]
            + (15/2) * f[i-2]
            + (-20/3) * f[i-3]
            + (15/4) * f[i-4]
            + (-6/5) * f[i-5]
            + (1/6) * f[i-6]) * inverse_dx

@jax.jit
def fd6_centered(f, i, inverse_dx):
    """6th order centered finite difference scheme."""
    return ((-1/60) * f[i-3]
            + (3/20) * f[i-2]
            + (-3/4) * f[i-1]
            + (3/4) * f[i+1]
            + (-3/20) * f[i+2]
            + (1/60) * f[i+3]) * inverse_dx
    
@jax.jit
def fd6_forward(f, i, inverse_dx):
    """6th order forward finite difference scheme."""
    return ((-49/20) * f[i]
            + (6) * f[i+1]
            + (-15/2) * f[i+2]
            + (20/3) * f[i+3]
            + (-15/4) * f[i+4]
            + (6/5) * f[i+5]
            + (-1/6) * f[i+6]) * inverse_dx
    
# 8th order 
@jax.jit
def fd8_backward(f, i, inverse_dx):
    """8th order backward finite difference scheme."""
    return ((761/280) * f[i]
            + (-8) * f[i-1]
            + (14) * f[i-2]
            + (-56/3) * f[i-3]
            + (35/2) * f[i-4]
            + (-56/5) * f[i-5]
            + (14/3) * f[i-6]
            + (-8/7) * f[i-7]
            + (1/8) * f[i-8]) * inverse_dx
    
@jax.jit
def fd8_centered(f, i, inverse_dx):
    """8th order centered finite difference scheme."""
    return ((1/280) * f[i-4]
            + (-4/105) * f[i-3]
            + (1/5) * f[i-2]
            + (-4/5) * f[i-1]
            + (4/5) * f[i+1]
            + (-1/5) * f[i+2]
            + (4/105) * f[i+3]
            + (-1/280) * f[i+4]) * inverse_dx
    
@jax.jit
def fd8_forward(f, i, inverse_dx):
    """8th order forward finite difference scheme."""
    return ((-761/280) * f[i]
            + (8) * f[i+1]
            + (-14) * f[i+2]
            + (56/3) * f[i+3]
            + (-35/2) * f[i+4]
            + (56/5) * f[i+5]
            + (-14/3) * f[i+6]
            + (8/7) * f[i+7]
            + (-1/8) * f[i+8]) * inverse_dx

def fd_map(func, farray, idx, imin, imax):
    """Map the finite difference function over the grid."""
    return jax.vmap(
        lambda i: func(farray, i, idx))(
            jnp.arange(imin, imax))

###############################################################################
# Finite differencing class applying the schemes to data grid.
###############################################################################

class FiniteDifference():
    """This class applies the FD schemes to the entire data grid.

    Parameters
    ----------
    param : dict
        Dictionary containing the data grid parameters:

        'xmin', 'ymin', 'zmin': float, minimum x, y, and z coordinates

        'dx', 'dy', 'dz' : float, elementary grid size in x, y, and z direction

        'Nx', 'Ny', 'Nz' : int, number of data points in x, y, and z direction

    boundary : string, default 'no boundary'
        Options are: 'periodic', 'symmetric', or 'no boundary'.
        If 'periodic' or 'symmetric' a centered FD scheme is used,
        otherwise a combination of forward + centered + backward
        FD schemes are used.
    fd_order : int, default 4
        4, 6 or 8 order of FD schemes used.

    Attributes
    ----------
    xarray, yarray, zarray : jax.numpy.ndarray
        (*jax.numpy.ndarray*) - 1D arrays of x, y, and z coordinates.
    ixcenter, iycenter, izcenter : int
        (*int*) - Indexes of the x, y, and z coordinates closest to zero.
    x, y, z : jax.numpy.ndarray
        (*jax.numpy.ndarray*) - 3D arrays of x, y, and z coordinates.
    cartesian_coords : jax.numpy.ndarray
        (*jax.numpy.ndarray*) - 3D array of x, y, and z coordinates.
    r, phi, theta : jax.numpy.ndarray
        (*jax.numpy.ndarray*) - 3D arrays of radius, azimuth, and inclination 
        coordinates.
    spherical_coords : jax.numpy.ndarray
        (*jax.numpy.ndarray*) - 3D array of radius, azimuth, and inclination 
        coordinates.
    mask_len : int
        (*int*) - Length of the finite difference mask.
    """
    def __init__(
            self,
            param, 
            boundary='no boundary', 
            fd_order=4,
            verbose=True):
        """Initialize the FiniteDifference class."""

        self.param = param
        self.boundary = boundary
        self.inverse_dx = 1 / self.param['dx']
        self.inverse_dy = 1 / self.param['dy']
        self.inverse_dz = 1 / self.param['dz']

        self.xarray = jnp.arange(
            self.param['xmin'], 
            self.param['xmin'] + self.param['Nx'] * self.param['dx'], 
            self.param['dx'])
        self.yarray = jnp.arange(
            self.param['ymin'], 
            self.param['ymin'] + self.param['Ny'] * self.param['dy'], 
            self.param['dy'])
        self.zarray = jnp.arange(
            self.param['zmin'], 
            self.param['zmin'] + self.param['Nz'] * self.param['dz'], 
            self.param['dz'])
        
        self.ixcenter = jnp.argmin(abs(self.xarray))
        self.iycenter = jnp.argmin(abs(self.yarray))
        self.izcenter = jnp.argmin(abs(self.zarray))
        
        self.x, self.y, self.z = jnp.meshgrid(
            self.xarray, self.yarray, self.zarray, 
            indexing='ij')
        self.cartesian_coords = jnp.array([self.x, self.y, self.z])

        # radius
        self.r = jnp.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

        # azimuth
        self.phi = jnp.sign(self.y) * jnp.arccos(
            maths.safe_division(
                self.x, jnp.sqrt(self.x*self.x + self.y*self.y)))
        mask = jnp.logical_and(jnp.sign(self.y) == 0.0, 
                               jnp.sign(self.x)<0)
        self.phi = self.phi.at[mask].set(jnp.pi)

        # inclination
        self.theta = jnp.arccos(maths.safe_division(self.z, self.r))
        self.spherical_coords = jnp.array([self.r, self.phi, self.theta])

        if fd_order == 8:
            if verbose:
                print("8th order finite difference schemes are defined")
            self.backward = fd8_backward
            self.centered = fd8_centered
            self.forward = fd8_forward
            self.mask_len = 4
        elif fd_order == 6:
            if verbose:
                print("6th order finite difference schemes are defined")
            self.backward = fd6_backward
            self.centered = fd6_centered
            self.forward = fd6_forward
            self.mask_len = 3
        else:
            if verbose:
                print("4th order finite difference schemes are defined")
            self.backward = fd4_backward
            self.centered = fd4_centered
            self.forward = fd4_forward
            self.mask_len = 2

    def d3(self, f, idx, N):
        """Apply the finite difference scheme to the data grid."""
        if self.boundary == 'periodic':
            # Periodic boundaries are used.
            # The grid is extended along the x direction by the 
            # FD mask number of points from the opposite edge.
            flong = jnp.concatenate((
                f[-self.mask_len:, :, :], 
                f, 
                f[:self.mask_len, :, :]), 
                axis=0)
            # excluding the edge points.  We retrieve shape (Nx, Ny, Nz).
            
            return fd_map(
                self.centered, flong, idx, 
                self.mask_len, N+self.mask_len)
        elif self.boundary =='symmetric':
            # Symmetric boundaries are used.
            # The grid is extended along the x direction by the 
            # FD mask number of points from the opposite edge.
            iend = N - 1
            flong = jnp.concatenate((
                f[1:1+self.mask_len, :, :][::-1, :, :], 
                f, 
                f[iend-self.mask_len:iend, :, :][::-1, :, :]), 
                axis=0)
            return fd_map(
                self.centered, flong, idx, 
                self.mask_len, N+self.mask_len)
        else:
            # There are no periodic boundaries so a combination
            # of backward centered and forward schemes are used.
            # lhs : Apply the forward FD scheme to the edge points in the x
            # direction that can not use the centered FD scheme.
            lhs = fd_map(
                self.forward, f, idx,
                0, self.mask_len)
            # Apply the centered FD scheme to all points not affected
            # by the boundary condition.
            central_part = fd_map(
                self.centered, f, idx, 
                self.mask_len, N - self.mask_len)
            # rhs : Apply the forward FD scheme to the edge points in the x
            # direction that can not use the centered FD scheme.
            rhs = fd_map(
                self.backward, f, idx, 
                N - self.mask_len, N)
            # Concatenate all the points together
            return jnp.concatenate((lhs, central_part, rhs), axis=0)
    
    def d3x(self, f): 
        r"""Derivative along x of a scalar: $\partial_x (f)$."""
        return self.d3(f, self.inverse_dx, self.param['Nx'])
    
    def d3y(self, f):  
        r"""Derivative along y of a scalar: $\partial_y (f)$."""
        # Same as D3x but as we apply the FD schemes in the y direction 
        # we transpose
        f = jnp.transpose(f, (1, 0, 2))
        dyf = self.d3(f, self.inverse_dy, self.param['Ny'])
        return jnp.transpose(dyf, (1, 0, 2))
    
    def d3z(self, f):  
        r"""Derivative along z of a scalar: $\partial_z (f)$."""
        # Same as D3x but as we apply the FD schemes in the z direction 
        # we transpose
        f = jnp.transpose(f, (2, 1, 0))
        dzf = self.d3(f, self.inverse_dz, self.param['Nz'])
        return jnp.transpose(dzf, (2, 1, 0))
    
    def d3_scalar(self, f):
        r"""Spatial derivatives of a scalar: 
        $\partial_i (f)$."""
        return jnp.array([self.d3x(f), self.d3y(f), self.d3z(f)])
    
    def d3_rank1tensor(self, f):
        r"""Spatial derivatives of a spatial rank 1 tensor: 
        $\partial_i (f_{j})$ or $\partial_i (f^{j})$."""
        return jnp.stack(
            [jax.vmap(self.d3x)(f), 
             jax.vmap(self.d3y)(f), 
             jax.vmap(self.d3z)(f)], axis=0)
    
    def d3_rank2tensor(self, f):
        r"""Spatial derivatives of a spatial rank 2 tensor: 
        $\partial_i (f_{kj})$ or $\partial_i (f^{kj})$ 
        or $\partial_i (f^{k}_{j})$."""
        return jnp.array(
            [self.d3x_rank2tensor(f),
             self.d3y_rank2tensor(f),
             self.d3z_rank2tensor(f)])
    
    def d3x_rank2tensor(self, f):
        r"""Spatial derivatives along x of a spatial rank 2 tensor: 
        $\partial_x (f_{kj})$ or $\partial_x (f^{kj})$ 
        or $\partial_x (f^{k}_{j})$."""
        return jax.vmap(
            jax.vmap(self.d3x, in_axes=0),
            in_axes=0)(f)
    
    def d3y_rank2tensor(self, f):
        r"""Spatial derivatives along y of a spatial rank 2 tensor: 
        $\partial_y (f_{kj})$ or $\partial_y (f^{kj})$ 
        or $\partial_y (f^{k}_{j})$."""
        return jax.vmap(
            jax.vmap(self.d3y, in_axes=0),
            in_axes=0)(f)
    
    def d3z_rank2tensor(self, f):
        r"""Spatial derivatives along z of a spatial rank 2 tensor: 
        $\partial_z (f_{kj})$ or $\partial_z (f^{kj})$ 
        or $\partial_z (f^{k}_{j})$."""
        return jax.vmap(
            jax.vmap(self.d3z, in_axes=0),
            in_axes=0)(f)
    
    def cutoffmask(self, f):
        """Remove boundary points, for when FDs were applied once."""
        if len(f.shape) == 1:
            return f[self.mask_len:-self.mask_len]
        elif len(f.shape) == 2:
            return f[self.mask_len:-self.mask_len, 
                     self.mask_len:-self.mask_len]
        elif len(f.shape) == 3:
            return f[self.mask_len:-self.mask_len, 
                    self.mask_len:-self.mask_len, 
                    self.mask_len:-self.mask_len]
    
    def cutoffmask2(self, f):
        """Remove boundary points, for when FDs were applied twice."""
        if len(f.shape) == 1:
            return f[2*self.mask_len:-2*self.mask_len]
        elif len(f.shape) == 2:
            return f[2*self.mask_len:-2*self.mask_len, 
                     2*self.mask_len:-2*self.mask_len]
        elif len(f.shape) == 3:
            return f[2*self.mask_len:-2*self.mask_len, 
                    2*self.mask_len:-2*self.mask_len, 
                    2*self.mask_len:-2*self.mask_len]
    
    def excision(self, finput, isingularity='find'):
        """Excision of the singularity, for when FDs were applied once."""
        f = jnp.copy(finput)
        if isingularity == 'find':
            isx, isy, isz = self.ixcenter, self.iycenter, self.izcenter
        else:
            isx, isy, isz = isingularity
        b = 1 # b for buffer
        if isx is not None:
            f = f.at[
                isx-self.mask_len-b: isx+self.mask_len+1+b, 
                isy, isz].set(jnp.nan)
        if isy is not None:
            f = f.at[
                isx, isy-self.mask_len-b: isy+self.mask_len+1+b, 
                isz].set(jnp.nan)
        if isz is not None:
            f = f.at[
                isx, isy, 
                isz-self.mask_len-b: isz+self.mask_len+1+b].set(jnp.nan)
        return f
    
    def excision2(self, finput, isingularity='find'):
        """Double singularity excision, for when FDs were applied twice."""
        f = jnp.copy(finput)
        if isingularity == 'find':
            isx, isy, isz = self.ixcenter, self.iycenter, self.izcenter
        else:
            isx, isy, isz = isingularity
        b = 1 # b for buffer
        if (isx is not None) and (isy is not None) and (isz is not None):
            f = f.at[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b].set(jnp.nan)
            f = f.at[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b].set(jnp.nan)
            f = f.at[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b].set(jnp.nan)
            f = f.at[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b].set(jnp.nan)
        elif (isx is not None) and (isy is not None):
            f = f.at[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz].set(jnp.nan)
            f = f.at[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz].set(jnp.nan)
        elif (isx is not None) and (isz is not None):
            f = f.at[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy, 
              isz-self.mask_len-b: isz+self.mask_len+1+b].set(jnp.nan)
            f = f.at[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b].set(jnp.nan)
        elif (isy is not None) and (isz is not None):
            f = f.at[isx, 
              isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b].set(jnp.nan)
            f = f.at[isx, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b].set(jnp.nan)
        elif (isx is not None):
            f = f.at[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy, isz].set(jnp.nan)
        elif (isy is not None):
            f = f.at[isx, isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz].set(jnp.nan)
        elif (isz is not None):
            f = f.at[isx, isy, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b].set(jnp.nan)
        return f