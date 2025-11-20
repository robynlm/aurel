"""
finitedifference.py

This module provides finite difference schemes for computing spatial
derivatives of scalar or tensor fields on 3D grids.

It contains the following classes and functions:

- 2nd, 4th, 6th, and 8th order finite difference schemes for backward,
  centered, and forward differences.

- FiniteDifference: A class that applies finite difference schemes to
  3D data grids.

  - Provides Cartesian and Spherical coordinates, and function to convert
    between them.
  - Functions for computing the spatial derivatives of rank 0, 1, and 2
    tensors along the x, y, and z axes.
  - Function for removing points affected by the boundary condition.
"""

import numpy as np
from . import maths

###############################################################################
# Finite differencing schemes.
###############################################################################
# FD coefficients listed in doi:10.1090/S0025-5718-1988-0935077-0

# 2nd order
def fd2_backward(f, i, inverse_dx):
    """2nd order backward finite difference scheme."""
    return ((3/2) * f[i]
            + (-2) * f[i-1]
            + (1/2) * f[i-2]) * inverse_dx

def fd2_centered(f, i, inverse_dx):
    """2nd order centered finite difference scheme."""
    return ((-1/2) * f[i-1]
            + (1/2) * f[i+1]) * inverse_dx

def fd2_forward(f, i, inverse_dx):
    """2nd order forward finite difference scheme."""
    return ((-3/2) * f[i]
            + (2) * f[i+1]
            + (-1/2) * f[i+2]) * inverse_dx

# 4th order
def fd4_backward(f, i, inverse_dx):
    """4th order backward finite difference scheme."""
    return ((25/12) * f[i]
            + (-4) * f[i-1]
            + (3) * f[i-2]
            + (-4/3) * f[i-3]
            + (1/4) * f[i-4]) * inverse_dx

def fd4_centered(f, i, inverse_dx):
    """4th order centered finite difference scheme."""
    return ((1/12) * f[i-2]
            + (-2/3) * f[i-1]
            + (2/3) * f[i+1]
            + (-1/12) * f[i+2]) * inverse_dx

def fd4_forward(f, i, inverse_dx):
    """4th order forward finite difference scheme."""
    return ((-25/12) * f[i]
            + (4) * f[i+1]
            + (-3) * f[i+2]
            + (4/3) * f[i+3]
            + (-1/4) * f[i+4]) * inverse_dx
    
# 6th order
def fd6_backward(f, i, inverse_dx):
    """6th order backward finite difference scheme."""
    return ((49/20) * f[i]
            + (-6) * f[i-1]
            + (15/2) * f[i-2]
            + (-20/3) * f[i-3]
            + (15/4) * f[i-4]
            + (-6/5) * f[i-5]
            + (1/6) * f[i-6]) * inverse_dx

def fd6_centered(f, i, inverse_dx):
    """6th order centered finite difference scheme."""
    return ((-1/60) * f[i-3]
            + (3/20) * f[i-2]
            + (-3/4) * f[i-1]
            + (3/4) * f[i+1]
            + (-3/20) * f[i+2]
            + (1/60) * f[i+3]) * inverse_dx

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
    return np.array([func(farray, i, idx) for i in np.arange(imin, imax)])
    
def map1(func, f):
    """Map a function over the indice of a rank 1 tensor."""
    dimj = np.shape(f)[0]
    return np.array([func(f[j]) for j in range(dimj)])

def map2(func, f):
    """Map a function over the two indices of a rank 2 tensor."""
    dimk = np.shape(f)[0]
    dimj = np.shape(f)[1]
    return np.array([[func(f[k, j]) 
                      for j in range(dimj)] 
                      for k in range(dimk)])

def map3(func, f):
    """Map a function over the three indices of a rank 3 tensor."""
    dimk = np.shape(f)[0]
    dimj = np.shape(f)[1]
    dimi = np.shape(f)[2]
    return np.array([[[func(f[k, j, i]) 
                       for i in range(dimi)]
                       for j in range(dimj)] 
                       for k in range(dimk)])

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
        2, 4, 6 or 8 order of FD schemes used.

    Attributes
    ----------
    xarray, yarray, zarray : numpy.ndarray
        (*numpy.ndarray*) - 1D arrays of x, y, and z coordinates.
    xmin, ymin, zmin : float
        (*float*) - Minimum x, y, and z coordinates.
    xmax, ymax, zmax : float
        (*float*) - Maximum x, y, and z coordinates.
    Nx, Ny, Nz : int
        (*int*) - Number of data points in x, y, and z directions.
    ixcenter, iycenter, izcenter : int
        (*int*) - Indexes of the x, y, and z coordinates closest to zero.
    x, y, z : numpy.ndarray
        (*numpy.ndarray*) - 3D arrays of x, y, and z coordinates.
    cartesian_coords : numpy.ndarray
        (*numpy.ndarray*) - 3D array of x, y, and z coordinates.
    r, phi, theta : numpy.ndarray
        (*numpy.ndarray*) - 3D arrays of radius, inclination/polar and azimuth 
        coordinates.
    spherical_coords : numpy.ndarray
        (*numpy.ndarray*) - 3D array of radius, inclination/polar and azimuth 
        coordinates.
    mask_len : int
        (*int*) - Length of the finite difference mask.
    """
    def __init__(
            self,
            param, 
            boundary='no boundary', 
            fd_order=4,
            verbose=True,
            veryverbose=False):
        """Initialize the FiniteDifference class."""

        self.param = param
        self.boundary = boundary
        self.fd_order = fd_order
        self.verbose = verbose
        self.veryverbose = veryverbose
        
        self.dx = self.param['dx']
        self.dy = self.param['dy']
        self.dz = self.param['dz']
        self.inverse_dx = 1 / self.param['dx']
        self.inverse_dy = 1 / self.param['dy']
        self.inverse_dz = 1 / self.param['dz']

        self.xmin = self.param['xmin']
        self.ymin = self.param['ymin']
        self.zmin = self.param['zmin']

        self.xarray = np.arange(
            self.param['xmin'], 
            self.param['xmin'] + self.param['Nx'] * self.param['dx'], 
            self.param['dx'])
        self.yarray = np.arange(
            self.param['ymin'], 
            self.param['ymin'] + self.param['Ny'] * self.param['dy'], 
            self.param['dy'])
        self.zarray = np.arange(
            self.param['zmin'], 
            self.param['zmin'] + self.param['Nz'] * self.param['dz'], 
            self.param['dz'])
        
        self.xmax = self.xarray[-1]
        self.ymax = self.yarray[-1]
        self.zmax = self.zarray[-1]
        
        self.Nx = len(self.xarray)
        self.Ny = len(self.yarray)
        self.Nz = len(self.zarray)
        
        self.ixcenter = np.argmin(abs(self.xarray))
        self.iycenter = np.argmin(abs(self.yarray))
        self.izcenter = np.argmin(abs(self.zarray))
        
        self.x, self.y, self.z = np.meshgrid(
            self.xarray, self.yarray, self.zarray, 
            indexing='ij')
        self.cartesian_coords = np.array([self.x, self.y, self.z])

        self.r, self.theta, self.phi = self.cartesian_to_spherical(
            self.x, self.y, self.z)
        self.spherical_coords = np.array([self.r, self.phi, self.theta])

        if self.fd_order == 8:
            if self.verbose:
                print("8th order finite difference schemes are defined")
            self.backward = fd8_backward
            self.centered = fd8_centered
            self.forward = fd8_forward
        elif self.fd_order == 6:
            if self.verbose:
                print("6th order finite difference schemes are defined")
            self.backward = fd6_backward
            self.centered = fd6_centered
            self.forward = fd6_forward
        elif self.fd_order == 2:
            if self.verbose:
                print("2nd order finite difference schemes are defined")
            self.backward = fd2_backward
            self.centered = fd2_centered
            self.forward = fd2_forward
        else:
            self.fd_order = 4
            if self.verbose:
                print("4th order finite difference schemes are defined")
            self.backward = fd4_backward
            self.centered = fd4_centered
            self.forward = fd4_forward
        self.mask_len = int(self.fd_order / 2)

    def d3(self, f, idx, N):
        """Apply the finite difference scheme to the whole data grid."""
        if self.boundary == 'periodic':
            return self.d3_periodic(f, idx, N)
        elif self.boundary =='symmetric':
            return self.d3_symmetric(f, idx, N)
        else:
            return self.d3_onesided(f, idx, N)
    
    def d3_periodic(self, f, idx, N):
        """Apply the finite difference scheme with periodic boundaries."""
        # The grid is extended along the x direction by the 
        # FD mask number of points from the opposite edge.
        flong = np.concatenate((
            f[-self.mask_len:], f, f[:self.mask_len]), 
            axis=0)
        # excluding the edge points.  We retrieve shape (Nx, Ny, Nz).
        return fd_map(
            self.centered, flong, idx, 
            self.mask_len, N+self.mask_len)
    
    def d3_symmetric(self, f, idx, N):
        """Apply the finite difference scheme with symmetric boundaries."""
        # The grid is extended along the x direction by the 
        # FD mask number of points from the opposite edge.
        iend = N - 1
        flong = np.concatenate((
            f[1:1+self.mask_len][::-1], f, f[iend-self.mask_len:iend][::-1]), 
            axis=0)
        return fd_map(
            self.centered, flong, idx, 
            self.mask_len, N+self.mask_len)
    
    def d3_onesided(self, f, idx, N):
        """Apply the finite difference scheme with one-sided boundaries."""
        # Combination of backward centered and forward schemes are used.
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
        return np.concatenate((lhs, central_part, rhs), axis=0)
    
    def d3x(self, f): 
        r"""Derivative along x of a scalar: $\partial_x (f)$."""
        return self.d3(f, self.inverse_dx, self.param['Nx'])
    
    def d3y(self, f):  
        r"""Derivative along y of a scalar: $\partial_y (f)$."""
        # Same as D3x but as we apply the FD schemes in the y direction 
        # we transpose
        f = np.transpose(f, (1, 0, 2))
        dyf = self.d3(f, self.inverse_dy, self.param['Ny'])
        return np.transpose(dyf, (1, 0, 2))
    
    def d3z(self, f):  
        r"""Derivative along z of a scalar: $\partial_z (f)$."""
        # Same as D3x but as we apply the FD schemes in the z direction 
        # we transpose
        f = np.transpose(f, (2, 1, 0))
        dzf = self.d3(f, self.inverse_dz, self.param['Nz'])
        return np.transpose(dzf, (2, 1, 0))
    
    def d3_scalar(self, f):
        r"""Spatial derivatives of a scalar: 
        $\partial_i (f)$."""
        return np.array([self.d3x(f), self.d3y(f), self.d3z(f)])
    
    def d3_rank1tensor(self, f):
        r"""Spatial derivatives of a spatial rank 1 tensor: 
        $\partial_i (f_{j})$ or $\partial_i (f^{j})$."""
        return np.stack(
            [map1(self.d3x, f), 
             map1(self.d3y, f), 
             map1(self.d3z, f)], axis=0)
    
    def d3x_rank1tensor(self, f):
        r"""Spatial derivatives of a spatial rank 1 tensor: 
        $\partial_x (f_{j})$ or $\partial_x (f^{j})$."""
        return map1(self.d3x, f)
    
    def d3y_rank1tensor(self, f):
        r"""Spatial derivatives of a spatial rank 1 tensor: 
        $\partial_y (f_{j})$ or $\partial_y (f^{j})$."""
        return map1(self.d3y, f)
    
    def d3z_rank1tensor(self, f):
        r"""Spatial derivatives of a spatial rank 1 tensor: 
        $\partial_z (f_{j})$ or $\partial_z (f^{j})$."""
        return map1(self.d3z, f)
    
    def d3_rank2tensor(self, f):
        r"""Spatial derivatives of a spatial rank 2 tensor: 
        $\partial_i (f_{kj})$ or $\partial_i (f^{kj})$ 
        or $\partial_i (f^{k}_{j})$."""
        return np.array(
            [self.d3x_rank2tensor(f),
             self.d3y_rank2tensor(f),
             self.d3z_rank2tensor(f)])
    
    def d3x_rank2tensor(self, f):
        r"""Spatial derivatives along x of a spatial rank 2 tensor: 
        $\partial_x (f_{kj})$ or $\partial_x (f^{kj})$ 
        or $\partial_x (f^{k}_{j})$."""
        return map2(self.d3x, f)
    
    def d3y_rank2tensor(self, f):
        r"""Spatial derivatives along y of a spatial rank 2 tensor: 
        $\partial_y (f_{kj})$ or $\partial_y (f^{kj})$ 
        or $\partial_y (f^{k}_{j})$."""
        return map2(self.d3y, f)
    
    def d3z_rank2tensor(self, f):
        r"""Spatial derivatives along z of a spatial rank 2 tensor: 
        $\partial_z (f_{kj})$ or $\partial_z (f^{kj})$ 
        or $\partial_z (f^{k}_{j})$."""
        return map2(self.d3z, f)
    
    def d3_rank3tensor(self, f):
        r"""Spatial derivatives of a spatial rank 3 tensor."""
        return np.array(
            [self.d3x_rank3tensor(f),
             self.d3y_rank3tensor(f),
             self.d3z_rank3tensor(f)])
    
    def d3x_rank3tensor(self, f):
        r"""Spatial derivatives along x of a spatial rank 3 tensor."""
        return map3(self.d3x, f)
    
    def d3y_rank3tensor(self, f):
        r"""Spatial derivatives along y of a spatial rank 3 tensor."""
        return map3(self.d3y, f)
    
    def d3z_rank3tensor(self, f):
        r"""Spatial derivatives along z of a spatial rank 3 tensor."""
        return map3(self.d3z, f)
    
    def cartesian_to_spherical(self, x, y, z):
        """Convert Cartesian coordinates to Spherical coordinates.
        
        Parameters
        ----------
        x, y, z : numpy.ndarray
            arrays of Cartesian coordinates.
        
        Returns
        -------
        r, theta, phi : numpy.ndarray
            arrays of radius, inclination/polar and azimuth coordinates.
        """
        # radius
        r = np.sqrt(x*x + y*y + z*z)

        # azimuth -pi to pi
        phi = np.sign(y) * np.arccos(
            maths.safe_division(x, np.sqrt(x*x + y*y)))
        mask = np.logical_and(np.sign(y) == 0.0, np.sign(x)<0)
        phi[mask] = -np.pi

        # inclination 0 to pi
        theta = np.arccos(maths.safe_division(z, r))
        return r, theta, phi
    
    def spherical_to_cartesian(self, r, theta, phi):
        """Convert Spherical coordinates to Cartesian coordinates.
        
        Parameters
        ----------
        r, theta, phi : numpy.ndarray
            arrays of radius, inclination/polar and 
            azimuth coordinates.
        
        Returns
        -------
        x, y, z : numpy.ndarray
            arrays of Cartesian coordinates.
        
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
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
        f = np.copy(finput)
        if isingularity == 'find':
            isx, isy, isz = self.ixcenter, self.iycenter, self.izcenter
        else:
            isx, isy, isz = isingularity
        b = 1 # b for buffer
        if isx is not None:
            f[
                isx-self.mask_len-b: isx+self.mask_len+1+b, 
                isy, isz] = np.nan
        if isy is not None:
            f[
                isx, isy-self.mask_len-b: isy+self.mask_len+1+b, 
                isz] = np.nan
        if isz is not None:
            f[
                isx, isy, 
                isz-self.mask_len-b: isz+self.mask_len+1+b] = np.nan
        return f
    
    def excision2(self, finput, isingularity='find'):
        """Double singularity excision, for when FDs were applied twice."""
        f = np.copy(finput)
        if isingularity == 'find':
            isx, isy, isz = self.ixcenter, self.iycenter, self.izcenter
        else:
            isx, isy, isz = isingularity
        b = 1 # b for buffer
        if (isx is not None) and (isy is not None) and (isz is not None):
            f[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b] = np.nan
            f[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b] = np.nan
            f[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b] = np.nan
            f[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b] = np.nan
        elif (isx is not None) and (isy is not None):
            f[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz] = np.nan
            f[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz] = np.nan
        elif (isx is not None) and (isz is not None):
            f[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy, 
              isz-self.mask_len-b: isz+self.mask_len+1+b] = np.nan
            f[isx-self.mask_len-b: isx+self.mask_len+1+b, 
              isy, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b] = np.nan
        elif (isy is not None) and (isz is not None):
            f[isx, 
              isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz-self.mask_len-b: isz+self.mask_len+1+b] = np.nan
            f[isx, 
              isy-self.mask_len-b: isy+self.mask_len+1+b, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b] = np.nan
        elif (isx is not None):
            f[isx-2*self.mask_len-b: isx+2*self.mask_len+1+b, 
              isy, isz] = np.nan
        elif (isy is not None):
            f[isx, isy-2*self.mask_len-b: isy+2*self.mask_len+1+b, 
              isz] = np.nan
        elif (isz is not None):
            f[isx, isy, 
              isz-2*self.mask_len-b: isz+2*self.mask_len+1+b] = np.nan
        return f

