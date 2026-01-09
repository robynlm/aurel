"""Symbolic version of AurelCore, using sympy for symbolic calculations.

This class is designed in a similar manner to the AurelCore class,
but it takes in different inputs,
has no cache management,
and there are much fewer quantities available.
"""

import numpy as np
import sympy as sp

from .utils.descriptions import load_symbolic_descriptions

# Load symbolic descriptions from YAML file
symbolic_descriptions = load_symbolic_descriptions()

class AurelCoreSymbolic:
    """Symbolic version of AurelCore, using sympy for symbolic calculations.

    Parameters
    ----------
    coords : list of sympy symbols
        List of coordinates.
    verbose : bool, optional
        If True, print the calculation description. Default is True.
    simplify : bool, optional
        If True, simplify the expressions after calculation. Default is True.

    Attributes
    ----------
    dim : int
        (*int*) - Dimension of the metric tensor.
    data : dict
        (*dict*) - Dictionary to store calculated quantities.
    """

    def __init__(self, coords, verbose=True, simplify=True):
        """Initialize the AurelCoreSymbolic class."""
        self.coords = coords
        self.dim = len(coords)
        self.verbose = verbose
        self.simplify = simplify
        self.data = {}

    def __getitem__(self, key):
        """Get data[key] or compute it if not present."""
        # First check if the key is already cached
        if key in self.data:
            return self.data[key]

        # Dynamically get the function by name
        func = getattr(self, key)

        # Call the function if it takes no additional arguments
        if func.__code__.co_argcount == 1:
            self.data[key] = func()
            # Print the calculation description if available
            if self.verbose:
                print(f"Calculated symbolic {key}: "
                      + symbolic_descriptions[key])
            if self.simplify:
                self.data[key] = sp.simplify(self.data[key])
            return self.data[key]

        # Return the function itself if it requires arguments
        return func

    def gdown(self):
        if self.dim == 4:
            return sp.Matrix([[-1, 0, 0, 0],
                              [ 0, 1, 0, 0],
                              [ 0, 0, 1, 0],
                              [ 0, 0, 0, 1]
                              ])
        elif self.dim == 3:
            return sp.Matrix([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                              ])
        else:
            raise ValueError("Dimension not supported for default gdown.")

    def gup(self):
        return self["gdown"].inv()

    def gdet(self):
        return self["gdown"].det()

    def Gamma_down(self):
        Gamma3 = sp.MutableDenseNDimArray([0]*(self.dim**3),
                                          (self.dim, self.dim, self.dim))
        done = np.zeros((self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if not done[i,j,k]:
                        val = 0
                        for m in range(self.dim):
                            val += (self["gdown"][i, m]
                                    * self["Gamma_udd"][m, j, k])
                        if self.simplify:
                            val = sp.simplify(val)
                        Gamma3[i, j, k] = val
                        done[i, j, k] = 1
                        Gamma3[i, k, j] = Gamma3[i, j, k]
                        done[i, k, j] = 1
        return Gamma3

    def Gamma_udd(self):
        Gamma3 = sp.MutableDenseNDimArray([0]*(self.dim**3),
                                          (self.dim, self.dim, self.dim))
        done = np.zeros((self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if not done[i,j,k]:
                        val = 0
                        for m in range(self.dim):
                            val += self["gup"][i, m] * (
                                sp.diff(self["gdown"][m, k], self.coords[j]) +
                                sp.diff(self["gdown"][m, j], self.coords[k]) -
                                sp.diff(self["gdown"][j, k], self.coords[m])
                            )
                        if self.simplify:
                            val = sp.simplify(0.5 * val)
                        Gamma3[i, j, k] = val
                        done[i, j, k] = 1
                        Gamma3[i, k, j] = Gamma3[i, j, k]
                        done[i, k, j] = 1
        return Gamma3

    def Riemann_down(self):
        if "Riemann_uddd" in self.data.keys():
            RiemannD_down = sp.MutableDenseNDimArray(
                [0]*(self.dim**4),
                (self.dim, self.dim, self.dim, self.dim))
            done = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for h in range(self.dim):
                for i in range(self.dim):
                    if h == i:
                        done[h,i,:,:] = 1
                        pass
                    else:
                        for j in range(self.dim):
                            for k in range(self.dim):
                                if j == k:
                                    done[h,i,j,k] = 1
                                    pass
                                else:
                                    if not done[h,i,j,k]:
                                        RiemannD_down[h, i, j, k] = sum(
                                            self["gdown"][h, m]
                                            * self["Riemann_uddd"][m, i, j, k]
                                            for m in range(self.dim)
                                        )
                                        if self.simplify:
                                            Rhjjk = sp.simplify(
                                                RiemannD_down[h, i, j, k])
                                        else:
                                            Rhjjk = RiemannD_down[h, i, j, k]
                                        RiemannD_down[h, i, j, k] = Rhjjk
                                        RiemannD_down[h, i, k, j] = - Rhjjk
                                        RiemannD_down[i, h, j, k] = - Rhjjk
                                        RiemannD_down[i, h, k, j] = Rhjjk
                                        done[h, i, j, k] = 1
                                        done[h, i, k, j] = 1
                                        done[i, h, j, k] = 1
                                        done[i, h, k, j] = 1
                                        RiemannD_down[j, k, h, i] = Rhjjk
                                        RiemannD_down[j, k, i, h] = - Rhjjk
                                        RiemannD_down[k, j, h, i] = - Rhjjk
                                        RiemannD_down[k, j, i, h] = Rhjjk
                                        done[j, k, h, i] = 1
                                        done[j, k, i, h] = 1
                                        done[k, j, h, i] = 1
                                        done[k, j, i, h] = 1
        else:
            RiemannD_down = sp.MutableDenseNDimArray(
                [0]*(self.dim**4), (self.dim, self.dim, self.dim, self.dim))
            done = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if i == j:
                        done[i,j,:,:] = 1
                        pass
                    else:
                        for k in range(self.dim):
                            for h in range(self.dim):
                                if k == h:
                                    done[i,j,k,h] = 1
                                    pass
                                else:
                                    if not done[i, j, k, h]:
                                        term1 = sp.diff(
                                            self["Gamma_down"][i, j, h],
                                            self.coords[k])
                                        term2 = sp.diff(
                                            self["Gamma_down"][i, j, k],
                                            self.coords[h])
                                        term3 = sum(
                                            self["Gamma_down"][i, k, m]
                                            *self["Gamma_udd"][m, j, h]
                                            for m in range(self.dim))
                                        term4 = sum(
                                            self["Gamma_down"][i, h, m]
                                            *self["Gamma_udd"][m, j, k]
                                            for m in range(self.dim))
                                        if self.simplify:
                                            Rijkh = sp.simplify(
                                                sp.simplify(term1 - term2)
                                                + sp.simplify(term3 - term4))
                                        else:
                                            Rijkh = (term1 - term2
                                                     + term3 - term4)
                                        RiemannD_down[i, j, k, h] = Rijkh
                                        RiemannD_down[i, j, h, k] = - Rijkh
                                        RiemannD_down[j, i, k, h] = - Rijkh
                                        RiemannD_down[j, i, h, k] = Rijkh
                                        done[i, j, k, h] = 1
                                        done[i, j, h, k] = 1
                                        done[j, i, k, h] = 1
                                        done[j, i, h, k] = 1

                                        RiemannD_down[k, h, i, j] = Rijkh
                                        RiemannD_down[k, h, j, i] = - Rijkh
                                        RiemannD_down[h, k, i, j] = - Rijkh
                                        RiemannD_down[h, k, j, i] = Rijkh
                                        done[k, h, i, j] = 1
                                        done[k, h, j, i] = 1
                                        done[h, k, i, j] = 1
                                        done[h, k, j, i] = 1
        return RiemannD_down

    def Riemann_uddd(self):
        Riemann3_up = sp.MutableDenseNDimArray(
            [0]*(self.dim**4), (self.dim, self.dim, self.dim, self.dim))
        done = np.zeros((self.dim, self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    done[i,j,:,:] = 1
                    pass
                else:
                    for k in range(self.dim):
                        for h in range(self.dim):
                            if k == h:
                                done[i,j,k,h] = 1
                                pass
                            else:
                                if not done[i, j, k, h]:
                                    term1 = sp.diff(self["Gamma_udd"][i, j, h],
                                                    self.coords[k])
                                    term2 = sp.diff(self["Gamma_udd"][i, j, k],
                                                    self.coords[h])
                                    term3 = sum(self["Gamma_udd"][i, k, m]
                                                *self["Gamma_udd"][m, j, h]
                                                for m in range(self.dim))
                                    term4 = sum(self["Gamma_udd"][i, h, m]
                                                *self["Gamma_udd"][m, j, k]
                                                for m in range(self.dim))
                                    if self.simplify:
                                        Rijkh = sp.simplify(
                                            sp.simplify(term1 - term2)
                                            + sp.simplify(term3 - term4))
                                    else:
                                        Rijkh = (term1 - term2
                                                 + term3 - term4)
                                    Riemann3_up[i, j, k, h] = Rijkh
                                    Riemann3_up[i, j, h, k] = - Rijkh
                                    done[i, j, k, h] = 1
                                    done[i, j, h, k] = 1
        return Riemann3_up

    def Ricci_down(self):
        Ricci3_down = sp.MutableDenseNDimArray(
            [0]*(self.dim**2), (self.dim, self.dim))
        done = np.zeros((self.dim, self.dim))
        if "Riemann_uddd" in self.data.keys():
            for i in range(self.dim):
                for j in range(self.dim):
                    if not done[i,j]:
                        val = 0
                        for k in range(self.dim):
                            val += self["Riemann_uddd"][k, i, k, j]
                        if self.simplify:
                            val = sp.simplify(val)
                        Ricci3_down[i, j] = val
                        done[i,j] = 1
                        Ricci3_down[j, i] = Ricci3_down[i, j]
                        done[j,i] = 1
        else:
            for i in range(self.dim):
                for j in range(self.dim):
                    if not done[i,j]:
                        val = 0
                        for k in range(self.dim):
                            if i == k or j == k:
                                pass
                            else:
                                term1 = sp.diff(self["Gamma_udd"][k, i, j],
                                                self.coords[k])
                                term2 = sp.diff(self["Gamma_udd"][k, i, k],
                                                self.coords[j])
                                term3 = sum(self["Gamma_udd"][k, k, m]
                                            *self["Gamma_udd"][m, i, j]
                                            for m in range(self.dim))
                                term4 = sum(self["Gamma_udd"][k, j, m]
                                            *self["Gamma_udd"][m, i, k]
                                            for m in range(self.dim))
                                if self.simplify:
                                    val += sp.simplify(
                                        sp.simplify(term1 - term2)
                                        + sp.simplify(term3 - term4))
                                else:
                                    val += (term1 - term2 + term3 - term4)
                        if self.simplify:
                            val = sp.simplify(val)
                        Ricci3_down[i, j] = val
                        done[i,j] = 1
                        Ricci3_down[j, i] = Ricci3_down[i, j]
                        done[j,i] = 1
        return Ricci3_down

    def RicciS(self):
        Ricci3 = 0
        for i in range(self.dim):
            for j in range(self.dim):
                Ricci3 += self["gup"][i, j] * self["Ricci_down"][i, j]
        return Ricci3

    def Einstein_down(self):
        Einstein_down = sp.MutableDenseNDimArray(
            [0]*(self.dim**2), (self.dim, self.dim))
        done = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if not done[i,j]:
                    Einstein_down[i, j] = ( self["Ricci_down"][i, j]
                                            - 0.5 * self["gdown"][i, j]
                                            * self["RicciS"] )
                    done[i,j] = 1
                    Einstein_down[j, i] = Einstein_down[i, j]
                    done[j,i] = 1
        return Einstein_down

# Update __doc__ of the functions listed in descriptions
for func_name, doc in symbolic_descriptions.items():
    func = getattr(AurelCoreSymbolic, func_name, None)
    if func is not None:
        func.__doc__ = doc
