"""Analytic version of AurelCore, using sympy for symbolic calculations.

This class is designed in a similar manner to the AurelCore class, 
but it takes in different inputs,
has no cache management,
and there are much fewer quantities available.
"""

import sympy as sp
import numpy as np

analytic_descriptions = {
    "gdown": "Metric tensor in the down index form (need to input)",
    "gup": "Metric tensor in the up index form",
    "Gamma_udd": "Christoffel symbols in the up-down-down index form",
    "Gamma_down": "Christoffel symbols in the down index form",
    "Riemann_down": "Riemann curvature tensor in the down index form",
    "Riemann_uddd": ("Riemann curvature tensor in the up-down-down-down"
                     + " index form"),
    "Ricci_down": "Ricci curvature tensor in the down index form",
    "RicciS": "Ricci scalar",
}

class AurelCoreAnalytic():
    """Analytic version of AurelCore, using sympy for symbolic calculations.
        
    Parameters
    ----------
    coords : list of sympy symbols
        List of coordinates.
    g : sympy.Matrix
        Metric tensor in the down index form.
    verbose : bool, optional
        If True, print the calculation description. Default is True.
    
    Attributes
    ----------
    dim : int
        (*int*) - Dimension of the metric tensor.
    data : dict
        (*dict*) - Dictionary to store calculated quantities.
    """
    def __init__(self, coords, g, verbose=True):
        """Initialize the AurelCoreAnalytic class."""
        self.coords = coords
        self.dim = len(coords)
        self.verbose = verbose
        self.data = {"gdown": g}

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
                print(f"Calculated analytic {key}: " 
                      + analytic_descriptions[key])
            return self.data[key]

        # Return the function itself if it requires arguments
        return func
    
    def gup(self):
        return self["gdown"].inv()
    
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
                        Gamma3[i, j, k] = sp.simplify(val)
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
                        Gamma3[i, j, k] = sp.simplify(0.5 * val)
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
            for l in range(self.dim):
                for i in range(self.dim):
                    if l == i:
                        done[l,i,:,:] = 1
                        pass
                    else:
                        for j in range(self.dim):
                            for k in range(self.dim):
                                if j == k:
                                    done[l,i,j,k] = 1
                                    pass
                                else:
                                    if not done[l,i,j,k]:
                                        print(l, i, j, k, end= ",  ")
                                        RiemannD_down[l, i, j, k] = sum(
                                            self["gdown"][l, m] 
                                            * self["Riemann_uddd"][m, i, j, k]
                                            for m in range(self.dim)
                                        )
                                        Rljjk = sp.simplify(
                                            RiemannD_down[l, i, j, k])
                                        RiemannD_down[l, i, j, k] = Rljjk
                                        RiemannD_down[l, i, k, j] = - Rljjk
                                        RiemannD_down[i, l, j, k] = - Rljjk
                                        RiemannD_down[i, l, k, j] = Rljjk
                                        done[l, i, j, k] = 1
                                        done[l, i, k, j] = 1
                                        done[i, l, j, k] = 1
                                        done[i, l, k, j] = 1
                                        RiemannD_down[j, k, l, i] = Rljjk
                                        RiemannD_down[j, k, i, l] = - Rljjk
                                        RiemannD_down[k, j, l, i] = - Rljjk
                                        RiemannD_down[k, j, i, l] = Rljjk
                                        done[j, k, l, i] = 1
                                        done[j, k, i, l] = 1
                                        done[k, j, l, i] = 1
                                        done[k, j, i, l] = 1
            print()
            return RiemannD_down
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
                            for l in range(self.dim):
                                if k == l:
                                    done[i,j,k,l] = 1
                                    pass
                                else:
                                    if not done[i, j, k, l]:
                                        print(i, j, k, l, end= ",  ")
                                        term1 = sp.diff(
                                            self["Gamma_down"][i, j, l], 
                                            self.coords[k])
                                        term2 = sp.diff(
                                            self["Gamma_down"][i, j, k], 
                                            self.coords[l])
                                        term3 = sum(
                                            self["Gamma_down"][i, k, m]
                                            *self["Gamma_udd"][m, j, l] 
                                            for m in range(self.dim))
                                        term4 = sum(
                                            self["Gamma_down"][i, l, m]
                                            *self["Gamma_udd"][m, j, k] 
                                            for m in range(self.dim))
                                        Rijkl = sp.simplify(
                                            sp.simplify(term1 - term2) 
                                            + sp.simplify(term3 - term4))
                                        RiemannD_down[i, j, k, l] = Rijkl
                                        RiemannD_down[i, j, l, k] = - Rijkl
                                        RiemannD_down[j, i, k, l] = - Rijkl
                                        RiemannD_down[j, i, l, k] = Rijkl
                                        done[i, j, k, l] = 1
                                        done[i, j, l, k] = 1
                                        done[j, i, k, l] = 1
                                        done[j, i, l, k] = 1

                                        RiemannD_down[k, l, i, j] = Rijkl
                                        RiemannD_down[k, l, j, i] = - Rijkl
                                        RiemannD_down[l, k, i, j] = - Rijkl
                                        RiemannD_down[l, k, j, i] = Rijkl
                                        done[k, l, i, j] = 1
                                        done[k, l, j, i] = 1
                                        done[l, k, i, j] = 1
                                        done[l, k, j, i] = 1
            print()
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
                        for l in range(self.dim):
                            if k == l:
                                done[i,j,k,l] = 1
                                pass
                            else:
                                if not done[i, j, k, l]:
                                    print(i, j, k, l, end= ",  ")
                                    term1 = sp.diff(self["Gamma_udd"][i, j, l], 
                                                    self.coords[k])
                                    term2 = sp.diff(self["Gamma_udd"][i, j, k], 
                                                    self.coords[l])
                                    term3 = sum(self["Gamma_udd"][i, k, m]
                                                *self["Gamma_udd"][m, j, l] 
                                                for m in range(self.dim))
                                    term4 = sum(self["Gamma_udd"][i, l, m]
                                                *self["Gamma_udd"][m, j, k] 
                                                for m in range(self.dim))
                                    Rijkl = sp.simplify(
                                        sp.simplify(term1 - term2) 
                                        + sp.simplify(term3 - term4))
                                    Riemann3_up[i, j, k, l] = Rijkl
                                    Riemann3_up[i, j, l, k] = - Rijkl
                                    done[i, j, k, l] = 1
                                    done[i, j, l, k] = 1
        print()
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
                        Ricci3_down[i, j] = sp.simplify(val)
                        done[i,j] = 1
                        Ricci3_down[j, i] = Ricci3_down[i, j]
                        done[j,i] = 1
            return Ricci3_down
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
                                val = sp.simplify(sp.simplify(term1 - term2) 
                                                  + sp.simplify(term3 - term4))
                        Ricci3_down[i, j] = sp.simplify(val)
                        done[i,j] = 1
                        Ricci3_down[j, i] = Ricci3_down[i, j]
                        done[j,i] = 1

    
    def RicciS(self):
        Ricci3 = 0
        for i in range(self.dim):
            for j in range(self.dim):
                Ricci3 += self["gup"][i, j] * self["Ricci_down"][i, j]
        return Ricci3
    
# Update __doc__ of the functions listed in descriptions
for func_name, doc in analytic_descriptions.items():
    func = getattr(AurelCoreAnalytic, func_name, None)
    if func is not None:
        func.__doc__ = doc