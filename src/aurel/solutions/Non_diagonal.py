"""See section 3.2 of 2211.08133"""

import numpy as np
import sympy as sp

kappa = 8 * np.pi # Einstein's gravitational constant
Lambda = 10.0 # Wavelength in the z direction
fq = 2 * np.pi / Lambda

def A(z, analytical=False):
    """Conformal factor"""
    if analytical:
        return 2.3 + 0.2 * sp.sin(fq * z)
    else:
        return 2.3 + 0.2 * np.sin(fq * z)

def dzA(z):
    """Conformal factor 1st derivative"""
    return 0.2 * fq * np.cos(fq * z)

def dzdzA(z):
    """Conformal factor 2nd derivative"""
    return - 0.2 * fq * fq * np.sin(fq * z)

def gammadown3(t, x, y, z, analytical=False):
    """Spatial metric"""
    B = t*A(z, analytical=analytical)
    if analytical:
        ones = 1
        zeros = 0
        return sp.Matrix([
        [B, ones, ones],
        [ones, B, zeros],
        [ones, zeros, B]
    ])
    else:
        ones = np.ones(np.shape(x))
        zeros = np.zeros(np.shape(x))
        return np.array([
        [B, ones, ones],
        [ones, B, zeros],
        [ones, zeros, B]
    ])

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
    
def Kdown3(t, x, y, z):
    """Extrinsic curvature"""
    zeros = np.zeros(np.shape(x))
    return (-1/2)*np.array([[A(z), zeros, zeros],
                            [zeros, A(z), zeros],
                            [zeros, zeros, A(z)]])

def Tdown4(t, x, y, z):
    """Stress-energy tensor, from Einstein's field equations"""
    ones = np.ones(np.shape(x))
    zeros = np.zeros(np.shape(x))
    udown = np.array([-ones, zeros, zeros, zeros])
    u_axu_b = np.einsum('a...,b...->ab...', udown, udown)
    hdown = gdown4(t, x, y, z) + u_axu_b

    # energy density 
    aux0=(8.*((A(z))*((t)*(dzdzA(z)))))+(-2.*(((A(z))**3.)*(3.+(2.*(((t)**3.)*(dzdzA(z)))))))
    aux1=(2.*((t)*(((dzA(z))**2))))+((3.*((((A(z))**2))*(((t)**3.)*(((dzA(z))**2)))))+aux0)
    aux2=((-2.+((((A(z))**2))*(((t)**2))))**-2.)*((3.*(((A(z))**5.)*(((t)**2))))+aux1)
    rho = 0.25*aux2/(A(z)*kappa)
    
    # pressure
    aux0=((A(z))*(8.+(-8.*(((t)**3.)*(dzdzA(z))))))+(((A(z))**3.)*((6.*(((t)**2)))+(4.*(((t)**5.)*(dzdzA(z))))))
    aux1=(-2.*(((t)**3.)*(((dzA(z))**2))))+((-3.*((((A(z))**2))*(((t)**5.)*(((dzA(z))**2)))))+aux0)
    aux2=((-2.+((((A(z))**2))*(((t)**2))))**-2.)*((3.*(((A(z))**5.)*((t)**4.)))+aux1)
    p=((0.0833333*(((t)**-2.)*aux2))/(A(z)))/kappa

    # momentum density
    aux0=(-6.+((((A(z))**2))*(((t)**2))))*(((-2.+((((A(z))**2))*(((t)**2))))**-2.)*(dzA(z)));
    outputx=(0.25*aux0)/kappa;
    aux0=((-2.+((((A(z))**2))*(((t)**2))))**-2.)*((2.+((((A(z))**2))*(((t)**2))))*(dzA(z)));
    outputy=(((0.25*aux0)/(t))/(A(z)))/kappa;
    aux0=((-2.+((((A(z))**2))*(((t)**2))))**-2.)*((-2.+(7.*((((A(z))**2))*(((t)**2)))))*(dzA(z)));
    outputz=(((-0.25*aux0)/(t))/(A(z)))/kappa;
    qdown = np.array([zeros, outputx, outputy, outputz])

    # anistropic pressure
    aux0=(2.*(((A(z))**5.)*(((t)**5.)*(dzdzA(z)))))+(2.*(((A(z))**3.)*(8.+(((t)**3.)*(dzdzA(z))))))
    aux1=(-3.*(((A(z))**4.)*(((t)**5.)*(((dzA(z))**2)))))+((-12.*((A(z))*((t)*(dzdzA(z)))))+aux0)
    aux2=(6.*((t)*(((dzA(z))**2))))+((-7.*((((A(z))**2))*(((t)**3.)*(((dzA(z))**2)))))+aux1);
    outputxx=(t)*aux2;
        
    aux0=(8.*(((A(z))**3.)*(((t)**5.)*(dzdzA(z)))))+((A(z))*(4.+(-16.*(((t)**3.)*(dzdzA(z))))));
    aux1=(8.*(((t)**3.)*(((dzA(z))**2))))+((-12.*((((A(z))**2))*(((t)**5.)*(((dzA(z))**2)))))+aux0);
    outputxy=(A(z))*((3.*(((A(z))**5.)*((t)**4.)))+aux1);
        
    aux0=(2.*(((A(z))**3.)*(((t)**5.)*(dzdzA(z)))))+((A(z))*(4.+(-4.*(((t)**3.)*(dzdzA(z))))));
    aux1=(2.*(((t)**3.)*(((dzA(z))**2))))+((-3.*((((A(z))**2))*(((t)**5.)*(((dzA(z))**2)))))+aux0);
    outputxz=(A(z))*((3.*(((A(z))**5.)*((t)**4.)))+aux1);
        
    aux0=(2.*(((A(z))**3.)*(2.+(((t)**3.)*(dzdzA(z))))))+(2.*(((A(z))**5.)*((((t)**2))*(3.+(((t)**3.)*(dzdzA(z)))))));
    aux1=(-3.*(((A(z))**4.)*(((t)**5.)*(((dzA(z))**2)))))+((-12.*((A(z))*((t)*(dzdzA(z)))))+aux0);
    aux2=(6.*((t)*(((dzA(z))**2))))+((-7.*((((A(z))**2))*(((t)**3.)*(((dzA(z))**2)))))+aux1);
    outputyy=(t)*aux2;
        
    aux0=(-4.*((A(z))*((t)*(dzdzA(z)))))+(2.*(((A(z))**3.)*(2.+(((t)**3.)*(dzdzA(z))))));
    aux1=(2.*((t)*(((dzA(z))**2))))+((-3.*((((A(z))**2))*(((t)**3.)*(((dzA(z))**2)))))+aux0);
    outputyz=3.*((t)*((-2.*(((A(z))**5.)*(((t)**2))))+aux1));
        
    aux0=(2.*(((A(z))**3.)*(2.+(7.*(((t)**3.)*(dzdzA(z)))))))+(((A(z))**5.)*((6.*(((t)**2)))+(-4.*(((t)**5.)*(dzdzA(z))))));
    aux1=(6.*(((A(z))**4.)*(((t)**5.)*(((dzA(z))**2)))))+((-12.*((A(z))*((t)*(dzdzA(z)))))+aux0);
    aux2=(6.*((t)*(((dzA(z))**2))))+((-13.*((((A(z))**2))*(((t)**3.)*(((dzA(z))**2)))))+aux1);
    outputzz=(t)*aux2;

    pidown = np.array([[zeros, zeros, zeros, zeros],
                       [zeros, outputxx, outputxy, outputxz],
                       [zeros, outputxy, outputyy, outputyz],
                       [zeros, outputxz, outputyz, outputzz]])
    At2 = (A(z)*t)**2
    pidown /= 12*kappa*At2*(-2 + At2)**2

    return ( rho * u_axu_b
            + p * hdown
            + np.einsum('a...,b...->ab...', qdown, udown)
            + np.einsum('b...,a...->ab...', qdown, udown)
            + pidown)

def data(t, x, y, z):
    """Returns dictionary of Non-diagonal data"""
    return {'gammadown3': gammadown3(t, x, y, z),
            'Kdown3': Kdown3(t, x, y, z),
            'Tdown4': Tdown4(t, x, y, z)}