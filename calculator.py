import numpy as np

R = 8.31432 # gas const[J/mol.K]
M = 28.9644e-3 # molar mass of dry air[kg/mol]
Na = 6.022e23 # avogadro const[molec/mol]
Cv = 717. # specific heat of dry air at constant volume[J/kg.K]
Cp = Cv + R/M # specific heat of dry air at constant pressure[J/kg.K]
kB = R/Na # Boltzmann constant [J/K]
g0 = 9.80665 # gravitational acceleration constant [m/sec2]
h = 6.63e-34 # Plank's constant [J.sec]
c = 299792458. # m/s

def ept(T,P,P0=1000.):
    '''
    ept(T[K],P[hpa],P0[hpa,1000 as default])
    return potential temperature
    '''
    return np.r_[T]*(P0/np.r_[P])**(R/M/Cp)

def Q2q(T,theta,Q):
    '''
    return descent rate q from diabatic heating rate Q
    '''
    return (theta/T)*(Q/Cp)

def Q2q2(Q,P,P0=1000.):
    '''
    return descent rate q from diabatic heating rate Q
    '''
    return (P0/P)**(R/M/Cp)*(Q/Cp)

def number_dencity(T,P):
    '''
    P: [hPa]
    T: [K]
    return number dencity [molec/m3]
    '''
    P = P*100. #[hpa->pa]
    return P/(kB*T)

def dencity(T,P):
    '''
    P: [hPa]
    T: [K]
    return mass dencity [kg/m3]
    '''
    P = P*100. #[hpa->pa]
    return P/(R*T)

def geth(p,h0=0, t0=273.,p0=1013.25):
    '''
    return height [m]
    p [hPa]
    '''
    h = h0 + (R*t0*np.log(p/p0))/(-g0*M)
    return h

def nd2du(cnd):
    '''
    convert the unit for the partial column of species in [molecs/m2] to Dobson Unit (DU)
    '''
    return cnd/2.69e20

def atm2pa(var,return_inv=False):
    '''
    convert atm to pa
    '''
    c = 101325.
    if return_inv:
        return var*c
    return var/c

def planck(freq,T):
    '''
    planck function [Jm-2 --> Wsm-2]
    '''
    bnu = 2*h*freq**3/c**2/(np.exp(h*freq/kB/T)-1)
    return bnu

def brightness_temperature(freq,T,alpha=1):
    '''
    Brightness Temperature [K]
    '''
    #tb = planck(freq,T)*c**2 / (2*kB * freq**2)
    emis = alpha # LTE
    tb = h*freq/kB / np.log(1 + (np.exp(h*freq/(kB*T)) - 1)/emis)
    return tb
