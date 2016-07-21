#! /usr/bin/python
# -*- coding: utf-8 -*-
########################
'''
created on 10 July 2016
@Author Kazutoshi Sagi

'''

__version__ = "1.0"
__filename__ = "spectro.py"
__user__ = 'sagi' #operator
__usage__ = None

########################
import numpy as np
import os,sys,copy
from scipy.special import wofz
from calculator import *
########################
ln2 = np.log(2)

def wn2freq(var,return_inv=False):
    '''
    convert wave number [cm-1] to frequency [Hz]
    '''
    if not return_inv:
        return 1e2*var*c
    else:
        return 1e-2*var/c

def wl2wn(var,return_inv=False):
    '''
    convert wave length [m] to wave number [cm-1]
    '''
    var *= 1e2 #[cm-1 -> m-1 or m -> cm]
    return 1./var

def pressure_shift(freq0,delta_air,n_air,P,T,T0=296.):
    '''
    P [hPa]
    '''
    freq0_shifted = freq0 + delta_air*P*(T0/T)**(0.25+1.5*n_air)
    return freq0_shifted

def doppler_width(m, freq0, T, return_fwhm=True):
    '''
       return doppler line shape FWHM

       Input
       m: molar mass [g]
       T: Temperature [K]
       freq0: frequency
    '''
    dw = freq0/c * np.sqrt(2*R*T/m*1e3)
    if return_fwhm:
        dw *= np.sqrt(ln2)
    return dw

def doppler(m,freq0,freq,T):
    '''
       return doppler line shape

       Input
       m: molar mass [g]
       T: Temperature [K]
       freq0: frequency
       freq: frequency
    '''
    _dw = doppler_width(m,freq0,T,return_fwhm=0)
    doppler_shape = 1/(_dw*np.sqrt(np.pi)) * np.exp(-(freq-freq0)**2/_dw**2)
    return doppler_shape

def gamma_natural(a):
    '''
    return gamma of natural broadening
    '''
    gamma_n = a/(2*np.pi)
    return gamma_n

def gamma_lorentzian(gamma_air,gamma_self,q,T,P,T0=296.,n_air=0.5,n_self=0.5):
    '''
    return gamma_lorentzian
    '''
    gamma_l = P*(1-q)*gamma_air*(T0/T)**n_air + P*q*gamma_self*(T0/T)**n_self
    return gamma_l

def lorentz(freq0,freq,gamma_air,gamma_self,q,T,P,T0=296.,n_air=0.5,n_self=0.5):
    '''
    return lorentzian line shape function
    '''
    gamma_l = gamma_lorentzian(gamma_air,gamma_self,q,T,P,T0=T0,n_air=n_air,n_self=n_self)
    lorentz_shape = gamma_l/np.pi /((freq-freq0)**2+gamma_l**2)
    return lorentz_shape

def _voigt(x, y):
   '''
   The Voigt function is also the real part of
   w(z) = exp(-z^2) erfc(iz), the complex probability function,
   which is also known as the Faddeeva function
   '''
   z = x + 1j*y
   I = wofz(z).real
   return I

def voigt(nu, nu_0, gamma_d, gamma_l):
   '''
   return the Voigt line shape in terms of its physical parameters
   '''
   x = (nu-nu_0)/gamma_d
   y = gamma_l/gamma_d
   V = 1./(gamma_d*np.sqrt(np.pi)) * _voigt(x, y)
   return V

def ckd(freq,freq0,S,gamma_l):
    '''
    CKD continuum
    freq, freq0 [Hz]
    gamma_l [/Hz]
    '''
    nu = freq/c *1e-2 # [Hz --> cm-1]
    nu0 = freq0/c *1e-2 # [Hz --> cm-1]
    dnu = np.abs(nu-nu0) # |nu-nu0|
    gamma_l = gamma_l*c*1e2 # [/Hz --> /cm-1]
    pass
