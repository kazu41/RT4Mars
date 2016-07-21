#! /usr/bin/python
# -*- coding: utf-8 -*-
########################
'''
created on 10 July 2016
@Author Kazutoshi Sagi

'''

__version__ = "1.0"
__filename__ = "geometry.py"
__user__ = 'sagi' #operator
__usage__ = None

########################
import numpy as np
########################

def deg2tanz(angle,r_p,z_inst,geometry):
    '''
    return tangent altitude
    '''
    rad = np.deg2rad(angle) # deg --> radian
    if geometry == 'up-looking':
        return np.cos(rad)*(r_p+z_inst) - r_p
    elif geometry == 'down-looking':
        return np.sin(rad)*(r_p+z_inst) - r_p
    else: raise IOError

def tanz2deg(tanz,r_p,z_inst,geometry):
    '''
    return angle corresponding to the given tangent altitude
    '''
    if geometry == 'up-looking':
        rad = np.arccos((r_p + tanz)/(r_p + z_inst))
    elif geometry == 'down-looking':
        rad = np.arcsin((r_p + tanz)/(r_p + z_inst))
    else: raise IOError
    return np.rad2deg(rad)

def convert_los(z,z0,r_p):
    '''
    convert vertical grids into the LOS grids

    Input
    ------------
    z : altitude grids [km]
    z0 : tangent altitude [km]
    r_p : planet radius
    '''
    return np.sqrt((z + r_p)**2 - (z0 + r_p)**2)
