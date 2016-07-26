'''
created on 26 July 2016
@Author Kazutoshi Sagi

'''

__version__ = "1.0"
__filename__ = "control.py"
__user__ = 'sagi' #operator
__usage__ = None

########################
# Multiprocessing
# Number of processors
n_proc = 4

# JPL molecules files
molelist = {'O2':1,
            'O-18-O':2,
            'H2O':1,
            'H2O-18':2,
            'H2O-17':3,
            'HDO':4,
            'CO':1,
            'C-13-O':2,
            'CO-18':3,
            'CO-17':4
            }
files_jpl = {'O2':'JPL/c032001.cat',
            'O-18-O':'JPL/c034001.cat',
            'H2O':'JPL/c018003.cat',
            'H2O-18':'JPL/c020003.cat',
            'H2O-17':'JPL/c019003.cat',
            'HDO':'JPL/c019002.cat',
            'CO':'JPL/c028001.cat',
            'C-13-O':'JPL/c029001.cat',
            'CO-17':'JPL/c029006.cat',
            'CO-18':'JPL/c030001.cat'
            }
file_atm = 'ATM/mars_atm_renyu_iso.npz'

# observational geometry
geometry = 'up-looking'# 'down-looking' or 'up-looking'
z0 = None # Tangent altitudes [km]
z_inst = 0. # instrumental altitude [km]
obsangle = 90. # observational angle [deg]
             # SZA for 'down-looking'
             # Eelvation angle for 'up-looking'
R_p = 3390.0 # Planet radius [km] [Mars]
T_surf = 240 # surface Temperature [K]
T_bg = 2.725 # Back ground Temperature [K]
divider_layers = 100 # addition given numbers layers of atmospheric data, i.e p, t, vmr, are provided according to the cumulative number density distribution.

# frequency range
fre_min = 400.e9 # [Hz]
fre_max = 450.e9 # [Hz]
fre_delta = 1e6 # channel resolution (1MHz) [Hz]
fre_cutoff = 10e9 # cut off frequency (10GHz) [Hz]
lineshape = 'v' # l for lorentz, d for doppler, v for voigt
