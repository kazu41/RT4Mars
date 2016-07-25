#! /usr/bin/python
# -*- coding: utf-8 -*-
########################
'''
created on 13 July 2016
@Author Kazutoshi Sagi

'''

__version__ = "1.0"
__filename__ = "main.py"
__user__ = 'sagi' #operator
__usage__ = None

########################
import os,sys,copy,time
import numpy as np
import read_jpl as JPL
import spectro as sp
import geometry as geo
import calculator as c
from scipy import integrate
import multiprocessing as mp
from IPython.core.debugger import Pdb
pdb = Pdb()

# settings
# JPL molecules files
molelist = {'O2':1,'O-18-O':2,'H2O':1,'H2O-18':2,'H2O-17':3,'HDO':4,'CO':1,'C-13-O':2,'CO-18':3,'CO-17':4}
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

class RT:
    def __init__(self,molelist):
        self.molelist = molelist.keys()
        self.isos = molelist.values()
        self.set_conditions()

    def __repr__(self):
        def get_freqstr(nu):
            freq = np.log10(nu)
            freq_pow = np.floor(freq)
            freq_base = 10**(freq - freq_pow)
            str = '%.2f * 10^%i'%(freq_base,freq_pow)
            return str

        strings = '#'*40 + '\n'
        strings += 'Radiative transfer calculation\n'
        strings += '#'*40 + '\n'
        strings += 'Geometry: %s\n'%self.geometry
        strings += 'Instrumental altitude: %.1fkm with angle = %.1f\n'%(self.z_inst,self.angle)
        strings += '(* No refraction approximation)\n'
        strings += '#'*40 + '\n'
        strings += 'Absorption by %s\n'%self.molelist
        strings += "%s lines : %s -- %s Hz\n"%(self.spectro['spectro']['nline'],get_freqstr(self.fre_min),get_freqstr(self.fre_max))
        strings += 'Line shape function : %s\n'%self.lineshape
        strings += "l for lorentz, d for doppler, v for voigt\n"
        strings += '#'*40 + '\n'
        return strings

    def set_conditions(self):
        # settings
        # load atmospheric profiles
        #atm = np.load('ATM/mars_atm.npz')
        self.atm = self.load_atm()
        self.n_levels = self.atm['z'].size

        # observational geometry
        self.z_inst = 0. # instrumental altitude [km]
        if self.z_inst > 0.:
            self.geometry = 'down-looking'
        elif self.z_inst == 0.:
            self.geometry = 'up-looking'
        else: raise IOError
        self.angle = 90. # observational angle [deg]
                         # SZA for 'down-looking'
                         # Eelvation angle for 'up-looking'
        self.R_p = 3390.0 # Planet radius [km] [Mars]
        self.T_surf = 240 # surface Temperature [K]
        self.T_bg = 2.725 # Back ground Temperature [K]

        # frequency range
        self.fre_min = 300.e9 # [Hz]
        self.fre_max = 500.e9 # [Hz]
        self.fre_delta = 1e6 # channel resolution (1MHz) [Hz]
        self.fre_cutoff = 10e9 # cut off frequency (10GHz) [Hz]
        self.freq = np.arange(self.fre_min,self.fre_max+self.fre_delta,self.fre_delta)
        self.n_channel = self.freq.size
        self.lineshape = 'v' # l for lorentz, d for doppler, v for voigt
        self.spectro = self.merge_jplcat(self.molelist)

    # functions
    def load_atm(self,n=100):
        '''
        load atmospheric data with interpolation by cumulative number density

        Input
        --------
        n : divider (100)
        '''
        out = dict(np.load(file_atm))
        z = out.pop('z')
        csnd_0 = np.cumsum(out['nd'])
        csnd_1 = np.linspace(csnd_0[0],csnd_0[-1],n)
        zz = np.interp(csnd_1,csnd_0,z).round(2)
        z_new = np.unique(np.r_[z,zz])
        keys = out.keys()
        for k in keys:
            out[k] = np.interp(z_new,z,out[k])
        out['z'] = z_new
        return out

    def load_jplcat(self,mole,iso):
        jpl = JPL.JPL(files_jpl[mole])
        jpl.limit_lines(self.fre_min,self.fre_max)
        hitran = JPL.HITRAN(jpl.name,iso)
        cond_hitran = hitran.coincise_freq(jpl.freq)
        out = jpl.get_data()
        out['line']['stg296'] = hitran.stg296[cond_hitran]
        out['line']['gamma_air'] = hitran.gamma_air[cond_hitran]
        out['line']['gamma_self'] = hitran.gamma_self[cond_hitran]
        out['line']['gamma_natural'] = sp.gamma_natural(hitran.a[cond_hitran])
        out['line']['n_air'] = hitran.n_air[cond_hitran]
        out['line']['n_self'] = hitran.n_self[cond_hitran]
        out['line']['delta_air'] =  hitran.delta_air[cond_hitran]
        out['line']['delta_self'] = hitran.delta_self[cond_hitran]
        return out

    # load spectroscopic data
    def merge_jplcat(self,molelist):
        for i,mole in enumerate(molelist):
            if i==0:
                jpl = self.load_jplcat(mole,self.isos[i])
            else:
                jpl_tmp = self.load_jplcat(mole,self.isos[i])
                # update spectroscopic dictionary
                jpl['spectro']['nline'] += jpl_tmp['spectro'].pop('nline')
                for k in jpl_tmp['spectro'].keys():
                    jpl['spectro'][k].update(jpl_tmp['spectro'][k])
                # update line dictionary
                for k in jpl_tmp['line'].keys():
                    jpl['line'][k] = np.append(jpl['line'][k],jpl_tmp['line'][k])
                # sort by frequencies
                id_sort = np.argsort(jpl['line']['freq'])
                for k in jpl_tmp['line'].keys():
                    jpl['line'][k] = jpl['line'][k][id_sort]
        return jpl

    # main script
    def _abscoef(self,id_line):
        '''
        calculate Absorption coefficient of a certain line
        '''
        temp = self.atm['t'] # K
        pres = self.atm['p'] # hPa
        nd = self.atm['nd'] # molec/m3
        mole_name = self.spectro['line']['name'][id_line]
        freq0 = self.spectro['line']['freq'][id_line]
        n_air = self.spectro['line']['n_air'][id_line]
        n_self = self.spectro['line']['n_self'][id_line]
        d_air = self.spectro['line']['delta_air'][id_line]
        #d_self = self.spectro['line']['delta_self'][id_line]
        gair = self.spectro['line']['gamma_air'][id_line]
        gself = self.spectro['line']['gamma_self'][id_line]
        gntr = self.spectro['line']['gamma_natural'][id_line]
        stg300 = self.spectro['line']['stg300'][id_line]
        elo = self.spectro['line']['elo'][id_line]
        qs = self.spectro['spectro']['qs'][mole_name][0]
        temp4qs = self.spectro['spectro']['qs'][mole_name][1]
        mole_vmr = self.atm[mole_name]
        molar_mass = self.spectro['spectro']['molar_mass'][mole_name]

        cond_freq = (self.freq >= freq0-self.fre_cutoff) & (self.freq <= freq0+self.fre_cutoff)
        freq_tmp = self.freq[cond_freq]
        abscoef = np.zeros([self.n_levels,freq_tmp.size])
        for i in xrange(self.n_levels):
            # sys.stdout.write("\r%i line : %i / %i levels"%(id_line+1,i+1,self.n_levels))
            # sys.stdout.flush()
            # pressure shift
            freq0 = sp.pressure_shift(freq0,d_air,n_air,pres[i],temp[i])
            # Line shape
            if self.lineshape=='d': # doppler
                f_ls = sp.doppler(molar_mass,freq0,freq_tmp,temp[i])
            elif self.lineshape=='l': # lorentz
                f_ls = sp.lorentz(freq0,freq_tmp,gair,gself,mole_vmr[i],temp[i],pres[i],n_air=n_air,n_self=n_self)
            elif self.lineshape=='v':
                # lorentzian line width
                gamma_l = sp.gamma_lorentzian(gair,gself,mole_vmr[i],temp[i],pres[i],n_air=n_air,n_self=n_self)
                # adding natural broadening
                gamma_l += gntr
                # doppler line width
                gamma_d = sp.doppler_width(molar_mass,freq0,temp[i],return_fwhm=False)
                # voigt lineshape
                f_ls = sp.voigt(freq_tmp,freq0,gamma_d,gamma_l)
            else: raise ValueError
            # line intensity
            stg_t = JPL.get_Stg(freq0,stg300,elo,temp4qs,qs,temp[i]) # Hzm2/molec
            abscoef[i] = stg_t*f_ls*mole_vmr[i]*nd[i]
        # print('')
        return cond_freq,abscoef

    def get_abscoef_tmp(self):
        '''
        main script to calculate radiative transfer
        '''
        print('#'*30)
        print('Absorption coefficient calculation')
        t0 = time.time()
        #p = mp.Pool()
        nline = self.spectro['spectro']['nline']
        #result = p.map(MulHelper(self,'_abscoef'),xrange(nline))
        self.abscoef = np.zeros([self.n_levels,self.n_channel])
        for l in xrange(nline):
            cond_freq,absc = self._abscoef(l)
            #cond_freq,absc = result[l]
            self.abscoef[:,cond_freq] += absc
        # Abs. finish
        print('%f sec'%(time.time()-t0))
        print('#'*30)

    def get_abscoef(self):
        '''
        main script to calculate radiative transfer
        '''
        t0 = time.time()
        temp = self.atm['t'] # K
        pres = self.atm['p'] # hPa
        nd = self.atm['nd'] # molec/m3
        nline = self.spectro['spectro']['nline']

        print('#'*30)
        print('Absorption coefficient calculation')
        self.abscoef = np.zeros([self.n_levels,self.n_channel])
        for i in xrange(self.n_levels):
            # line by line
            for l in xrange(nline):
                sys.stdout.write("\r%i / %i levels : %i / %i lines"%(i+1,self.n_levels,l+1,nline))
                sys.stdout.flush()
                mole_name = self.spectro['line']['name'][l]
                freq0 = self.spectro['line']['freq'][l]
                n_air = self.spectro['line']['n_air'][l]
                n_self = self.spectro['line']['n_self'][l]
                d_air = self.spectro['line']['delta_air'][l]
                #d_self = self.spectro['line']['delta_self'][l]
                gair = self.spectro['line']['gamma_air'][l]
                gself = self.spectro['line']['gamma_self'][l]
                gntr = self.spectro['line']['gamma_natural'][l]
                stg300 = self.spectro['line']['stg300'][l]
                elo = self.spectro['line']['elo'][l]
                qs = self.spectro['spectro']['qs'][mole_name][0]
                temp4qs = self.spectro['spectro']['qs'][mole_name][1]
                mole_vmr = self.atm[mole_name][i]
                molar_mass = self.spectro['spectro']['molar_mass'][mole_name]
                cond_freq = (self.freq >= freq0-self.fre_cutoff)&(self.freq <= freq0+self.fre_cutoff)
                freq_tmp = self.freq[cond_freq]
                # pressure shift
                freq0 = sp.pressure_shift(freq0,d_air,n_air,pres[i],temp[i])
                # Line shape
                if self.lineshape=='d': # doppler
                    f_ls = sp.doppler(molar_mass,freq0,freq_tmp,temp[i])
                elif self.lineshape=='l': # lorentz
                    f_ls = sp.lorentz(freq0,freq_tmp,gair,gself,mole_vmr,temp[i],pres[i],n_air=n_air,n_self=n_self)
                elif self.lineshape=='v':
                    # lorentzian line width
                    gamma_l = sp.gamma_lorentzian(gair,gself,mole_vmr,temp[i],pres[i],n_air=n_air,n_self=n_self)
                    # adding natural broadening
                    gamma_l += gntr
                    # doppler line width
                    gamma_d = sp.doppler_width(molar_mass,freq0,temp[i],return_fwhm=False)
                    # voigt lineshape
                    f_ls = sp.voigt(freq_tmp,freq0,gamma_d,gamma_l)
                else: raise IOError
                # line intensity
                stg_t = JPL.get_Stg(freq0,stg300,elo,temp4qs,qs,temp[i]) # Hzm2/molec
                self.abscoef[i,cond_freq] += stg_t*f_ls*mole_vmr*nd[i]
        print('') # Abs. finish
        print('%f sec'%(time.time()-t0))
        print('#'*30)

    def radiative_transfer(self):
        '''
        return Tb
        '''
        # Geometrical change of the path
        print('#'*30)
        print('Radiative transfer')
        print('-- Geometrical setting')
        # tangent altitude
        z0 = geo.deg2tanz(self.angle,self.R_p,self.z_inst,self.geometry)
        # LOS
        z = self.atm['z']
        s_tmp = geo.convert_los(z, z0, self.R_p)*1e3 # [km --> m]
        s_inst = geo.convert_los(self.z_inst, z0, self.R_p)*1e3 # [km --> m]
        absc = self.abscoef
        temp_phis = self.atm['t']
        if self.geometry=='down-looking':
            if z0 > 0: # Limb
                geometry = 'Limb'
                t0 = self.T_bg
                cond_alt = (z < self.z_inst)&(np.isnan(s_tmp)==0)
                cond_alt_far = (np.isnan(s_tmp)==0)
                absc = np.r_[absc[cond_alt][::-1],absc[cond_alt_far][1:]]
                temp_phis = np.r_[temp_phis[cond_alt][::-1],temp_phis[cond_alt_far][1:]]
                s_tmp = np.r_[s_tmp[cond_alt][::-1],s_tmp[cond_alt_far][1:]]
                ds_tmp = np.abs(np.gradient(s_tmp))
            else: # down-looking
                geometry = self.geometry
                t0 = self.T_surf
                cond_alt = (z < self.z_inst)&(np.isnan(s_tmp)==0)
                absc = absc[cond_alt][::-1]
                temp_phis = temp_phis[cond_alt][::-1]
                s_tmp = s_tmp[cond_alt][::-1]
                ds_tmp = np.abs(np.gradient(s_tmp))[:,np.newaxis]
        elif self.geometry=='up-looking':
            geometry = self.geometry
            # back ground temperature
            t0 = self.T_bg # space
            cond_alt = (z > self.z_inst)&(np.isnan(s_tmp)==0)
            absc = absc[cond_alt]
            temp_phis = temp_phis[cond_alt]
            s_tmp = s_tmp[cond_alt]
            ds_tmp = np.abs(np.gradient(s_tmp))[:,np.newaxis]
        else: raise ValueError
        print('%s setting, tangent height : %.1f km'%(geometry,z0))
        ds_0 = np.abs(s_inst-s_tmp[0]) # distance from the instrument to the 1st layer
        s = np.cumsum(ds_tmp)
        print('-- Calculation')
        # Radiative transfer
        # Brightness Temperature
        # calc. tau for each level
        tau = integrate.cumtrapz(absc,x=s,axis=0,initial=0)
        tau[0] = absc[0]*ds_0
        eta = np.exp(-tau)
        Inu = np.r_[[c.planck(self.freq,temp_phis[i]) for i in xrange(s_tmp.size)]]
        I0 = eta[-1]*c.planck(self.freq,t0)
        Ib = integrate.trapz(absc*eta*Inu,x=s,axis=0)
        Tb_out = c.I2Tb(self.freq,I0+Ib)
        print('#'*30)
        return Tb_out

class MulHelper(object):
    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name

    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)
