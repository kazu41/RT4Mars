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

class CTL:
    def __init__(self,ctlfile):
        self.ctlfile = ctlfile
        self.loadctl()

    def loadctl(self):
        '''
        settings loaded from the control file
        '''
        execfile(self.ctlfile)
        ctldata = locals()
        # settings
        self.n_proc = ctldata['n_proc']
        self.files_jpl = ctldata['files_jpl']
        self.molelist = ctldata['molelist'].keys()
        self.isos = ctldata['molelist'].values()
        # load atmospheric profiles
        self.atm = self.load_atm(ctldata['file_atm'],ctldata['divider_layers'])
        self.n_levels = self.atm['z'].size

        # observational geometry
        self.geometry = ctldata['geometry']
        # instrumental altitude [km]
        self.z_inst = ctldata['z_inst']
        # observational angle [deg]
        self.angle = ctldata['obsangle']
        # Planet radius [km] [Mars]
        self.R_p = ctldata['R_p']
        # surface Temperature [K]
        self.T_surf = ctldata['T_surf']
        # Back ground Temperature [K]
        self.T_bg = ctldata['T_bg']
        # Tangent altitude [km]
        self.z0 = ctldata['z0']

        # frequency range
        self.lineshape = ctldata['lineshape'] # l for lorentz, d for doppler, v for voigt
        self.fre_min = ctldata['fre_min'] # [Hz]
        self.fre_max = ctldata['fre_max'] # [Hz]
        self.fre_delta = ctldata['fre_delta'] # channel resolution (1MHz) [Hz]
        self.fre_cutoff = ctldata['fre_cutoff'] # cut off frequency (10GHz) [Hz]
        self.set_freqs()
        self.spectro = self.merge_jplcat(self.molelist)

    # functions
    def set_freqs(self):
        '''
        set self.freq and self.n_channel
        '''
        self.freq = np.arange(self.fre_min,self.fre_max+self.fre_delta,self.fre_delta)
        self.n_channel = self.freq.size
        self.spectro = self.merge_jplcat(self.molelist)

    def load_atm(self,file_atm,n=100):
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
        jpl = JPL.JPL(self.files_jpl[mole])
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

class RT(CTL):
    def __init__(self,*args,**opt):
        self.ctlfile = args[0]
        self.loadctl()

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
        strings += 'Instrumental altitude: %.1f km\nwith angle = %.1f deg\n'%(self.z_inst,self.angle)
        strings += '(* No refraction approximation)\n'
        strings += '#'*40 + '\n'
        strings += 'Absorption by %s\n'%self.molelist
        strings += "%s lines : %s -- %s Hz\n"%(self.spectro['spectro']['nline'],get_freqstr(self.fre_min),get_freqstr(self.fre_max))
        strings += 'Line shape function : %s\n'%self.lineshape
        strings += "l,d and v for lorentz, doppler and voigt\n"
        strings += '#'*40 + '\n'
        return strings

    def __call__(self,*args):
        '''
        execute Absorption coeff. and Radiative transfer calculations
        '''
        # reset freqs
        flag_freq = args[0]
        if flag_freq==1:
            self.set_freqs()
        # print current settings
        print(self)
        # Absorption coefficient
        flag_abs = args[1]
        if flag_abs==1:
            self.get_abscoef()
        # Radiative transfer
        flag_rt = args[2]
        if flag_rt==1:
            Tb = self.radiative_transfer()
            return Tb

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
        abscoef = np.zeros([self.n_levels,self.n_channel])
        for i in xrange(self.n_levels):
            # pressure shift
            freq0_shifted = sp.pressure_shift(freq0,d_air,n_air,pres[i],temp[i])
            # Line shape
            if self.lineshape=='d': # doppler
                f_ls = sp.doppler(molar_mass,freq0_shifted,freq_tmp,temp[i])
            elif self.lineshape=='l': # lorentz
                f_ls = sp.lorentz(freq0_shifted,freq_tmp,gair,gself,mole_vmr[i],temp[i],pres[i],n_air=n_air,n_self=n_self)
            elif self.lineshape=='v':
                # lorentzian line width
                gamma_l = sp.gamma_lorentzian(gair,gself,mole_vmr[i],temp[i],pres[i],n_air=n_air,n_self=n_self)
                # adding natural broadening
                gamma_l += gntr
                # doppler line width
                gamma_d = sp.doppler_width(molar_mass,freq0_shifted,temp[i],return_fwhm=False)
                # voigt lineshape
                f_ls = sp.voigt(freq_tmp,freq0_shifted,gamma_d,gamma_l)
            else: raise ValueError
            # line intensity
            stg_t = JPL.get_Stg(freq0,stg300,elo,temp4qs,qs,temp[i]) # Hzm2/molec
            abscoef[i][cond_freq] = stg_t*f_ls*mole_vmr[i]*nd[i]
        # print('')
        return abscoef

    def subcalc(self,queue,p):
        '''
        multiprocessing sub routine
        '''
        nline = self.spectro['spectro']['nline']
        ini = nline * p/self.n_proc
        fin = nline * (p+1)/self.n_proc
        out = np.zeros([self.n_levels,self.n_channel])
        for i in xrange(ini,fin):
            out += self._abscoef(i)
        queue.put(out)

    def get_abscoef(self):
        '''
        main script to calculate radiative transfer
        '''
        print('#'*30)
        print('Absorption coefficient calculation')
        t0 = time.time()
        self.abscoef = np.zeros([self.n_levels,self.n_channel])
        # set queue
        queue = mp.Queue()
        # set processes
        ps = [mp.Process(target=self.subcalc, args=(queue, i)) for i in xrange(self.n_proc)]
        # start Process
        for p in ps:
            p.start()
        # store result
        for i in xrange(self.n_proc):
            self.abscoef += queue.get()
        # close queue
        queue.close()
        # terminate processes
        for p in ps:
            p.terminate()
        # Abs. finish
        print('%f sec'%(time.time()-t0))
        print('#'*30)

    def params4los(self,z0):
        '''
        return parameters on LOS for the given tangent altitude z0 [km]
        '''
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
        else: raise ValueError('Given geometry setting is wrong.')
        print('%s setting, tangent height : %.1f km'%(geometry,z0))
        ds_0 = np.abs(s_inst-s_tmp[0]) # distance from the instrument to the 1st layer
        s = np.cumsum(ds_tmp)
        return ds_0,s,t0,temp_phis,absc

    def radiative_transfer(self):
        '''
        return Tb
        '''
        # Geometrical change of the path
        print('#'*30)
        print('Radiative transfer')
        print('-- Geometrical setting')
        # Tangent altitude [km]
        if self.z0==None:
            z0 = geo.deg2tanz(self.angle,self.R_p,self.z_inst,self.geometry)
        else:
            z0 = self.z0
        # lOS
        ds_0,s,t0,temp_phis,absc = self.params4los(z0)
        print('-- Calculation')
        # Radiative transfer
        # Brightness Temperature
        # calc. tau for each level
        tau = integrate.cumtrapz(absc,x=s,axis=0,initial=0)
        tau[0] = absc[0]*ds_0
        eta = np.exp(-tau)
        Inu = np.r_[[c.planck(self.freq,temp_phis[i]) for i in xrange(s.size)]]
        I0 = eta[-1]*c.planck(self.freq,t0)
        Ib = integrate.trapz(absc*eta*Inu,x=s,axis=0)
        Tb_out = c.I2Tb(self.freq,I0+Ib)
        print('#'*30)
        return Tb_out
