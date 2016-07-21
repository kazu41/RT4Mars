#! /usr/bin/python
# -*- coding: utf-8 -*-
########################
'''
created on 8 July 2016
@Author Kazutoshi Sagi

'''

__version__ = "1.0"
__filename__ = "read_jpl.py"
__user__ = 'sagi' #operator
__usage__ = None

########################
import numpy as np
import os,sys,copy
import hapi
from spectro import wn2freq,wl2wn
from calculator import *
########################
url_jpl = 'http://spec.jpl.nasa.gov'
dir = 'JPL/'
dir_hitran = 'HITRAN'
file_catdir = dir + 'catdir.cat'
file_molecules = dir + 'c%06i.cat'
########################

# prepare for using HITRAN database
hapi.db_begin(dir_hitran)
hitran_db_table = hapi.getTableList()

def read_catdir():
    format = 'I6,X, A13,    I6, 7F7.4,  I2'
    f = open(file_catdir,mode='r')
    flines = f.readlines()
    out = {}
    for fl in flines:
        tag = np.int(fl[:7])
        namefactors = fl[7:13+7].split()
        if len(namefactors) > 1:
            for i in xrange(len(namefactors)-1):
                namefactors[i+1] = ' ' + namefactors[i+1]
        name = ''.join(namefactors)
        others = fl[13+7:].split()
        nline = np.int(others.pop(0))
        version = others.pop(-1)
        if len(others) == 7:
            qlogs = [np.float(others[i]) for i in xrange(7)]
        else:
            qlogs = []
            for _qlog in others:
                try:
                    qlogs.append(np.float(_qlog))
                except:
                    _qlogs = _qlog.split('-')
                    _qlogs = np.asarray([np.float(_qlogs[i]) for i in xrange(len(_qlogs))])
                    _qlogs[1:] = -1 * _qlogs[1:]
                    qlogs.extend(_qlogs)

        out[tag] = {
                    'name'      :name,
                    'nline'     :nline,
                    'qs'     :10**np.asarray(qlogs),
                    'temp4qs':np.r_[300., 225.,150., 75., 37.5, 18.75, 9.375],
                    'version'   :version
                    }
    f.close()
    return out
    raise ValueError

class JPL:
    def __init__(self,filename):
        jplcat = read_catdir()
        self.filename = filename
        tag = np.int(os.path.basename(filename).split('.')[0][1:])
        self.tag = tag
        self.name = jplcat[tag]['name']
        self.molar_mass = self.tag/1000
        self.nline = jplcat[tag]['nline']
        self.qs = jplcat[tag]['qs']
        self.temp4qs = jplcat[tag]['temp4qs']
        self.version = jplcat[tag]['version']
        self.freq = np.zeros(self.nline)*np.nan
        self.freq_err = np.zeros(self.nline)*np.nan
        self.stg300 = np.zeros(self.nline)*np.nan
        self.readlines()

    def __repr__(self):
        def get_freqstr(nu):
            freq = np.log10(nu)
            freq_pow = np.floor(freq)
            freq_base = 10**(freq - freq_pow)
            str = '%.2f * 10^%i'%(freq_base,freq_pow)
            return str

        strings = "JPL %i %s (v%s)\n"%(self.tag, self.name, self.version)
        strings += "%s lines : %s -- %s Hz\n"%(self.nline,get_freqstr(self.freq[0]),get_freqstr(self.freq[-1]))
        return strings

    def readlines(self):
        """
        read frequencies and intensities of each line

        definition of parameters in the catalogue file
          FREQ, ERR, LGINT, DR,  ELO, GUP, TAG, QNFMT,  QN',  QN"
        (F13.4,F8.4, F8.4,  I2,F10.4,  I3,  I7,    I4,  6I2,  6I2)
        """
        f = open(self.filename,mode='r')
        flines = f.readlines()
        freq = []
        freq_err = []
        stg = []
        dr = []
        elo = []
        gup = []
        for i in xrange(self.nline):
            freq.append(np.float(flines[i][:13]))
            freq_err.append(np.float(flines[i][13:13+8]))
            stg.append(np.float(flines[i][13+8:13+8+8]))
            dr.append(np.int(flines[i][13+8+8:13+8+8+2]))
            elo.append(np.float(flines[i][13+8+8+2:13+8+8+2+10]))
            gup.append(np.int(flines[i][13+8+8+2+10:13+8+8+2+10+3]))

        self.freq = np.asarray(freq)*1e6 # MHz --> Hz
        self.freq_err = np.asarray(freq_err)*1e6 # MHz --> Hz
        self.stg300 = 10**np.asarray(stg) # log10 --> 10^x
        self.stg300 *= 1e-18 *1e6 #MHznm2/molec --> Hzm2/molec
        self.elo = np.asarray(elo) * 100. * c * h # wn --> freq

    def get_data(self):
        '''
        return dictionary containing spectroscopic data
        '''
        spec_dic = {'molar_mass':{self.name:self.molar_mass},
                    'qs':{self.name:[self.qs,self.temp4qs]},
                    'nline':self.nline}

        line_dic = { 'name':np.asarray([self.name]*self.nline),
                'freq':self.freq,
                'freq_err':self.freq_err,
                'stg300':self.stg300,
                'elo':self.elo}

        out = {'spectro':spec_dic,'line':line_dic}
        return out

    def limit_lines(self,fre_min,fre_max):
        cond = (self.freq >= fre_min)&(self.freq <= fre_max)
        self.freq = self.freq[cond]
        self.freq_err = self.freq_err[cond]
        self.stg300 = self.stg300[cond]
        self.elo = self.elo[cond]
        self.nline = cond.sum()

def get_Q(temp4qs,qs,T,n=3):
    '''
    return interpolated Q
    '''
    p = np.polyfit(temp4qs,qs,n)
    q_poly = np.polyval(p,T)
    return q_poly

def get_Stg(freq,stg300,elo,temp4qs,qs,T,n=3):
    '''
    return intensities
    '''
    e_lower = elo
    de = h * freq
    e_upper = de + e_lower
    qt = get_Q(temp4qs,qs,T,n=n)
    stg = stg300 * (qs[0]/qt) * (np.exp(-(e_lower)/kB/T) - np.exp(-(e_upper)/kB/T)) / (np.exp(-(e_lower)/kB/300.) - np.exp(-(e_upper)/kB/300.))
    return stg

class HITRAN:
    def __init__(self,molcname,iso):
        self.name = molcname
        self.iso = iso
        isoids = hapi.getColumn(molcname,'local_iso_id')
        cond = np.asarray(isoids) == iso
        freq = hapi.getColumn(molcname,'nu')
        self.freq = wn2freq(np.asarray(freq))[cond]
        a,sw,gamma_self,gamma_air,n_air,n_self,delta_air,delta_self = np.asarray(hapi.getColumns(molcname,['a','sw','gamma_self','gamma_air','n_air','n_self','delta_air','delta_self']))
        self.a = a[cond]
        self.n_air = n_air[cond] # temperature dependency for gamma_air
        self.n_self = n_self[cond] # temperature dependency for gamma_self
        self.stg296 = sw[cond]*1e2*c*1e-4 #Hz/m2/molec
        self.gamma_self = gamma_self[cond]*1e2*c/1013.25 #Hz/hPa
        self.gamma_air = gamma_air[cond]*1e2*c/1013.25 #Hz/hPa
        self.delta_self = delta_self[cond]*1e2*c/1013.25 #Hz/hPa
        self.delta_air = delta_air[cond]*1e2*c/1013.25 #Hz/hPa

    def coincise_freq(self,freq):
        def distance(f0):
            return (self.freq-f0)**2
        out = []
        for f0 in freq:
            out.append(np.argmin(distance(f0)))
        return out
