import numpy as np
import miepython
from scipy.interpolate import interp1d
import miepython
import math

def planck(wn,T):
#Evaluate the emission intensity for a blackbody of temperature T as a function of wavelength
#Inputs:
#wn :: numpy array containing wavenumber [cm-1] T :: temperature [K]
#Outputs:
#B :: intensity [W / m2 / sr / cm-1]
    wn = np.array(wn) # if the input is a list or a tuple, make it an array
    c1=1.191066e-8 # [ W / m2 / sr / cm-1 ] 
    c2=1.438833; # [ K / cm-1 ]
    B = c1*wn**3/(np.exp(c2*wn/T)-1)
    return B

def convert_lblOD_to_band(source,var_in,id_wave):
# Average a spectral variable var_in[nelement,wavenumber] 
# over a range of wavenumber index id_wave[:,]
# with a given source function source[wavenumber,]
# var_our[nelement,]
    var_out = np.zeros(len(var_in))
    for i in range(len(var_in)):
        tmp = np.reshape(var_in[i,:],np.shape(source))
        rad_tmp = source[id_wave]*tmp[id_wave]
        rad_tmp2= source[id_wave]
        var_out[i] = np.nansum(rad_tmp)/np.nansum(rad_tmp2)
    return var_out

def read_solar_spectrum(path='./data/solar_flux.csv'):
# data source: GFDL AM4 model solar spectrum
    f       = open(path,"r")
    lines   = f.readlines()
    wavenum = [] # cm-1
    solar   = []
    for x in lines:
        wavenum.append(float(x.split(',')[0]))
        solar.append(float(x.split(',')[1]))
    f.close()
    return wavenum, solar

def read_refractive_index(freq_out,path='./data/refractive_index'):
# loading refractive index http://www.philiplaven.com/p20.html
# freq_out: output frequency um
    f       = open(path,"r")
    f.readline() # skip 
    f.readline() # skip
    f.readline() # skip
    lines   = f.readlines()
    freq = []
    n    = []
    k    = []
    for x in lines:
        freq.append(float(x.split('\t')[0]))
        n.append(float(x.split('\t')[1]))
        k.append(float(x.split('\t')[2]))
    f.close()
    n_new = interp1d(freq[:],n[:])(freq_out[:])  # Real part
    k_new = interp1d(freq[:],k[:])(freq_out[:])  # imaginary part
    m = n_new - k_new * 1j;    # following description in https://pypi.org/project/miepython/
    # please be aware that in some codes, the imaginary part of the refractive index is defined positively
    return m
    
def mie_coefficients(wavenum, Re, A, m):
# wavenum: cm-1, vector
# Re: droplet radii, micron
# m: refractive index
# A: Geometric cross-sectional area, um**2
    nmed = 1.0003 # refractive index of air
    cm_to_um = 10**4
    x = 2 * math.pi * Re * wavenum / cm_to_um * nmed
    qext,qsca,tmp,asy = miepython.mie(m,x)               
    #print(A)
    ext_cross_section     = qext*A             # Extinction cross-section, um**2
    sca_cross_section     = qsca*A             # scattering cross-section, um**2
    return ext_cross_section, sca_cross_section, asy