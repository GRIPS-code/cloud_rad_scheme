import math
import numpy as np
from .optics import optics_var
from .spec_util import mie_coefficients, read_refractive_index_aerosol


def compute_mie_aerosol(sigma, min_re,max_re,geore,rau,refractive_index_input,file_out):
    cm_to_um = 10**4
    kg_to_g = 1000
    rau = rau * kg_to_g # dry mass density kg/m**3 to g/m**3
    freq, m = read_refractive_index_aerosol(refractive_index_input)
    wavenum = cm_to_um/freq
    nwav = len(wavenum)
    nsize = len(min_re)
    asy = np.zeros((nsize,nwav))
    ext = np.zeros((nsize,nwav)) 
    ssa = np.zeros((nsize,nwav))
    sca = np.zeros((nsize,nwav))
    r_out = np.zeros((nsize,))
    v = np.zeros((nsize,))
    s = np.zeros((nsize,))
    for i in range(nsize):
        mu = math.exp(math.log(geore[i])-sigma[i]**2/2)
        #mu = geore[i] * math.exp(-2.5*sigma[i]**2)
        #mu = geore[i]
        r_out[i], s[i], v[i], ext[i,:], sca[i,:], ssa[i,:], asy[i,:] = compute_mie_aerosol_singlesize(mu,sigma[i],min_re[i],max_re[i],rau,wavenum,m)

    aersol_optics_outres=optics_var(r_out, s, v, ext, sca, ssa, asy, rau, wavenum=wavenum)
    
    aersol_optics_outres.write_lut_spectralpoints(file_out)

def compute_mie_aerosol_singlesize(mu,sigma,min_re,max_re,rau,wavenum,m):
    nwav = len(wavenum)
    dr = min([0.1, min_re]) # um, integrate step of droplet particle size
    if dr<=(max_re-min_re)/10:
        dr = 0.01
    r_hres = np.append(np.array([]), np.arange(min_re, max_re, dr)) # droplet dimension

    nr = len(r_hres)
    d_hres = 2*r_hres
    s_hres = np.zeros(np.shape(r_hres))
    v_hres = np.zeros(np.shape(r_hres))

    ext_hres = np.zeros((nwav,nr))
    scat_hres = np.zeros((nwav,nr))
    asy_hres = np.zeros((nwav,nr))

    # generate cross sections and asymmetry factor for droplet mie scattering 
    v_hres = 4.0/3.0*math.pi*r_hres[:]**3.0
    s_hres = math.pi*r_hres[:]**2.0 

    for i in range(nr):
        ext_hres[:,i], scat_hres[:,i], asy_hres[:,i] = mie_coefficients(wavenum[:], r_hres[i], s_hres[i], m[:])
    # fixing out-of-range values
    scat_hres[scat_hres<0] = 0
    asy_hres[asy_hres<0] = 0
    # integrate over droplet particle size distribution (PSD)
    r_out, s, v, ext, sca, ssa, asy = optics_var.lognormal_int(wavenum, mu, sigma, dr,d_hres,v_hres,s_hres,ext_hres,scat_hres,asy_hres,rau)
    return r_out, s, v, ext, sca, ssa, asy