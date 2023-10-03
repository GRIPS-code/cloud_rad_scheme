import math

import numpy as np
from scipy.interpolate import interp1d

from .optics import optics_var
from .spec_util import mie_coefficients, read_refractive_index, create_list
from os.path import exists

def compute_liq(file_outres, file_band, file_pade, a, wavenum_out, source, band_limit, 
                re_range_pade, re_ref_pade, thin_flag):
    rau = 996*10**3 # droplet density g/m**3
    cm_to_um = 10**4
    freq = cm_to_um/wavenum_out
    m = read_refractive_index(freq)
    nwav = len(wavenum_out)

    r = create_list(0.1,re_range_pade[-1,-1]*3,5) # droplet dimension
    nr = len(r)
    asy = np.zeros((nr,nwav))
    ext = np.zeros((nr,nwav))
    ssa = np.zeros((nr,nwav))
    sca = np.zeros((nr,nwav))
    r_out = np.zeros((nr,))
    v = np.zeros((nr,))
    s = np.zeros((nr,))

    if not exists(file_outres):
        # Mie Theory & integrate over gamma PSD
        for i in range(nr):
            r_out[i], s[i], v[i], ext[i,:], sca[i,:], ssa[i,:], asy[i,:] = compute_mie_singlesize(a,2.0*r[i],rau,wavenum_out,m)

        optics_outres=optics_var(r_out, s, v, ext, sca, ssa, asy, rau, wavenum=wavenum_out)
        optics_outres.write_lut_spectralpoints(file_outres)
    else:
        optics_outres=optics_var.load_from_nc(file_outres)

    if thin_flag==True:
        optics_band = optics_outres.thin_average(source,band_limit)
    else:
        optics_band = optics_outres.thick_average(source,band_limit)
    optics_band.write_lut_spectralpoints(file_band)
    
    v_range = np.zeros(np.shape(re_range_pade))
    try:
        v_range[0,:] = interp1d(optics_band.r,optics_band.v**(1/3.0))(re_range_pade[0,:])**3
    except:
        print('WARNING: Padé approximant size range re_range exceeds lower-limit at '+'{:4.1f}'.format(optics_band.r[0])+' microns')
        v_range[0,:] = interp1d(optics_band.r,optics_band.v**(1/3.0),fill_value="extrapolate")(re_range_pade[0,:])**3
    try:
        v_range[1,:] = interp1d(optics_band.r,optics_band.v**(1/3.0))(re_range_pade[1,:])**3
    except:
        print('WARNING: Padé approximant size range re_range exceeds upper-limit at '+'{:4.1f}'.format(optics_band.r[-1])+' microns')
        v_range[1,:] = interp1d(optics_band.r,optics_band.v**(1/3.0),fill_value="extrapolate")(re_range_pade[1,:])**3
    # output parameterization netcdf file following Padé approximant
    optics_band.create_pade_coeff(re_range_pade,re_ref_pade,v_range,file_pade)


def compute_mie_singlesize(a,d,rau,wavenum,m):
    nwav = len(wavenum)
    dr = d/10.0 # um, integrate step of particle size
    d_hres = np.append(np.array([]), np.arange(dr, d*3.5, dr)) # radius
    d = np.array([d])

    nr = len(d_hres)
    r_hres = d_hres/2.0
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

    a = np.array([a])
    optics_singlepsd = optics_var.gamma_int(wavenum, a, d_hres,v_hres,s_hres,d,dr,ext_hres,scat_hres,asy_hres,rau)
    r_out = optics_singlepsd.r
    s = optics_singlepsd.s
    v = optics_singlepsd.v 
    ext = optics_singlepsd.ext
    sca = optics_singlepsd.sca
    ssa = optics_singlepsd.ssa
    asy  = optics_singlepsd.asy
    return r_out, s, v, ext, sca, ssa, asy
