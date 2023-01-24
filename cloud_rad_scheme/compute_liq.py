import math

import numpy as np
from scipy.interpolate import interp1d

from .cloud_optics import cloud_optics_var
from .spec_util import mie_coefficients, read_refractive_index


def compute_liq(file_lut, file_pade, a, wavenum_out, source, band_limit, re_range_lut,
                re_range_pade, re_ref_pade, thin_flag):
    rau_liq = 996*10**3 # droplet density g/m**3
    cm_to_um = 10**4
    dr = 0.1 # um, integrate step of droplet particle size
    d = np.append(np.append(np.arange(2, 20,0.2), np.arange(22, 200, 2)), np.arange(210, 1000, 10)) # droplet dimension
    a = np.repeat(a, len(d), axis=0) 

    freq = cm_to_um/wavenum_out
    m = read_refractive_index(freq)
    d_hres = np.arange(dr*2, d[-1], dr*2)

    r_hres = d_hres/2.0
    nr = len(r_hres)
    nwav = len(wavenum_out)
    s_hres = np.zeros(np.shape(r_hres))
    v_hres = np.zeros(np.shape(r_hres))
    ext_hres = np.zeros((nwav,nr))
    scat_hres = np.zeros((nwav,nr))
    asy_hres = np.zeros((nwav,nr))

    # generate cross sections and asymmetry factor for droplet mie scattering 
    v_hres = 4.0/3.0*math.pi*r_hres[:]**3.0
    s_hres = math.pi*r_hres[:]**2.0 
    for i in range(nr):
            ext_hres[:,i], scat_hres[:,i], asy_hres[:,i] = mie_coefficients(wavenum_out[:], r_hres[i], s_hres[i], m[:])
    # fixing out-of-range values
    scat_hres[scat_hres<0] = 0
    asy_hres[asy_hres<0] = 0

    # integrate over droplet particle size distribution (PSD)
    cloud_optics_outres = cloud_optics_var.gamma_int(wavenum_out, a,d_hres,v_hres,s_hres,d,dr,ext_hres,scat_hres,asy_hres,rau_liq)

    if thin_flag==True:
        cloud_optics_band = cloud_optics_outres.thin_average(source,band_limit)
    else:
        cloud_optics_band = cloud_optics_outres.thick_average(source,band_limit)
    
    v_range = np.zeros(np.shape(re_range_lut))
    try:
        v_range[0,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0))(re_range_lut[0,:])**3
    except:
        print('WARNING: look-up-table size range re_range exceeds lower-limit at '+'{:4.1f}'.format(cloud_optics_band.r[0])+' microns')
        v_range[0,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0),fill_value="extrapolate")(re_range_lut[0,:])**3
    try:
        v_range[1,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0))(re_range_lut[1,:])**3
    except:
        print('WARNING: look-up-table size range re_range exceeds upper-limit at '+'{:4.1f}'.format(cloud_optics_band.r[-1])+' microns')
        v_range[1,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0),fill_value="extrapolate")(re_range_lut[1,:])**3
    # output parameterization netcdf file following piece-wise linear interpolation
    cloud_optics_band.create_lut_coeff(re_range_lut,v_range,file_lut)
    print(cloud_optics_band.ext[10,:])
    v_range = np.zeros(np.shape(re_range_pade))
    try:
        v_range[0,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0))(re_range_pade[0,:])**3
    except:
        print('WARNING: Padé approximant size range re_range exceeds lower-limit at '+'{:4.1f}'.format(cloud_optics_band.r[0])+' microns')
        v_range[0,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0),fill_value="extrapolate")(re_range_pade[0,:])**3
    try:
        v_range[1,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0))(re_range_pade[1,:])**3
    except:
        print('WARNING: Padé approximant size range re_range exceeds upper-limit at '+'{:4.1f}'.format(cloud_optics_band.r[-1])+' microns')
        v_range[1,:] = interp1d(cloud_optics_band.r,cloud_optics_band.v**(1/3.0),fill_value="extrapolate")(re_range_pade[1,:])**3
    # output parameterization netcdf file following Padé approximant
    cloud_optics_band.create_pade_coeff(re_range_pade,re_ref_pade,v_range,file_pade)
