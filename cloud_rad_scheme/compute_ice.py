import netCDF4 as nc
import numpy as np
from scipy.interpolate import interp1d

from .optics import optics_var
from .read_yang_ice_library import read_yang_ice_library


def compute_ice(path_ori, habit, roughness, file_lut, file_pade, a, wavenum_out, source,
                band_limit, re_range_lut, re_range_pade, re_ref_pade, thin_flag):
    rau_ice = 917*10**3 # ice density g/m**3
    dr = 0.1 # um, integrate step of ice particle size
    d = np.append(np.append(np.arange(1, 20,0.2), np.arange(22, 200, 2)), np.arange(210, 1000, 10)) # ice dimension
    a = np.repeat(a, len(d), axis=0)

    [wavenum_in, r_in, d_in, s_in, v_in, ext_cross_section_in, sca_cross_section_in, asy_in] =\
        read_yang_ice_library(path_ori, habit, roughness)
    # linearly interpolate the library into a higher resolution
    d_hres = np.arange(dr*2, d_in[-1], dr*2)
    try:
        v_hres = interp1d(d_in[:], v_in[:]**(1/3.0))(d_hres[:])**3.0
        s_hres = interp1d(d_in[:], s_in[:]**(1/2.0))(d_hres[:])**2.0
        ext_hres = interp1d(s_in[:], ext_cross_section_in[:,:], axis=1)(s_hres[:])
        scat_hres = interp1d(s_in[:], sca_cross_section_in[:,:], axis=1)(s_hres[:])
        asy_hres = interp1d(s_in[:], asy_in[:,:]*sca_cross_section_in[:,:], axis=1)(s_hres[:])/scat_hres
    except:
        print('WARNING: Yang[2013] library is being extrapolated out of the size range')
        v_hres = interp1d(d_in[:], v_in[:]**(1/3.0),fill_value="extrapolate")(d_hres[:])**3.0
        s_hres = interp1d(d_in[:], s_in[:]**(1/2.0),fill_value="extrapolate")(d_hres[:])**2.0
        ext_hres = interp1d(s_in[:], ext_cross_section_in[:,:], axis=1, fill_value="extrapolate")(s_hres[:])
        scat_hres = interp1d(s_in[:], sca_cross_section_in[:,:], axis=1, fill_value="extrapolate")(s_hres[:])
        asy_hres = interp1d(s_in[:], asy_in[:,:]*sca_cross_section_in[:,:], axis=1, fill_value="extrapolate")(s_hres[:])/scat_hres

    # fixing out-of-range values
    scat_hres[scat_hres<0] = 0
    asy_hres[asy_hres<0] = 0

    # integrate over ice particle size distribution (PSD)
    cloud_optics_inres = optics_var.gamma_int(wavenum_in, a, d_hres, v_hres, s_hres, d, dr, ext_hres, scat_hres, asy_hres, rau_ice)
    del ext_hres, scat_hres, asy_hres
    cloud_optics_outres = cloud_optics_inres.interp_cloud_optics(wavenum_out)

    if thin_flag == True:
        cloud_optics_band = cloud_optics_outres.thin_average(source, band_limit)
    else:
        cloud_optics_band = cloud_optics_outres.thick_average(source, band_limit)

    v_range = np.zeros(np.shape(re_range_lut))
    try:
        v_range[0,:] = interp1d(cloud_optics_band.r, cloud_optics_band.v**(1/3.0))(re_range_lut[0,:])**3
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
