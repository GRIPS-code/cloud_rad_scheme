import math
import netCDF4 as nc
import numpy as np
from scipy.interpolate import interp1d
from .spec_util import create_list
from .optics import optics_var
from .read_yang_ice_library import read_yang_ice_library

def compute_ice(path_ori, habit, roughness, file_outres, file_band, file_pade, a, wavenum_out, source,
                band_limit, re_range_pade, re_ref_pade, thin_flag):
    rau = 917*10**3 # ice density g/m**3
    nwav = len(wavenum_out)
    r = create_list(1,10000,20)
    nr = len(r)
    r_out = np.zeros((nr,))
    v = np.zeros((nr,))
    s = np.zeros((nr,))

    [wavenum_in, r_in, d_in, s_in, v_in, ext_cross_section_in, sca_cross_section_in, asy_in] =\
        read_yang_ice_library(path_ori, habit, roughness)
    nwav = len(wavenum_in)
    asy = np.zeros((nr,nwav))
    ext = np.zeros((nr,nwav)) 
    ssa = np.zeros((nr,nwav))
    sca = np.zeros((nr,nwav))
    for i in range(nr):
        r_out[i], s[i], v[i], ext[i,:], sca[i,:], ssa[i,:], asy[i,:] = compute_Yang_singlesize(a,2.0*r[i],rau,wavenum_in,wavenum_in,d_in, s_in, v_in, ext_cross_section_in, sca_cross_section_in, asy_in)
    optics_outres=optics_var(r_out, s, v, ext, sca, ssa, asy, rau, wavenum=wavenum_in)
    optics_outres.write_lut_spectralpoints(file_outres)

    nwav = len(wavenum_out)
    asy = np.zeros((nr,nwav))
    ext = np.zeros((nr,nwav)) 
    ssa = np.zeros((nr,nwav))
    sca = np.zeros((nr,nwav))
    # Mie Theory & integrate over gamma PSD
    for i in range(nr):
        r_out[i], s[i], v[i], ext[i,:], sca[i,:], ssa[i,:], asy[i,:] = compute_Yang_singlesize(a,2.0*r[i],rau,wavenum_out,wavenum_in,d_in, s_in, v_in, ext_cross_section_in, sca_cross_section_in, asy_in)
    optics_outres=optics_var(r_out, s, v, ext, sca, ssa, asy, rau, wavenum=wavenum_out)

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


def compute_Yang_singlesize(a,d,rau,wavenum_out,wavenum_in,d_in, s_in, v_in, ext_cross_section_in, sca_cross_section_in, asy_in):
    nwav = len(wavenum_in)
    dr = min(d/100.0,1) # um, integrate step of particle size
    d_hres = np.append(np.array([]), np.arange(dr, d*3.5, dr)) # radius
    d = np.array([d])

    nr = len(d_hres)
    r_hres = d_hres/2.0
    s_hres = np.zeros(np.shape(r_hres))
    v_hres = np.zeros(np.shape(r_hres))

    ext_hres = np.zeros((nwav,nr))
    scat_hres = np.zeros((nwav,nr))
    asy_hres = np.zeros((nwav,nr))

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

    a = np.array([a])
    optics_inres = optics_var.gamma_int(wavenum_in, a, d_hres,v_hres,s_hres,d,dr,ext_hres,scat_hres,asy_hres,rau)
    optics_singlepsd = optics_inres.interp_cloud_optics(wavenum_out)
    r_out = optics_singlepsd.r
    s = optics_singlepsd.s
    v = optics_singlepsd.v 
    ext = optics_singlepsd.ext
    sca = optics_singlepsd.sca
    ssa = optics_singlepsd.ssa
    asy  = optics_singlepsd.asy
    return r_out, s, v, ext, sca, ssa, asy