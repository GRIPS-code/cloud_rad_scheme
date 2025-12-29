from array import *
from cloud_rad_scheme import compute_ice, compute_liq, planck, read_solar_spectrum
import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d


def main():
    """Yang [2013] library path
    https://doi.org/10.1175/JAS-D-12-039.1
    format: *.tar.gz file for FIR, Rough*.tar.gz for MIR.
    """
    # This line is ignored if ./data/MIR and ./data.FIR contain required files
    path_ori = '/scratch/gpfs/jf7775/data/ice_optics_yang/' 

    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_pade = np.zeros((2, 5)) # Padé approximantsize size range, micron
    re_range_pade[0,:] = [2.5,  15., 50., 100., 1000.]
    re_range_pade[1,:] = [15., 50., 100., 1000., 5000.] 
    re_ref_pade = np.zeros(np.shape(re_range_pade)[1],)

    band_limit = np.zeros((3249,2))
    band_limit[:,0] = np.arange(0.5, 3249.5, 1)
    band_limit[:,1] = np.arange(1.5, 3250.5, 1)

    wavenum = np.arange(1, 3260, 1)
    # initialize longwave source function
    source = planck(wavenum, 250)

    # generate parameterization for longwave ice
    compute_ice(path_ori, 'solid_column', 50,
                'hres_ice_lw_solid_column_severlyroughen_gamma_aeq1_lbl.nc',
                'lbl_lut_ice_lw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                'lbl_pade_ice_lw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_pade,
                re_ref_pade,True)
    
    band_limit = np.zeros((4975,2))
    band_limit[:,0] = np.arange(250, 50000, 10)
    band_limit[:,1] = np.arange(260, 50010, 10)
    wavenum = np.arange(band_limit[0,0], band_limit[-1,-1], 1)
    # read-in shortwave spectrum
    wavenum_solar, solar = read_solar_spectrum()
    source = interp1d(wavenum_solar[:], solar[:])(wavenum[:])

    # generate parameterization for shortwave ice
    compute_ice(path_ori, 'solid_column', 50,
                'hres_ice_sw_solid_column_severlyroughen_gamma_aeq1.nc',
                'lbl_lut_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                'lbl_pade_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit,re_range_pade,
                re_ref_pade,False)


def main_liq():

    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_pade = np.zeros((2, 5)) # Padé approximantsize size range, micron
    re_range_pade[0,:] = [2.5,  15., 50., 100., 1000.]
    re_range_pade[1,:] = [15., 50., 100., 1000., 5000.] 
    re_ref_pade = np.zeros(np.shape(re_range_pade)[1],)

    band_limit = np.zeros((3249,2))
    band_limit[:,0] = np.arange(0.5, 3249.5, 1)
    band_limit[:,1] = np.arange(1.5, 3250.5, 1)

    wavenum = np.arange(1, 3260, 1)
    # initialize longwave source function
    source = planck(wavenum, 250)

    compute_liq('hres_liq_lw_mie_gamma_aeq12_lbl.nc',
                'lbl_band_liq_lw_mie_gamma_aeq12_thick.nc',
                'lbl_pade_liq_lw_mie_gamma_aeq12_thick.nc',
                12, wavenum, source, band_limit, re_range_pade,
                re_ref_pade, True)

    # initialize shortwave band limits that matches with rrtmgp gas optics
    band_limit = np.zeros((4975,2))
    band_limit[:,0] = np.arange(250, 50000, 10)
    band_limit[:,1] = np.arange(260, 50010, 10)
    wavenum = np.arange(band_limit[0,0], band_limit[-1,-1], 10)
    # read-in shortwave spectrum
    wavenum_solar, solar = read_solar_spectrum()
    source = interp1d(wavenum_solar[:], solar[:])(wavenum[:])

    compute_liq('hres_liq_sw_mie_gamma_aeq12_ecckd.nc',
                'lbl_band_liq_sw_mie_gamma_aeq12_thick.nc',
                'lbl_pade_liq_sw_mie_gamma_aeq12_thick.nc',
                12, wavenum, source, band_limit, re_range_pade,
                re_ref_pade, False)
if __name__ == "__main__":
    main()
