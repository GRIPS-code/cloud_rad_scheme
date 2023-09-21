from array import *
from cloud_rad_scheme import compute_snow, planck, read_solar_spectrum, create_list
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
    re_range_pade = np.zeros((2,5)) # Padé approximantsize size range, micron
    tmp = create_list(10.,700., 5.)
    re_range_lut = np.zeros((2,len(tmp)-2))
    re_range_lut[0,:] = tmp[0:-2]
    re_range_lut[1,:] = tmp[1:-1]
    re_range_pade[0,:] = [2.5,  15., 50., 100., 1000.]
    re_range_pade[1,:] = [15., 50., 100., 1000., 5000.] 
    re_ref_pade = np.zeros(np.shape(re_range_pade)[1],)

    # initialize longwave band limits that matches with rrtmgp gas optics
    band_limit = np.array([[  10.,  250.], 
                               [ 250.,  500.], 
                               [ 500.,  630.], 
                               [ 630.,  700.],
                               [ 700.,  820.], 
                               [ 820.,  980.], 
                               [ 980., 1080.], 
                               [1080., 1180.],
                               [1180., 1390.], 
                               [1390., 1480.], 
                               [1480., 1800.], 
                               [1800., 2080.],
                               [2080., 2250.], 
                               [2250., 2390.], 
                               [2390., 2680.], 
                               [2680., 3250.]])
    print(np.shape(band_limit))
    wavenum = np.arange(band_limit[0,0], band_limit[-1,-1], 1)

    # initialize longwave source function
    source = planck(wavenum, 250)

    # generate parameterization for longwave ice
    compute_snow(path_ori, 'solid_column', 50,
                'hres_ice_lw_solid_column_severlyroughen_gamma_aeq1.nc',
                'lut_ice_lw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                'pade_ice_lw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade, True)

    # initialize shortwave band limits that matches with rrtmgp gas optics
    band_limit = np.array([[  820., 2680.], 
                               [ 2680., 3250.], 
                               [ 3250., 4000.], 
                               [ 4000., 4650.],
                               [ 4650., 5150.], 
                               [ 5150., 6150.], 
                               [ 6150., 7700.], 
                               [ 7700., 8050.], 
                               [ 8050., 12850.], 
                               [12850., 16000.], 
                               [16000., 22650.], 
                               [22650., 29000.],
                               [29000., 38000.],
                               [38000., 50000.]])

    wavenum = np.arange(band_limit[0,0], band_limit[-1,-1], 10)
    # read-in shortwave spectrum
    wavenum_solar, solar = read_solar_spectrum()
    source = interp1d(wavenum_solar[:], solar[:])(wavenum[:])

    # generate parameterization for shortwave ice
    compute_snow(path_ori, 'solid_column', 50,
                'hres_ice_sw_solid_column_severlyroughen_gamma_aeq1.nc',
                'lut_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                'pade_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade,False)


if __name__ == "__main__":
    main()
