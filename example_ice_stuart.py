from array import *
from cloud_rad_scheme import compute_ice, planck, read_solar_spectrum
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
    re_range_lut = np.zeros((2, 15)) # look-up-table (piecewise linear coefficients) size range, micron
    re_range_pade = np.zeros((2, 3)) # Padé approximantsize size range, micron
    re_range_lut[0,:] = np.append(np.append([1.1, 2, 3, 5, 7], np.arange(10, 30, 5)), np.arange(40, 100, 10))
    re_range_lut[1,:] = np.append(np.append([2, 3, 5, 7], np.arange(10, 30, 5)), np.arange(40, 110, 10))
    re_range_pade[0,:] = [1.1, 10, 50]
    re_range_pade[1,:] = [10, 50, 100] # Please check compute_ice module variable 'd', if a wider range is required
    re_ref_pade = np.zeros(np.shape(re_range_pade)[1],)

    # initialize shortwave band limits that matches with rrtmgp gas optics
    band_limit = np.array([[  1., 2500.],
                               [ 2500., 2900.],
                               [ 2900., 3400.],
                               [ 3400., 4200.],
                               [4200., 4700.],
                               [4700., 5600.],
                               [5600., 6200.],
                               [6200., 8200.],
                               [8200., 11500.],
                               [11500., 14600.],
                               [14600., 16700.],
                               [16700., 20000.],
                               [20000., 22300.],
                               [22300., 24600.],
                               [24600., 27500.],
                               [27500., 30000.],
                               [30000., 31900.],
                               [31900., 33000.],
                               [33000., 33800.],
                               [33800., 34500.],
                               [34500., 35300.],
                               [35300., 36500.],
                               [36500., 40000.],
                               [40000., 43300.],
                               [43300., 57600.]])

    wavenum = np.arange(band_limit[0,0], band_limit[-1,-1], 10)
    # read-in shortwave spectrum
    wavenum_solar, solar = read_solar_spectrum()
    source = interp1d(wavenum_solar[:], solar[:])(wavenum[:])

    # generate parameterization for shortwave ice
    compute_ice(path_ori, 'solid_column', 50,
                '25bands_lut_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                '25bands_pade_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade,False)


if __name__ == "__main__":
    main()
