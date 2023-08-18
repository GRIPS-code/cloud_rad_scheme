from cloud_rad_scheme import compute_liq, planck, read_solar_spectrum
import numpy as np
from scipy.interpolate import interp1d


def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_lut = np.zeros((2,15)) # look-up-table (piecewise linear coefficients) size range, micron
    re_range_pade = np.zeros((2,3)) # Padé approximantsize size range, micron
    re_range_lut[0,:] = np.append(np.append([1.1,2,3,5,7],np.arange(10,30,5)),np.arange(40,100,10))
    re_range_lut[1,:] = np.append(np.append([2,3,5,7],np.arange(10,30,5)),np.arange(40,110,10))
    re_range_pade[0,:] = [1.2, 10, 50]
    re_range_pade[1,:] = [10, 50, 100] # Please check compute_liquid module variable 'd', if a wider range is required
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

    # generate parameterization for shortwave liquid
    compute_liq('25bands_lut_liq_sw_mie_gamma_aeq12_thick.nc',
                '25bands_pade_liq_sw_mie_gamma_aeq12_thick.nc',
                12, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade, False)


if __name__ == "__main__":
    main()
