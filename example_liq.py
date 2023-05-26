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
    wavenum = np.arange(band_limit[0,0],band_limit[-1,-1],1)

    # initialize longwave source function
    source = planck(wavenum, 250) # use 250 K as a reference
    # generate parameterization for longwave liquid
    compute_liq('lut_liq_lw_mie_gamma_aeq12_thick.nc',
                'pade_liq_lw_mie_gamma_aeq12_thick.nc',
                12, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade, False)

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

    # generate parameterization for shortwave liquid
    compute_liq('lut_liq_sw_mie_gamma_aeq12_thick.nc',
                'pade_liq_sw_mie_gamma_aeq12_thick.nc',
                12, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade, False)


if __name__ == "__main__":
    main()
