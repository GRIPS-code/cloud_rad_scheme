from cloud_rad_scheme import compute_rain, planck, read_solar_spectrum
import numpy as np
from scipy.interpolate import interp1d

def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_lut = np.zeros((2,3)) # look-up-table (piecewise linear coefficients) size range, micron
    re_range_pade = np.zeros((2,6)) # Padé approximantsize size range, micron
    re_range_lut[0,:] = [16., 100., 1000.]
    re_range_lut[1,:] = [100., 1000., 5000.]
    re_range_pade[0,:] = [0.1, 1., 10., 50., 100., 1000.]
    re_range_pade[1,:] = [1., 10, 50, 100., 1000., 5000.]
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
    compute_rain('lut_rain_lw_mie_gamma_aeq1_thick.nc',
                'pade_rain_lw_mie_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_lut, re_range_pade,
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
    compute_rain('lut_rain_sw_mie_gamma_aeq1_thick.nc',
                'pade_rain_sw_mie_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade, False)


if __name__ == "__main__":
    main()
