from cloud_rad_scheme import compute_rain, planck, read_solar_spectrum, create_list
import numpy as np
from scipy.interpolate import interp1d

def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_pade = np.zeros((2,6)) # Padé approximantsize size range, micron
    tmp = create_list(0.2,5000,2.5)
    re_range_lut = np.zeros((2,len(tmp)-2))
    re_range_lut[0,:] = tmp[0:-2]
    re_range_lut[1,:] = tmp[1:-1]
    print(tmp[0:-2])
    re_range_pade[0,:] = [0.2, 1., 10., 50., 100., 1000.]
    re_range_pade[1,:] = [1., 10, 50, 100., 1000., 5000.] # Please check compute_liquid module variable 'd', if a wider range is required
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
    compute_rain('hres_rain_lw_mie_gamma_aeq12.nc',
                 'lut_rain_lw_mie_gamma_aeq12_thick.nc',
                'pade_rain_lw_mie_gamma_aeq12_thick.nc',
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
    compute_rain('hres_rain_sw_mie_gamma_aeq12.nc',
                 'lut_rain_sw_mie_gamma_aeq12_thick.nc',
                'pade_rain_sw_mie_gamma_aeq12_thick.nc',
                12, wavenum, source, band_limit, re_range_lut, re_range_pade,
                re_ref_pade, False)

if __name__ == "__main__":
    main()
