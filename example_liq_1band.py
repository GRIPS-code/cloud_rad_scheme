from cloud_rad_scheme import compute_liq, planck, read_solar_spectrum, create_list
import numpy as np
from scipy.interpolate import interp1d

def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_pade = np.zeros((2,7)) # Padé approximantsize size range, micron
    tmp = create_list(0.25,5000,2.5)
    re_range_pade[0,:] = [0.2, 1., 5.,  15., 50., 100., 1000.]
    re_range_pade[1,:] = [1., 5., 15., 50, 100., 1000., 17300.] # Please check compute_liquid module variable 'd', if a wider range is required
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
    compute_liq('hres_liq_lw_mie_gamma_aeq1.nc',
                 'band_liq_lw_mie_gamma_aeq1_thick.nc',
                'pade_liq_lw_mie_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_pade,
                re_ref_pade, True)

    # initialize shortwave band limits that matches with rrtmgp gas optics
    band_limit = np.array([[  820.,  50000.]])
    wavenum = np.arange(band_limit[0,0], band_limit[-1,-1], 10)
    # read-in shortwave spectrum
    wavenum_solar, solar = read_solar_spectrum()
    source = interp1d(wavenum_solar[:], solar[:])(wavenum[:])

    # generate parameterization for shortwave liquid
    compute_liq('hres_liq_sw_mie_gamma_aeq1.nc',
                 '1band_band_liq_sw_mie_gamma_aeq1_thick.nc',
                '1band_pade_liq_sw_mie_gamma_aeq1_thick.nc',
                1, wavenum, source, band_limit, re_range_pade,
                re_ref_pade, False)

if __name__ == "__main__":
    main()
