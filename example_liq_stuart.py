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

    # initialize shortwave band limits that matches with rrtmgp gas optics
    nband = 14
    if nband==25:
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
    if nband==18:
        band_limit = np.array([[  1., 2500.],
                               [ 2500., 4200.],
                               [4200., 8200.],
                               [8200., 11500.],
                               [11500., 14600.],
                               [14600., 16700.],
                               [16700., 20000.],
                               [20000., 22300.],
                               [22300., 24600.],
                               [24600., 27500.],
                               [27500., 32400.],
                               [32400., 33000.],
                               [33000., 34500.],
                               [34500., 35300.],
                               [35300., 36500.],
                               [36500., 40000.],
                               [40000., 43300.],
                               [43300., 50000.]])            
    if nband==14:
        band_limit = np.array([[820., 2500.],
                               [2680., 4350.],
                               [3250., 4000.],
                               [4000., 4650.],
                               [4650., 5150.],
                               [5150., 6150.],
                               [6150., 7700.],
                               [7700., 8050.],
                               [8050., 12850.],
                               [12850., 16000.],
                               [16000., 22650.],
                               [22650., 29000.],
                               [29000., 38000.],
                               [38000., 50000.]])      
    
    wavenum = np.arange(1, 57600., 50)
    # read-in shortwave spectrum
    wavenum_solar, solar = read_solar_spectrum()
    source = interp1d(wavenum_solar[:], solar[:])(wavenum[:])

    # generate parameterization for shortwave liquid
    compute_liq('hres_liq_sw_mie_gamma_aeq12_25band.nc',
                '14band_band_liq_sw_mie_gamma_aeq12_thick.nc',
                '14band_pade_liq_sw_mie_gamma_aeq12_thick.nc',
                12, wavenum, source, band_limit, re_range_pade,
                re_ref_pade, False)


if __name__ == "__main__":
    main()
