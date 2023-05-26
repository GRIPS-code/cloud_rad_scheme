from cloud_rad_scheme import compute_liq, planck, read_solar_spectrum, compute_existingLUT
from cloud_rad_scheme.optics import optics_var
import numpy as np
import math
from scipy.interpolate import interp1d

rau = 996*10**3 
def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_lut = np.zeros((2,61)) # look-up-table (piecewise linear coefficients) size range, micron
    re_range_lut[0,:] = np.arange(4.2,16.4,0.2)
    re_range_lut[1,:] = np.arange(4.4,16.6,0.2)
    re_range_pade=np.zeros((2,2))
    re_range_pade[0,:] = [4.2, 8]
    re_range_pade[1,:] = [8, 16.6]
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

    # Held 
    r = np.arange(4.2,16.6,0.1)
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0

    nsize = len(r)
    nwav = len(wavenum)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))

    ext_in[:] = 0.1
    sca_in[:] = 0.1

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               wavenum)
    
    compute_existingLUT(cloud_optics_in,
                        'lut_liq_lw_held_thick.nc', 
                        'pade_liq_lw_held_thick.nc',
                        source, 
                        band_limit,re_range_lut,
                        re_range_pade, re_ref_pade, False)
    
    ################################
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

    # Slingo
    # generate parameterization for shortwave liquid
    r = np.arange(4.2,16.6,0.1)
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0
    a = np.array([-1.023E+00, 1.950E+00, 1.579E+00, 1.850E+00, 1.970E+00, 
        2.237E+00, 2.463E+00, 2.551E+00, 2.589E+00, 2.632E+00, 
        2.497E+00, 2.622E+00, 2.650E+00, 3.115E+00, 2.895E+00, 
        2.831E+00, 2.838E+00, 2.672E+00, 2.698E+00, 2.668E+00, 
        2.801E+00, 3.308E+00, 2.944E+00, 3.094E+00])
    b = np.array([1.933E+00, 1.540E+00, 1.611E+00, 1.556E+00, 1.501E+00, 
        1.452E+00, 1.420E+00, 1.401E+00, 1.385E+00, 1.365E+00, 
        1.376E+00, 1.362E+00, 1.349E+00, 1.244E+00, 1.315E+00, 
        1.317E+00, 1.300E+00, 1.320E+00, 1.315E+00, 1.307E+00, 
        1.293E+00, 1.246E+00, 1.270E+00, 1.252E+00])
    c = np.array([2.500E-02, 4.490E-01, 1.230E-01, 1.900E-04, 1.200E-03, 
               1.200E-04, 2.400E-04, 6.200E-05,-2.800E-05,-4.600E-05, 
               9.800E-06, 3.300E-06, 2.300E-06,-2.700E-07,-1.200E-07, 
              -1.200E-06, 0.000E+00, 0.000E+00, 1.000E-06, 0.000E+00, 
               1.000E-06,-3.000E-07,-6.500E-07, 7.900E-07])
    d = np.array([1.220E-02, 1.540E-03, 9.350E-03, 2.540E-03, 2.160E-03, 
               6.670E-04, 8.560E-04, 2.600E-04, 8.000E-05, 5.000E-05, 
               2.100E-05, 2.800E-06, 1.700E-06, 1.400E-06, 4.400E-07, 
               4.000E-07, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 
               0.000E+00, 2.360E-07, 4.330E-07, 3.690E-07])
    e = np.array([7.260E-01, 8.310E-01, 8.510E-01, 7.690E-01, 7.400E-01, 
               7.490E-01, 7.540E-01, 7.730E-01, 7.800E-01, 7.840E-01, 
               7.830E-01, 8.060E-01, 8.090E-01, 8.040E-01, 8.180E-01, 
               8.280E-01, 8.250E-01, 8.280E-01, 8.200E-01, 8.400E-01, 
               8.360E-01, 8.390E-01, 8.410E-01, 8.440E-01])
    f = np.array([6.652E+00, 6.102E+00, 2.814E+00, 5.171E+00, 7.469E+00, 
               6.931E+00, 6.555E+00, 5.405E+00, 4.989E+00, 4.745E+00, 
               5.035E+00, 3.355E+00, 3.387E+00, 3.520E+00, 2.989E+00, 
               2.492E+00, 2.776E+00, 2.467E+00, 3.004E+00, 1.881E+00, 
               2.153E+00, 1.946E+00, 1.680E+00, 1.558E+00])
    band_in = np.array([2924,  3437,  4202,  4695,  6098,  6536,  7813,   
                     8404,  9091, 10000, 11494, 12821, 13333, 14493,   
                    15625, 17544, 19231, 20833, 22727, 25000, 27778,   
                    30303, 33333, 57600])
    
    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))

    for i in range(nsize):
        ext_in[i,:] = 1.0E-02*a + b/r[i]
        ssa_in[i,:] = 1.0 - c + d * r[i]
        asy_in[i,:] = e + 1.0E-03* f * r[i]
        sca_in[i,:] = ext_in[i,:] * ssa_in[i,:]

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               band_limit=band_in)
    
    compute_existingLUT(cloud_optics_in.band2wav_cloud_optics(wavenum),
                        'lut_liq_sw_slingo_thick.nc', 
                        'pade_liq_sw_slingo_thick.nc',
                        source, 
                        band_limit,re_range_lut,
                        re_range_pade, re_ref_pade, False)


if __name__ == "__main__":
    main()

