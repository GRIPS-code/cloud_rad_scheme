from cloud_rad_scheme import compute_liq, planck, read_solar_spectrum, compute_existingLUT
from cloud_rad_scheme.optics import optics_var
import numpy as np
import math
from scipy.interpolate import interp1d

rau = 996*10**3 
def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_lut = np.zeros((2,1)) # look-up-table (piecewise linear coefficients) size range, micron
    re_range_lut[0,:] = 60.
    re_range_lut[1,:] = 1800.
    r = np.zeros((1))
    r[0] = 100.
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0

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

    # Fu rain lw
    #the absorption coefficient for 
   # rain water for longwave radiation (Fu et al., 1995, J. Atmos. Sci.)
   # it is designed for use with rain drops with radii between 60 um 
   # and 1.8 mm. see also notes from Q. Fu (4 Sept 98). note that the 
   # size of the rain water from the microphysical model (if existing) 
   # does not enter into the calculations.
   # Leo Donner, GFDL, 20 Mar 99
    # generate parameterization for shortwave liquid

    brn = np.array([1.6765,  1.6149,  1.5993,  1.5862,  1.5741,  1.5647,
             1.5642,  1.5600,  1.5559,  1.5512,  1.5478,  1.5454])
    
    wrnf = np.array([.55218,  .55334,  .55488,  .55169,  .53859,  .51904, 
              .52321,  .52716,  .52969,  .53192,  .52884,  .53233])
    
    grn = np.array([.90729,  .92990,  .93266,  .94218,  .96374,  .98584,   
              .98156,  .97745,  .97467,  .97216,  .97663,  .97226 ])
    
    rwc0 = 0.5

    band_in =   np.array([280,    400,    540,   670,  800,  980,  1100,  1250, 
                1400,   1700,   1900,   2200])

    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))

    ext_in[0,:] = brn/rwc0/1000 #* (1-wrnf)
    ssa_in[0,:] = wrnf #0.0
    asy_in[0,:] = grn #1.0 
    sca_in[0,:] = ext_in[0,:] * ssa_in[0,:]

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               band_limit=band_in)

    compute_existingLUT(cloud_optics_in.band2wav_cloud_optics(wavenum),
                        'lut_rain_lw_fu_thick.nc', 
                        'tmp.nc',
                        source, 
                        band_limit,re_range_lut,
                        [], [], True)
    
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

    re_range_lut = np.zeros((2,3)) # look-up-table (piecewise linear coefficients) size range, micron
    re_range_lut[0,:] = [16.6, 100., 1000.]
    re_range_lut[1,:] = [100., 1000., 5000.]
    re_range_pade=np.zeros((2,1))
    re_range_pade[0,:] = [16.6]
    re_range_pade[1,:] = [5000.]
    re_ref_pade = np.zeros(np.shape(re_range_pade)[1],)

    r = np.array([16.6, 30., 50., 100., 200., 500., 1000., 2500., 5000.])
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0

#----------------------------------------------------------------------
#    subroutine savijarvi defines the single scattering parameters for 
#    rain drops using the Savijarvi parameterization for his spectral 
#    intervals. references:                                     
#    savijarvi, h., shortwave optical properties of rain., tellus, 49a, 
#    177-181, 1997.                                                   
#---------------------------------------------------------------------- 
    a = np.array([4.65E-01, 2.64E-01, 1.05E-02, 8.00E-05])
    
    b = np.array([1.00E-03, 9.00E-02, 2.20E-01, 2.30E-01])
    
    asymm = np.array([ 9.70E-01, 9.40E-01, 8.90E-01, 8.80E-01])

    band_in =   np.array([4202, 8403, 14493, 57600])
    
    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))
    
    rcap = pow(r/500, 4.348E+00)

    for i in range(nsize):
        ext_in[i,:] = 1.505E+00/r[i] 
        ssa_in[i,:] = 1.0 - (a * rcap[i] * b)
        asy_in[i,:] = asymm
        sca_in[i,:] = ext_in[i,:] * ssa_in[i,:]

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               band_limit=band_in)
    
    compute_existingLUT(cloud_optics_in.band2wav_cloud_optics(wavenum),
                        'lut_rain_sw_savijarvi_thick.nc', 
                        'pade_rain_sw_savijarvi_thick.nc',
                        source, 
                        band_limit,re_range_lut,
                        re_range_pade, re_ref_pade, False)


if __name__ == "__main__":
    main()

