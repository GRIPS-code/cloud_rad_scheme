from cloud_rad_scheme import compute_liq, planck, read_solar_spectrum, compute_existingLUT
from cloud_rad_scheme.optics import optics_var
import numpy as np
import math
from scipy.interpolate import interp1d

rau = 917*10**3 
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

    brn = np.array([.87477,  .85421,  .84825,  .84418,  .84286,  .84143, 
              .84097,  .84058,  .84029,  .83995,  .83979,  .83967])
    
    wrnf = np.array([.55474,  .53160,  .54307,  .55258,  .54914,  .52342, 
              .52446,  .52959,  .53180,  .53182,  .53017,  .53296])
    
    grn = np.array([.93183,  .97097,  .95539,  .94213,  .94673,  .98396, 
              .98274,  .97626,  .97327,  .97330,  .97559,  .97173 ])
    
    swc0 = 0.5

    band_in =   np.array([280,    400,    540,   670,  800,  980,  1100,  1250, 
                1400,   1700,   1900,   2200])

    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))

    ext_in[0,:] = brn/swc0 * (1-wrnf)
    ssa_in[0,:] = 0.0
    asy_in[0,:] = 1.0 #grn
    sca_in[0,:] = ext_in[0,:] * ssa_in[0,:]

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               band_limit=band_in)

    compute_existingLUT(cloud_optics_in.band2wav_cloud_optics(wavenum),
                        'lut_snow_lw_fu_thick.nc', 
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

    ext = np.array([8.3951E-01,8.3946E-01,8.3941E-01,8.3940E-01,    
                   8.3940E-01,8.3939E-01])
    
    ssalb = np.array([5.3846E-01,5.2579E-01,5.3156E-01,5.6192E-01,   
                   9.7115E-01,9.99911E-01])
    
    asymm = np.array([9.6373E-01,9.8141E-01,9.7816E-01,9.6820E-01,   
                   8.9940E-01,8.9218E-01])

    band_in =   np.array([2857, 4000, 5263, 7692, 14493, 57600
                          ])
    
    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))
    
    ext_in[0,:] = ext
    ssa_in[0,:] = ssalb
    asy_in[0,:] = asymm
    sca_in[0,:] = ext_in[0,:] * ssa_in[0,:]

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               band_limit=band_in)
    

    compute_existingLUT(cloud_optics_in.band2wav_cloud_optics(wavenum),
                        'lut_snow_sw_fu_thick.nc', 
                        'tmp.nc',
                        source, 
                        band_limit,re_range_lut,
                        [], [], False)


if __name__ == "__main__":
    main()

