from cloud_rad_scheme import compute_liq, planck, read_solar_spectrum, compute_existingLUT, create_list
from cloud_rad_scheme.optics import optics_var
import numpy as np
import math
from scipy.interpolate import interp1d

rau = 917*10**3 
def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    re_range_lut = np.zeros((2,22)) # look-up-table (piecewise linear coefficients) size range, micron
    re_range_lut[0,:] = np.arange(18.6,125.2,5)/2.0
    re_range_lut[1,:] = np.arange(23.6,130.2,5)/2.0
    re_range_pade=np.zeros((2,5))
    re_range_pade[0,:] = [18.6, 48.6, 78, 100.2, 120]
    re_range_pade[1,:] = [48.6, 78, 100.2, 120, 1800*2]
    re_range_pade = re_range_pade/2.0
    re_ref_pade = np.zeros(np.shape(re_range_pade)[1],)
    
    flag_nolwscat = True

    r = np.append(np.arange(18.6/2.0,130.2/2.0,0.1),np.arange(100.,1800.,100))
    d = r*2.0
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0
    range_cloud = np.where(d<=120)
    range_precip = np.where(d>120)

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
    # fu, q., an accurate parameterization of the solar radiative    
    # properties of cirrus clouds for climate models., j. climate,     
    # 9, 2058-2082, 1996

    a0 = np.array([-7.752E-03,  -1.741E-02,  -1.704E-02,  -1.151E-02,    
          -1.026E-02,  -8.294E-03,  -1.153E-02,  -9.609E-03,         
          -9.061E-03,  -8.441E-03,  -8.088E-03,  -7.770E-03])
    
    a1 = np.array([4.624,   5.541,   4.830,   4.182,   4.105,   3.925,       
                   4.109,   3.768,   3.741,   3.715,   3.717,   3.734])
    
    a2 = np.array([-42.010, -58.420,  16.270,  31.130,  16.360,   1.315,      
                   17.320,  34.110,  26.480,  19.480,  17.170,  11.850])
    b0 = np.array([0.8079,   0.3964,   0.1028,   0.3254,   0.5207,   0.5631,  
          0.2307,   0.2037,   0.3105,   0.3908,   0.3014,   0.1996])
    b1 = np.array([-0.7004E-02, -0.3155E-02,  0.5019E-02,  0.3434E-02,         
                   -0.9778E-03, -0.1434E-02,  0.3830E-02,  0.4247E-02,         
                   0.2603E-02,  0.1272E-02,  0.2639E-02,  0.3780E-02])
    b2= np.array([0.5209E-04,  0.6417E-04, -0.2024E-04, -0.3081E-04,         
                  0.3725E-05,  0.6298E-05, -0.1616E-04, -0.1810E-04,         
                  -0.1139E-04, -0.5564E-05, -0.1116E-04, -0.1491E-04])
    b3= np.array([-0.1425E-06, -0.2979E-06,  0.0000E+00,  0.9143E-07,         
                  0.0000E+00,  0.0000E+00,  0.0000E+00,  0.0000E+00,         
                  0.0000E+00,  0.0000E+00,  0.0000E+00,  0.0000E+00])
    cpr0 = np.array([0.2292,   0.7024,   0.7290,   0.7678,   0.8454,   0.9092,  
                     0.9167,   0.8815,   0.8765,   0.8915,   0.8601,   0.7955])
    cpr1 = np.array([1.724E-02,   4.581E-03,   2.132E-03,   2.571E-03,         
                     1.429E-03,   9.295E-04,   5.499E-04,   9.858E-04,         
                     1.198E-03,   1.060E-03,   1.599E-03,   2.524E-03])
    cpr2 = np.array([-1.573E-04,  -3.054E-05,  -5.584E-06,  -1.041E-05,         
                     -5.859E-06,  -3.877E-06,  -1.507E-06,  -3.116E-06,         
                     -4.485E-06,  -4.171E-06,  -6.465E-06,  -1.022E-05])
    cpr3 = np.array([4.995E-07,   6.684E-08,   0.000E+00,   0.000E+00,         
                     0.000E+00,   0.000E+00,   0.000E+00,   0.000E+00,         
                     0.000E+00,   0.000E+00,   0.000E+00,   0.000E+00])

    band_in = np.array([280,    400,    540,   670,  800,  980,  1100,  1250, 
                1400,   1700,   1900,   2200])  ## Be aware that Fu1996 in the longwave does not cover the entire RRTMGP LW bands
    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))

    for i in range(nsize):
        if flag_nolwscat: 
            ext_in[i,:] = (a0 + a1/d[i] + a2/d[i]**2)*(b0 + b1*d[i] + b2*d[i]**2 + b3*d[i]**3)
            ssa_in[i,:] = 0.0
            asy_in[i,:] = 1.0
        else:
            ext_in[i,:] = (a0 + a1/d[i] + a2/d[i]**2)
            ssa_in[i,:] = 1 - (b0 + b1*d[i] + b2*d[i]**2 + b3*d[i]**3)
            asy_in[i,:] = (cpr0 + cpr1*d[i] + cpr2*d[i]**2 + cpr3*d[i]**3)

    sca_in[:,:] = ext_in[:,:] * ssa_in[:,:]

    optics_cloud=optics_var(r[range_cloud], s[range_cloud], v[range_cloud],
               ext_in[range_cloud,:], sca_in[range_cloud,:], ssa_in[range_cloud,:], asy_in[range_cloud,:], rau,
               band_limit=band_in)
    optics_cloud=optics_cloud.band2wav_cloud_optics(wavenum)
    optics_cloud_band = optics_cloud.thin_average(source,band_limit)

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

    if flag_nolwscat: 
        ext_in[:,:] = brn/swc0/1000* (1-wrnf)
        ssa_in[:,:] = 0.0 #wrnf
        asy_in[:,:] = 1.0 #grn
    else:
        ext_in[:,:] = brn/swc0/1000
        ssa_in[:,:] = wrnf
        asy_in[:,:] = grn

    sca_in[:,:] = ext_in[:,:] * ssa_in[:,:]

    optics_precip=optics_var(r[range_precip], s[range_precip], v[range_precip],
               ext_in[range_precip,:], sca_in[range_precip,:], ssa_in[range_precip,:], asy_in[range_precip,:], rau,
               band_limit=band_in)
    optics_precip=optics_precip.band2wav_cloud_optics(wavenum)
    optics_precip_band = optics_precip.thin_average(source,band_limit)
    
    v_range_pade = re_range_pade **3.0 * 4.0 / 3.0 * math.pi

    optics_band = optics_var.combine(optics_cloud_band,optics_precip_band)
    optics_band.create_pade_coeff(re_range_pade,re_ref_pade,v_range_pade,'pade_ice_lw_AM4MG2_noscat_thick.nc')
    
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

    # fu and Liou (1993, JAS) 
    # generate parameterization for shortwave liquid
    r = np.append(np.arange(18.6/2.0,130.2/2.0,0.1),np.arange(100.,1800.,100))
    d = r*2.0
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0
    range_cloud = np.where(d<=120)
    range_precip = np.where(d>120)

    a0 = -6.656e-03
    a1 =  3.686
    b0 = np.array([.10998e-5,  .20208e-4, .1359e-3,  -.16598e-2,  .4618,      .42362e-1])
    b1 = np.array([ -.26101e-7,  .96483e-5, .73453e-3,  .20933e-2,  .24471e-3,  .86425e-2])
    b2 = np.array([ .10896e-8,  .83009e-7, .28281e-5, -.13977e-5, -.27839e-5, -.75519e-4])
    b3 = np.array([ -.47387e-11,-.32217e-9,-.18272e-7, -.18703e-7,  .10379e-7,  .24056e-6])

    c0 = np.array([2.211,      2.2151,    2.2376,    2.3012,    2.7975,    1.9655])
    c1 = np.array([-.10398e-2, -.77982e-3, .10293e-2, .33854e-2, .29741e-2, .20094e-1])
    c2 = np.array([.65199e-4,  .6375e-4,  .50842e-4, .23528e-4,-.32344e-4,-.17067e-3])
    c3 = np.array([-.34498e-6, -.34466e-6,-.30135e-6,-.20068e-6, .11636e-6, .50806e-6])

    d0 = np.array([.12495,    .12363,    .12117,    .11581,   -.15968e-3, .1383])
    d1 = np.array([-.43582e-3,-.44419e-3,-.48474e-3,-.55031e-3, .10115e-4,-.18921e-2])
    d2 = np.array([.14092e-4, .14038e-4, .12495e-4, .98776e-5,-.12472e-6, .1203e-4])
    d3 = np.array([-.69565e-7,-.68851e-7,-.62411e-7,-.50193e-7, .48667e-9,-.31698e-7])
    
    band_in = np.array([2857, 4000, 5263, 7692, 14493, 57600])
    
    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))

    for ni in range(6):
        for i in range(nsize):
            ext_in[i,ni] = a0 + (a1/d[i])
            ssa_in[i,ni] = 1.0 - (b0[5-ni] + b1[5-ni]*d[i] + b2[5-ni]*d[i]**2 + b3[5-ni]*d[i]**3)
            fgam2  = c0[5-ni] + c1[5-ni]*d[i] + c2[5-ni]*d[i] **2 + c3[5-ni]*d[i] **3
            fdel2  = d0[5-ni] + d1[5-ni]*d[i] + d2[5-ni]*d[i] **2 + d3[5-ni]*d[i] **3
            asy_in[i,ni] = ((1. - fdel2)*fgam2 + 3.*fdel2)/3.
            sca_in[i,ni] = ext_in[i,ni] * ssa_in[i,ni]

    optics_cloud=optics_var(r[range_cloud], s[range_cloud], v[range_cloud],
               ext_in[range_cloud,:], sca_in[range_cloud,:], ssa_in[range_cloud,:], asy_in[range_cloud,:], rau,
               band_limit=band_in)
    optics_cloud=optics_cloud.band2wav_cloud_optics(wavenum)
    optics_cloud_band = optics_cloud.thick_average(source,band_limit)
    
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
    
    ext_in[:,:] = ext/1000
    ssa_in[:,:] = ssalb
    asy_in[:,:] = asymm
    sca_in[:,:] = ext_in[:,:] * ssa_in[:,:]

    optics_precip=optics_var(r[range_precip], s[range_precip], v[range_precip],
               ext_in[range_precip,:], sca_in[range_precip,:], ssa_in[range_precip,:], asy_in[range_precip,:], rau,
               band_limit=band_in)

    optics_precip=optics_precip.band2wav_cloud_optics(wavenum)
    optics_precip_band = optics_precip.thick_average(source,band_limit)
    v_range_pade = re_range_pade **3.0 * 4.0 / 3.0 * math.pi

    optics_band = optics_var.combine(optics_cloud_band,optics_precip_band)
    optics_band.create_pade_coeff(re_range_pade,re_ref_pade,v_range_pade,'pade_ice_sw_AM4MG2_thick.nc')

if __name__ == "__main__":
    main()

