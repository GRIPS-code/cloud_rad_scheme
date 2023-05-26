from cloud_rad_scheme import compute_liq, planck, read_solar_spectrum, compute_existingLUT
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
    re_range_pade=np.zeros((2,4))
    re_range_pade[0,:] = [18.6, 48.6, 78, 100.2]
    re_range_pade[1,:] = [48.6, 78, 100.2, 130.2]
    re_range_pade[:] = re_range_pade/2.0
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
    # fu, q., an accurate parameterization of the solar radiative    
    # properties of cirrus clouds for climate models., j. climate,     
    # 9, 2058-2082, 1996

    r = np.arange(18.6/2.0,130.2/2.0,0.1)
    d = r*2.0
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0
    a0fu  = np.array([-2.54823E-04, 1.87598E-04, 2.97295E-04, 2.34245E-04, 
                   4.89477E-04,-8.37325E-05, 6.44675E-04,-8.05155E-04, 
                   6.51659E-05, 4.13595E-04,-6.14288E-04, 7.31638E-05, 
                   8.10443E-05, 2.26539E-04,-3.04991E-04, 1.61983E-04, 
                   9.82244E-05,-3.03108E-05,-9.45458E-05, 1.29121E-04, 
                  -1.06451E-04,-2.58858E-04,-2.93599E-04,-2.66955E-04, 
                  -2.36447E-04])
    a1fu = np.array([2.52909E+00, 2.51396E+00, 2.48895E+00, 2.48573E+00, 
                   2.48776E+00, 2.52504E+00, 2.47060E+00, 2.57600E+00, 
                   2.51660E+00, 2.48783E+00, 2.56520E+00, 2.51051E+00, 
                   2.51619E+00, 2.49909E+00, 2.54412E+00, 2.50746E+00, 
                   2.50875E+00, 2.51805E+00, 2.52061E+00, 2.50410E+00, 
                   2.52684E+00, 2.53815E+00, 2.54540E+00, 2.54179E+00, 
                   2.53817E+00])
    b0fu = np.array([2.60155E-01, 1.96793E-01, 4.64416E-01, 9.05631E-02, 
                   5.83469E-04, 2.53234E-03, 2.01931E-03,-2.85518E-05, 
                  -1.48012E-07, 6.47675E-06,-9.38455E-06,-2.32733E-07, 
                  -1.57963E-07,-2.75031E-07, 3.12168E-07,-7.78001E-08, 
                  -8.93276E-08, 9.89368E-08, 5.08447E-07, 7.10418E-07, 
                   3.25057E-08,-1.98529E-07, 1.82299E-07,-1.00570E-07, 
                  -2.69916E-07])
    b1fu = np.array([5.45547E-03, 5.75235E-03, 2.04716E-05, 2.93035E-03, 
                   1.18127E-03, 1.75078E-03, 1.83364E-03, 1.71993E-03, 
                   9.02355E-05, 2.18111E-05, 1.77414E-05, 6.41602E-06, 
                   1.72475E-06, 9.72285E-07, 4.93304E-07, 2.53360E-07, 
                   1.14916E-07, 5.44286E-08, 2.73206E-08, 1.42205E-08, 
                   5.43665E-08, 9.39480E-08, 1.12454E-07, 1.60441E-07, 
                   2.12909E-07])
    b2fu= np.array([-5.58760E-05,-5.29220E-05,-4.60375E-07,-1.89176E-05, 
                  -3.40011E-06,-8.00994E-06,-7.00232E-06,-7.43697E-06, 
                  -1.98190E-08, 1.83054E-09,-1.13004E-09, 1.97733E-10, 
                   9.02156E-11,-2.23685E-10, 1.79019E-10,-1.15489E-10, 
                  -1.62990E-10,-1.00877E-10, 4.96553E-11, 1.99874E-10, 
                  -9.24925E-11,-2.54540E-10,-1.08031E-10,-2.05663E-10, 
                  -2.65397E-10])
    b3fu = np.array([1.97086E-07, 1.76618E-07, 2.03198E-09, 5.93361E-08, 
                   8.78549E-09, 2.31309E-08, 1.84287E-08, 2.09647E-08, 
                   4.01914E-11,-8.28710E-12, 2.37196E-12,-6.96836E-13, 
                  -3.79423E-13, 5.75512E-13,-7.31058E-13, 4.65084E-13, 
                   6.53291E-13, 4.56410E-13,-1.86001E-13,-7.81101E-13, 
                   4.53386E-13, 1.10876E-12, 4.99801E-13, 8.88595E-13,  
                   1.12983E-12])
    c0fu = np.array([7.99084E-01, 7.59183E-01, 9.19599E-01, 8.29283E-01,  
                   7.75916E-01, 7.58748E-01, 7.51497E-01, 7.52528E-01,  
                   7.51277E-01, 7.52292E-01, 7.52048E-01, 7.51715E-01,  
                   7.52318E-01, 7.51779E-01, 7.53393E-01, 7.49693E-01,  
                   7.52131E-01, 7.51135E-01, 7.49856E-01, 7.48613E-01,  
                   7.47054E-01, 7.43546E-01, 7.40926E-01, 7.37809E-01,  
                   7.33260E-01])
    c1fu = np.array([4.81706E-03, 4.93765E-03, 5.03025E-04, 2.06865E-03,  
                   1.74517E-03, 2.02709E-03, 2.05963E-03, 1.95748E-03,  
                   1.29824E-03, 1.14395E-03, 1.12044E-03, 1.10166E-03,  
                   1.04224E-03, 1.03341E-03, 9.61630E-04, 1.05446E-03,  
                   9.37763E-04, 9.09208E-04, 8.89161E-04, 8.90545E-04,  
                   8.86508E-04, 9.08674E-04, 8.90216E-04, 8.97515E-04,  
                   9.18317E-04])
    c2fu = np.array([-5.13220E-05,-4.84059E-05,-5.74771E-06,-1.59247E-05,  
                  -9.21314E-06,-1.17029E-05,-1.12135E-05,-1.02495E-05,  
                  -4.99075E-06,-3.27944E-06,-3.11826E-06,-2.91300E-06,  
                  -2.26618E-06,-2.13121E-06,-1.32519E-06,-2.32576E-06,  
                  -9.72292E-07,-6.34939E-07,-3.49578E-07,-3.44038E-07,  
                  -2.59305E-07,-4.65326E-07,-1.87919E-07,-2.17099E-07,  
                  -4.22974E-07])
    c3fu = np.array([1.84420E-07, 1.65801E-07, 2.01731E-08, 5.01791E-08,  
                   2.15003E-08, 2.95195E-08, 2.73998E-08, 2.35479E-08,  
                   6.33757E-09,-2.42583E-10,-5.70868E-10,-1.37242E-09,  
                  -3.68283E-09,-4.24308E-09,-7.17071E-09,-3.58307E-09,  
                  -8.62063E-09,-9.84390E-09,-1.09913E-08,-1.10117E-08,  
                  -1.13305E-08,-1.05786E-08,-1.16760E-08,-1.16090E-08,  
                  -1.07976E-08])
 
    band_in = np.array([280,    400,    540,   670,  800,  980,  1100,  1250, 
                1400,   1700,   1900,   2200])  ## Be aware that Fu1996 in the longwave does not cover the entire RRTMGP LW bands
    nsize = len(r)
    nwav = len(band_in)
    ext_in = np.zeros((nsize,nwav))
    sca_in = np.zeros((nsize,nwav))
    ssa_in = np.zeros((nsize,nwav))
    asy_in = np.zeros((nsize,nwav))

    for i in range(nsize):
        ext_in[i,:] = a0fu[:nwav] + (a1fu[:nwav]/d[i])
        ssa_in[i,:] = 1-(b0fu[:nwav] + b1fu[:nwav]*d[i] + b2fu[:nwav]*d[i]**2 + b3fu[:nwav]*d[i]**3)
        asy_in[i,:] = c0fu[:nwav] + c1fu[:nwav]*d[i] + c2fu[:nwav]*d[i]**2 + c3fu[:nwav]*d[i]**3
        sca_in[i,:] = ext_in[i,:] * ssa_in[i,:]

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               band_limit=band_in)
    
    compute_existingLUT(cloud_optics_in.band2wav_cloud_optics(wavenum),
                        'lut_ice_lw_fu1996_thick.nc', 
                        'pade_ice_lw_fu1996_thick.nc',
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

    # fu and Liou (1993, JAS) 
    # generate parameterization for shortwave liquid
    r = np.arange(18.6/2.0,130.2/2.0,0.1)
    d = r*2.0
    s = np.power(r, 2) * math.pi
    v = np.power(r, 3) * math.pi * 4.0 / 3.0
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

    cloud_optics_in=optics_var(r, s, v,
               ext_in, sca_in, ssa_in, asy_in, rau,
               band_limit=band_in)
    
    compute_existingLUT(cloud_optics_in.band2wav_cloud_optics(wavenum),
                        'lut_ice_sw_fuliou1993_thick.nc', 
                        'pade_ice_sw_fuliou1993_thick.nc',
                        source, 
                        band_limit,re_range_lut,
                        re_range_pade, re_ref_pade, False) 
if __name__ == "__main__":
    main()

