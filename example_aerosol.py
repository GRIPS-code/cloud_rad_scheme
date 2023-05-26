from cloud_rad_scheme import compute_mie_aerosol
import numpy as np
import math


def main():
    # initialize parameterization size range for look-up-table and Padé approximantsize
    rau = 2160 # kg/m3, dry mass
    min_re = [0.01,    0.1,      1.0,       2.5,      5.0  ] # um
    max_re = [0.1,     1.0,      2.5,       5.0,     10.0  ] # um
    re = ([0.078222,0.3580154,0.7310735, 1.125,    2.2774594]) # um
    sigma = np.log([2.0,     2.0,      2.0,       2.0,      2.0      ]) 

    refractive_index_input = './data/aerosol/refractive_index_RH30'
    file_out = 'RH30_rau2160_mu.nc'
    compute_mie_aerosol(sigma, min_re,max_re,re,rau,refractive_index_input,file_out)
   
    print(re)

    #print(re*np.exp(-2*sigma**2))

if __name__ == "__main__":
    main()
