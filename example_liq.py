import numpy as np
import netCDF4 as nc
from compute_ice import compute_ice
from compute_liq import compute_liq
from scipy.interpolate import interp1d
from spec_util import planck, read_solar_spectrum

# initialize parameterization size range for look-up-table and Padé approximantsize
re_range_lut = np.zeros((2,15)) # look-up-table (piecewise linear coefficients) size range, micron
re_range_pade = np.zeros((2,3)) # Padé approximantsize size range, micron
re_range_lut[0,:] = np.append(np.append([1.1,2,3,5,7],np.arange(10,30,5)),np.arange(40,100,10))
re_range_lut[1,:] = np.append(np.append([2,3,5,7],np.arange(10,30,5)),np.arange(40,110,10))
re_range_pade[0,:] = [1.2, 10, 50]
re_range_pade[1,:] = [10, 50, 100] # Please check compute_liquid module variable 'd', if a wider range is required
re_ref_pade = np.zeros(np.shape(re_range_pade)[1],)

# initialize longwave band limits that matches with rrtmgp gas optics
file_rrtmgp = '/scratch/gpfs/jf7775/projects/rte-rrtmgp/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc'
data_rrtmgp = nc.Dataset(file_rrtmgp)
band_limit = data_rrtmgp['bnd_limits_wavenumber'][:,:] 
wavenum = np.arange(band_limit[0,0],band_limit[-1,-1],1)

# initialize longwave source function
source = planck(wavenum,250) # use 250 K as a reference
# generate parameterization for longwave liquid
compute_liq('lut_liq_lw_mie_gamma_aeq12_thick.nc',\
'pade_liq_lw_mie_gamma_aeq12_thick.nc',\
12,wavenum,source,band_limit,re_range_lut,re_range_pade,re_ref_pade,False)

# initialize shortwave band limits that matches with rrtmgp gas optics
file_rrtmgp = '/scratch/gpfs/jf7775/projects/rte-rrtmgp/rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc'
data_rrtmgp = nc.Dataset(file_rrtmgp)
band_limit = data_rrtmgp['bnd_limits_wavenumber'][:,:]
wavenum = np.arange(band_limit[0,0],band_limit[-1,-1],10)
# read-in shortwave spectrum
wavenum_solar, solar = read_solar_spectrum()
source = interp1d(wavenum_solar[:],solar[:])(wavenum[:])

# generate parameterization for shortwave liquid
compute_liq('lut_liq_sw_mie_gamma_aeq12_thick.nc',\
'pade_liq_sw_mie_gamma_aeq12_thick.nc',\
12,wavenum,source,band_limit,re_range_lut,re_range_pade,re_ref_pade,False)