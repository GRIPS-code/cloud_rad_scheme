import numpy as np
import netCDF4 as nc
def read_from_lut(file,Re):
    ds = nc.Dataset(file)
    band = np.zeros(2,ds.nband)
    band[1,:]  = ds.Band_limits_lwr
    band[2,:]  = ds.Band_limits_upr
    r_lwr      = ds.Effective_Radius_limits_lwr
    r_upr      = ds.Effective_Radius_limits_upr

    iRe = np.where(Re>=r_lwr & Re<r_upr)

    lut_ext_a = ds.ext_coef_a
    lut_ext_b = ds.ext_coef_b
    lut_ssa_a = ds.ssa_coef_a
    lut_ssa_b = ds.ssa_coef_b
    lut_asy_a = ds.asy_coef_a
    lut_asy_b = ds.asy_coef_b

    ext = compute_from_lut(Re,np.squeeze(lut_ext_a[:,iRe,:]),np.squeeze(lut_ext_b[:,iRe,:]))
    ssa = compute_from_lut(Re,np.squeeze(lut_ssa_a[:,iRe,:]),np.squeeze(lut_ssa_b[:,iRe,:]))
    asy = compute_from_lut(Re,np.squeeze(lut_asy_a[:,iRe,:]),np.squeeze(lut_asy_b[:,iRe,:]))
return ext, ssa, asy

def compute_from_lut(x,a,b):
    y = a + b * x
    return y