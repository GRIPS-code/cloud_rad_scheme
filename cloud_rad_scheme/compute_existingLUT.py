import math
from .optics import optics_var

def compute_existingLUT(cloud_optics_in,file_lut, file_pade,  source, band_limit, re_range_lut,
                re_range_pade, re_ref_pade, thin_flag):
    if thin_flag==True:
        cloud_optics_band = cloud_optics_in.thin_average(source,band_limit)
    else:
        cloud_optics_band = cloud_optics_in.thick_average(source,band_limit)
    
    v_range_lut = re_range_lut **3.0 * 4.0 / 3.0 * math.pi
    # output parameterization netcdf file following piece-wise linear interpolation
    cloud_optics_band.create_lut_coeff(re_range_lut,v_range_lut,file_lut)

    v_range_pade = re_range_pade **3.0 * 4.0 / 3.0 * math.pi
    cloud_optics_band.create_pade_coeff(re_range_pade,re_ref_pade,v_range_pade,file_pade)
