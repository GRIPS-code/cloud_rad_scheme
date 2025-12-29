import os
from netCDF4 import Dataset
import numpy as np

# File paths
src_path = 'pade_ice_sw_voronoi_gamma_aeq1_thick.nc'
ref_path = 'pade_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc'
temp_path = 'temp_extended.nc'

# Open source and reference files
src = Dataset(src_path, 'r')
ref = Dataset(ref_path, 'r')
temp = Dataset(temp_path, 'w')

# Copy dimensions, updating Re_range to 5
for name, dim in src.dimensions.items():
    if name == 'Re_range':
        temp.createDimension(name, 5)
    else:
        temp.createDimension(name, len(dim) if not dim.isunlimited() else None)

# List of variables that depend on Re_range
re_range_vars = [
    'Effective_Radius_Ref',
    'Effective_Radius_limits_lwr',
    'Effective_Radius_limits_upr',
    'Particle_Volume_limits_lwr',
    'Particle_Volume_limits_upr',
    'Pade_ext_p',
    'Pade_ext_q',
    'Pade_ssa_p',
    'Pade_ssa_q',
    'Pade_asy_p',
    'Pade_asy_q'
]

# Copy variables and extend Re_range-dependent ones
for name, var in src.variables.items():
    # Replace 'Re_range' with updated size in dimensions
    new_dims = tuple('Re_range' if d in ['Re_range', 'Re_range_old'] else d for d in var.dimensions)
    temp_var = temp.createVariable(name, var.datatype, new_dims)


    # Copy attributes
    for attr in var.ncattrs():
        setattr(temp_var, attr, getattr(var, attr))

    # Extend Re_range-dependent variables
    if name in re_range_vars:
        ref_var = ref.variables[name]
        data = var[:]

        if data.ndim == 1:
            new_data = np.append(data, ref_var[4])
        elif data.ndim == 2:
            new_data = np.concatenate((data, ref_var[:, 4:5]), axis=1)
        elif data.ndim == 3:
            new_data = np.concatenate((data, ref_var[:, 4:5, :]), axis=1)
        else:
            raise ValueError(f"Unexpected shape for variable {name}")

        temp_var[:] = new_data
    else:
        temp_var[:] = var[:]

# Copy global attributes
for attr in src.ncattrs():
    setattr(temp, attr, getattr(src, attr))

# Close files
src.close()
ref.close()
temp.close()

# Replace original file with extended version
os.replace(temp_path, src_path)
print(f"Overwritten {src_path} with extended Re_range = 5.")
