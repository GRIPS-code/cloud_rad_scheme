# Cloud_rad_scheme
Contact Jing Feng <Jing.Feng@noaa.gov> for questions

## Module required:
pip install netCDF4
pip install miepython

## Library required:
Yang et al., 2013 https://doi.org/10.1175/JAS-D-12-039.1
a example file under ./data/ is attached for severly roughened solid column

## Input variables:
See example_ice.py and example_liq.py
python3 example_ice.py
python3 example_liq.py

1. Parameterization size range [microns] to generate look-up-table for piecewise linear fit or Padé approximant
2. Upper and lower wavenumber [cm-1] limits to match with the band limits of gas optics parameterization
3. Crystal habit and roughness, following the description of Yang et al., 2013. 
4. Gamma shape parameter 'a' to match with cloud microphysics scheme. 

## Output: parameterization coefficients over effective radii for given spectral bands
1. Look-up-table coefficients for ice and liquid. 
Output from example_ice.py:
lut_ice_lw_solid_column_severlyroughen_gamma_aeq1_thick.nc
lut_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc
Output from example_liq.py:
lut_liq_lw_mie_gamma_aeq12_thick.nc 
lut_liq_sw_mie_gamma_aeq12_thick.nc

2. Padé approximant coefficients for ice and liquid.
Output from example_ice.py:
pade_ice_lw_solid_column_severlyroughen_gamma_aeq1_thick.nc
pade_ice_sw_solid_column_severlyroughen_gamma_aeq1_thick.nc
Output from example_liq.py:
pade_liq_lw_mie_gamma_aeq12_thick.nc
pade_liq_sw_mie_gamma_aeq12_thick.nc

## To use parameterization coefficients:
Check './extensions'