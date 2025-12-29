import xarray as xr
import numpy as np
from collections import Counter

cm_to_um = 10**4
m_to_micron = 10**6


def read_voronoi_ice(path="./data/voronoi/voronoi_raw.csv"):
    # Load the NetCDF file you created
    ds_rmax = xr.open_dataset(path)

    wavelength_um = ds_rmax.wavelength.values * m_to_micron
    rmax_um = ds_rmax.rmax.values * m_to_micron  

    # Extract variables from your dataset
    ext_eff = ds_rmax["extinction_efficiency"].values
    ssa = ds_rmax["single_scattering_albedo"].values
    asy = ds_rmax["asymmetry_factor"].values

    # Reconstruct cross sections and volumes using scaling laws
    projected_area_um2 = ds_rmax["geometrical_cross_section"].values * m_to_micron**2
    volume_um3 = ds_rmax["volume"].values * m_to_micron**3


    # Convert wavelength to wavenumber (cm⁻¹)
    wavenum_table = 1e4 / wavelength_um

    # Use rmax as r_table
    r_table = rmax_um

    # Approximate maximum dimension (d_table) as 2 × rmax
    d_table = 2 * r_table

    # Projected area and volume
    s_table = projected_area_um2
    v_table = volume_um3

    # Compute cross sections
    ext_cross_section_table = ext_eff * s_table
    sca_cross_section_table = ssa * ext_cross_section_table

    # Asymmetry factor
    asy_table = asy

    print(s_table)
    print(r_table)

    # --- Data Screening ---
    # Create a mask based on the most frequent value in each row of s_table
    mask = np.zeros_like(s_table, dtype=bool)
    for i in range(s_table.shape[1]):
        col = s_table[:, i]
        mean = np.mean(col)
        std = np.std(col)
        z = (col - mean) / std
        mask[:, i] = np.abs(z) < 2  

    # Apply mask to all relevant tables
    s_table = np.where(mask, s_table, np.nan)
    s_table =np.nanmean(s_table,axis=0)
    v_table = np.where(mask, v_table, np.nan)
    v_table =np.nanmean(v_table,axis=0)

    ext_cross_section_table = np.where(mask, ext_cross_section_table, np.nan)
    sca_cross_section_table = np.where(mask, sca_cross_section_table, np.nan)
    asy_table = np.where(mask, asy_table, np.nan)

    return wavenum_table, r_table, d_table, s_table, v_table, ext_cross_section_table, sca_cross_section_table, asy_table

