!> @file
!! @brief Reads in cloud optics look-up tables.
!! @author J. Feng
!! @email gfdl.climate.model.info@noaa.gov
! Adapted from RTE-RRTMGP. Author: Robert Pincus and Eli Mlawer:  email:  rrtmgp@aer.com
module load_cloud_coefficients
use mo_rte_kind, only: wp
use define_cloud_optics, only: ty_cloud_optics
use fms_mod, only: error_mesg, fatal
use fms2_io_mod, only: close_file, FmsNetcdfFile_t, get_dimension_size, &
                       open_file, read_data
use utilities, only: catch_error
implicit none
private


public :: load_cld_padecoeff


contains


!> @brief Read cloud optical property Pade coefficients from NetCDF file.
subroutine load_cld_padecoeff(cloud_spec, cld_coeff_file)

  class(ty_cloud_optics), intent(inout) :: cloud_spec !< Cloud optics parameterization.
  character(len=*), intent(in) :: cld_coeff_file !< Path to the input dataset.

  integer :: nband, nsizereg, n, m
  real(wp), dimension(:,:), allocatable :: band_lims_wvn ! Spectral discretization
  !Pade coefficients nom p
  real(wp), dimension(:,:,:), allocatable :: pade_ext_p ! extinction coefficient [m2 g-1]: liquid
  real(wp), dimension(:,:,:), allocatable :: pade_ssa_p ! single scattering albedo: liquid
  real(wp), dimension(:,:,:), allocatable :: pade_asy_p ! asymmetry parameter: liquid
  !Pade coefficients denom q
  real(wp), dimension(:,:,:), allocatable :: pade_ext_q ! extinction coefficient [m2 g-1]: liquid
  real(wp), dimension(:,:,:), allocatable :: pade_ssa_q ! single scattering albedo: liquid
  real(wp), dimension(:,:,:), allocatable :: pade_asy_q ! asymmetry parameter: liquid
  !Pade particle size regime boundaries
  real(wp), dimension(:,:), allocatable :: pade_sizereg
  real(wp), dimension(:), allocatable :: pade_sizeref
  type(FmsNetcdfFile_t) :: dataset

  !Open cloud optical property coefficient file
  if (.not. open_file(dataset, trim(cld_coeff_file), "read")) then
    call error_mesg("load_cld_padecoeff", "can't open file "//trim(cld_coeff_file), fatal)
  endif

  !Read Pade coefficient dimensions
  call get_dimension_size(dataset, "Band", nband)
  call get_dimension_size(dataset, "Re_range", nsizereg)
  call get_dimension_size(dataset, "n", n)
  call get_dimension_size(dataset, "m", m)

  allocate(band_lims_wvn(2, nband))
  call read_data(dataset, 'Band_limits_lwr', band_lims_wvn(1,:))
  call read_data(dataset, 'Band_limits_upr', band_lims_wvn(2,:))

  !Allocate cloud property Pade coefficient input arrays
  allocate(pade_ext_p(nband, nsizereg, n), &
           pade_ssa_p(nband, nsizereg, n), &
           pade_asy_p(nband, nsizereg, n), &
           pade_ext_q(nband, nsizereg, m), &
           pade_ssa_q(nband, nsizereg, m), &
           pade_asy_q(nband, nsizereg, m))

  call read_data(dataset, "Pade_ext_p", pade_ext_p)
  call read_data(dataset, "Pade_ssa_p", pade_ssa_p)
  call read_data(dataset, "Pade_asy_p", pade_asy_p)
  call read_data(dataset, "Pade_ext_q", pade_ext_q)
  call read_data(dataset, "Pade_ssa_q", pade_ssa_q)
  call read_data(dataset, "Pade_asy_q", pade_asy_q)

  !Allocate cloud property Pade coefficient particle size boundary input arrays
  allocate(pade_sizereg(2, nsizereg))
  allocate(pade_sizeref(nsizereg))

  call read_data(dataset, "Effective_Radius_limits_lwr", pade_sizereg(1,:))
  call read_data(dataset, "Effective_Radius_limits_upr", pade_sizereg(2,:))
  call read_data(dataset, "Effective_Radius_Ref", pade_sizeref)
  call close_file(dataset)

  if (pade_sizereg(2,nsizereg) .le. 0) then
    call error_mesg("load_cld_padecoeff", &
                    "cloud coeff size region is not assigned properly!", fatal)
  endif
  call catch_error(cloud_spec%load(band_lims_wvn, &
                                   pade_ext_p, pade_ssa_p, pade_asy_p, &
                                   pade_ext_q, pade_ssa_q, pade_asy_q, &
                                   pade_sizereg, pade_sizeref))
  deallocate(band_lims_wvn, &
             pade_ext_p, &
             pade_ssa_p, &
             pade_asy_p, &
             pade_ext_q, &
             pade_ssa_q, &
             pade_asy_q, &
             pade_sizereg, &
             pade_sizeref)
end subroutine load_cld_padecoeff
end module load_cloud_coefficients
