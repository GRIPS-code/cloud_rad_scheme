! Contacts: Jing Feng
! email:  jing.feng@noaa.gov
! Adapted from RTE-RRTMGP. Author: Robert Pincus and Eli Mlawer:  email:  rrtmgp@aer.com
! -------------------------------------------------------------------------------------------------
! Compute cloud optical properties as a function of effective radius for the RRTMGP bands
!   Required Fortran library: RTE-RRTMGP https://earth-system-radiation.github.io/rte-rrtmgp/
!   Required input: file generated from example_ice.py and example_liq.py 
! -------------------------------------------------------------------------------------------------

module define_cloud_optics
  use fms_mod, only: error_mesg, fatal, NOTE, string
  use mo_rte_kind,      only: wp, wl
  use mo_rte_util_array,only: any_vals_less_than, any_vals_outside, extents_are
  use mo_optical_props, only: ty_optical_props,      &
                              ty_optical_props_arry, &
                              ty_optical_props_1scl, &
                              ty_optical_props_2str, &
                              ty_optical_props_nstr
 use fms_mod, only: error_mesg, fatal
   implicit none
  interface pade_eval
    module procedure pade_eval_nbnd, pade_eval_1
  end interface pade_eval
  private 
 ! -----------------------------------------------------------------------------------
  type, extends(ty_optical_props), public :: ty_cloud_optics
    !
    ! Lookup table information
    !
    ! Upper and lower limits of the tables
     integer, public :: nbnd, nsizereg, n,m
    ! Spectral discretization
    real(wp), public :: rad_lwr = 0._wp, rad_upr = 0._wp
    !
    ! Pade approximant coefficients
    !
    real(wp), dimension(:,:,:), allocatable :: pade_ext_p, pade_ssa_p, pade_asy_p ! (nbnd, nsizereg, n)
    real(wp), dimension(:,:,:), allocatable :: pade_ext_q, pade_ssa_q, pade_asy_q ! (nbnd, nsizereg, m)
    ! Particle size regimes for Pade formulations
    real(wp), dimension(:,:), allocatable :: pade_sizereg  ! (2,nsize)
    real(wp), dimension(:),   allocatable :: pade_sizeref  ! (nsize)
    ! -----
  contains
    generic,   public :: load  => load_pade
    procedure, public :: finalize
!   procedure, public :: cloud_optics
    procedure, public :: get_min_radius
    procedure, public :: get_max_radius
    procedure, public :: compute_all_from_pade
    ! Internal procedures
    procedure, private :: load_pade
  end type ty_cloud_optics
   
contains
  ! ------------------------------------------------------------------------------
  !
  ! Routines to load data needed for cloud optics calculations. 
  !
  ! ------------------------------------------------------------------------------
! ------------------------------------------------------------------------------
  !
  ! Cloud optics initialization function - Pade
  !
  ! ------------------------------------------------------------------------------
  function load_pade(this, band_lims_wvn, &
                     pade_ext_p, pade_ssa_p, pade_asy_p, &
                     pade_ext_q, pade_ssa_q, pade_asy_q, &
                     pade_sizereg, pade_sizeref) result(error_msg)

     class(ty_cloud_optics),       intent(inout) :: this          ! cloud specification data
    real(wp), dimension(:,:),     intent(in   ) :: band_lims_wvn ! Spectral discretization
    !
    ! Pade coefficients: extinction, single-scattering albedo, and asymmetry factor for liquid and ice
    !
    real(wp), dimension(:,:,:),   intent(in)    :: pade_ext_p, pade_ssa_p, pade_asy_p
    real(wp), dimension(:,:,:),   intent(in)    :: pade_ext_q, pade_ssa_q, pade_asy_q
    !
    ! Boundaries of size regimes. Liquid and ice are separate;
    !   extinction is fit to different numbers of size bins than single-scattering albedo and asymmetry factor
    !
    real(wp),  dimension(:,:),       intent(in)    :: pade_sizereg
    real(wp),  dimension(:),         intent(in)    :: pade_sizeref
    character(len=128)    :: error_msg
! ------- Local -------

    integer               :: nbnd, nsizereg, n, m

! ------- Definitions -------
    error_msg = this%init(band_lims_wvn, name="RRTMGP cloud optics")
    ! Pade coefficient dimensions
    nbnd         = size(pade_ext_p,dim=1)
    nsizereg     = size(pade_ext_p,dim=2)
    n            = size(pade_ext_p,dim=3)
    m            = size(pade_ext_q,dim=3)

    this%nbnd = nbnd
    this%nsizereg = nsizereg
    this%n = n
    this%m = m
    this%band_lims_wvn = band_lims_wvn
    this%rad_lwr = pade_sizereg(1,1)
    this%rad_upr = pade_sizereg(2,nsizereg)
    !
    ! Allocate Pade coefficients
    !
    allocate(this%pade_ext_p(nbnd, nsizereg, n),   &
             this%pade_ssa_p(nbnd, nsizereg, n), &
             this%pade_asy_p(nbnd, nsizereg, n), &
             this%pade_ext_q(nbnd, nsizereg, m),   &
             this%pade_ssa_q(nbnd, nsizereg, m), &
             this%pade_asy_q(nbnd, nsizereg, m))
    !
    ! Allocate Pade coefficient particle size regime boundaries
    !
    allocate(this%pade_sizereg(2,nsizereg),this%pade_sizeref(nsizereg))
    !
    ! Load data
    !
    !$acc kernels
    this%pade_ext_p = pade_ext_p
    this%pade_ssa_p = pade_ssa_p
    this%pade_asy_p = pade_asy_p
    this%pade_ext_q = pade_ext_q
    this%pade_ssa_q = pade_ssa_q
    this%pade_asy_q = pade_asy_q
    this%pade_sizereg = pade_sizereg
    this%pade_sizeref = pade_sizeref
    !$acc end kernels
    !
    !
  end function load_pade
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Finalize
  !
  !--------------------------------------------------------------------------------------------------------------------
  subroutine finalize(this)
    class(ty_cloud_optics), intent(inout) :: this

    this%rad_lwr = 0._wp
    this%rad_upr = 0._wp

    ! Pade cloud optics coefficients
    if(allocated(this%pade_ext_p)) then
      deallocate(this%pade_ext_p, this%pade_ssa_p, this%pade_asy_p, &
                 this%pade_ext_q, this%pade_ssa_q, this%pade_asy_q, &
                 this%pade_sizereg)
    end if
  end subroutine finalize

  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Inquiry functions
  !
  !--------------------------------------------------------------------------------------------------------------------
  function get_min_radius(this) result(r)
    class(ty_cloud_optics), intent(in   ) :: this
    real(wp)                              :: r

    r = this%rad_lwr
  end function get_min_radius
  !-----------------------------------------------
  function get_max_radius(this) result(r)
    class(ty_cloud_optics), intent(in   ) :: this
    real(wp)                              :: r

    r = this%rad_upr
  end function get_max_radius
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Ancillary functions
  !
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Read from pade curve fiting 
  ! Extinction coefficient: m^2/g
  ! Returns 0 where the mask is false.
  !
  !
  !---------------------------------------------------------------------------
  subroutine compute_all_from_pade(this, nlay, ibnd,  &
                                   mask, lwp, re,  &
                                   tau, ssa, g)
    class(ty_cloud_optics), intent(inout) :: this
    integer,                        intent(in) :: nlay, ibnd
    logical(wl),  &
              dimension(nlay), intent(in)   :: mask
    real(wp), dimension(nlay), intent(in)   :: lwp, re ! Cloud water path: [kg/kg] * [g/m^3] * [m] = [g/m^2]
                                                          ! Effective radius: microns
    real(wp), dimension(:), intent(inout) :: tau, ssa, g ! [nlay, number of output gpoints]
    ! ---------------------------
    integer  ::  ilay,  irad, count
    real(wp) :: t, ts

    tau   (:) = 0._wp
    ssa   (:) = 0._wp
    g     (:) = 0._wp

    !$acc parallel loop gang vector default(present) collapse(3)
      do ilay = 1, nlay
          if(mask(ilay)) then
            !
            ! Finds index into size regime table
            !
            do irad = 1, this%nsizereg
                if (this%pade_sizereg(1,irad) .le. re(ilay) .AND. this%pade_sizereg(2,irad) .ge. re(ilay)) then
                        exit
                endif
            enddo

            if (irad .gt. this%nsizereg) call  error_mesg('compute_all_from_pade:','re out of range',fatal)
            tau (ilay)  = lwp(ilay) *     &
                 pade_eval_1(ibnd, this%nbnd, this%nsizereg, this%n, this%m, irad, re(ilay) - this%pade_sizeref(irad), this%pade_ext_p, this%pade_ext_q)

            ssa (ilay)  = &
                 pade_eval_1(ibnd, this%nbnd, this%nsizereg, this%n, this%m, irad, re(ilay) - this%pade_sizeref(irad), this%pade_ssa_p, this%pade_ssa_q)

            g(ilay)    =  &
                 pade_eval_1(ibnd, this%nbnd, this%nsizereg, this%n, this%m, irad, re(ilay) - this%pade_sizeref(irad), this%pade_asy_p, this%pade_asy_q)
          end if
        end do

    where(g(:) .gt. 1._wp)
      g(:) = 1._wp ! pade approximate may produce out-of-range values for g
    endwhere

    where(ssa(:) .gt. 1._wp)
      ssa(:) = 1._wp ! pade approximate may produce out-of-range values for g
    endwhere

    where(ssa(:) .lt. 0._wp)
      ssa(:) = 0._wp ! pade approximate may produce negative values for g
    endwhere

    where(g(:) .lt. 0._wp)
      g(:) = 0._wp ! pade approximate may produce negative values for g
    endwhere
  end subroutine compute_all_from_pade

  function pade_eval_nbnd(nbnd, nrads, n, m, irad, re,  pade_coeffs_p, pade_coeffs_q)
    integer,                intent(in) :: nbnd, nrads, m, n, irad
    real(wp), dimension(nbnd, nrads, n), &
                            intent(in) :: pade_coeffs_p
    real(wp), dimension(nbnd, nrads, m), &
                            intent(in) :: pade_coeffs_q
    real(wp),               intent(in) :: re
    real(wp), dimension(nbnd)          :: pade_eval_nbnd

    integer :: iband
    real(wp) :: numer, denom
    integer  :: i

    do iband = 1, nbnd
      denom = pade_coeffs_q(iband,irad,1)
      do i = 2, m
        denom = pade_coeffs_q(iband,irad,i)+re*denom
      end do

      numer = pade_coeffs_p(iband,irad,1)
      do i = 2, n
        numer = pade_coeffs_p(iband,irad,i)+re*numer
      end do

      pade_eval_nbnd(iband) = numer/denom
    end do
  end function pade_eval_nbnd
  !---------------------------------------------------------------------------
  !
  ! Evaluate Pade approximant of order [n/m]
  !
  function pade_eval_1(iband, nbnd, nrads, n, m, irad, re, pade_coeffs_p, pade_coeffs_q)
    !$acc routine seq
    !
    integer,                intent(in) :: iband, nbnd, nrads, m, n, irad
    real(wp), dimension(nbnd, nrads, n), &
                            intent(in) :: pade_coeffs_p
    real(wp), dimension(nbnd, nrads, m), &
                            intent(in) :: pade_coeffs_q                    
    real(wp),               intent(in) :: re
    real(wp)                           :: pade_eval_1

    real(wp) :: numer, denom
    integer  :: i

    denom = pade_coeffs_q(iband,irad,1)
    do i = 2, m
      denom = pade_coeffs_q(iband,irad,i)+re*denom
    end do

    numer = pade_coeffs_p(iband,irad,1)
    do i = 2, n
      numer = pade_coeffs_p(iband,irad,i)+re*numer
    end do

    pade_eval_1 = numer/denom
  end function pade_eval_1

end module define_cloud_optics
