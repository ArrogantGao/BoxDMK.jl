module bdmk_debug_store
  use iso_c_binding
  implicit none

  private
  public :: bdmk_debug_reset_internal
  public :: bdmk_debug_store_pot
  public :: bdmk_debug_store_proxy
  public :: boxdmk_debug_reset
  public :: boxdmk_debug_get_meta
  public :: boxdmk_debug_copy_step2_pot
  public :: boxdmk_debug_copy_step3_proxycharge
  public :: boxdmk_debug_copy_step6_proxypotential
  public :: boxdmk_debug_copy_step7_pot
  public :: boxdmk_debug_copy_step8_pot
  public :: boxdmk_debug_copy_step9_pot

  integer(c_int), save :: debug_nd = 0
  integer(c_int), save :: debug_npbox = 0
  integer(c_int), save :: debug_ncbox = 0
  integer(c_int), save :: debug_nboxes = 0

  real(c_double), allocatable, save :: step2_pot(:,:,:)
  real(c_double), allocatable, save :: step3_proxycharge(:,:,:)
  real(c_double), allocatable, save :: step6_proxypotential(:,:,:)
  real(c_double), allocatable, save :: step7_pot(:,:,:)
  real(c_double), allocatable, save :: step8_pot(:,:,:)
  real(c_double), allocatable, save :: step9_pot(:,:,:)

contains

  subroutine bdmk_debug_reset_internal()
    if (allocated(step2_pot)) deallocate(step2_pot)
    if (allocated(step3_proxycharge)) deallocate(step3_proxycharge)
    if (allocated(step6_proxypotential)) deallocate(step6_proxypotential)
    if (allocated(step7_pot)) deallocate(step7_pot)
    if (allocated(step8_pot)) deallocate(step8_pot)
    if (allocated(step9_pot)) deallocate(step9_pot)

    debug_nd = 0
    debug_npbox = 0
    debug_ncbox = 0
    debug_nboxes = 0
  end subroutine bdmk_debug_reset_internal

  subroutine boxdmk_debug_reset() bind(C, name="boxdmk_debug_reset")
    call bdmk_debug_reset_internal()
  end subroutine boxdmk_debug_reset

  subroutine bdmk_debug_store_pot(step, nd, npbox, nboxes, src)
    integer(c_int), intent(in) :: step
    integer(c_int), intent(in) :: nd
    integer(c_int), intent(in) :: npbox
    integer(c_int), intent(in) :: nboxes
    real(c_double), intent(in) :: src(nd, npbox, nboxes)

    debug_nd = nd
    debug_npbox = npbox
    debug_nboxes = nboxes

    select case (step)
    case (2)
      if (allocated(step2_pot)) deallocate(step2_pot)
      allocate(step2_pot(nd, npbox, nboxes))
      step2_pot = src
    case (7)
      if (allocated(step7_pot)) deallocate(step7_pot)
      allocate(step7_pot(nd, npbox, nboxes))
      step7_pot = src
    case (8)
      if (allocated(step8_pot)) deallocate(step8_pot)
      allocate(step8_pot(nd, npbox, nboxes))
      step8_pot = src
    case (9)
      if (allocated(step9_pot)) deallocate(step9_pot)
      allocate(step9_pot(nd, npbox, nboxes))
      step9_pot = src
    end select
  end subroutine bdmk_debug_store_pot

  subroutine bdmk_debug_store_proxy(step, ncbox, nd, nboxes, src)
    integer(c_int), intent(in) :: step
    integer(c_int), intent(in) :: ncbox
    integer(c_int), intent(in) :: nd
    integer(c_int), intent(in) :: nboxes
    real(c_double), intent(in) :: src(ncbox, nd, nboxes)

    debug_nd = nd
    debug_ncbox = ncbox
    debug_nboxes = nboxes

    select case (step)
    case (3)
      if (allocated(step3_proxycharge)) deallocate(step3_proxycharge)
      allocate(step3_proxycharge(ncbox, nd, nboxes))
      step3_proxycharge = src
    case (6)
      if (allocated(step6_proxypotential)) deallocate(step6_proxypotential)
      allocate(step6_proxypotential(ncbox, nd, nboxes))
      step6_proxypotential = src
    end select
  end subroutine bdmk_debug_store_proxy

  subroutine boxdmk_debug_get_meta(nd, npbox, ncbox, nboxes, has_step2, has_step3, has_step6, has_step7, has_step8, has_step9) &
      bind(C, name="boxdmk_debug_get_meta")
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: npbox
    integer(c_int), intent(out) :: ncbox
    integer(c_int), intent(out) :: nboxes
    integer(c_int), intent(out) :: has_step2
    integer(c_int), intent(out) :: has_step3
    integer(c_int), intent(out) :: has_step6
    integer(c_int), intent(out) :: has_step7
    integer(c_int), intent(out) :: has_step8
    integer(c_int), intent(out) :: has_step9

    nd = debug_nd
    npbox = debug_npbox
    ncbox = debug_ncbox
    nboxes = debug_nboxes
    has_step2 = merge(1_c_int, 0_c_int, allocated(step2_pot))
    has_step3 = merge(1_c_int, 0_c_int, allocated(step3_proxycharge))
    has_step6 = merge(1_c_int, 0_c_int, allocated(step6_proxypotential))
    has_step7 = merge(1_c_int, 0_c_int, allocated(step7_pot))
    has_step8 = merge(1_c_int, 0_c_int, allocated(step8_pot))
    has_step9 = merge(1_c_int, 0_c_int, allocated(step9_pot))
  end subroutine boxdmk_debug_get_meta

  subroutine copy_rank3(src, dest, expected_size)
    real(c_double), intent(in) :: src(:,:,:)
    real(c_double) :: dest(*)
    integer(c_int), intent(in) :: expected_size
    integer :: i, j, k, idx

    if (size(src, 1) * size(src, 2) * size(src, 3) /= expected_size) then
      stop 1
    endif

    idx = 1
    do k = 1, size(src, 3)
      do j = 1, size(src, 2)
        do i = 1, size(src, 1)
          dest(idx) = src(i, j, k)
          idx = idx + 1
        enddo
      enddo
    enddo
  end subroutine copy_rank3

  subroutine boxdmk_debug_copy_step2_pot(dest, expected_size) bind(C, name="boxdmk_debug_copy_step2_pot")
    real(c_double) :: dest(*)
    integer(c_int), value :: expected_size

    if (.not. allocated(step2_pot)) stop 1
    call copy_rank3(step2_pot, dest, expected_size)
  end subroutine boxdmk_debug_copy_step2_pot

  subroutine boxdmk_debug_copy_step3_proxycharge(dest, expected_size) bind(C, name="boxdmk_debug_copy_step3_proxycharge")
    real(c_double) :: dest(*)
    integer(c_int), value :: expected_size

    if (.not. allocated(step3_proxycharge)) stop 1
    call copy_rank3(step3_proxycharge, dest, expected_size)
  end subroutine boxdmk_debug_copy_step3_proxycharge

  subroutine boxdmk_debug_copy_step6_proxypotential(dest, expected_size) bind(C, name="boxdmk_debug_copy_step6_proxypotential")
    real(c_double) :: dest(*)
    integer(c_int), value :: expected_size

    if (.not. allocated(step6_proxypotential)) stop 1
    call copy_rank3(step6_proxypotential, dest, expected_size)
  end subroutine boxdmk_debug_copy_step6_proxypotential

  subroutine boxdmk_debug_copy_step7_pot(dest, expected_size) bind(C, name="boxdmk_debug_copy_step7_pot")
    real(c_double) :: dest(*)
    integer(c_int), value :: expected_size

    if (.not. allocated(step7_pot)) stop 1
    call copy_rank3(step7_pot, dest, expected_size)
  end subroutine boxdmk_debug_copy_step7_pot

  subroutine boxdmk_debug_copy_step8_pot(dest, expected_size) bind(C, name="boxdmk_debug_copy_step8_pot")
    real(c_double) :: dest(*)
    integer(c_int), value :: expected_size

    if (.not. allocated(step8_pot)) stop 1
    call copy_rank3(step8_pot, dest, expected_size)
  end subroutine boxdmk_debug_copy_step8_pot

  subroutine boxdmk_debug_copy_step9_pot(dest, expected_size) bind(C, name="boxdmk_debug_copy_step9_pot")
    real(c_double) :: dest(*)
    integer(c_int), value :: expected_size

    if (.not. allocated(step9_pot)) stop 1
    call copy_rank3(step9_pot, dest, expected_size)
  end subroutine boxdmk_debug_copy_step9_pot

end module bdmk_debug_store
