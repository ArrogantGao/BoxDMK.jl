module bdmk_c_api
  use iso_c_binding
  implicit none

  private
  public :: boxdmk_vol_tree_mem
  public :: boxdmk_vol_tree_build
  public :: boxdmk_bdmk

  abstract interface
    subroutine boxdmk_rhs_c(nd, xyz, dpars, zpars, ipars, f) bind(C)
      import :: c_int, c_double, c_double_complex
      integer(c_int), value :: nd
      real(c_double) :: xyz(*)
      real(c_double) :: dpars(*)
      complex(c_double_complex) :: zpars(*)
      integer(c_int) :: ipars(*)
      real(c_double) :: f(*)
    end subroutine boxdmk_rhs_c
  end interface

  procedure(boxdmk_rhs_c), pointer, save :: rhs_cb => null()

contains

  subroutine rhsfun_bridge(nd, xyz, dpars, zpars, ipars, f)
    implicit none
    integer(c_int) :: nd
    real(c_double) :: xyz(*)
    real(c_double) :: dpars(*)
    complex(c_double_complex) :: zpars(*)
    integer(c_int) :: ipars(*)
    real(c_double) :: f(*)

    if (.not. associated(rhs_cb)) then
      stop 1
    endif
    call rhs_cb(nd, xyz, dpars, zpars, ipars, f)
  end subroutine rhsfun_bridge

  subroutine boxdmk_vol_tree_mem(ndim, ipoly, iperiod, eps, zk, boxlen, norder, &
      iptype, eta, funptr, nd, dpars, zpars, ipars, ifnewtree, nboxes, nlevels, &
      ltree, rintl) bind(C, name="boxdmk_vol_tree_mem")
    implicit none
    integer(c_int) :: ndim, ipoly, iperiod, norder, iptype, nd
    integer(c_int) :: ifnewtree, nboxes, nlevels, ltree
    real(c_double) :: eps, boxlen, eta
    complex(c_double_complex) :: zk
    real(c_double) :: dpars(*)
    complex(c_double_complex) :: zpars(*)
    integer(c_int) :: ipars(*)
    type(c_funptr), value :: funptr
    real(c_double) :: rintl(201)

    external :: vol_tree_mem

    call c_f_procpointer(funptr, rhs_cb)
    call vol_tree_mem(ndim, ipoly, iperiod, eps, zk, boxlen, norder, &
      iptype, eta, rhsfun_bridge, nd, dpars, zpars, ipars, ifnewtree, &
      nboxes, nlevels, ltree, rintl)
  end subroutine boxdmk_vol_tree_mem

  subroutine boxdmk_vol_tree_build(ndim, ipoly, iperiod, eps, zk, boxlen, norder, &
      iptype, eta, funptr, nd, dpars, zpars, ipars, rintl, nboxes, nlevels, ltree, &
      itree, iptr, centers, boxsize, fvals) bind(C, name="boxdmk_vol_tree_build")
    implicit none
    integer(c_int) :: ndim, ipoly, iperiod, norder, iptype, nd
    integer(c_int) :: nboxes, nlevels, ltree
    real(c_double) :: eps, boxlen, eta
    complex(c_double_complex) :: zk
    real(c_double) :: dpars(*)
    complex(c_double_complex) :: zpars(*)
    integer(c_int) :: ipars(*)
    real(c_double) :: rintl(*)
    integer(c_int) :: itree(*)
    integer(c_int) :: iptr(8)
    real(c_double) :: centers(*)
    real(c_double) :: boxsize(*)
    real(c_double) :: fvals(*)
    type(c_funptr), value :: funptr

    external :: vol_tree_build

    call c_f_procpointer(funptr, rhs_cb)
    call vol_tree_build(ndim, ipoly, iperiod, eps, zk, boxlen, norder, &
      iptype, eta, rhsfun_bridge, nd, dpars, zpars, ipars, rintl, nboxes, &
      nlevels, ltree, itree, iptr, centers, boxsize, fvals)
  end subroutine boxdmk_vol_tree_build

  subroutine boxdmk_bdmk(nd, ndim, eps, ikernel, beta, ipoly, norder, npbox, &
      nboxes, nlevels, ltree, itree, iptr, centers, boxsize, fvals, ifpgh, pot, grad, &
      hess, ntarg, targs, ifpghtarg, pote, grade, hesse, tottimeinfo) &
      bind(C, name="boxdmk_bdmk")
    implicit none
    integer(c_int) :: nd, ndim, ikernel, ipoly, norder, npbox
    integer(c_int) :: nboxes, nlevels, ltree, ifpgh, ntarg, ifpghtarg
    integer(c_int) :: itree(*)
    integer(c_int) :: iptr(8)
    real(c_double) :: eps, beta
    real(c_double) :: centers(*)
    real(c_double) :: boxsize(*)
    real(c_double) :: fvals(*)
    real(c_double) :: pot(*)
    real(c_double) :: grad(*)
    real(c_double) :: hess(*)
    real(c_double) :: targs(*)
    real(c_double) :: pote(*)
    real(c_double) :: grade(*)
    real(c_double) :: hesse(*)
    real(c_double) :: tottimeinfo(*)

    external :: bdmk

    call bdmk(nd, ndim, eps, ikernel, beta, ipoly, norder, npbox, nboxes, nlevels, &
      ltree, itree, iptr, centers, boxsize, fvals, ifpgh, pot, grad, hess, ntarg, targs, &
      ifpghtarg, pote, grade, hesse, tottimeinfo)
  end subroutine boxdmk_bdmk

end module bdmk_c_api
