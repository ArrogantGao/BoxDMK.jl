program bench_fortran_same_problem
  implicit none

  integer :: nd, ndim, ikernel, ipoly, norder, npbox
  integer :: iperiod, iptype, ifnewtree, nboxes, nlevels, ltree
  integer :: ntarg, ifpgh, ifpghtarg
  integer :: i, idp
  integer, allocatable :: itree(:), iptr(:), ipars(:)

  real*8 :: eps, beta, rsig, delta, rsign, boxlen, eta, epstree
  integer :: c0, c1, cr
  real*8 :: tree_build_s, solve_s
  real*8 :: pnorm

  real*8, allocatable :: dpars(:), rintl(:), centers(:), boxsize(:), fvals(:)
  real*8, allocatable :: pot(:), grad(:), hess(:), targs(:), pote(:), grade(:), hesse(:), tottimeinfo(:)
  complex*16 :: zk
  complex*16, allocatable :: zpars(:)

  nd = 1
  ndim = 3
  ikernel = 1
  beta = 1.0d0
  eps = 1.0d-6
  rsig = 1.0d-4

  norder = 16
  ipoly = 0
  ifpgh = 1
  ifpghtarg = 0

  iperiod = 0
  iptype = 2
  eta = 0.0d0
  ifnewtree = 0
  boxlen = 1.18d0
  zk = (30.0d0, 0.0d0)
  epstree = eps * 500.0d0

  allocate(ipars(256), dpars(1000), zpars(16), rintl(201))
  ipars = 0
  dpars = 0.0d0
  zpars = (0.0d0, 0.0d0)

  ipars(1) = ndim
  ipars(2) = ikernel
  ipars(3) = 2
  ipars(5) = iperiod
  ipars(10) = max(ifpgh, ifpghtarg)

  delta = 1.0d0
  rsign = (rsig * delta)**(ndim/2.0d0)
  dpars(201) = beta

  dpars(1) = 0.1d0
  dpars(2) = 0.02d0
  dpars(3) = 0.04d0
  dpars(4) = rsig
  dpars(5) = 1.0d0 / 3.141592653589793d0 / rsign

  dpars(6) = 0.03d0
  dpars(7) = -0.1d0
  dpars(8) = 0.05d0
  dpars(9) = rsig / 2.0d0
  dpars(10) = -0.5d0 / 3.141592653589793d0 / rsign

  call system_clock(c0, cr)
  call vol_tree_mem(ndim, ipoly, iperiod, epstree, zk, boxlen, norder, iptype, eta, &
    rhs_cb, nd, dpars, zpars, ipars, ifnewtree, nboxes, nlevels, ltree, rintl)

  npbox = norder**ndim
  allocate(itree(ltree), iptr(8), centers(ndim*nboxes), boxsize(nlevels+1), fvals(nd*npbox*nboxes))

  call vol_tree_build(ndim, ipoly, iperiod, epstree, zk, boxlen, norder, iptype, eta, &
    rhs_cb, nd, dpars, zpars, ipars, rintl, nboxes, nlevels, ltree, itree, iptr, centers, boxsize, fvals)
  call system_clock(c1)
  tree_build_s = dble(c1 - c0) / dble(cr)

  ntarg = 1
  allocate(pot(max(1, nd*npbox*nboxes)), grad(1), hess(1))
  allocate(targs(ndim*ntarg), pote(nd*ntarg), grade(max(1, nd*ndim*ntarg)), hesse(1), tottimeinfo(20))

  targs = 0.0d0

  call system_clock(c0)
  call bdmk(nd, ndim, eps, ikernel, beta, ipoly, norder, npbox, nboxes, nlevels, ltree, &
    itree, iptr, centers, boxsize, fvals, ifpgh, pot, grad, hess, ntarg, targs, ifpghtarg, &
    pote, grade, hesse, tottimeinfo)
  call system_clock(c1)
  solve_s = dble(c1 - c0) / dble(cr)

  pnorm = dsqrt(sum(pot*pot))

  write(*,'(A,1X,F10.6,1X,A,1X,F10.6,1X,A,1X,F10.6,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,I8,1X,A,1X,I8)') &
    'BENCH_FORTRAN tree_build_s=', tree_build_s, 'solve_s=', solve_s, 'total_s=', tree_build_s + solve_s, &
    'eps=', eps, 'pnorm=', pnorm, 'nboxes=', nboxes, 'nlevels=', nlevels

contains

  subroutine rhs_cb(nd, xyz, dpars, zpars, ipars, f)
    implicit none
    integer, intent(in) :: nd
    real*8, intent(in) :: xyz(*)
    real*8, intent(in) :: dpars(*)
    complex*16, intent(in) :: zpars(*)
    integer, intent(in) :: ipars(*)
    real*8, intent(out) :: f(*)

    integer :: i, k, ng, ndim, idp
    real*8 :: fi, rr, xk, sigma, strength

    ndim = ipars(1)
    ng = ipars(3)

    do i = 1, nd
      fi = 0.0d0
      do k = 1, ng
        idp = (k - 1) * 5
        rr = 0.0d0
        xk = xyz(1) - dpars(idp + 1)
        rr = rr + xk * xk
        xk = xyz(2) - dpars(idp + 2)
        rr = rr + xk * xk
        xk = xyz(3) - dpars(idp + 3)
        rr = rr + xk * xk
        sigma = dpars(idp + 4)
        strength = dpars(idp + 5)
        fi = fi + strength * dexp(-rr / sigma) * (-2.0d0 * ndim + 4.0d0 * rr / sigma) / sigma
      end do
      f(i) = fi
    end do
  end subroutine rhs_cb

end program bench_fortran_same_problem
