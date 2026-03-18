using BoxDMK
using LinearAlgebra
using Test

@testset "Proxy System" begin
    @test BoxDMK.select_porder(1e-3) == 16
    @test BoxDMK.select_porder(0.8e-3) == 16
    @test BoxDMK.select_porder(1e-6) == 30
    @test BoxDMK.select_porder(1e-12) == 62

    legendre_basis = LegendreBasis()
    chebyshev_basis = ChebyshevBasis()
    norder = 6
    porder = 16
    ndim = 3
    nd = 2

    proxy_legendre = BoxDMK.build_proxy_data(legendre_basis, norder, porder, ndim)
    proxy_chebyshev = BoxDMK.build_proxy_data(chebyshev_basis, norder, porder, ndim)

    @test proxy_legendre.porder == porder
    @test proxy_legendre.ncbox == porder^ndim
    @test size(proxy_legendre.den2pc_mat) == (porder, norder)
    @test size(proxy_legendre.poteval_mat) == (norder, porder)
    @test size(proxy_legendre.p2c_transmat) == (porder, porder, ndim, 2^ndim)
    @test size(proxy_legendre.c2p_transmat) == (porder, porder, ndim, 2^ndim)
    @test proxy_legendre.p2c_transmat ≈ BoxDMK.p2c_transform(LegendreBasis(), porder, ndim)
    @test proxy_legendre.c2p_transmat ≈ BoxDMK.c2p_transform(LegendreBasis(), porder, ndim)

    @test proxy_chebyshev.p2c_transmat ≈ proxy_legendre.p2c_transmat
    @test proxy_chebyshev.c2p_transmat ≈ proxy_legendre.c2p_transmat
    @test proxy_chebyshev.den2pc_mat ≉ proxy_legendre.den2pc_mat

    fvals = randn(nd, norder^ndim, 3)
    charge = zeros(proxy_legendre.ncbox, nd, 3)
    BoxDMK.density_to_proxy!(charge, fvals, proxy_legendre)
    @test size(charge) == (proxy_legendre.ncbox, nd, 3)

    pot_back = zeros(nd, norder^ndim, 3)
    BoxDMK.proxy_to_potential!(pot_back, charge, proxy_legendre)
    @test size(pot_back) == size(fvals)
    @test norm(pot_back[:, :, 1] - fvals[:, :, 1]) / norm(fvals[:, :, 1]) < 0.1

    fvals2 = randn(nd, norder^ndim)
    charge2 = zeros(proxy_legendre.ncbox, nd)
    BoxDMK.density_to_proxy!(charge2, fvals2, proxy_legendre)
    pot_back2 = zeros(nd, norder^ndim)
    BoxDMK.proxy_to_potential!(pot_back2, charge2, proxy_legendre)
    @test norm(pot_back2 - fvals2) / norm(fvals2) < 0.1
end
