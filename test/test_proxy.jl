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

    rect_out = zeros(nd, porder^ndim)
    rect_out_ws = similar(rect_out)
    rect_workspace = BoxDMK._rect_tensor_apply_workspace(Float64, nd, norder, porder, ndim)

    BoxDMK._tensor_product_apply_rect!(rect_out, proxy_legendre.den2pc_mat, @view(fvals[:, :, 1]), norder, porder, ndim, nd)
    BoxDMK._tensor_product_apply_rect!(rect_out_ws, proxy_legendre.den2pc_mat, @view(fvals[:, :, 1]), norder, porder, ndim, nd, rect_workspace)
    rect_out_3d = similar(rect_out)
    BoxDMK._tensor_product_apply_rect_3d!(rect_out_3d, proxy_legendre.den2pc_mat, @view(fvals[:, :, 1]), norder, porder, nd, rect_workspace)
    @test rect_out_ws ≈ rect_out
    @test rect_out_3d ≈ rect_out

    if BoxDMK._FORTRAN_HOTPATHS_AVAILABLE[]
        charge_direct = fill(-1.0, proxy_legendre.ncbox, nd)
        BoxDMK._f_density2proxycharge!(charge_direct, @view(fvals[:, :, 1]), proxy_legendre.den2pc_mat, ndim, nd, norder, porder)
        @test charge_direct ≈ transpose(rect_out)

        proxy_pot_box = reshape(collect(range(-2.0, step = 0.125, length = proxy_legendre.ncbox * nd)), proxy_legendre.ncbox, nd)
        expected_pot_box = zeros(nd, norder^ndim)
        src_box = Matrix{Float64}(undef, nd, proxy_legendre.ncbox)
        src_box .= transpose(proxy_pot_box)
        BoxDMK._tensor_product_apply_rect!(expected_pot_box, proxy_legendre.poteval_mat, src_box, porder, norder, ndim, nd)

        direct_accumulated = fill(3.0, nd, norder^ndim)
        BoxDMK._f_proxypot2pot!(direct_accumulated, proxy_pot_box, proxy_legendre.poteval_mat, ndim, nd, porder, norder)
        @test direct_accumulated ≈ fill(3.0, nd, norder^ndim) .+ expected_pot_box

        pot_overwrite = fill(7.0, nd, norder^ndim, 1)
        BoxDMK.proxy_to_potential!(pot_overwrite, reshape(proxy_pot_box, proxy_legendre.ncbox, nd, 1), proxy_legendre)
        @test @view(pot_overwrite[:, :, 1]) ≈ expected_pot_box
    end
end
