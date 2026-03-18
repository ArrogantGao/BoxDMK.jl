using BoxDMK
using Test

function _pw_test_tree()
    ndim = 3
    nchildren = 2^ndim
    nboxes = 1 + nchildren
    centers = zeros(Float64, ndim, nboxes)
    centers[:, 1] .= 0.5

    for bits in 0:(nchildren - 1)
        child = bits + 2
        for d in 1:ndim
            shift = ((bits >> (d - 1)) & 0x1) == 0 ? -0.25 : 0.25
            centers[d, child] = 0.5 + shift
        end
    end

    children = zeros(Int, nchildren, nboxes)
    children[:, 1] .= 2:nboxes

    return BoxDMK.BoxTree(
        ndim,
        1,
        centers,
        [1.0, 0.5],
        [0; fill(1, nchildren)],
        children,
        [Int[] for _ in 1:nboxes],
        [0; fill(1, nchildren)],
        BoxDMK.LegendreBasis(),
        4,
    )
end

@testset "Plane Wave Setup" begin
    @test BoxDMK.get_pw_term_count(1e-6, 0) == 44
    @test BoxDMK.get_pw_term_count(1e-6, -10) == 10
    @test BoxDMK.get_pw_term_count(1e-6, 2) == 44
    @test BoxDMK.get_pw_term_count(1e-6, -11) == 22

    npw = BoxDMK.get_pw_term_count(1e-6, 0)
    boxdim = 2.0
    nodes, weights = BoxDMK.get_pw_nodes(1e-6, 0, boxdim)
    spacing = 2pi / (boxdim * npw)

    @test length(nodes) == npw
    @test length(weights) == npw
    @test nodes[1] ≈ -pi / boxdim atol = 1e-12
    @test nodes[end] ≈ pi / boxdim - spacing atol = 1e-12
    @test all(diff(nodes) .≈ spacing)
    @test all(weights .≈ spacing)
    @test sum(weights) ≈ 2pi / boxdim atol = 1e-12

    porder = 4
    tab_coefs2pw, tab_pw2pot = BoxDMK.build_pw_conversion_tables(porder, npw, nodes, boxdim)
    xs_ref, quadweights_ref = BoxDMK.nodes_and_weights(BoxDMK.LegendreBasis(), porder)
    xs = xs_ref .* (boxdim / 2)
    quadweights = quadweights_ref .* (boxdim / 2)
    weight = spacing

    @test size(tab_coefs2pw) == (npw, porder)
    @test size(tab_pw2pot) == (npw, porder)
    @test tab_coefs2pw[1, 1] ≈ exp(-im * nodes[1] * xs[1]) * sqrt(weight) atol = 1e-12
    @test tab_pw2pot[1, 1] ≈ exp(im * nodes[1] * xs[1]) * sqrt(weight) * quadweights[1] atol = 1e-12

    shift = BoxDMK.build_pw_shift_matrices(npw, nodes, boxdim)
    @test size(shift) == (npw, 3)
    @test shift[:, 2] ≈ ones(ComplexF64, npw) atol = 1e-12
    @test shift[:, 1] ≈ exp.(-im .* nodes .* boxdim) atol = 1e-12
    @test shift[:, 3] ≈ exp.(im .* nodes .* boxdim) atol = 1e-12

    kernel_ft = BoxDMK.kernel_fourier_transform([4.0], [2.0], npw, nodes)
    @test length(kernel_ft) == npw
    @test eltype(kernel_ft) == ComplexF64
    @test kernel_ft[1] ≈ ComplexF64(16 * exp(-nodes[1]^2)) atol = 1e-12

    tree = _pw_test_tree()
    proxy = BoxDMK.build_proxy_data(BoxDMK.LegendreBasis(), tree.norder, 6, tree.ndim)
    sog = BoxDMK.SOGNodes([1.0, 0.5], [0.25, 0.5], 0.1)
    data = BoxDMK.setup_planewave_data(tree, proxy, BoxDMK.LaplaceKernel(), sog, 1e-6)

    @test data isa BoxDMK.PlaneWaveData
    @test data.npw == [44, 44]
    @test length(data.pw_nodes) == tree.nlevels + 1
    @test length(data.pw_weights) == tree.nlevels + 1
    @test length(data.wpwshift) == tree.nlevels + 1
    @test length(data.tab_coefs2pw) == tree.nlevels + 1
    @test length(data.tab_pw2pot) == tree.nlevels + 1
    @test size(data.tab_coefs2pw[1]) == (44, proxy.porder)
    @test size(data.tab_pw2pot[2]) == (44, proxy.porder)
    @test size(data.iaddr) == (2, BoxDMK.nboxes(tree))
    @test length(data.ifpwexp) == BoxDMK.nboxes(tree)
    @test all(data.ifpwexp)
    @test length(data.rmlexp) == 2 * sum(data.npw[tree.level .+ 1])
end
