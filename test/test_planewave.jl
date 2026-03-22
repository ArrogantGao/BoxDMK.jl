using BoxDMK
using Test

function _planewave_test_tree()
    centers = reshape([0.5, 0.25, 0.75], 1, 3)
    children = zeros(Int, 2, 3)
    children[:, 1] .= (2, 3)

    return BoxDMK.BoxTree(
        1,
        1,
        centers,
        [1.0, 0.5],
        [0, 1, 1],
        children,
        [[1], [2, 3], [2, 3]],
        [0, 1, 1],
        LegendreBasis(),
        2,
    )
end

@testset "Plane Wave Setup" begin
    eps = 1e-6
    @test BoxDMK.get_pw_term_count(eps, 0) == 44
    @test BoxDMK.get_pw_term_count(eps, -10) == 10
    @test BoxDMK.get_pw_term_count(eps, 2) == 44
    @test BoxDMK.get_pw_term_count(eps, -11) == 22
    @test BoxDMK.pw_expansion_size_half(6, 3) == 108

    boxdim = 2.0
    npw = BoxDMK.get_pw_term_count(eps, 0)
    nodes, weights = BoxDMK.get_pw_nodes(eps, 0, boxdim)
    step = 2pi / boxdim
    weight = step / (2 * sqrt(pi))

    @test length(nodes) == npw
    @test length(weights) == npw
    @test nodes[1] ≈ -step * (npw - 1) / 2 atol = 1e-12
    @test nodes[end] ≈ step * (npw - 1) / 2 atol = 1e-12
    @test all(diff(nodes) .≈ step)
    @test all(weights .≈ weight)
    @test sum(weights) ≈ npw * weight atol = 1e-12

    basis = LegendreBasis()
    tab_coefs2pw_a, tab_pw2pot_a = BoxDMK.build_pw_conversion_tables(basis, 4, npw, boxdim)
    tab_coefs2pw_b, tab_pw2pot_b = BoxDMK.build_pw_conversion_tables(basis, 4, nodes, boxdim)

    @test size(tab_coefs2pw_a) == (npw, 4)
    @test size(tab_pw2pot_a) == (4, npw)
    @test tab_coefs2pw_a ≈ tab_coefs2pw_b
    @test tab_pw2pot_a ≈ tab_pw2pot_b
    @test tab_coefs2pw_a ≈ adjoint(tab_pw2pot_a)

    deltas = [1.0, 4.0]
    strengths = [2.0, -0.5]
    pw_nodes = [-1.0, 2.0]
    pw_weights = [0.25, 0.75]
    kernel_ft = BoxDMK.kernel_fourier_transform(deltas, strengths, pw_nodes, pw_weights, 2)
    npw2 = (length(pw_nodes) + 1) ÷ 2
    expected_kernel_ft = Float64[]

    for multi_index in Iterators.product(1:npw2, 1:length(pw_nodes))
        total = 0.0
        for k in eachindex(deltas)
            contribution = strengths[k]
            for index in multi_index
                contribution *= pw_weights[index] * sqrt(deltas[k]) * exp(-(pw_nodes[index]^2) * deltas[k] / 4)
            end
            total += contribution
        end
        push!(expected_kernel_ft, total)
    end

    @test length(kernel_ft) == BoxDMK.pw_expansion_size_half(length(pw_nodes), 2)
    @test kernel_ft ≈ expected_kernel_ft

    shift_nodes, _ = BoxDMK.get_pw_nodes(1e-2, 0, 1.0)
    shift = BoxDMK.build_pw_shift_matrices(shift_nodes, 1.0, 1; nmax = 2)
    shift_from_npw = BoxDMK.build_pw_shift_matrices(length(shift_nodes), 1.0, 1; nmax = 2)
    shift_count = BoxDMK.pw_expansion_size_half(length(shift_nodes), 1)

    @test size(shift) == (shift_count, 5)
    @test shift ≈ shift_from_npw
    @test shift[:, 3] ≈ ones(ComplexF64, shift_count)
    @test shift[:, 1] ≈ exp.(-2im .* shift_nodes[1:shift_count])
    @test shift[:, 5] ≈ exp.(2im .* shift_nodes[1:shift_count])

    tree = _planewave_test_tree()
    proxy = BoxDMK.build_proxy_data(LegendreBasis(), tree.norder, 4, tree.ndim)
    ifpwexp = [true, false, true]
    data = BoxDMK.setup_planewave_data(tree, proxy, 1e-2; nd = 2, ifpwexp = ifpwexp, nmax = 1)

    expected_npw = [BoxDMK.get_pw_term_count(1e-2, level) for level in 0:tree.nlevels]
    expected_iaddr = zeros(Int, 2, BoxDMK.nboxes(tree))
    expected_total = 0
    for ibox in 1:BoxDMK.nboxes(tree)
        if ifpwexp[ibox]
            nexp_half = BoxDMK.pw_expansion_size_half(expected_npw[tree.level[ibox] + 1], tree.ndim)
            len = nexp_half * 2
            expected_iaddr[1, ibox] = expected_total + 1
            expected_total += len
            expected_iaddr[2, ibox] = expected_total + 1
            expected_total += len
        end
    end

    @test data isa BoxDMK.PlaneWaveData
    @test data.npw == expected_npw
    @test length(data.pw_nodes) == tree.nlevels + 1
    @test length(data.pw_weights) == tree.nlevels + 1
    @test length(data.ww_1d) == tree.nlevels + 1
    @test length(data.tab_coefs2pw) == tree.nlevels + 1
    @test length(data.tab_pw2pot) == tree.nlevels + 1
    @test size(data.tab_coefs2pw[1]) == (expected_npw[1], proxy.porder)
    @test size(data.tab_pw2pot[2]) == (proxy.porder, expected_npw[2])
    @test size(data.iaddr) == (2, BoxDMK.nboxes(tree))
    @test data.ifpwexp == ifpwexp
    @test length(data.rmlexp) == expected_total
    @test data.iaddr == expected_iaddr
end
