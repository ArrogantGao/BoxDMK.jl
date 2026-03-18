using BoxDMK
using LinearAlgebra
using Test

function _classification_tree()
    return BoxDMK.BoxTree(
        1,
        2,
        reshape([0.5], 1, 1),
        [1.0, 0.5, 0.25],
        [0],
        zeros(Int, 2, 1),
        [[1]],
        [0],
        LegendreBasis(),
        2,
    )
end

function _two_child_tree()
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

function _simple_pw_data(tree::BoxDMK.BoxTree; eps::Float64 = 1e-6)
    rmlexp = zeros(ComplexF64, 8)
    iaddr = zeros(Int, 2, BoxDMK.nboxes(tree))
    iaddr[:, 2] .= (1, 3)
    iaddr[:, 3] .= (5, 7)

    return BoxDMK.PlaneWaveData(
        rmlexp,
        iaddr,
        [1, 2],
        [[0.0], [0.0, 0.0]],
        [[1.0], [1.0, 1.0]],
        [ones(ComplexF64, 1, 1), ones(ComplexF64, 2, 3)],
        [ones(ComplexF64, 1, 1), Matrix{ComplexF64}(I, 2, 2)],
        [ones(ComplexF64, 1, 1), Matrix{ComplexF64}(I, 2, 2)],
        [false, true, true],
        eps,
    )
end

@testset "Delta Classification" begin
    tree = _classification_tree()
    sog = BoxDMK.SOGNodes([1.0, 2.0, 3.0, 4.0], [0.1, 0.01, 5e-4, 1e-4], 1.0)

    groups = BoxDMK.group_deltas_by_level(sog, tree, 1e-6)

    @test groups.fat == [(0.1, 1.0)]
    @test groups.asymptotic == [(1e-4, 4.0)]
    @test length(groups.normal) == 2
    @test groups.normal[1][1] == 1
    @test groups.normal[1][2] == [0.01]
    @test groups.normal[1][3] == [2.0]
    @test groups.normal[2][1] == 2
    @test groups.normal[2][2] == [5e-4]
    @test groups.normal[2][3] == [3.0]
end

@testset "Box FGT" begin
    tree = _two_child_tree()
    pw_data = _simple_pw_data(tree)
    lists = BoxDMK.InteractionLists([Int[] for _ in 1:3], [Int[], [3], [2]])

    proxy_charges = zeros(Float64, 2, 1, 3)
    proxy_charges[:, 1, 2] .= [1.0, 2.0]
    proxy_charges[:, 1, 3] .= [10.0, 20.0]

    proxy_pot = zeros(Float64, 2, 1, 3)
    deltas = [0.01, 0.016]
    weights = [2.0, -1.0]

    BoxDMK.boxfgt!(proxy_pot, tree, proxy_charges, deltas, weights, pw_data, lists)

    scale = sum(weights .* sqrt.(deltas))
    @test proxy_pot[:, 1, 1] == [0.0, 0.0]
    @test proxy_pot[:, 1, 2] ≈ scale .* [11.0, 22.0]
    @test proxy_pot[:, 1, 3] ≈ scale .* [11.0, 22.0]
end

@testset "Fat Gaussian" begin
    tree = _two_child_tree()
    pw_data = _simple_pw_data(tree)
    proxy_charges = zeros(Float64, 1, 1, 3)
    proxy_charges[1, 1, 1] = 2.0
    proxy_charges[1, 1, 2] = 100.0
    proxy_charges[1, 1, 3] = -50.0
    proxy_pot = zeros(Float64, 1, 1, 3)

    delta = 0.1
    weight = 0.75
    npwlevel = BoxDMK.get_delta_cutoff_level(tree, delta, pw_data.eps)
    @test npwlevel < 0

    BoxDMK.handle_fat_gaussian!(proxy_pot, tree, proxy_charges, delta, weight, pw_data)

    pw_nodes, pw_weights = BoxDMK.get_pw_nodes(pw_data.eps, npwlevel, 2.0)
    kernel_ft = BoxDMK.kernel_fourier_transform([delta], [weight], pw_nodes, pw_weights, tree.ndim)
    expected_root = 2.0 * sum(kernel_ft)

    @test proxy_pot[1, 1, 1] ≈ expected_root
    @test proxy_pot[1, 1, 2] == 0.0
    @test proxy_pot[1, 1, 3] == 0.0
end
