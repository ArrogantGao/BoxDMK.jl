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
    rmlexp = zeros(ComplexF64, 4)
    iaddr = zeros(Int, 2, BoxDMK.nboxes(tree))
    iaddr[:, 2] .= (1, 2)
    iaddr[:, 3] .= (3, 4)

    return BoxDMK.PlaneWaveData(
        rmlexp,
        iaddr,
        [1, 2],
        [[0.0], [0.0, 0.0]],
        [[1.0], [1.0, 1.0]],
        [ones(ComplexF64, 1, 3), ones(ComplexF64, 2, 3)],
        1,
        [ones(ComplexF64, 1, 1), ones(ComplexF64, 2, 1)],
        [ones(ComplexF64, 1, 1), ComplexF64[0.5 0.5]],
        [false, true, true],
        eps,
    )
end

function _reconstruct_full_pw(half_pw::AbstractMatrix{ComplexF64}, npw::Int, ndim::Int)
    npw2 = (npw + 1) ÷ 2
    nfull = npw^ndim
    full_pw = Matrix{ComplexF64}(undef, nfull, size(half_pw, 2))

    for id in axes(half_pw, 2)
        idx = 1
        full_tensor = Array{ComplexF64}(undef, ntuple(_ -> npw, ndim))
        if ndim == 3
            for i1 in 1:npw2
                for i2 in 1:npw
                    for i3 in 1:npw
                        full_tensor[i1, i2, i3] = half_pw[idx, id]
                        idx += 1
                    end
                end
            end
            for i1 in (npw2 + 1):npw
                for i2 in 1:npw
                    for i3 in 1:npw
                        full_tensor[i1, i2, i3] = conj(full_tensor[npw + 1 - i1, npw + 1 - i2, npw + 1 - i3])
                    end
                end
            end
        else
            for multi_index in Iterators.product(1:npw2, ntuple(_ -> 1:npw, ndim - 1)...)
                full_tensor[multi_index...] = half_pw[idx, id]
                idx += 1
            end
            for multi_index in Iterators.product((npw2 + 1):npw, ntuple(_ -> 1:npw, ndim - 1)...)
                symmetric_index = ntuple(dim -> npw + 1 - multi_index[dim], ndim)
                full_tensor[multi_index...] = conj(full_tensor[symmetric_index...])
            end
        end
        full_pw[:, id] .= vec(full_tensor)
    end

    return full_pw
end

function _half_linear_indices(npw::Int, ndim::Int)
    npw2 = (npw + 1) ÷ 2
    indices = Int[]

    if ndim == 3
        for i1 in 1:npw2
            for i2 in 1:npw
                for i3 in 1:npw
                    push!(indices, i1 + (i2 - 1) * npw + (i3 - 1) * npw * npw)
                end
            end
        end
        return indices
    end

    for multi_index in Iterators.product(1:npw2, ntuple(_ -> 1:npw, ndim - 1)...)
        index = 1
        stride = 1
        for dim in 1:ndim
            index += (multi_index[dim] - 1) * stride
            stride *= npw
        end
        push!(indices, index)
    end

    return indices
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

    proxy_charges = zeros(Float64, 1, 1, 3)
    proxy_charges[1, 1, 2] = 1.0
    proxy_charges[1, 1, 3] = 10.0

    proxy_pot = zeros(Float64, 1, 1, 3)
    deltas = [0.01, 0.016]
    weights = [2.0, -1.0]

    BoxDMK.boxfgt!(proxy_pot, tree, proxy_charges, deltas, weights, pw_data, lists)

    scale = sum(weights .* sqrt.(deltas))
    @test proxy_pot[:, 1, 1] == [0.0]
    @test proxy_pot[:, 1, 2] ≈ scale .* [11.0]
    @test proxy_pot[:, 1, 3] ≈ scale .* [11.0]
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
    kernel_ft_half = BoxDMK.kernel_fourier_transform([delta], [weight], pw_nodes, pw_weights, tree.ndim)
    kernel_ft_full = vec(_reconstruct_full_pw(reshape(ComplexF64.(kernel_ft_half), :, 1), length(pw_nodes), tree.ndim))
    expected_root = 2.0 * real(sum(kernel_ft_full))

    @test proxy_pot[1, 1, 1] ≈ expected_root
    @test proxy_pot[1, 1, 2] == 0.0
    @test proxy_pot[1, 1, 3] == 0.0
end

@testset "Fat Gaussian Batching Reuses Cached Tables" begin
    tree = _two_child_tree()
    proxy = BoxDMK.build_proxy_data(LegendreBasis(), tree.norder, 1, tree.ndim)
    pw_data = _simple_pw_data(tree)
    proxy_charges = zeros(Float64, 1, 1, 3)
    proxy_charges[1, 1, 1] = 2.0

    deltas = [0.1, 0.2]
    weights = [0.75, -0.25]
    level = BoxDMK.get_delta_cutoff_level(tree, deltas[1], pw_data.eps)

    @test level < 0
    @test all(BoxDMK.get_delta_cutoff_level(tree, delta, pw_data.eps) == level for delta in deltas)

    cached_tables = BoxDMK.build_fat_gaussian_tables(tree, proxy, pw_data.eps, level)
    batched = zeros(Float64, 1, 1, 3)
    sequential = zeros(Float64, 1, 1, 3)

    BoxDMK.handle_fat_gaussian!(batched, tree, proxy_charges, deltas, weights, cached_tables)

    for (delta, weight) in zip(deltas, weights)
        BoxDMK.handle_fat_gaussian!(sequential, tree, proxy_charges, delta, weight, pw_data)
    end

    @test batched ≈ sequential
end

@testset "PW Conversion Workspaces" begin
    ndim = 3
    porder = 2
    npw = 2
    nd = 2
    nproxy = porder^ndim
    npwexp = npw^ndim
    npwexp_half = BoxDMK.pw_expansion_size_half(npw, ndim)
    tab = Matrix{ComplexF64}(I, npw, porder)

    charges = reshape(collect(1.0:(nproxy * nd)), nproxy, nd)
    multipole = Matrix{ComplexF64}(undef, npwexp_half, nd)
    src_workspace = Matrix{ComplexF64}(undef, nd, nproxy)
    work_workspace = Matrix{ComplexF64}(undef, nd, npwexp)

    BoxDMK._proxycharge_to_pw!(
        multipole,
        charges,
        tab,
        ndim,
        porder,
        npw,
        nd,
        src_workspace,
        work_workspace,
    )
    @test multipole == ComplexF64.(charges[_half_linear_indices(npw, ndim), :])

    local_pw = ComplexF64.(reshape(collect(1:(npwexp_half * nd)), npwexp_half, nd) .+ 2im)
    proxy_pot = fill(3.0, nproxy, nd)
    expected_full = _reconstruct_full_pw(local_pw, npw, ndim)

    BoxDMK._pw_to_proxy!(
        proxy_pot,
        local_pw,
        tab,
        ndim,
        porder,
        npw,
        nd,
        src_workspace,
        work_workspace,
    )
    @test proxy_pot == fill(3.0, nproxy, nd) .+ real.(expected_full)
end
