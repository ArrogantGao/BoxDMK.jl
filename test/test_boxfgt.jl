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

function _two_child_tree_3d()
    centers = [
        0.5 0.25 0.75
        0.5 0.25 0.75
        0.5 0.25 0.75
    ]
    children = zeros(Int, 8, 3)
    children[1, 1] = 2
    children[8, 1] = 3

    return BoxDMK.BoxTree(
        3,
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

function _explicit_proxycharge_to_pw_3d(charges, tab_coefs2pw, porder::Int, npw::Int, nd::Int)
    npw_half = (npw + 1) ÷ 2
    coefs = reshape(charges, porder, porder, porder, nd)
    pwexp = Array{ComplexF64}(undef, npw, npw, npw_half, nd)

    @inbounds for id in 1:nd, k3 in 1:npw_half, k2 in 1:npw, k1 in 1:npw
        total = 0.0 + 0.0im
        for m3 in 1:porder, m2 in 1:porder, m1 in 1:porder
            total += coefs[m1, m2, m3, id] *
                     tab_coefs2pw[k1, m1] *
                     tab_coefs2pw[k2, m2] *
                     tab_coefs2pw[k3, m3]
        end
        pwexp[k1, k2, k3, id] = total
    end

    return reshape(pwexp, :, nd)
end

function _explicit_pw2proxypot_3d!(dest, pwexp, tab_pw2coefs, porder::Int, npw::Int, nd::Int)
    npw_half = (npw + 1) ÷ 2
    npw_sym = npw ÷ 2
    pw_tensor = reshape(pwexp, npw, npw, npw_half, nd)
    coefs = reshape(dest, porder, porder, porder, nd)

    @inbounds for id in 1:nd, k3 in 1:porder, k2 in 1:porder, k1 in 1:porder
        total = 0.0 + 0.0im
        for m3 in 1:npw_half, m2 in 1:npw, m1 in 1:npw
            weight = m3 > npw_sym ? 0.5 : 1.0
            total += weight *
                     tab_pw2coefs[m1, k1] *
                     tab_pw2coefs[m2, k2] *
                     tab_pw2coefs[m3, k3] *
                     pw_tensor[m1, m2, m3, id]
        end
        coefs[k1, k2, k3, id] += 2 * real(total)
    end

    return dest
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

@testset "Fortran PW Hotpaths" begin
    @test isdefined(BoxDMK, :_LIBBOXDMK_PATH)
    @test isdefined(BoxDMK, :_FORTRAN_HOTPATHS_AVAILABLE)

    if isdefined(BoxDMK, :_LIBBOXDMK_PATH) && isdefined(BoxDMK, :_FORTRAN_HOTPATHS_AVAILABLE)
        @test BoxDMK._FORTRAN_HOTPATHS_AVAILABLE[] == isfile(BoxDMK._LIBBOXDMK_PATH)

        if BoxDMK._FORTRAN_HOTPATHS_AVAILABLE[]
            porder = 2
            npw = 3
            nd = 2
            nproxy = porder^3
            nexp_half = BoxDMK.pw_expansion_size_half(npw, 3)
            charges = reshape(collect(1.0:(nproxy * nd)), nproxy, nd)
            tab_coefs2pw, _ = BoxDMK.build_pw_conversion_tables(LegendreBasis(), porder, npw, 1.0)
            tab_pw2coefs = conj.(tab_coefs2pw)

            expected_mp = _explicit_proxycharge_to_pw_3d(charges, tab_coefs2pw, porder, npw, nd)
            actual_mp = Matrix{ComplexF64}(undef, nexp_half, nd)
            BoxDMK._f_proxycharge2pw_3d!(actual_mp, charges, tab_coefs2pw, nd, porder, npw)
            @test actual_mp ≈ expected_mp atol = 1e-12 rtol = 1e-12

            shift_vec = Vector{ComplexF64}(undef, nexp_half)
            ww = BoxDMK.build_pw_1d_phase_table([-1.0, 0.0, 1.0], 1.0; nmax = 1)
            BoxDMK.compute_shift_vector!(shift_vec, ww, (1, 0, -1), npw, 1)

            expected_loc = copy(actual_mp)
            expected_loc .+= actual_mp .* shift_vec
            actual_loc = copy(actual_mp)
            BoxDMK._f_shiftpw!(actual_loc, actual_mp, shift_vec, nd, nexp_half)
            @test actual_loc ≈ expected_loc atol = 1e-12 rtol = 1e-12

            kernel_ft = collect(range(0.5, step = 0.25, length = nexp_half))
            expected_scaled = actual_loc .* kernel_ft
            actual_scaled = copy(actual_loc)
            BoxDMK._f_multiply_kernelft!(actual_scaled, kernel_ft, nd, nexp_half)
            @test actual_scaled ≈ expected_scaled atol = 1e-12 rtol = 1e-12

            expected_proxy = fill(3.0, nproxy, nd)
            _explicit_pw2proxypot_3d!(expected_proxy, actual_scaled, tab_pw2coefs, porder, npw, nd)
            actual_proxy = fill(3.0, nproxy, nd)
            BoxDMK._f_pw2proxypot_3d!(actual_proxy, actual_scaled, tab_pw2coefs, nd, porder, npw)
            @test actual_proxy ≈ expected_proxy atol = 1e-12 rtol = 1e-12
        end
    end
end

@testset "Box FGT 3D Matches Fortran PW Formulas" begin
    tree = _two_child_tree_3d()
    proxy = BoxDMK.build_proxy_data(LegendreBasis(), tree.norder, 2, tree.ndim)
    pw_data = BoxDMK.setup_planewave_data(tree, proxy, 1e-2; nd = 1, ifpwexp = [false, true, true], nmax = 1)
    lists = BoxDMK.InteractionLists([Int[] for _ in 1:3], [Int[], [3], [2]])

    proxy_charges = zeros(Float64, proxy.ncbox, 1, 3)
    proxy_charges[:, 1, 2] .= 1:proxy.ncbox
    proxy_charges[:, 1, 3] .= 10 .* (1:proxy.ncbox)

    proxy_pot = zeros(Float64, proxy.ncbox, 1, 3)
    deltas = [0.01, 0.016]
    weights = [2.0, -1.0]

    BoxDMK.boxfgt!(proxy_pot, tree, proxy_charges, deltas, weights, pw_data, lists)

    level = BoxDMK.get_delta_cutoff_level(tree, deltas[1], pw_data.eps)
    npw = pw_data.npw[level + 1]
    porder = proxy.porder
    nd = 1
    nexp_half = BoxDMK.pw_expansion_size_half(npw, tree.ndim)
    tab_coefs2pw = pw_data.tab_coefs2pw[level + 1]
    tab_pw2coefs = conj.(tab_coefs2pw)
    kernel_ft = BoxDMK.kernel_fourier_transform(
        deltas,
        weights,
        pw_data.pw_nodes[level + 1],
        pw_data.pw_weights[level + 1],
        tree.ndim,
    )
    ww = pw_data.ww_1d[level + 1]

    expected = zeros(Float64, proxy.ncbox, 1, 3)
    multipoles = Dict(
        2 => _explicit_proxycharge_to_pw_3d(@view(proxy_charges[:, :, 2]), tab_coefs2pw, porder, npw, nd),
        3 => _explicit_proxycharge_to_pw_3d(@view(proxy_charges[:, :, 3]), tab_coefs2pw, porder, npw, nd),
    )

    for ibox in (2, 3)
        loc = copy(multipoles[ibox])
        shift_vec = Vector{ComplexF64}(undef, nexp_half)
        for jbox in lists.listpw[ibox]
            offset = BoxDMK._box_offset(tree, ibox, jbox, level)
            BoxDMK.compute_shift_vector!(shift_vec, ww, offset, npw, pw_data.nmax)
            loc .+= multipoles[jbox] .* shift_vec
        end
        loc = loc .* kernel_ft
        _explicit_pw2proxypot_3d!(@view(expected[:, :, ibox]), loc, tab_pw2coefs, porder, npw, nd)
    end

    @test proxy_pot ≈ expected atol = 1e-11 rtol = 1e-11
end
