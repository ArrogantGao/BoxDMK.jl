struct DeltaGroups
    normal::Vector{Tuple{Int, Vector{Float64}, Vector{Float64}}}
    fat::Vector{Tuple{Float64, Float64}}
    asymptotic::Vector{Tuple{Float64, Float64}}
end

function _boxsize_at_level(tree::BoxTree, level::Integer)
    level_int = Int(level)
    if level_int >= 0
        0 <= level_int <= tree.nlevels || throw(ArgumentError("level $level_int is out of bounds"))
        return Float64(tree.boxsize[level_int + 1])
    end

    return Float64(tree.boxsize[1]) * 2.0^(-level_int)
end

function get_delta_cutoff_level(tree::BoxTree, delta::Real, eps::Real)
    delta_value = Float64(delta)
    eps_value = Float64(eps)
    delta_value > 0 || throw(ArgumentError("delta must be positive"))
    eps_value > 0 || throw(ArgumentError("eps must be positive"))

    dcutoff = sqrt(delta_value * log(inv(eps_value)))
    if dcutoff <= Float64(tree.boxsize[end])
        return tree.nlevels
    end

    for level in tree.nlevels:-1:0
        if Float64(tree.boxsize[level + 1]) >= dcutoff
            return level
        end
    end

    return -ceil(Int, log2(dcutoff / Float64(tree.boxsize[1])))
end

function _is_asymptotic_delta(tree::BoxTree, delta::Real, eps::Real)
    finest_box = Float64(tree.boxsize[end])
    return (4 * Float64(delta) / finest_box^2)^3 <= Float64(eps)
end

function group_deltas_by_level(sog::SOGNodes, tree::BoxTree, eps::Float64)
    length(sog.deltas) == length(sog.weights) || throw(ArgumentError("SOG deltas and weights must have the same length"))

    normal_map = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    fat = Tuple{Float64, Float64}[]
    asymptotic = Tuple{Float64, Float64}[]

    for (delta_raw, weight_raw) in zip(sog.deltas, sog.weights)
        delta = Float64(delta_raw)
        weight = Float64(weight_raw)
        level = get_delta_cutoff_level(tree, delta, eps)

        if level < 0
            push!(fat, (delta, weight))
        elseif _is_asymptotic_delta(tree, delta, eps)
            push!(asymptotic, (delta, weight))
        else
            deltas, weights = get!(normal_map, level) do
                (Float64[], Float64[])
            end
            push!(deltas, delta)
            push!(weights, weight)
        end
    end

    normal = Tuple{Int, Vector{Float64}, Vector{Float64}}[]
    for level in sort(collect(keys(normal_map)))
        deltas, weights = normal_map[level]
        push!(normal, (level, deltas, weights))
    end

    return DeltaGroups(normal, fat, asymptotic)
end

function _infer_proxy_order(ncbox::Integer, ndim::Integer)
    porder = round(Int, Float64(ncbox)^(1 / Int(ndim)))
    porder^Int(ndim) == ncbox || throw(ArgumentError("proxy array leading dimension must equal porder^ndim"))
    return porder
end

function _pw_nmax(shift::AbstractMatrix, ndim::Integer)
    ndim_int = Int(ndim)
    side = round(Int, size(shift, 2)^(1 / ndim_int))
    side^ndim_int == size(shift, 2) || throw(ArgumentError("invalid PW shift matrix width"))
    return (side - 1) ÷ 2
end

function _pw_expansion_view(pw_data::PlaneWaveData, which::Int, ibox::Int, nexp_half::Int, nd::Int)
    start = pw_data.iaddr[which, ibox]
    start > 0 || throw(ArgumentError("missing plane-wave workspace for box $ibox"))
    stop = start + nexp_half * nd - 1
    return reshape(@view(pw_data.rmlexp[start:stop]), nexp_half, nd)
end

function _box_offset(tree::BoxTree, ibox::Int, jbox::Int, level::Int)
    boxdim = _boxsize_at_level(tree, level)
    return ntuple(tree.ndim) do dim
        offset = (Float64(tree.centers[dim, ibox]) - Float64(tree.centers[dim, jbox])) / boxdim
        rounded = round(Int, offset)
        isapprox(offset, rounded; atol = 1e-8, rtol = 1e-8) || throw(ArgumentError("box centers are not aligned with level-$level spacing"))
        rounded
    end
end

function _use_fortran_pw_hotpath(dest, src, ndim::Int)
    return _FORTRAN_HOTPATHS_AVAILABLE[] &&
           ndim == 3 &&
           dest isa StridedArray{Float64,3} &&
           src isa StridedArray{Float64,3}
end

@inline function _pw_linear_index(multi_index, npw::Int)
    index = 1
    stride = 1

    for dim in eachindex(multi_index)
        index += (multi_index[dim] - 1) * stride
        stride *= npw
    end

    return index
end

function _proxycharge_to_pw!(
    dest,
    charges,
    tab_coefs2pw,
    ndim::Int,
    porder::Int,
    npw::Int,
    nd::Int,
    src_workspace::Union{Nothing, AbstractMatrix{ComplexF64}} = nothing,
    work_workspace::Union{Nothing, AbstractMatrix{ComplexF64}} = nothing,
    tensor_workspace = nothing,
)
    nproxy = porder^ndim
    npwexp = npw^ndim
    nexp_half = pw_expansion_size_half(npw, ndim)
    src = src_workspace === nothing ? Matrix{ComplexF64}(undef, nd, nproxy) : src_workspace
    work = work_workspace === nothing ? Matrix{ComplexF64}(undef, nd, npwexp) : work_workspace

    size(src) == (nd, nproxy) || throw(DimensionMismatch("src_workspace must have size ($nd, $nproxy)"))
    size(work) == (nd, npwexp) || throw(DimensionMismatch("work_workspace must have size ($nd, $npwexp)"))
    size(dest) == (nexp_half, nd) || throw(DimensionMismatch("dest must have size ($nexp_half, $nd)"))
    size(charges) == (nproxy, nd) || throw(DimensionMismatch("charges must have size ($nproxy, $nd)"))

    rect_workspace = tensor_workspace === nothing ?
        _rect_tensor_apply_workspace(ComplexF64, nd, porder, npw, ndim) :
        tensor_workspace

    @views src .= ComplexF64.(transpose(charges))
    _tensor_product_apply_rect!(work, tab_coefs2pw, src, porder, npw, ndim, nd, rect_workspace)
    npw2 = (npw + 1) ÷ 2

    if ndim == 3
        @inbounds for id in 1:nd
            idx = 1
            for i1 in 1:npw2
                for i2 in 1:npw
                    for i3 in 1:npw
                        j_full = i1 + (i2 - 1) * npw + (i3 - 1) * npw * npw
                        dest[idx, id] = work[id, j_full]
                        idx += 1
                    end
                end
            end
        end
        return dest
    end

    index_ranges = (1:npw2, ntuple(_ -> 1:npw, ndim - 1)...)
    @inbounds for id in 1:nd
        idx = 1
        for multi_index in Iterators.product(index_ranges...)
            dest[idx, id] = work[id, _pw_linear_index(multi_index, npw)]
            idx += 1
        end
    end
    return dest
end

function _expand_half_pw!(full_pw, half_pw, npw::Int, ndim::Int, nd::Int)
    npw2 = (npw + 1) ÷ 2

    if ndim == 3
        @inbounds for id in 1:nd
            idx = 1
            for i1 in 1:npw2
                for i2 in 1:npw
                    for i3 in 1:npw
                        j_full = i1 + (i2 - 1) * npw + (i3 - 1) * npw * npw
                        full_pw[id, j_full] = half_pw[idx, id]
                        idx += 1
                    end
                end
            end
            for i1 in (npw2 + 1):npw
                for i2 in 1:npw
                    for i3 in 1:npw
                        j_full = i1 + (i2 - 1) * npw + (i3 - 1) * npw * npw
                        i1s = npw + 1 - i1
                        i2s = npw + 1 - i2
                        i3s = npw + 1 - i3
                        j_sym = i1s + (i2s - 1) * npw + (i3s - 1) * npw * npw
                        full_pw[id, j_full] = conj(full_pw[id, j_sym])
                    end
                end
            end
        end
        return full_pw
    end

    half_ranges = (1:npw2, ntuple(_ -> 1:npw, ndim - 1)...)
    mirror_ranges = ((npw2 + 1):npw, ntuple(_ -> 1:npw, ndim - 1)...)

    @inbounds for id in 1:nd
        idx = 1
        for multi_index in Iterators.product(half_ranges...)
            full_pw[id, _pw_linear_index(multi_index, npw)] = half_pw[idx, id]
            idx += 1
        end
        for multi_index in Iterators.product(mirror_ranges...)
            symmetric_index = ntuple(dim -> npw + 1 - multi_index[dim], ndim)
            full_pw[id, _pw_linear_index(multi_index, npw)] = conj(full_pw[id, _pw_linear_index(symmetric_index, npw)])
        end
    end

    return full_pw
end

function _pw_to_proxy!(
    dest,
    local_pw,
    tab_pw2pot,
    ndim::Int,
    porder::Int,
    npw::Int,
    nd::Int,
    src_workspace::Union{Nothing, AbstractMatrix{ComplexF64}} = nothing,
    work_workspace::Union{Nothing, AbstractMatrix{ComplexF64}} = nothing,
    tensor_workspace = nothing,
)
    nproxy = porder^ndim
    npwexp = npw^ndim
    nexp_half = pw_expansion_size_half(npw, ndim)
    src = src_workspace === nothing ? Matrix{ComplexF64}(undef, nd, npwexp) : src_workspace
    work = work_workspace === nothing ? Matrix{ComplexF64}(undef, nd, nproxy) : work_workspace

    size(src) == (nd, npwexp) || throw(DimensionMismatch("src_workspace must have size ($nd, $npwexp)"))
    size(work) == (nd, nproxy) || throw(DimensionMismatch("work_workspace must have size ($nd, $nproxy)"))
    size(dest) == (nproxy, nd) || throw(DimensionMismatch("dest must have size ($nproxy, $nd)"))
    size(local_pw) == (nexp_half, nd) || throw(DimensionMismatch("local_pw must have size ($nexp_half, $nd)"))

    rect_workspace = tensor_workspace === nothing ?
        _rect_tensor_apply_workspace(ComplexF64, nd, npw, porder, ndim) :
        tensor_workspace

    _expand_half_pw!(src, local_pw, npw, ndim, nd)
    _tensor_product_apply_rect!(work, tab_pw2pot, src, npw, porder, ndim, nd, rect_workspace)
    @views dest .+= real.(transpose(work))
    return dest
end

function build_fat_gaussian_tables(tree::BoxTree, porder::Integer, eps::Real, level::Integer)
    level_int = Int(level)
    porder_int = _check_basis_order(porder)
    eps_value = Float64(eps)
    level_int < 0 || throw(ArgumentError("fat Gaussian cutoff level must be negative"))

    pw_boxdim = _boxsize_at_level(tree, level_int)
    pw_nodes, pw_weights = get_pw_nodes(eps_value, level_int, pw_boxdim)
    tab_coefs2pw, tab_pw2pot = build_pw_conversion_tables(LegendreBasis(), porder_int, pw_nodes, tree.boxsize[1])

    return (
        level = level_int,
        eps = eps_value,
        pw_nodes = pw_nodes,
        pw_weights = pw_weights,
        tab_coefs2pw = tab_coefs2pw,
        tab_pw2pot = tab_pw2pot,
    )
end

build_fat_gaussian_tables(tree::BoxTree, proxy::ProxyData, eps::Real, level::Integer) =
    build_fat_gaussian_tables(tree, proxy.porder, eps, level)

function _batch_level(tree::BoxTree, deltas::AbstractVector, eps::Real)
    isempty(deltas) && throw(ArgumentError("deltas must be nonempty"))
    level = get_delta_cutoff_level(tree, first(deltas), eps)
    for delta in deltas
        get_delta_cutoff_level(tree, delta, eps) == level ||
            throw(ArgumentError("all deltas in a Box FGT batch must share the same cutoff level"))
    end
    return level
end

function boxfgt!(
    proxy_pot,
    tree::BoxTree,
    proxy_charges,
    deltas,
    weights,
    pw_data::PlaneWaveData,
    lists::InteractionLists,
)
    size(proxy_pot) == size(proxy_charges) || throw(DimensionMismatch("proxy_pot and proxy_charges must have the same size"))
    size(proxy_charges, 3) == nboxes(tree) || throw(DimensionMismatch("proxy arrays must have one slice per tree box"))
    level = _batch_level(tree, deltas, pw_data.eps)
    level >= 0 || throw(ArgumentError("fat Gaussians must be handled with handle_fat_gaussian!"))
    level <= tree.nlevels || throw(ArgumentError("invalid cutoff level $level"))

    porder = _infer_proxy_order(size(proxy_charges, 1), tree.ndim)
    nd = size(proxy_charges, 2)
    npw = pw_data.npw[level + 1]
    nexp_half = pw_expansion_size_half(npw, tree.ndim)
    ww = pw_data.ww_1d[level + 1]
    nmax = pw_data.nmax
    level_boxes = collect(boxes_at_level(tree, level))
    use_fortran_hotpaths = _use_fortran_pw_hotpath(proxy_pot, proxy_charges, tree.ndim)
    tab_pw2coefs_f = use_fortran_hotpaths ? conj.(pw_data.tab_coefs2pw[level + 1]) : nothing
    nthreadslots = Threads.maxthreadid()
    shift_vecs = [Vector{ComplexF64}(undef, nexp_half) for _ in 1:nthreadslots]
    proxy_workspaces = [Matrix{ComplexF64}(undef, nd, porder^tree.ndim) for _ in 1:nthreadslots]
    pw_workspaces = [Matrix{ComplexF64}(undef, nd, npw^tree.ndim) for _ in 1:nthreadslots]
    to_pw_tensor_workspaces = [_rect_tensor_apply_workspace(ComplexF64, nd, porder, npw, tree.ndim) for _ in 1:nthreadslots]
    to_proxy_tensor_workspaces = [_rect_tensor_apply_workspace(ComplexF64, nd, npw, porder, tree.ndim) for _ in 1:nthreadslots]
    kernel_ft = kernel_fourier_transform(
        deltas,
        weights,
        pw_data.pw_nodes[level + 1],
        pw_data.pw_weights[level + 1],
        tree.ndim,
    )

    Threads.@threads for index in eachindex(level_boxes)
        ibox = level_boxes[index]
        pw_data.ifpwexp[ibox] || continue
        tid = Threads.threadid()
        mp = _pw_expansion_view(pw_data, 1, ibox, nexp_half, nd)
        loc = _pw_expansion_view(pw_data, 2, ibox, nexp_half, nd)
        fill!(mp, 0)
        fill!(loc, 0)

        if use_fortran_hotpaths
            _f_proxycharge2pw_3d!(mp, @view(proxy_charges[:, :, ibox]), pw_data.tab_coefs2pw[level + 1], nd, porder, npw)
        else
            _proxycharge_to_pw!(
                mp,
                @view(proxy_charges[:, :, ibox]),
                pw_data.tab_coefs2pw[level + 1],
                tree.ndim,
                porder,
                npw,
                nd,
                proxy_workspaces[tid],
                pw_workspaces[tid],
                to_pw_tensor_workspaces[tid],
            )
        end

        loc .= mp
    end

    Threads.@threads for index in eachindex(level_boxes)
        ibox = level_boxes[index]
        pw_data.ifpwexp[ibox] || continue
        tid = Threads.threadid()
        loc = _pw_expansion_view(pw_data, 2, ibox, nexp_half, nd)

        for jbox in lists.listpw[ibox]
            pw_data.ifpwexp[jbox] || continue
            tree.level[jbox] == level || continue

            offset = _box_offset(tree, ibox, jbox, level)
            compute_shift_vector!(shift_vecs[tid], ww, offset, npw, nmax)
            src = _pw_expansion_view(pw_data, 1, jbox, nexp_half, nd)

            if use_fortran_hotpaths
                _f_shiftpw!(loc, src, shift_vecs[tid], nd, nexp_half)
            else
                @views loc .+= src .* shift_vecs[tid]
            end
        end

        if use_fortran_hotpaths
            _f_multiply_kernelft!(loc, kernel_ft, nd, nexp_half)
            _f_pw2proxypot_3d!(@view(proxy_pot[:, :, ibox]), loc, tab_pw2coefs_f, nd, porder, npw)
        else
            @views loc .*= kernel_ft
            _pw_to_proxy!(
                @view(proxy_pot[:, :, ibox]),
                loc,
                pw_data.tab_pw2pot[level + 1],
                tree.ndim,
                porder,
                npw,
                nd,
                pw_workspaces[tid],
                proxy_workspaces[tid],
                to_proxy_tensor_workspaces[tid],
            )
        end
    end

    return proxy_pot
end

function handle_fat_gaussian!(proxy_pot, tree::BoxTree, proxy_charges, delta, weight, pw_data::PlaneWaveData)
    level = get_delta_cutoff_level(tree, delta, pw_data.eps)
    porder = _infer_proxy_order(size(proxy_charges, 1), tree.ndim)
    tables = build_fat_gaussian_tables(tree, porder, pw_data.eps, level)
    return handle_fat_gaussian!(proxy_pot, tree, proxy_charges, [delta], [weight], tables)
end

function handle_fat_gaussian!(proxy_pot, tree::BoxTree, proxy_charges, deltas::AbstractVector, weights::AbstractVector, tables)
    size(proxy_pot) == size(proxy_charges) || throw(DimensionMismatch("proxy_pot and proxy_charges must have the same size"))
    size(proxy_charges, 3) == nboxes(tree) || throw(DimensionMismatch("proxy arrays must have one slice per tree box"))
    length(deltas) == length(weights) || throw(DimensionMismatch("deltas and weights must have the same length"))
    isempty(deltas) && return proxy_pot
    tables.level < 0 || throw(ArgumentError("fat Gaussian cutoff level must be negative"))

    nd = size(proxy_charges, 2)
    porder = _infer_proxy_order(size(proxy_charges, 1), tree.ndim)
    for delta in deltas
        get_delta_cutoff_level(tree, delta, tables.eps) == tables.level ||
            throw(ArgumentError("all fat Gaussians in a batch must share cutoff level $(tables.level)"))
    end

    npw = length(tables.pw_nodes)
    nexp_half = pw_expansion_size_half(npw, tree.ndim)
    kernel_ft = kernel_fourier_transform(deltas, weights, tables.pw_nodes, tables.pw_weights, tree.ndim)
    use_fortran_hotpaths = _use_fortran_pw_hotpath(proxy_pot, proxy_charges, tree.ndim)
    tab_pw2coefs_f = use_fortran_hotpaths ? conj.(tables.tab_coefs2pw) : nothing
    proxy_workspace = Matrix{ComplexF64}(undef, nd, porder^tree.ndim)
    pw_workspace = Matrix{ComplexF64}(undef, nd, npw^tree.ndim)
    to_pw_workspace = _rect_tensor_apply_workspace(ComplexF64, nd, porder, npw, tree.ndim)
    to_proxy_workspace = _rect_tensor_apply_workspace(ComplexF64, nd, npw, porder, tree.ndim)

    mp = Matrix{ComplexF64}(undef, nexp_half, nd)
    if use_fortran_hotpaths
        _f_proxycharge2pw_3d!(mp, @view(proxy_charges[:, :, 1]), tables.tab_coefs2pw, nd, porder, npw)
    else
        _proxycharge_to_pw!(
            mp,
            @view(proxy_charges[:, :, 1]),
            tables.tab_coefs2pw,
            tree.ndim,
            porder,
            npw,
            nd,
            proxy_workspace,
            pw_workspace,
            to_pw_workspace,
        )
    end
    @views mp .*= kernel_ft
    if use_fortran_hotpaths
        _f_pw2proxypot_3d!(@view(proxy_pot[:, :, 1]), mp, tab_pw2coefs_f, nd, porder, npw)
    else
        _pw_to_proxy!(
            @view(proxy_pot[:, :, 1]),
            mp,
            tables.tab_pw2pot,
            tree.ndim,
            porder,
            npw,
            nd,
            pw_workspace,
            proxy_workspace,
            to_proxy_workspace,
        )
    end

    return proxy_pot
end
