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

function _pw_expansion_view(pw_data::PlaneWaveData, which::Int, ibox::Int, nexp::Int, nd::Int)
    start = pw_data.iaddr[which, ibox]
    start > 0 || throw(ArgumentError("missing plane-wave workspace for box $ibox"))
    stop = start + nexp * nd - 1
    return reshape(@view(pw_data.rmlexp[start:stop]), nexp, nd)
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

function _proxycharge_to_pw!(dest, charges, tab_coefs2pw, ndim::Int, porder::Int, npw::Int, nd::Int)
    src = Matrix{ComplexF64}(undef, nd, porder^ndim)
    work = Matrix{ComplexF64}(undef, nd, npw^ndim)
    @views src .= ComplexF64.(transpose(charges))
    _tensor_product_apply_rect!(work, tab_coefs2pw, src, porder, npw, ndim, nd)
    dest .= transpose(work)
    return dest
end

function _pw_to_proxy!(dest, local_pw, tab_pw2pot, ndim::Int, porder::Int, npw::Int, nd::Int)
    src = Matrix{ComplexF64}(undef, nd, npw^ndim)
    work = Matrix{ComplexF64}(undef, nd, porder^ndim)
    src .= transpose(local_pw)
    _tensor_product_apply_rect!(work, tab_pw2pot, src, npw, porder, ndim, nd)
    @views dest .+= real.(transpose(work))
    return dest
end

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
    nexp = pw_expansion_size(npw, tree.ndim)
    shift = pw_data.wpwshift[level + 1]
    nmax = _pw_nmax(shift, tree.ndim)
    kernel_ft = kernel_fourier_transform(
        deltas,
        weights,
        pw_data.pw_nodes[level + 1],
        pw_data.pw_weights[level + 1],
        tree.ndim,
    )

    size(shift, 1) == nexp || throw(DimensionMismatch("PW shift matrix has incompatible expansion length"))

    for ibox in boxes_at_level(tree, level)
        pw_data.ifpwexp[ibox] || continue
        mp = _pw_expansion_view(pw_data, 1, ibox, nexp, nd)
        loc = _pw_expansion_view(pw_data, 2, ibox, nexp, nd)
        fill!(mp, 0)
        fill!(loc, 0)
        _proxycharge_to_pw!(mp, @view(proxy_charges[:, :, ibox]), pw_data.tab_coefs2pw[level + 1], tree.ndim, porder, npw, nd)
        loc .= mp
    end

    for ibox in boxes_at_level(tree, level)
        pw_data.ifpwexp[ibox] || continue
        loc = _pw_expansion_view(pw_data, 2, ibox, nexp, nd)

        for jbox in lists.listpw[ibox]
            pw_data.ifpwexp[jbox] || continue
            tree.level[jbox] == level || continue

            offset = _box_offset(tree, ibox, jbox, level)
            column = _translation_index(offset, nmax)
            src = _pw_expansion_view(pw_data, 1, jbox, nexp, nd)
            @views loc .+= src .* shift[:, column]
        end

        @views loc .*= kernel_ft
        _pw_to_proxy!(@view(proxy_pot[:, :, ibox]), loc, pw_data.tab_pw2pot[level + 1], tree.ndim, porder, npw, nd)
    end

    return proxy_pot
end

function handle_fat_gaussian!(proxy_pot, tree::BoxTree, proxy_charges, delta, weight, pw_data::PlaneWaveData)
    size(proxy_pot) == size(proxy_charges) || throw(DimensionMismatch("proxy_pot and proxy_charges must have the same size"))
    size(proxy_charges, 3) == nboxes(tree) || throw(DimensionMismatch("proxy arrays must have one slice per tree box"))

    level = get_delta_cutoff_level(tree, delta, pw_data.eps)
    level < 0 || throw(ArgumentError("delta=$delta is not a fat Gaussian for eps=$(pw_data.eps)"))

    nd = size(proxy_charges, 2)
    porder = _infer_proxy_order(size(proxy_charges, 1), tree.ndim)
    pw_boxdim = _boxsize_at_level(tree, level)
    pw_nodes, pw_weights = get_pw_nodes(pw_data.eps, level, pw_boxdim)
    npw = length(pw_nodes)
    nexp = pw_expansion_size(npw, tree.ndim)
    tab_coefs2pw, tab_pw2pot = build_pw_conversion_tables(LegendreBasis(), porder, pw_nodes, tree.boxsize[1])
    kernel_ft = kernel_fourier_transform([delta], [weight], pw_nodes, pw_weights, tree.ndim)

    mp = Matrix{ComplexF64}(undef, nexp, nd)
    _proxycharge_to_pw!(mp, @view(proxy_charges[:, :, 1]), tab_coefs2pw, tree.ndim, porder, npw, nd)
    @views mp .*= kernel_ft
    _pw_to_proxy!(@view(proxy_pot[:, :, 1]), mp, tab_pw2pot, tree.ndim, porder, npw, nd)

    return proxy_pot
end
