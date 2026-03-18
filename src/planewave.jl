const _PW_TERM_TABLES = Dict(
    2 => (2, 2, 2, 2, 2, 2, 2, 3, 4, 6, 8),
    3 => (3, 3, 3, 3, 3, 3, 3, 4, 5, 8, 11),
    4 => (4, 4, 4, 4, 3, 4, 4, 5, 7, 10, 15),
    5 => (4, 4, 4, 4, 4, 4, 5, 6, 8, 12, 19),
    6 => (5, 5, 5, 5, 5, 5, 6, 7, 9, 14, 22),
    7 => (5, 6, 6, 6, 5, 6, 7, 8, 11, 16, 26),
    8 => (6, 6, 6, 6, 6, 7, 7, 9, 12, 18, 29),
    9 => (7, 7, 7, 7, 7, 7, 8, 10, 13, 20, 33),
    10 => (7, 7, 8, 8, 8, 8, 9, 11, 15, 22, 36),
    11 => (8, 8, 8, 8, 8, 9, 10, 12, 16, 25, 40),
)

function _pw_table_digits(eps::Real)
    eps_value = Float64(eps)
    eps_value > 0 || throw(ArgumentError("eps must be positive"))

    if 1e-3 < eps_value <= 1e-2
        return 2
    elseif 1e-4 < eps_value <= 1e-3
        return 3
    elseif 1e-5 < eps_value <= 1e-4
        return 4
    elseif 1e-6 < eps_value <= 1e-5
        return 5
    elseif 1e-7 < eps_value <= 1e-6
        return 6
    elseif 1e-8 < eps_value <= 1e-7
        return 7
    elseif 1e-9 < eps_value <= 1e-8
        return 8
    elseif 1e-10 < eps_value <= 1e-9
        return 9
    elseif 1e-11 < eps_value <= 1e-10
        return 10
    elseif 1e-12 < eps_value <= 1e-11
        return 11
    end

    return nothing
end

function get_pw_term_count(eps::Real, level::Integer)
    eps_value = Float64(eps)
    level_int = Int(level)
    digits = _pw_table_digits(eps_value)

    if digits === nothing || level_int < -10
        return 2 * ceil(Int, 2 * log(10 / eps_value) / pi)
    end

    table = _PW_TERM_TABLES[digits]
    return 2 * table[min(level_int, 0) + 11]
end

function _default_pw_quadrature(npw::Int, boxdim::Real)
    boxdim_value = Float64(boxdim)
    boxdim_value > 0 || throw(ArgumentError("boxdim must be positive"))
    step = 2pi / boxdim_value
    offsets = collect(1:npw) .- (npw + 1) / 2
    nodes = step .* offsets
    weights = fill(step / (2 * sqrt(pi)), npw)
    return nodes, weights
end

function get_pw_nodes(eps::Real, level::Integer, boxdim::Real = 2.0^(-Int(level)))
    return _default_pw_quadrature(get_pw_term_count(eps, level), boxdim)
end

function build_pw_conversion_tables(basis::AbstractBasis, porder::Integer, npw::Integer, boxdim::Real)
    pw_nodes, _ = _default_pw_quadrature(Int(npw), boxdim)
    return build_pw_conversion_tables(basis, porder, pw_nodes, boxdim)
end

function build_pw_conversion_tables(
    basis::AbstractBasis,
    porder::Integer,
    pw_nodes::AbstractVector,
    boxdim::Real,
)
    porder_int = _check_basis_order(porder)
    source_nodes, _ = nodes_and_weights(basis, porder_int)
    forward = ComplexF64.(forward_transform(basis, porder_int))
    scaled_nodes = (Float64(boxdim) / 2) .* Float64.(source_nodes)
    npw = length(pw_nodes)

    tab_pw2pot = Matrix{ComplexF64}(undef, porder_int, npw)
    for (j, node) in pairs(pw_nodes)
        tab_pw2pot[:, j] .= forward * exp.(im * Float64(node) .* scaled_nodes)
    end

    tab_coefs2pw = conj.(transpose(tab_pw2pot))
    return tab_coefs2pw, tab_pw2pot
end

pw_expansion_size(npw::Integer, ndim::Integer) = Int(npw)^Int(ndim)

function kernel_fourier_transform(
    deltas::AbstractVector,
    weights::AbstractVector,
    pw_nodes::AbstractVector,
    pw_weights::AbstractVector,
    ndim::Integer,
)
    length(deltas) == length(weights) || throw(DimensionMismatch("deltas and weights must have the same length"))
    npw = length(pw_nodes)
    length(pw_weights) == npw || throw(DimensionMismatch("pw_nodes and pw_weights must have the same length"))
    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))

    ww = Matrix{Float64}(undef, npw, length(deltas))
    for (k, delta_raw) in pairs(deltas)
        delta = Float64(delta_raw)
        delta > 0 || throw(ArgumentError("deltas must be positive"))
        for i in 1:npw
            ww[i, k] = Float64(pw_weights[i]) * sqrt(delta) * exp(-(Float64(pw_nodes[i])^2) * delta / 4)
        end
    end

    kernel_ft = Vector{Float64}(undef, pw_expansion_size(npw, ndim_int))
    index = 1
    for multi_index in Iterators.product(ntuple(_ -> 1:npw, ndim_int)...)
        total = 0.0
        for k in eachindex(deltas)
            contribution = Float64(weights[k])
            for dim in 1:ndim_int
                contribution *= ww[multi_index[dim], k]
            end
            total += contribution
        end
        kernel_ft[index] = total
        index += 1
    end

    return kernel_ft
end

function kernel_fourier_transform!(
    out::AbstractVector,
    deltas::AbstractVector,
    weights::AbstractVector,
    pw_nodes::AbstractVector,
    pw_weights::AbstractVector,
    ndim::Integer,
)
    kernel_ft = kernel_fourier_transform(deltas, weights, pw_nodes, pw_weights, ndim)
    length(out) == length(kernel_ft) || throw(DimensionMismatch("output must have length $(length(kernel_ft))"))
    out .= kernel_ft
    return out
end

function _translation_index(offset::NTuple{N, Int}, nmax::Int) where {N}
    base = 2 * nmax + 1
    index = 1
    stride = 1

    for dim in 1:N
        component = offset[dim]
        abs(component) <= nmax || throw(ArgumentError("translation offset $offset exceeds nmax=$nmax"))
        index += (component + nmax) * stride
        stride *= base
    end

    return index
end

function build_pw_shift_matrices(pw_nodes::AbstractVector, boxdim::Real, ndim::Integer; nmax::Integer = 3)
    npw = length(pw_nodes)
    ndim_int = Int(ndim)
    nmax_int = Int(nmax)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))
    nmax_int >= 0 || throw(ArgumentError("nmax must be nonnegative"))

    nexp = pw_expansion_size(npw, ndim_int)
    ntranslations = (2 * nmax_int + 1)^ndim_int
    shift = Matrix{ComplexF64}(undef, nexp, ntranslations)
    frequencies = collect(Iterators.product(ntuple(_ -> 1:npw, ndim_int)...))

    for offset in Iterators.product(ntuple(_ -> (-nmax_int):nmax_int, ndim_int)...)
        column = _translation_index(offset, nmax_int)
        for (row, multi_index) in pairs(frequencies)
            phase = 0.0
            for dim in 1:ndim_int
                phase += Float64(pw_nodes[multi_index[dim]]) * offset[dim]
            end
            shift[row, column] = exp(im * Float64(boxdim) * phase)
        end
    end

    return shift
end

function build_pw_shift_matrices(npw::Integer, boxdim::Real, ndim::Integer; nmax::Integer = 3)
    pw_nodes, _ = _default_pw_quadrature(Int(npw), boxdim)
    return build_pw_shift_matrices(pw_nodes, boxdim, ndim; nmax = nmax)
end

function setup_planewave_data(
    tree::BoxTree,
    proxy::ProxyData,
    eps::Real;
    nd::Integer = 1,
    ifpwexp = trues(nboxes(tree)),
    nmax::Integer = 3,
)
    nd_int = Int(nd)
    nd_int > 0 || throw(ArgumentError("nd must be positive"))
    length(ifpwexp) == nboxes(tree) || throw(DimensionMismatch("ifpwexp must have length $(nboxes(tree))"))

    npw = Vector{Int}(undef, tree.nlevels + 1)
    pw_nodes = Vector{Vector{Float64}}(undef, tree.nlevels + 1)
    pw_weights = Vector{Vector{Float64}}(undef, tree.nlevels + 1)
    wpwshift = Vector{Matrix{ComplexF64}}(undef, tree.nlevels + 1)
    tab_coefs2pw = Vector{Matrix{ComplexF64}}(undef, tree.nlevels + 1)
    tab_pw2pot = Vector{Matrix{ComplexF64}}(undef, tree.nlevels + 1)

    for level in 0:tree.nlevels
        level_index = level + 1
        boxdim = Float64(tree.boxsize[level_index])
        npw[level_index] = get_pw_term_count(eps, level)
        pw_nodes[level_index], pw_weights[level_index] = get_pw_nodes(eps, level, boxdim)
        tab_coefs2pw[level_index], tab_pw2pot[level_index] = build_pw_conversion_tables(
            LegendreBasis(),
            proxy.porder,
            pw_nodes[level_index],
            boxdim,
        )
        wpwshift[level_index] = build_pw_shift_matrices(pw_nodes[level_index], boxdim, tree.ndim; nmax = nmax)
    end

    iaddr = zeros(Int, 2, nboxes(tree))
    total = 0
    ifpwexp_vec = Bool.(ifpwexp)

    for ibox in 1:nboxes(tree)
        if !ifpwexp_vec[ibox]
            continue
        end

        nexp = pw_expansion_size(npw[tree.level[ibox] + 1], tree.ndim)
        len = nexp * nd_int
        iaddr[1, ibox] = total + 1
        total += len
        iaddr[2, ibox] = total + 1
        total += len
    end

    return PlaneWaveData(
        zeros(ComplexF64, total),
        iaddr,
        npw,
        pw_nodes,
        pw_weights,
        wpwshift,
        tab_coefs2pw,
        tab_pw2pot,
        ifpwexp_vec,
        Float64(eps),
    )
end
