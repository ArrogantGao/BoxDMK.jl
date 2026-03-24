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
pw_expansion_size_half(npw::Integer, ndim::Integer) = ((Int(npw) + 1) ÷ 2) * Int(npw)^(Int(ndim) - 1)

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

    npw2 = (npw + 1) ÷ 2
    kernel_ft = Vector{Float64}(undef, pw_expansion_size_half(npw, ndim_int))
    index = 1
    if ndim_int == 3
        @inbounds for i1 in 1:npw2
            for i2 in 1:npw
                for i3 in 1:npw
                    total = 0.0
                    for k in eachindex(deltas)
                        total = muladd(Float64(weights[k]), ww[i1, k] * ww[i2, k] * ww[i3, k], total)
                    end
                    kernel_ft[index] = total
                    index += 1
                end
            end
        end
        return kernel_ft
    end

    index_ranges = (1:npw2, ntuple(_ -> 1:npw, ndim_int - 1)...)
    for multi_index in Iterators.product(index_ranges...)
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

function build_pw_1d_phase_table(pw_nodes::AbstractVector, boxdim::Real; nmax::Integer = 3)
    npw = length(pw_nodes)
    nmax_int = Int(nmax)
    nmax_int >= 0 || throw(ArgumentError("nmax must be nonnegative"))
    boxdim_value = Float64(boxdim)
    center = nmax_int + 1
    ww = Matrix{ComplexF64}(undef, npw, 2 * nmax_int + 1)

    for j in 1:npw
        base = exp(im * Float64(pw_nodes[j]) * boxdim_value)
        ww[j, center] = 1.0 + 0im
        ztmp = base
        for k in 1:nmax_int
            ww[j, center + k] = ztmp
            ww[j, center - k] = conj(ztmp)
            ztmp *= base
        end
    end

    return ww
end

function compute_shift_vector!(shift_vec::AbstractVector{ComplexF64}, ww::Matrix{ComplexF64}, offset::NTuple{N, Int}, npw::Int, nmax::Int) where {N}
    length(shift_vec) == pw_expansion_size_half(npw, N) ||
        throw(DimensionMismatch("shift_vec must have length $(pw_expansion_size_half(npw, N))"))
    center = nmax + 1
    npw2 = (npw + 1) ÷ 2
    idx = 1
    if N == 3
        @inbounds for i1 in 1:npw2
            w1 = ww[i1, offset[1] + center]
            for i2 in 1:npw
                w12 = w1 * ww[i2, offset[2] + center]
                for i3 in 1:npw
                    shift_vec[idx] = w12 * ww[i3, offset[3] + center]
                    idx += 1
                end
            end
        end
    else
        index_ranges = (1:npw2, ntuple(_ -> 1:npw, N - 1)...)
        for multi_index in Iterators.product(index_ranges...)
            value = 1.0 + 0im
            for dim in 1:N
                value *= ww[multi_index[dim], offset[dim] + center]
            end
            shift_vec[idx] = value
            idx += 1
        end
    end
    return shift_vec
end

# Keep old API for tests that use build_pw_shift_matrices directly
function build_pw_shift_matrices(pw_nodes::AbstractVector, boxdim::Real, ndim::Integer; nmax::Integer = 3)
    npw = length(pw_nodes)
    ndim_int = Int(ndim)
    nmax_int = Int(nmax)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))
    nmax_int >= 0 || throw(ArgumentError("nmax must be nonnegative"))

    ww = build_pw_1d_phase_table(pw_nodes, boxdim; nmax = nmax_int)
    nexp = pw_expansion_size_half(npw, ndim_int)
    ntranslations = (2 * nmax_int + 1)^ndim_int
    shift = Matrix{ComplexF64}(undef, nexp, ntranslations)

    column = 1
    for offset in Iterators.product(ntuple(_ -> (-nmax_int):nmax_int, ndim_int)...)
        compute_shift_vector!(@view(shift[:, column]), ww, offset, npw, nmax_int)
        column += 1
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
    nmax::Integer = 1,
    needed_levels::Union{Nothing, AbstractSet{<:Integer}} = nothing,
)
    nd_int = Int(nd)
    nmax_int = Int(nmax)
    nd_int > 0 || throw(ArgumentError("nd must be positive"))
    length(ifpwexp) == nboxes(tree) || throw(DimensionMismatch("ifpwexp must have length $(nboxes(tree))"))
    needed_levels_int = needed_levels === nothing ? nothing : Set(Int(level) for level in needed_levels)

    if needed_levels_int !== nothing
        for level in needed_levels_int
            0 <= level <= tree.nlevels || throw(ArgumentError("needed_levels contains invalid level $level"))
        end
    end

    npw = Vector{Int}(undef, tree.nlevels + 1)
    pw_nodes = Vector{Vector{Float64}}(undef, tree.nlevels + 1)
    pw_weights = Vector{Vector{Float64}}(undef, tree.nlevels + 1)
    ww_1d = Vector{Matrix{ComplexF64}}(undef, tree.nlevels + 1)
    tab_coefs2pw = Vector{Matrix{ComplexF64}}(undef, tree.nlevels + 1)
    tab_pw2pot = Vector{Matrix{ComplexF64}}(undef, tree.nlevels + 1)

    for level in 0:tree.nlevels
        level_index = level + 1
        boxdim = Float64(tree.boxsize[level_index])
        npw[level_index] = get_pw_term_count(eps, level)
        pw_nodes[level_index], pw_weights[level_index] = get_pw_nodes(eps, level, boxdim)
        if needed_levels_int === nothing || in(level, needed_levels_int)
            tab_coefs2pw[level_index], tab_pw2pot[level_index] = build_pw_conversion_tables(
                LegendreBasis(),
                proxy.porder,
                pw_nodes[level_index],
                boxdim,
            )
            ww_1d[level_index] = build_pw_1d_phase_table(pw_nodes[level_index], boxdim; nmax = nmax_int)
        else
            ww_1d[level_index] = Matrix{ComplexF64}(undef, 0, 0)
            tab_coefs2pw[level_index] = Matrix{ComplexF64}(undef, 0, 0)
            tab_pw2pot[level_index] = Matrix{ComplexF64}(undef, 0, 0)
        end
    end

    iaddr = zeros(Int, 2, nboxes(tree))
    total = 0
    ifpwexp_vec = collect(Bool.(ifpwexp))

    for ibox in 1:nboxes(tree)
        if !ifpwexp_vec[ibox]
            continue
        end

        nexp = pw_expansion_size_half(npw[tree.level[ibox] + 1], tree.ndim)
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
        ww_1d,
        nmax_int,
        tab_coefs2pw,
        tab_pw2pot,
        ifpwexp_vec,
        Float64(eps),
    )
end
