const _LOCAL_OFFSET_RADIUS = 1
const _LOCAL_OFFSET_COUNT = 2 * _LOCAL_OFFSET_RADIUS + 1
const _LOCAL_QUAD_ORDER = 50
const _LOCAL_SPARSE_TOL = 1.0e-12

function _basis_polynomial_values(basis::LegendreBasis, points::AbstractVector, norder::Int)
    values, _, _ = _legendre_eval_matrices(points, norder)
    return Float64.(values)
end

function _basis_polynomial_values(basis::ChebyshevBasis, points::AbstractVector, norder::Int)
    values, _, _ = _chebyshev_eval_matrices(points, norder)
    return Float64.(values)
end

function _compute_sparse_pattern(table::AbstractMatrix{<:Real}, tolerance::Float64)
    nrows, ncols = size(table)
    pattern = zeros(Int, 2, ncols + 1)
    pattern[2, :] .= -1

    dmax = maximum(abs, table; init = 0.0)
    dmax <= 2 * eps(Float64) && return pattern

    cutoff = tolerance * dmax / max(nrows, 1)

    for col in 1:ncols
        first_nonzero = 0
        last_nonzero = -1

        for row in 1:nrows
            if abs(table[row, col]) > cutoff
                first_nonzero = row
                break
            end
        end

        if first_nonzero != 0
            for row in nrows:-1:1
                if abs(table[row, col]) > cutoff
                    last_nonzero = row
                    break
                end
            end
        end

        pattern[1, col] = first_nonzero
        pattern[2, col] = last_nonzero
    end

    for col in 1:ncols
        if pattern[1, col] > 0
            pattern[1, ncols + 1] = col
            break
        end
    end

    for col in ncols:-1:1
        if pattern[2, col] > 0
            pattern[2, ncols + 1] = col
            break
        end
    end

    return pattern
end

function _merge_sparse_patterns(patterns::Vararg{AbstractMatrix{Int}})
    ncols_plus_one = size(first(patterns), 2)
    ncols = ncols_plus_one - 1
    merged = zeros(Int, 2, ncols_plus_one)
    merged[2, :] .= -1

    for col in 1:ncols
        first_nonzero = 0
        last_nonzero = -1

        for pattern in patterns
            candidate_first = pattern[1, col]
            candidate_last = pattern[2, col]

            if candidate_first != 0 && (first_nonzero == 0 || candidate_first < first_nonzero)
                first_nonzero = candidate_first
            end
            if candidate_last > last_nonzero
                last_nonzero = candidate_last
            end
        end

        if last_nonzero > 0 && first_nonzero == 0
            first_nonzero = 1
        end

        merged[1, col] = first_nonzero
        merged[2, col] = last_nonzero
    end

    for col in 1:ncols
        if merged[1, col] > 0
            merged[1, ncols_plus_one] = col
            break
        end
    end

    for col in ncols:-1:1
        if merged[2, col] > 0
            merged[2, ncols_plus_one] = col
            break
        end
    end

    return merged
end

function _build_local_modal_tables(
    target_nodes::AbstractVector{<:Real},
    quad_nodes::AbstractVector{<:Real},
    quad_weights::AbstractVector{<:Real},
    basis_values::AbstractMatrix{<:Real},
    boxsize::Float64,
    offset::Int,
    delta::Float64,
)
    source_positions = (boxsize / 2) .* quad_nodes
    target_positions = offset .* boxsize .+ (boxsize / 2) .* target_nodes
    weighted_quadrature = (boxsize / 2) .* quad_weights

    dx = reshape(target_positions, 1, :) .- reshape(source_positions, :, 1)
    gaussian = exp.(-(dx .^ 2) ./ delta)
    weighted_kernel = reshape(weighted_quadrature, :, 1) .* gaussian

    inv_delta = inv(delta)
    first_derivative = (-2 * inv_delta) .* dx .* weighted_kernel
    second_derivative = ((4 * inv_delta^2) .* (dx .^ 2) .- 2 * inv_delta) .* weighted_kernel

    polyv_t = transpose(basis_values)
    tab_modal = polyv_t * weighted_kernel
    tabx_modal = polyv_t * first_derivative
    tabxx_modal = polyv_t * second_derivative

    return tab_modal, tabx_modal, tabxx_modal
end

function build_local_tables(
    kernel::AbstractKernel,
    basis::AbstractBasis,
    norder::Int,
    ndim::Int,
    deltas::Vector{Float64},
    boxsizes::Vector{Float64},
    nlevels::Int,
)
    _ = kernel
    order = _check_basis_order(norder)
    ndim == 3 || throw(ArgumentError("build_local_tables currently expects ndim == 3"))
    nlevels >= 0 || throw(ArgumentError("nlevels must be nonnegative"))
    length(boxsizes) == nlevels + 1 || throw(DimensionMismatch("boxsizes must have length $(nlevels + 1)"))
    isempty(deltas) && throw(ArgumentError("deltas must be nonempty"))
    all(>(0.0), deltas) || throw(ArgumentError("deltas must be positive"))
    all(>(0.0), boxsizes) || throw(ArgumentError("boxsizes must be positive"))

    target_nodes, _ = nodes_and_weights(basis, order)
    quad_nodes, quad_weights = nodes_and_weights(LegendreBasis(), _LOCAL_QUAD_ORDER)
    basis_values = _basis_polynomial_values(basis, quad_nodes, order)
    source_transform = transpose(forward_transform(basis, order))

    tab = Array{Float64,5}(undef, order, order, _LOCAL_OFFSET_COUNT, length(deltas), nlevels + 1)
    tabx = similar(tab)
    tabxx = similar(tab)
    ind = Array{Int,5}(undef, 2, order + 1, _LOCAL_OFFSET_COUNT, length(deltas), nlevels + 1)

    for (ilevel_index, boxsize) in pairs(boxsizes)
        for (idelta, delta) in pairs(deltas)
            for offset in -_LOCAL_OFFSET_RADIUS:_LOCAL_OFFSET_RADIUS
                offset_index = offset + _LOCAL_OFFSET_RADIUS + 1
                modal, modalx, modalxx = _build_local_modal_tables(
                    target_nodes,
                    quad_nodes,
                    quad_weights,
                    basis_values,
                    boxsize,
                    offset,
                    delta,
                )

                tab_view = @view tab[:, :, offset_index, idelta, ilevel_index]
                tabx_view = @view tabx[:, :, offset_index, idelta, ilevel_index]
                tabxx_view = @view tabxx[:, :, offset_index, idelta, ilevel_index]

                mul!(tab_view, source_transform, modal)
                mul!(tabx_view, source_transform, modalx)
                mul!(tabxx_view, source_transform, modalxx)

                tab_pattern = _compute_sparse_pattern(tab_view, _LOCAL_SPARSE_TOL)
                tabx_pattern = _compute_sparse_pattern(tabx_view, _LOCAL_SPARSE_TOL)
                tabxx_pattern = _compute_sparse_pattern(tabxx_view, _LOCAL_SPARSE_TOL)
                @views ind[:, :, offset_index, idelta, ilevel_index] .= _merge_sparse_patterns(
                    tab_pattern,
                    tabx_pattern,
                    tabxx_pattern,
                )
            end
        end
    end

    return LocalTables(tab, tabx, tabxx, ind)
end
