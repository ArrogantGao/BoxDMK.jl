const _MAX_TREE_LEVELS = 200

mutable struct _TreeBoxState{T<:Real}
    center::Vector{T}
    size::T
    level::Int
    parent::Int
    children::Vector{Int}
    values::Matrix{T}
end

function _reference_grid(nodes::AbstractVector{T}, ndim::Int) where {T<:Real}
    np = length(nodes)^ndim
    grid = Matrix{T}(undef, ndim, np)
    idx = 1

    for point in Iterators.product(ntuple(_ -> nodes, ndim)...)
        @inbounds for d in 1:ndim
            grid[d, idx] = point[d]
        end
        idx += 1
    end

    return grid
end

function _reference_weights(weights::AbstractVector{T}, ndim::Int) where {T<:Real}
    np = length(weights)^ndim
    tensor_weights = Vector{Float64}(undef, np)
    idx = 1

    for point in Iterators.product(ntuple(_ -> weights, ndim)...)
        weight = 1.0
        @inbounds for d in 1:ndim
            weight *= Float64(point[d])
        end
        tensor_weights[idx] = weight
        idx += 1
    end

    return tensor_weights
end

function _sample_box(
    f,
    center::AbstractVector{T},
    boxsize::T,
    grid::AbstractMatrix{T},
    nd::Int,
    coord_shift::Real = 0.0,
) where {T<:Real}
    np = size(grid, 2)
    ndim = length(center)
    values = Matrix{Float64}(undef, nd, np)
    point = Vector{Float64}(undef, ndim)
    halfsize = Float64(boxsize) / 2
    shift = Float64(coord_shift)

    for ipoint in 1:np
        @inbounds for d in 1:ndim
            point[d] = Float64(center[d]) + halfsize * Float64(grid[d, ipoint]) + shift
        end

        raw = f(point)
        length(raw) == nd || throw(DimensionMismatch("f(x) must return a vector of length $nd"))

        @inbounds for idim in 1:nd
            values[idim, ipoint] = Float64(raw[idim])
        end
    end

    return values
end

function _child_center(center::AbstractVector{T}, boxsize::T, child::Int, ndim::Int) where {T<:Real}
    child_center = Vector{T}(undef, ndim)
    shift = boxsize / 4
    bits = child - 1

    @inbounds for d in 1:ndim
        sign = ((bits >> (d - 1)) & 0x1) == 0 ? -one(T) : one(T)
        child_center[d] = center[d] + sign * shift
    end

    return child_center
end

function _maxabs(values)
    maxval = zero(Float64)
    for value in values
        absvalue = abs(Float64(value))
        if absvalue > maxval
            maxval = absvalue
        end
    end
    return maxval
end

function _kernel_requires_refinement(kernel::AbstractKernel, boxsize::Real)
    return false
end

function _kernel_requires_refinement(kernel::YukawaKernel, boxsize::Real)
    return kernel.beta * boxsize > 5
end

function _modal_tail_mask(ndim::Int, norder::Int)
    npols = norder^ndim
    rmask = zeros(Float64, npols)
    morder = ndim == 1 ? norder - 1 : norder
    idx = 1
    rsum = 0.0

    for multi_index in Iterators.product(ntuple(_ -> 0:(norder - 1), ndim)...)
        degree_sum = 0
        @inbounds for d in 1:ndim
            degree_sum += (multi_index[d] + 1)^2
        end

        if degree_sum >= morder^2
            rmask[idx] = 1.0
            rsum += 1.0
        end

        idx += 1
    end

    return rmask, sqrt(rsum)
end

function _modal_tail_error(
    values,
    modal_transforms::Tuple,
    rmask::AbstractVector,
    rsum::Real,
    boxsize::Real,
    eta::Real,
    norder::Int,
    ndim::Int,
    nd::Int,
    coeffs,
    workspace,
)
    tensor_product_apply!(coeffs, modal_transforms, values, norder, ndim, nd, workspace)

    error = 0.0
    for idim in 1:nd
        tail_sq = 0.0
        @inbounds for index in eachindex(rmask)
            tail_sq += abs2(coeffs[idim, index]) * Float64(rmask[index])
        end
        error = max(error, sqrt(tail_sq))
    end

    return error * (Float64(boxsize) / 2)^Float64(eta) / Float64(rsum)
end

function _global_l2_scale(boxes, quadrature_weights::AbstractVector{<:Real}, ndim::Int)
    total = 0.0
    cell_count = 2^ndim

    for box in boxes
        all(iszero, box.children) || continue

        scale = Float64(box.size)^ndim / cell_count
        nd = size(box.values, 1)

        for ipoint in eachindex(quadrature_weights)
            weighted_scale = Float64(quadrature_weights[ipoint]) * scale
            @inbounds for idim in 1:nd
                total += abs2(box.values[idim, ipoint]) * weighted_scale
            end
        end
    end

    return sqrt(total)
end

function _box_l2_scale(values, quadrature_weights::AbstractVector{<:Real}, scale::Real)
    total = 0.0
    nd = size(values, 1)

    for ipoint in eachindex(quadrature_weights)
        weighted_scale = Float64(quadrature_weights[ipoint]) * Float64(scale)
        @inbounds for idim in 1:nd
            total += abs2(values[idim, ipoint]) * weighted_scale
        end
    end

    return sqrt(total)
end

function _split_box!(boxes, ibox::Int, child_values::Vector{<:AbstractMatrix})
    box = boxes[ibox]
    ndim = length(box.center)
    nchildren = length(box.children)
    child_size = box.size / 2

    for child in 1:nchildren
        center = _child_center(box.center, box.size, child, ndim)
        push!(boxes, _TreeBoxState(center, child_size, box.level + 1, ibox, zeros(Int, nchildren), copy(child_values[child])))
        box.children[child] = length(boxes)
    end

    return box.children
end

function _touches(box_a::_TreeBoxState, box_b::_TreeBoxState, tol::Real)
    for d in eachindex(box_a.center)
        if abs(box_a.center[d] - box_b.center[d]) > (box_a.size + box_b.size) / 2 + tol
            return false
        end
    end
    return true
end

function _enforce_level_restriction!(boxes, f, grid, nd::Int, maxlevels::Int, coord_shift::Real = 0.0)
    if isempty(boxes)
        return boxes
    end

    tol = 64 * eps(Float64) * max(boxes[1].size, 1.0)

    while true
        leaf_indices = [ibox for ibox in eachindex(boxes) if all(iszero, boxes[ibox].children)]
        to_refine = Int[]
        max_leaf_level = maximum(boxes[ibox].level for ibox in leaf_indices)

        for ibox in leaf_indices
            boxes[ibox].level < max_leaf_level - 3 || continue
            boxes[ibox].level < maxlevels && push!(to_refine, ibox)
        end

        for ibox in leaf_indices
            box_i = boxes[ibox]

            for jbox in eachindex(boxes)
                ibox == jbox && continue
                box_j = boxes[jbox]

                _touches(box_i, box_j, tol) || continue

                level_diff = box_i.level - box_j.level
                abs(level_diff) <= 1 && continue

                coarse = level_diff < 0 ? ibox : jbox
                all(iszero, boxes[coarse].children) || continue
                boxes[coarse].level < maxlevels && push!(to_refine, coarse)
            end
        end

        sort!(unique!(to_refine))
        isempty(to_refine) && break

        for ibox in to_refine
            all(iszero, boxes[ibox].children) || continue

            child_values = Vector{Matrix{Float64}}(undef, length(boxes[ibox].children))
            for child in eachindex(child_values)
                center = _child_center(boxes[ibox].center, boxes[ibox].size, child, length(boxes[ibox].center))
                child_values[child] = _sample_box(f, center, boxes[ibox].size / 2, grid, nd, coord_shift)
            end

            _split_box!(boxes, ibox, child_values)
        end
    end

    return boxes
end

function _build_colleagues(boxes)
    nboxes_total = length(boxes)
    colleagues = [Int[] for _ in 1:nboxes_total]
    tol = 64 * eps(Float64) * maximum(box -> Float64(box.size), boxes)

    for ibox in 1:nboxes_total
        box_i = boxes[ibox]

        for jbox in 1:nboxes_total
            boxes[jbox].level == box_i.level || continue
            _touches(box_i, boxes[jbox], tol) || continue
            push!(colleagues[ibox], jbox)
        end
    end

    return colleagues
end

function _pack_tree(boxes, basis::AbstractBasis, ndim::Int, norder::Int, boxlen::Real, center_shift::Real = 0.0)
    nboxes_total = length(boxes)
    np = size(boxes[1].values, 2)
    nd = size(boxes[1].values, 1)
    nlevels = maximum(box.level for box in boxes)

    centers = Matrix{Float64}(undef, ndim, nboxes_total)
    parent = Vector{Int}(undef, nboxes_total)
    children = Matrix{Int}(undef, 2^ndim, nboxes_total)
    level = Vector{Int}(undef, nboxes_total)
    fvals = Array{Float64,3}(undef, nd, np, nboxes_total)

    for ibox in 1:nboxes_total
        box = boxes[ibox]
        @inbounds for d in 1:ndim
            centers[d, ibox] = Float64(box.center[d]) + Float64(center_shift)
        end
        parent[ibox] = box.parent
        children[:, ibox] = box.children
        level[ibox] = box.level
        fvals[:, :, ibox] = box.values
    end

    boxsize = [Float64(boxlen) / (2.0^ilevel) for ilevel in 0:nlevels]
    colleagues = _build_colleagues(boxes)

    tree = BoxTree(
        ndim,
        nlevels,
        centers,
        boxsize,
        parent,
        children,
        colleagues,
        level,
        basis,
        norder,
    )

    return tree, fvals
end

function build_tree(
    f,
    kernel::AbstractKernel,
    basis::AbstractBasis;
    ndim::Integer = 3,
    norder::Integer = 6,
    eps::Real = 1e-6,
    boxlen::Real = 1.0,
    nd::Integer = 1,
    dpars = nothing,
    eta::Real = 1.0,
)
    _ = dpars
    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))
    nd_int = Int(nd)
    nd_int > 0 || throw(ArgumentError("nd must be positive"))
    norder_int = _check_basis_order(norder)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    boxlen > 0 || throw(ArgumentError("boxlen must be positive"))
    eps_value = Float64(eps)

    nodes, weights = nodes_and_weights(basis, norder_int)
    reference_grid = _reference_grid(Float64.(nodes), ndim_int)
    quadrature_weights = _reference_weights(Float64.(weights), ndim_int)
    modal_1d = forward_transform(basis, norder_int)
    modal_transforms = ntuple(_ -> modal_1d, ndim_int)
    rmask, rsum = _modal_tail_mask(ndim_int, norder_int)
    coeffs = Matrix{Float64}(undef, nd_int, norder_int^ndim_int)
    coeff_workspace = _tensor_apply_workspace(Float64, nd_int, norder_int, ndim_int)
    root_center = zeros(Float64, ndim_int)
    coord_shift = Float64(boxlen) / 2
    root_values = _sample_box(f, root_center, Float64(boxlen), reference_grid, nd_int, coord_shift)
    root_box = _TreeBoxState(root_center, Float64(boxlen), 0, 0, zeros(Int, 2^ndim_int), root_values)
    boxes = [root_box]
    level_first = [1]
    level_last = [1]
    root_rint_scale = Float64(boxlen)^2 / (2^ndim_int)
    current_rint = max(_box_l2_scale(root_values, quadrature_weights, root_rint_scale), Base.eps(Float64))
    root_scale = sqrt(inv(Float64(boxlen)^ndim_int))
    ilevel = 0

    while ilevel < _MAX_TREE_LEVELS
        ifirstbox = level_first[ilevel + 1]
        ilastbox = level_last[ilevel + 1]
        nbloc = ilastbox - ifirstbox + 1
        refine_flags = falses(nbloc)
        needs_any_refinement = false
        level_threshold = eps_value * root_scale * current_rint

        for offset in 1:nbloc
            ibox = ifirstbox + offset - 1
            box = boxes[ibox]

            needs_refinement = _kernel_requires_refinement(kernel, box.size)
            if !needs_refinement
                error = _modal_tail_error(
                    box.values,
                    modal_transforms,
                    rmask,
                    rsum,
                    box.size,
                    eta,
                    norder_int,
                    ndim_int,
                    nd_int,
                    coeffs,
                    coeff_workspace,
                )
                needs_refinement = error > level_threshold
            end

            refine_flags[offset] = needs_refinement
            needs_any_refinement |= needs_refinement
        end

        needs_any_refinement || break

        new_level_first = length(boxes) + 1
        for offset in 1:nbloc
            refine_flags[offset] || continue

            ibox = ifirstbox + offset - 1
            box = boxes[ibox]
            child_values = Vector{Matrix{Float64}}(undef, length(box.children))

            for child in eachindex(child_values)
                center = _child_center(box.center, box.size, child, ndim_int)
                child_values[child] = _sample_box(f, center, box.size / 2, reference_grid, nd_int, coord_shift)
            end

            _split_box!(boxes, ibox, child_values)
        end

        push!(level_first, new_level_first)
        push!(level_last, length(boxes))
        current_rint = max(_global_l2_scale(boxes, quadrature_weights, ndim_int), Base.eps(Float64))
        ilevel += 1
    end

    _enforce_level_restriction!(boxes, f, reference_grid, nd_int, _MAX_TREE_LEVELS, coord_shift)
    return _pack_tree(boxes, basis, ndim_int, norder_int, boxlen, coord_shift)
end
