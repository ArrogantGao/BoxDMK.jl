const _MAX_TREE_LEVELS = 6

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

function _sample_box(f, center::AbstractVector{T}, boxsize::T, grid::AbstractMatrix{T}, nd::Int) where {T<:Real}
    np = size(grid, 2)
    ndim = length(center)
    values = Matrix{Float64}(undef, nd, np)
    point = Vector{Float64}(undef, ndim)
    halfsize = Float64(boxsize) / 2

    for ipoint in 1:np
        @inbounds for d in 1:ndim
            point[d] = Float64(center[d]) + halfsize * Float64(grid[d, ipoint])
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

function _refinement_error(parent_values, child_values, child_transforms, norder::Int, ndim::Int, nd::Int)
    predicted = similar(parent_values)
    error = zero(eltype(parent_values))

    for child in eachindex(child_values)
        tensor_product_apply!(predicted, child_transforms[child], parent_values, norder, ndim, nd)
        diff_sq = 0.0
        count = length(predicted)

        for index in eachindex(predicted)
            delta = child_values[child][index] - predicted[index]
            diff_sq += abs2(delta)
        end

        error = max(error, sqrt(diff_sq / count))
    end

    return error
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

function _enforce_level_restriction!(boxes, f, grid, nd::Int, maxlevels::Int)
    if isempty(boxes)
        return boxes
    end

    tol = 64 * eps(Float64) * max(boxes[1].size, 1.0)

    while true
        leaf_indices = [ibox for ibox in eachindex(boxes) if all(iszero, boxes[ibox].children)]
        to_refine = Int[]

        for i in 1:(length(leaf_indices) - 1)
            ibox = leaf_indices[i]
            box_i = boxes[ibox]

            for j in (i + 1):length(leaf_indices)
                jbox = leaf_indices[j]
                box_j = boxes[jbox]

                _touches(box_i, box_j, tol) || continue

                level_diff = box_i.level - box_j.level
                abs(level_diff) <= 1 && continue

                coarse = level_diff < 0 ? ibox : jbox
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
                child_values[child] = _sample_box(f, center, boxes[ibox].size / 2, grid, nd)
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

function _pack_tree(boxes, basis::AbstractBasis, ndim::Int, norder::Int, boxlen::Real)
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
        centers[:, ibox] = box.center
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

    nodes, _ = nodes_and_weights(basis, norder_int)
    reference_grid = _reference_grid(Float64.(nodes), ndim_int)
    p2c = p2c_transform(basis, norder_int, ndim_int)
    child_transforms = map(1:(2^ndim_int)) do child
        ntuple(d -> view(p2c, :, :, d, child), ndim_int)
    end
    root_center = fill(Float64(boxlen) / 2, ndim_int)
    root_values = _sample_box(f, root_center, Float64(boxlen), reference_grid, nd_int)
    global_scale = max(_maxabs(root_values), Base.eps(Float64))
    root_box = _TreeBoxState(root_center, Float64(boxlen), 0, 0, zeros(Int, 2^ndim_int), root_values)
    boxes = [root_box]
    pending = [1]

    while !isempty(pending)
        ibox = pop!(pending)
        box = boxes[ibox]
        box.level >= _MAX_TREE_LEVELS && continue

        child_values = Vector{Matrix{Float64}}(undef, length(box.children))
        for child in eachindex(child_values)
            center = _child_center(box.center, box.size, child, ndim_int)
            child_values[child] = _sample_box(f, center, box.size / 2, reference_grid, nd_int)
        end

        needs_refinement = _kernel_requires_refinement(kernel, box.size)
        if !needs_refinement
            error = _refinement_error(box.values, child_values, child_transforms, norder_int, ndim_int, nd_int)
            needs_refinement = error * (box.size / 2)^eta > eps_value * global_scale
        end

        if needs_refinement
            children = _split_box!(boxes, ibox, child_values)
            append!(pending, reverse(children))
        end
    end

    _enforce_level_restriction!(boxes, f, reference_grid, nd_int, _MAX_TREE_LEVELS)
    return _pack_tree(boxes, basis, ndim_int, norder_int, boxlen)
end
