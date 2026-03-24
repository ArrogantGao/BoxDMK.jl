function _check_tree_data_box_array(name::AbstractString, arr, tree::BoxTree)
    nd = size(arr, 1)
    np = npbox(tree.norder, tree.ndim)
    size(arr, 2) == np || throw(DimensionMismatch("$name must have size (_, $np, _)"))
    size(arr, 3) == nboxes(tree) || throw(DimensionMismatch("$name must have size (_, _, $(nboxes(tree)))"))
    return nd, np
end

function _check_gradient_array(gvals, tree::BoxTree, nd::Int, np::Int)
    expected = (nd, tree.ndim, np, nboxes(tree))
    size(gvals) == expected || throw(DimensionMismatch("gvals must have size $expected"))
    return gvals
end

function _check_hessian_array(hvals, tree::BoxTree, nd::Int, np::Int)
    expected = (nd, nhess(tree.ndim), np, nboxes(tree))
    size(hvals) == expected || throw(DimensionMismatch("hvals must have size $expected"))
    return hvals
end

function _apply_matrix_along_dim!(dest, src, mat, tree::BoxTree, nd::Int, dim::Int)
    _tensor_product_apply_dim!(dest, src, mat, tree.norder, tree.ndim, nd, dim)
    return dest
end

function _level_boxsize(tree::BoxTree, ibox::Integer)
    return tree.boxsize[tree.level[ibox] + 1]
end

function _hessian_component_index(di::Int, dj::Int, ndim::Int)
    di == dj && return di

    index = ndim
    for i in 1:(ndim - 1)
        for j in (i + 1):ndim
            index += 1
            if i == di && j == dj
                return index
            end
        end
    end

    throw(ArgumentError("invalid Hessian component ($di, $dj) for ndim=$ndim"))
end

function _asymptotic_component(component)
    if component isa NamedTuple
        if :level in propertynames(component)
            return Int(component.level), component.delta, component.weight
        end
        return nothing, component.delta, component.weight
    end

    if component isa Tuple
        if length(component) == 2
            return nothing, component[1], component[2]
        elseif length(component) == 3
            return Int(component[1]), component[2], component[3]
        end
    end

    throw(ArgumentError("asymptotic component must be (delta, weight), (level, delta, weight), or a matching named tuple"))
end

function _eval_asymptotic_level!(pot, tree::BoxTree, fvals, flvals, fl2vals, delta, weight, level)
    factor = weight * (sqrt(pi * delta))^tree.ndim
    c2 = delta / 4
    c4 = delta^2 / 32

    for ibox in leaves(tree)
        level === nothing || tree.level[ibox] == level || continue

        @views pot[:, :, ibox] .+= factor .* (
            fvals[:, :, ibox] .+
            c2 .* flvals[:, :, ibox] .+
            c4 .* fl2vals[:, :, ibox]
        )
    end

    return pot
end

function compute_laplacian!(flvals, tree::BoxTree, fvals, basis::AbstractBasis)
    nd, np = _check_tree_data_box_array("fvals", fvals, tree)
    size(flvals) == (nd, np, nboxes(tree)) || throw(DimensionMismatch("flvals must have size $(size(fvals))"))

    fill!(flvals, zero(eltype(flvals)))

    d2mat = second_derivative_matrix(basis, tree.norder)
    work = similar(fvals, size(fvals, 1), size(fvals, 2))

    for ibox in leaves(tree)
        scale2 = (2 / _level_boxsize(tree, ibox))^2
        src = @view fvals[:, :, ibox]
        dest = @view flvals[:, :, ibox]

        for dim in 1:tree.ndim
            _apply_matrix_along_dim!(work, src, d2mat, tree, nd, dim)
            dest .+= scale2 .* work
        end
    end

    return flvals
end

function compute_bilaplacian!(fl2vals, tree::BoxTree, fvals, flvals, basis::AbstractBasis)
    _ = fvals
    return compute_laplacian!(fl2vals, tree, flvals, basis)
end

function compute_gradient_density!(gvals, tree::BoxTree, fvals, basis::AbstractBasis)
    nd, np = _check_tree_data_box_array("fvals", fvals, tree)
    _check_gradient_array(gvals, tree, nd, np)

    fill!(gvals, zero(eltype(gvals)))

    dmat = derivative_matrix(basis, tree.norder)
    work = similar(fvals, size(fvals, 1), size(fvals, 2))

    for ibox in leaves(tree)
        scale = 2 / _level_boxsize(tree, ibox)
        src = @view fvals[:, :, ibox]

        for dim in 1:tree.ndim
            _apply_matrix_along_dim!(work, src, dmat, tree, nd, dim)
            @views gvals[:, dim, :, ibox] .= scale .* work
        end
    end

    return gvals
end

function compute_hessian_density!(hvals, tree::BoxTree, fvals, basis::AbstractBasis)
    nd, np = _check_tree_data_box_array("fvals", fvals, tree)
    _check_hessian_array(hvals, tree, nd, np)

    fill!(hvals, zero(eltype(hvals)))

    dmat = derivative_matrix(basis, tree.norder)
    d2mat = second_derivative_matrix(basis, tree.norder)
    work1 = similar(fvals, size(fvals, 1), size(fvals, 2))
    work2 = similar(fvals, size(fvals, 1), size(fvals, 2))

    for ibox in leaves(tree)
        scale2 = (2 / _level_boxsize(tree, ibox))^2
        src = @view fvals[:, :, ibox]

        for dim in 1:tree.ndim
            _apply_matrix_along_dim!(work1, src, d2mat, tree, nd, dim)
            @views hvals[:, _hessian_component_index(dim, dim, tree.ndim), :, ibox] .= scale2 .* work1
        end

        for di in 1:(tree.ndim - 1)
            _apply_matrix_along_dim!(work1, src, dmat, tree, nd, di)

            for dj in (di + 1):tree.ndim
                _apply_matrix_along_dim!(work2, work1, dmat, tree, nd, dj)
                @views hvals[:, _hessian_component_index(di, dj, tree.ndim), :, ibox] .= scale2 .* work2
            end
        end
    end

    return hvals
end

function eval_asymptotic!(pot, tree::BoxTree, fvals, flvals, fl2vals, delta, weight)
    nd, np = _check_tree_data_box_array("fvals", fvals, tree)
    size(flvals) == (nd, np, nboxes(tree)) || throw(DimensionMismatch("flvals must have size $(size(fvals))"))
    size(fl2vals) == (nd, np, nboxes(tree)) || throw(DimensionMismatch("fl2vals must have size $(size(fvals))"))
    size(pot) == (nd, np, nboxes(tree)) || throw(DimensionMismatch("pot must have size $(size(fvals))"))

    return _eval_asymptotic_level!(pot, tree, fvals, flvals, fl2vals, delta, weight, nothing)
end

function apply_asymptotic!(pot, tree::BoxTree, fvals, flvals, fl2vals, asymptotic_deltas)
    for component in asymptotic_deltas
        level, delta, weight = _asymptotic_component(component)
        _eval_asymptotic_level!(pot, tree, fvals, flvals, fl2vals, delta, weight, level)
    end

    return pot
end
