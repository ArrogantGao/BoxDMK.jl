function _tensor_product_mats(identity_mat, diff_mat, ndim::Int, dims::Vararg{Int})
    return ntuple(i -> i in dims ? diff_mat : identity_mat, ndim)
end

function compute_gradient!(grad, pot, tree::BoxTree, basis::AbstractBasis)
    nd, np = _check_tree_data_box_array("pot", pot, tree)
    _check_gradient_array(grad, tree, nd, np)

    dmat = derivative_matrix(basis, tree.norder)
    imat = Matrix{eltype(dmat)}(I, tree.norder, tree.norder)
    grad_mats = ntuple(dim -> _tensor_product_mats(imat, dmat, tree.ndim, dim), tree.ndim)

    Threads.@threads for ibox in 1:nboxes(tree)
        scale = 2 / _level_boxsize(tree, ibox)
        src = @view pot[:, :, ibox]

        for dim in 1:tree.ndim
            dest = @view grad[:, dim, :, ibox]
            tensor_product_apply!(dest, grad_mats[dim], src, tree.norder, tree.ndim, nd)
            dest .*= scale
        end
    end

    return grad
end

function compute_hessian!(hess, pot, tree::BoxTree, basis::AbstractBasis)
    nd, np = _check_tree_data_box_array("pot", pot, tree)
    _check_hessian_array(hess, tree, nd, np)

    dmat = derivative_matrix(basis, tree.norder)
    d2mat = second_derivative_matrix(basis, tree.norder)
    imat = Matrix{promote_type(eltype(dmat), eltype(d2mat))}(I, tree.norder, tree.norder)

    pure_mats = ntuple(dim -> _tensor_product_mats(imat, d2mat, tree.ndim, dim), tree.ndim)
    mixed_pairs = [(di, dj, _hessian_component_index(di, dj, tree.ndim))
                   for di in 1:(tree.ndim - 1) for dj in (di + 1):tree.ndim]
    mixed_mats = [_tensor_product_mats(imat, dmat, tree.ndim, di, dj) for (di, dj, _) in mixed_pairs]

    Threads.@threads for ibox in 1:nboxes(tree)
        scale2 = (2 / _level_boxsize(tree, ibox))^2
        src = @view pot[:, :, ibox]

        for dim in 1:tree.ndim
            component = _hessian_component_index(dim, dim, tree.ndim)
            dest = @view hess[:, component, :, ibox]
            tensor_product_apply!(dest, pure_mats[dim], src, tree.norder, tree.ndim, nd)
            dest .*= scale2
        end

        for (idx, (_, _, component)) in pairs(mixed_pairs)
            dest = @view hess[:, component, :, ibox]
            tensor_product_apply!(dest, mixed_mats[idx], src, tree.norder, tree.ndim, nd)
            dest .*= scale2
        end
    end

    return hess
end
