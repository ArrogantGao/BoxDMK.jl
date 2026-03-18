function _check_tensor_product_inputs(out, mats, vals, n::Int, ndim::Int, nd::Int)
    n > 0 || throw(ArgumentError("tensor size n must be positive"))
    ndim > 0 || throw(ArgumentError("ndim must be positive"))
    nd > 0 || throw(ArgumentError("nd must be positive"))
    length(mats) == ndim || throw(ArgumentError("expected $ndim matrices, got $(length(mats))"))

    ngrid = n^ndim
    size(vals) == (nd, ngrid) || throw(DimensionMismatch("vals must have size ($nd, $ngrid)"))
    size(out) == (nd, ngrid) || throw(DimensionMismatch("out must have size ($nd, $ngrid)"))

    for (d, mat) in pairs(mats)
        size(mat) == (n, n) || throw(DimensionMismatch("matrix $d must have size ($n, $n)"))
    end

    return ngrid
end

function _tensor_product_apply_dim!(dest, src, mat, n::Int, ndim::Int, nd::Int, dim::Int)
    prefix = n^(dim - 1)
    suffix = n^(ndim - dim)
    mat_t = transpose(mat)

    for iv in 1:nd
        src_tensor = reshape(@view(src[iv, :]), prefix, n, suffix)
        dest_tensor = reshape(@view(dest[iv, :]), prefix, n, suffix)

        for isuffix in 1:suffix
            src_block = @view src_tensor[:, :, isuffix]
            dest_block = @view dest_tensor[:, :, isuffix]
            mul!(dest_block, src_block, mat_t)
        end
    end

    return dest
end

function tensor_product_apply!(out, mats::Tuple, vals, n, ndim, nd)
    n_int = Int(n)
    ndim_int = Int(ndim)
    nd_int = Int(nd)
    _check_tensor_product_inputs(out, mats, vals, n_int, ndim_int, nd_int)

    workspace = similar(out)
    src = vals

    for dim in 1:ndim_int
        dest = if dim == ndim_int
            src === out ? workspace : out
        else
            src === workspace ? out : workspace
        end

        _tensor_product_apply_dim!(dest, src, mats[dim], n_int, ndim_int, nd_int, dim)
        src = dest
    end

    src === out || copyto!(out, src)
    return out
end

function _child_half_nodes(parent_nodes::AbstractVector, side::Int)
    side == 0 && return 0.5 .* parent_nodes .- 0.5
    side == 1 && return 0.5 .* parent_nodes .+ 0.5
    throw(ArgumentError("child side must be 0 (left) or 1 (right)"))
end

function p2c_transform(basis::AbstractBasis, norder, ndim)
    norder_int = _check_basis_order(norder)
    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))

    parent_nodes, _ = nodes_and_weights(basis, norder_int)
    left_nodes = _child_half_nodes(parent_nodes, 0)
    right_nodes = _child_half_nodes(parent_nodes, 1)
    left_interp = interpolation_matrix(basis, parent_nodes, left_nodes)
    right_interp = interpolation_matrix(basis, parent_nodes, right_nodes)

    nchildren = 2^ndim_int
    p2c = Array{Float64,4}(undef, norder_int, norder_int, ndim_int, nchildren)

    for ichild in 1:nchildren
        child_bits = ichild - 1

        for d in 1:ndim_int
            bit = (child_bits >> (d - 1)) & 0x1
            p2c[:, :, d, ichild] .= bit == 0 ? left_interp : right_interp
        end
    end

    return p2c
end

function c2p_transform(basis::AbstractBasis, norder, ndim)
    p2c = p2c_transform(basis, norder, ndim)
    c2p = Array{Float64,4}(undef, size(p2c)...)

    for ichild in axes(p2c, 4), d in axes(p2c, 3)
        c2p[:, :, d, ichild] .= transpose(p2c[:, :, d, ichild])
    end

    return c2p
end
