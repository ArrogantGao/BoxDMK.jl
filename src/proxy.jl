function select_porder(eps::Float64)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    eps >= 0.8e-3 && return 16
    eps >= 0.8e-4 && return 22
    eps >= 0.8e-5 && return 26
    eps >= 0.8e-6 && return 30
    eps >= 0.8e-7 && return 36
    eps >= 0.8e-8 && return 42
    eps >= 0.8e-9 && return 46
    eps >= 0.8e-10 && return 50
    eps >= 0.8e-11 && return 56
    return 62
end

function _infer_proxy_ndim(proxy::ProxyData)
    ndim = 0
    ngrid = 1

    while ngrid < proxy.ncbox
        ngrid *= proxy.porder
        ndim += 1
    end

    ngrid == proxy.ncbox || throw(ArgumentError("proxy.ncbox must equal porder^ndim"))
    return ndim
end

function _tensor_product_apply_rect!(out, mat, vals, from_order::Int, to_order::Int, ndim::Int, nd::Int)
    ndim > 0 || throw(ArgumentError("ndim must be positive"))
    nd > 0 || throw(ArgumentError("nd must be positive"))
    size(mat) == (to_order, from_order) || throw(DimensionMismatch("matrix must have size ($to_order, $from_order)"))
    size(vals) == (nd, from_order^ndim) || throw(DimensionMismatch("vals must have size ($nd, $(from_order^ndim))"))
    size(out) == (nd, to_order^ndim) || throw(DimensionMismatch("out must have size ($nd, $(to_order^ndim))"))

    src = vals
    mat_t = transpose(mat)

    for dim in 1:ndim
        prefix = to_order^(dim - 1)
        suffix = from_order^(ndim - dim)
        ncols = to_order^dim * from_order^(ndim - dim)
        dest = dim == ndim ? out : Matrix{eltype(out)}(undef, nd, ncols)

        for idensity in 1:nd
            src_tensor = reshape(@view(src[idensity, :]), prefix, from_order, suffix)
            dest_tensor = reshape(@view(dest[idensity, :]), prefix, to_order, suffix)

            for isuffix in 1:suffix
                mul!(@view(dest_tensor[:, :, isuffix]), @view(src_tensor[:, :, isuffix]), mat_t)
            end
        end

        src = dest
    end

    return out
end

function build_proxy_data(basis::AbstractBasis, norder::Int, porder::Int, ndim::Int)
    norder_int = _check_basis_order(norder)
    porder_int = _check_basis_order(porder)
    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))

    proxy_basis = LegendreBasis()
    source_nodes, _ = nodes_and_weights(basis, norder_int)
    proxy_nodes, _ = nodes_and_weights(proxy_basis, porder_int)

    den2pc_mat = interpolation_matrix(basis, source_nodes, proxy_nodes)
    poteval_mat = interpolation_matrix(proxy_basis, proxy_nodes, source_nodes)

    return ProxyData(
        porder_int,
        porder_int^ndim_int,
        den2pc_mat,
        poteval_mat,
        p2c_transform(proxy_basis, porder_int, ndim_int),
        c2p_transform(proxy_basis, porder_int, ndim_int),
    )
end

function _density_to_proxy_impl!(charge, fvals, proxy::ProxyData)
    ndim = _infer_proxy_ndim(proxy)
    norder = size(proxy.den2pc_mat, 2)
    nd = size(fvals, 1)
    npbox_val = norder^ndim
    nboxes = size(fvals, 3)

    size(fvals, 2) == npbox_val || throw(DimensionMismatch("fvals must have size ($nd, $npbox_val, nboxes)"))
    size(charge) == (proxy.ncbox, nd, nboxes) || throw(DimensionMismatch("charge must have size ($(proxy.ncbox), $nd, $nboxes)"))

    work = Matrix{eltype(charge)}(undef, nd, proxy.ncbox)

    for ibox in 1:nboxes
        _tensor_product_apply_rect!(work, proxy.den2pc_mat, @view(fvals[:, :, ibox]), norder, proxy.porder, ndim, nd)
        @views charge[:, :, ibox] .= transpose(work)
    end

    return charge
end

function density_to_proxy!(charge::AbstractMatrix, fvals::AbstractMatrix, proxy::ProxyData)
    _density_to_proxy_impl!(
        reshape(charge, size(charge, 1), size(charge, 2), 1),
        reshape(fvals, size(fvals, 1), size(fvals, 2), 1),
        proxy,
    )
    return charge
end

function density_to_proxy!(charge::AbstractArray{<:Real,3}, fvals::AbstractArray{<:Real,3}, proxy::ProxyData)
    return _density_to_proxy_impl!(charge, fvals, proxy)
end

function _proxy_to_potential_impl!(pot, proxy_pot, proxy::ProxyData)
    ndim = _infer_proxy_ndim(proxy)
    norder = size(proxy.poteval_mat, 1)
    npbox_val = norder^ndim
    nd = size(pot, 1)
    nboxes = size(pot, 3)

    size(pot, 2) == npbox_val || throw(DimensionMismatch("pot must have size ($nd, $npbox_val, nboxes)"))
    size(proxy_pot) == (proxy.ncbox, nd, nboxes) || throw(DimensionMismatch("proxy_pot must have size ($(proxy.ncbox), $nd, $nboxes)"))

    src_box = Matrix{eltype(pot)}(undef, nd, proxy.ncbox)

    for ibox in 1:nboxes
        @views src_box .= transpose(proxy_pot[:, :, ibox])
        _tensor_product_apply_rect!(@view(pot[:, :, ibox]), proxy.poteval_mat, src_box, proxy.porder, norder, ndim, nd)
    end

    return pot
end

function proxy_to_potential!(pot::AbstractMatrix, proxy_pot::AbstractMatrix, proxy::ProxyData)
    _proxy_to_potential_impl!(
        reshape(pot, size(pot, 1), size(pot, 2), 1),
        reshape(proxy_pot, size(proxy_pot, 1), size(proxy_pot, 2), 1),
        proxy,
    )
    return pot
end

function proxy_to_potential!(pot::AbstractArray{<:Real,3}, proxy_pot::AbstractArray{<:Real,3}, proxy::ProxyData)
    return _proxy_to_potential_impl!(pot, proxy_pot, proxy)
end
