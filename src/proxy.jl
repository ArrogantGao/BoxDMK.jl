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

function _rect_tensor_scratch_cols(from_order::Int, to_order::Int, ndim::Int)
    ndim == 1 && return max(from_order, to_order)
    return maximum(to_order^dim * from_order^(ndim - dim) for dim in 1:(ndim - 1))
end

function _rect_tensor_middle_rows(from_order::Int, to_order::Int, ndim::Int)
    ndim <= 2 && return 0
    return maximum(to_order^(dim - 1) * from_order^(ndim - dim) for dim in 2:(ndim - 1))
end

function _rect_tensor_apply_workspace(::Type{T}, nd::Int, from_order::Int, to_order::Int, ndim::Int) where {T}
    scratch_cols = _rect_tensor_scratch_cols(from_order, to_order, ndim)
    middle_rows = _rect_tensor_middle_rows(from_order, to_order, ndim)
    tensor_in_len = ndim == 3 ? from_order^3 : 0
    tensor_out_len = ndim == 3 ? to_order^3 : 0
    ff_rows = ndim == 3 ? from_order^2 : 0
    fft_cols = ndim == 3 ? to_order * from_order : 0
    ff2_rows = ndim == 3 ? to_order^2 : 0

    return (
        scratch_a = Matrix{T}(undef, nd, scratch_cols),
        scratch_b = Matrix{T}(undef, nd, scratch_cols),
        tmp_perm = Matrix{T}(undef, middle_rows, from_order),
        result_2d = Matrix{T}(undef, middle_rows, to_order),
        packed_src = Vector{T}(undef, tensor_in_len),
        packed_dest = Vector{T}(undef, tensor_out_len),
        ff = Matrix{T}(undef, ff_rows, to_order),
        fft = Matrix{T}(undef, from_order, fft_cols),
        ff2 = Matrix{T}(undef, ff2_rows, from_order),
    )
end

function _check_rect_tensor_3d_workspace(workspace, from_order::Int, to_order::Int)
    for name in (:packed_src, :packed_dest, :ff, :fft, :ff2)
        hasproperty(workspace, name) || throw(ArgumentError("3D tensor workspace must provide $name"))
    end

    length(workspace.packed_src) >= from_order^3 ||
        throw(DimensionMismatch("workspace packed_src must have at least $(from_order^3) entries"))
    length(workspace.packed_dest) >= to_order^3 ||
        throw(DimensionMismatch("workspace packed_dest must have at least $(to_order^3) entries"))
    size(workspace.ff, 1) >= from_order^2 ||
        throw(DimensionMismatch("workspace ff must have at least $(from_order^2) rows"))
    size(workspace.ff, 2) >= to_order ||
        throw(DimensionMismatch("workspace ff must have at least $to_order columns"))
    size(workspace.fft, 1) >= from_order ||
        throw(DimensionMismatch("workspace fft must have at least $from_order rows"))
    size(workspace.fft, 2) >= to_order * from_order ||
        throw(DimensionMismatch("workspace fft must have at least $(to_order * from_order) columns"))
    size(workspace.ff2, 1) >= to_order^2 ||
        throw(DimensionMismatch("workspace ff2 must have at least $(to_order^2) rows"))
    size(workspace.ff2, 2) >= from_order ||
        throw(DimensionMismatch("workspace ff2 must have at least $from_order columns"))

    return workspace
end

function _tensor_product_apply_rect_3d!(out, mat, vals, from_order::Int, to_order::Int, nd::Int)
    workspace = _rect_tensor_apply_workspace(eltype(out), nd, from_order, to_order, 3)
    return _tensor_product_apply_rect_3d!(out, mat, vals, from_order, to_order, nd, workspace)
end

function _tensor_product_apply_rect_3d!(out, mat, vals, from_order::Int, to_order::Int, nd::Int, workspace)
    n = from_order
    m = to_order
    size(mat) == (m, n) || throw(DimensionMismatch("matrix must have size ($m, $n)"))
    size(vals) == (nd, n^3) || throw(DimensionMismatch("vals must have size ($nd, $(n^3))"))
    size(out) == (nd, m^3) || throw(DimensionMismatch("out must have size ($nd, $(m^3))"))

    _check_rect_tensor_3d_workspace(workspace, n, m)

    packed_src = @view(workspace.packed_src[1:(n^3)])
    packed_dest = @view(workspace.packed_dest[1:(m^3)])
    ff = @view(workspace.ff[1:(n^2), 1:m])
    fft = @view(workspace.fft[1:n, 1:(m * n)])
    ff2 = @view(workspace.ff2[1:(m^2), 1:n])
    mat_t = transpose(mat)

    for iv in 1:nd
        copyto!(packed_src, @view(vals[iv, :]))

        src_2d = reshape(packed_src, n * n, n)
        mul!(ff, src_2d, mat_t)

        ff_3d = reshape(ff, n, n, m)
        fft_3d = reshape(fft, n, m, n)
        @inbounds for i1 in 1:n, j3 in 1:m, i2 in 1:n
            fft_3d[i2, j3, i1] = ff_3d[i1, i2, j3]
        end

        ff2_step = reshape(ff2, m, m * n)
        mul!(ff2_step, mat, fft)

        dest_2d = reshape(packed_dest, m, m * m)
        mul!(dest_2d, mat, transpose(ff2))
        copyto!(@view(out[iv, :]), packed_dest)
    end

    return out
end

function _tensor_product_apply_rect!(out, mat, vals, from_order::Int, to_order::Int, ndim::Int, nd::Int)
    workspace = _rect_tensor_apply_workspace(eltype(out), nd, from_order, to_order, ndim)
    return _tensor_product_apply_rect!(out, mat, vals, from_order, to_order, ndim, nd, workspace)
end

function _tensor_product_apply_rect!(out, mat, vals, from_order::Int, to_order::Int, ndim::Int, nd::Int, workspace)
    ndim > 0 || throw(ArgumentError("ndim must be positive"))
    nd > 0 || throw(ArgumentError("nd must be positive"))
    size(mat) == (to_order, from_order) || throw(DimensionMismatch("matrix must have size ($to_order, $from_order)"))
    size(vals) == (nd, from_order^ndim) || throw(DimensionMismatch("vals must have size ($nd, $(from_order^ndim))"))
    size(out) == (nd, to_order^ndim) || throw(DimensionMismatch("out must have size ($nd, $(to_order^ndim))"))

    ndim == 3 && return _tensor_product_apply_rect_3d!(out, mat, vals, from_order, to_order, nd, workspace)

    scratch_cols = _rect_tensor_scratch_cols(from_order, to_order, ndim)
    middle_rows = _rect_tensor_middle_rows(from_order, to_order, ndim)
    size(workspace.scratch_a, 1) == nd || throw(DimensionMismatch("workspace scratch_a must have $nd rows"))
    size(workspace.scratch_b, 1) == nd || throw(DimensionMismatch("workspace scratch_b must have $nd rows"))
    size(workspace.scratch_a, 2) >= scratch_cols || throw(DimensionMismatch("workspace scratch_a must have at least $scratch_cols columns"))
    size(workspace.scratch_b, 2) >= scratch_cols || throw(DimensionMismatch("workspace scratch_b must have at least $scratch_cols columns"))
    size(workspace.tmp_perm, 1) >= middle_rows || throw(DimensionMismatch("workspace tmp_perm must have at least $middle_rows rows"))
    size(workspace.tmp_perm, 2) >= from_order || throw(DimensionMismatch("workspace tmp_perm must have at least $from_order columns"))
    size(workspace.result_2d, 1) >= middle_rows || throw(DimensionMismatch("workspace result_2d must have at least $middle_rows rows"))
    size(workspace.result_2d, 2) >= to_order || throw(DimensionMismatch("workspace result_2d must have at least $to_order columns"))

    src = vals
    mat_t = transpose(mat)
    src_slot = 0

    for dim in 1:ndim
        prefix = to_order^(dim - 1)
        suffix = from_order^(ndim - dim)
        ncols = to_order^dim * from_order^(ndim - dim)
        dest, dest_slot = if dim == ndim && src !== out
            (out, 3)
        elseif src_slot == 1
            (@view(workspace.scratch_b[:, 1:ncols]), 2)
        else
            (@view(workspace.scratch_a[:, 1:ncols]), 1)
        end

        for idensity in 1:nd
            if prefix == 1
                # Special case: src is (from_order, suffix) in memory, apply mat along leading dim
                src_flat = reshape(@view(src[idensity, :]), from_order, suffix)
                dest_flat = reshape(@view(dest[idensity, :]), to_order, suffix)
                mul!(dest_flat, mat, src_flat)
            elseif suffix == 1
                src_2d = reshape(@view(src[idensity, :]), prefix, from_order)
                dest_2d = reshape(@view(dest[idensity, :]), prefix, to_order)
                mul!(dest_2d, src_2d, mat_t)
            else
                src_3d = reshape(@view(src[idensity, :]), prefix, from_order, suffix)
                rows = prefix * suffix
                tmp_2d = @view(workspace.tmp_perm[1:rows, 1:from_order])
                tmp_3d = reshape(tmp_2d, prefix, suffix, from_order)
                permutedims!(tmp_3d, src_3d, (1, 3, 2))
                result_2d = @view(workspace.result_2d[1:rows, 1:to_order])
                mul!(result_2d, tmp_2d, mat_t)
                result_3d = reshape(result_2d, prefix, suffix, to_order)
                dest_3d = reshape(@view(dest[idensity, :]), prefix, to_order, suffix)
                permutedims!(dest_3d, result_3d, (1, 3, 2))
            end
        end

        src = dest
        src_slot = dest_slot
    end

    src === out || copyto!(out, src)
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

function _use_fortran_proxy_hotpath(dest, src, ndim::Int)
    return _FORTRAN_HOTPATHS_AVAILABLE[] &&
           ndim == 3 &&
           dest isa StridedArray{Float64,3} &&
           src isa StridedArray{Float64,3}
end

function _density_to_proxy_impl!(charge, fvals, proxy::ProxyData)
    ndim = _infer_proxy_ndim(proxy)
    norder = size(proxy.den2pc_mat, 2)
    nd = size(fvals, 1)
    npbox_val = norder^ndim
    nboxes = size(fvals, 3)

    size(fvals, 2) == npbox_val || throw(DimensionMismatch("fvals must have size ($nd, $npbox_val, nboxes)"))
    size(charge) == (proxy.ncbox, nd, nboxes) || throw(DimensionMismatch("charge must have size ($(proxy.ncbox), $nd, $nboxes)"))

    if _use_fortran_proxy_hotpath(charge, fvals, ndim)
        for ibox in 1:nboxes
            _f_density2proxycharge!(
                @view(charge[:, :, ibox]),
                @view(fvals[:, :, ibox]),
                proxy.den2pc_mat,
                ndim,
                nd,
                norder,
                proxy.porder,
            )
        end
        return charge
    end

    work = Matrix{eltype(charge)}(undef, nd, proxy.ncbox)
    tensor_workspace = _rect_tensor_apply_workspace(eltype(charge), nd, norder, proxy.porder, ndim)

    for ibox in 1:nboxes
        _tensor_product_apply_rect!(work, proxy.den2pc_mat, @view(fvals[:, :, ibox]), norder, proxy.porder, ndim, nd, tensor_workspace)
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

    if _use_fortran_proxy_hotpath(pot, proxy_pot, ndim)
        work = Matrix{Float64}(undef, nd, npbox_val)

        for ibox in 1:nboxes
            fill!(work, 0.0)
            _f_proxypot2pot!(
                work,
                @view(proxy_pot[:, :, ibox]),
                proxy.poteval_mat,
                ndim,
                nd,
                proxy.porder,
                norder,
            )
            @views pot[:, :, ibox] .= work
        end

        return pot
    end

    src_box = Matrix{eltype(pot)}(undef, nd, proxy.ncbox)
    tensor_workspace = _rect_tensor_apply_workspace(eltype(pot), nd, proxy.porder, norder, ndim)

    for ibox in 1:nboxes
        @views src_box .= transpose(proxy_pot[:, :, ibox])
        _tensor_product_apply_rect!(@view(pot[:, :, ibox]), proxy.poteval_mat, src_box, proxy.porder, norder, ndim, nd, tensor_workspace)
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
