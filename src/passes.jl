function _check_pass_inputs(data, tree::BoxTree, proxy::ProxyData, name::AbstractString)
    ndim = _infer_proxy_ndim(proxy)
    ndim == tree.ndim || throw(DimensionMismatch("tree.ndim ($(tree.ndim)) must match proxy ndim ($ndim)"))
    size(data, 1) == proxy.ncbox || throw(DimensionMismatch("$name must have size ($(proxy.ncbox), nd, nboxes)"))
    size(data, 3) == nboxes(tree) || throw(DimensionMismatch("$name must have size ($(proxy.ncbox), nd, $(nboxes(tree)))"))
    size(proxy.c2p_transmat) == (proxy.porder, proxy.porder, ndim, 2^ndim) ||
        throw(DimensionMismatch("proxy.c2p_transmat must have size ($(proxy.porder), $(proxy.porder), $ndim, $(2^ndim))"))
    size(proxy.p2c_transmat) == (proxy.porder, proxy.porder, ndim, 2^ndim) ||
        throw(DimensionMismatch("proxy.p2c_transmat must have size ($(proxy.porder), $(proxy.porder), $ndim, $(2^ndim))"))
    return ndim
end

_child_transfer_mats(transmat, ndim::Int, ic::Int) = ntuple(d -> view(transmat, :, :, d, ic), ndim)

function _child_transfer_array(transmat, ndim::Int, ic::Int)
    umat_nd = Array{Float64}(undef, size(transmat, 1), size(transmat, 2), ndim)
    for d in 1:ndim
        @views umat_nd[:, :, d] .= transmat[:, :, d, ic]
    end
    return umat_nd
end

function _use_fortran_pass_hotpath(data, ndim::Int)
    return _FORTRAN_HOTPATHS_AVAILABLE[] &&
           1 <= ndim <= 3 &&
           data isa StridedArray{Float64,3}
end

function upward_pass!(proxy_charges::Array{T,3}, tree::BoxTree, proxy::ProxyData) where {T}
    ndim = _check_pass_inputs(proxy_charges, tree, proxy, "proxy_charges")
    nd = size(proxy_charges, 2)
    nchildren = size(tree.children, 1)
    use_fortran_hotpath = _use_fortran_pass_hotpath(proxy_charges, ndim)

    if use_fortran_hotpath
        child_umats = [_child_transfer_array(proxy.c2p_transmat, ndim, ic) for ic in 1:nchildren]

        for level in (tree.nlevels - 1):-1:0
            level_boxes = boxes_at_level(tree, level)

            Threads.@threads for index in eachindex(level_boxes)
                ibox = level_boxes[index]
                isleaf(tree, ibox) && continue

                for ic in 1:nchildren
                    ichild = tree.children[ic, ibox]
                    ichild == 0 && continue
                    umat_nd = child_umats[ic]

                    for id in 1:nd
                        _f_tens_prod_trans!(
                            @view(proxy_charges[:, id, ibox]),
                            @view(proxy_charges[:, id, ichild]),
                            umat_nd,
                            ndim,
                            proxy.porder,
                            proxy.porder,
                            1,
                        )
                    end
                end
            end
        end

        return proxy_charges
    end

    nthreadslots = Threads.maxthreadid()
    src_boxes = [Matrix{T}(undef, nd, proxy.ncbox) for _ in 1:nthreadslots]
    works = [Matrix{T}(undef, nd, proxy.ncbox) for _ in 1:nthreadslots]
    tensor_workspaces = [_tensor_apply_workspace(T, nd, proxy.porder, ndim) for _ in 1:nthreadslots]

    for level in (tree.nlevels - 1):-1:0
        level_boxes = boxes_at_level(tree, level)

        Threads.@threads for index in eachindex(level_boxes)
            ibox = level_boxes[index]
            isleaf(tree, ibox) && continue

            tid = Threads.threadid()
            src_box = src_boxes[tid]
            work = works[tid]
            parent_box = @view proxy_charges[:, :, ibox]

            for ic in 1:nchildren
                ichild = tree.children[ic, ibox]
                ichild == 0 && continue

                @views src_box .= transpose(proxy_charges[:, :, ichild])
                tensor_product_apply!(work, _child_transfer_mats(proxy.c2p_transmat, ndim, ic), src_box, proxy.porder, ndim, nd, tensor_workspaces[tid])
                parent_box .+= transpose(work)
            end
        end
    end

    return proxy_charges
end

function downward_pass!(proxy_pot::Array{T,3}, tree::BoxTree, proxy::ProxyData) where {T}
    ndim = _check_pass_inputs(proxy_pot, tree, proxy, "proxy_pot")
    nd = size(proxy_pot, 2)
    nchildren = size(tree.children, 1)
    use_fortran_hotpath = _use_fortran_pass_hotpath(proxy_pot, ndim)

    if use_fortran_hotpath
        child_umats = [_child_transfer_array(proxy.p2c_transmat, ndim, ic) for ic in 1:nchildren]

        for level in 1:tree.nlevels
            parent_boxes = boxes_at_level(tree, level - 1)

            Threads.@threads for index in eachindex(parent_boxes)
                ibox = parent_boxes[index]
                isleaf(tree, ibox) && continue

                for ic in 1:nchildren
                    ichild = tree.children[ic, ibox]
                    ichild == 0 && continue
                    umat_nd = child_umats[ic]

                    for id in 1:nd
                        _f_tens_prod_trans!(
                            @view(proxy_pot[:, id, ichild]),
                            @view(proxy_pot[:, id, ibox]),
                            umat_nd,
                            ndim,
                            proxy.porder,
                            proxy.porder,
                            1,
                        )
                    end
                end
            end
        end

        return proxy_pot
    end

    nthreadslots = Threads.maxthreadid()
    src_boxes = [Matrix{T}(undef, nd, proxy.ncbox) for _ in 1:nthreadslots]
    works = [Matrix{T}(undef, nd, proxy.ncbox) for _ in 1:nthreadslots]
    tensor_workspaces = [_tensor_apply_workspace(T, nd, proxy.porder, ndim) for _ in 1:nthreadslots]

    for level in 1:tree.nlevels
        parent_boxes = boxes_at_level(tree, level - 1)

        Threads.@threads for index in eachindex(parent_boxes)
            ibox = parent_boxes[index]
            isleaf(tree, ibox) && continue

            tid = Threads.threadid()
            src_box = src_boxes[tid]
            work = works[tid]

            @views src_box .= transpose(proxy_pot[:, :, ibox])

            for ic in 1:nchildren
                ichild = tree.children[ic, ibox]
                ichild == 0 && continue

                tensor_product_apply!(work, _child_transfer_mats(proxy.p2c_transmat, ndim, ic), src_box, proxy.porder, ndim, nd, tensor_workspaces[tid])
                @views proxy_pot[:, :, ichild] .+= transpose(work)
            end
        end
    end

    return proxy_pot
end
