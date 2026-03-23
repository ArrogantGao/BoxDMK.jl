@inline function _local_table_index(offset::Int)
    abs(offset) <= _LOCAL_OFFSET_RADIUS || throw(ArgumentError("local offset $offset is outside supported range -$_LOCAL_OFFSET_RADIUS:$_LOCAL_OFFSET_RADIUS"))
    return offset + _LOCAL_OFFSET_RADIUS + 1
end

function _local_offset_indices(tree::BoxTree, ibox::Int, jbox::Int)
    target_boxsize = Float64(_boxsize(tree, ibox))
    source_boxsize = Float64(_boxsize(tree, jbox))
    tol = 128 * eps(Float64) * max(target_boxsize, source_boxsize, 1.0)

    abs(target_boxsize - source_boxsize) <= tol || throw(ArgumentError(
        "apply_local! currently supports same-level local interactions only; got box sizes $target_boxsize and $source_boxsize",
    ))

    offset_tol = 256 * eps(Float64)
    return ntuple(d -> begin
        offset_value = (Float64(tree.centers[d, ibox]) - Float64(tree.centers[d, jbox])) / target_boxsize
        offset = round(Int, offset_value)
        abs(offset_value - offset) <= offset_tol || throw(ArgumentError(
            "non-integer local offset $offset_value encountered between boxes $ibox and $jbox along dimension $d",
        ))
        _local_table_index(offset)
    end, tree.ndim)
end

@inline function _local_active_range(pattern::AbstractMatrix{Int}, n::Int)
    return pattern[1, n + 1], pattern[2, n + 1]
end

function _apply_local_sparse_3d!(
    pot_box,
    src_box,
    table_x,
    pattern_x,
    table_y,
    pattern_y,
    table_z,
    pattern_z,
    weight,
    ff,
    ff2,
    n::Int,
)
    xfirst, xlast = _local_active_range(pattern_x, n)
    yfirst, ylast = _local_active_range(pattern_y, n)
    zfirst, zlast = _local_active_range(pattern_z, n)

    (xfirst == 0 || yfirst == 0 || zfirst == 0) && return pot_box

    zero_work = zero(eltype(ff))
    weighted_scale = eltype(ff)(weight)

    @inbounds for id in axes(src_box, 1)
        src = reshape(@view(src_box[id, :]), n, n, n)
        dest = reshape(@view(pot_box[id, :]), n, n, n)

        for j3 in 1:n
            for j2 in 1:n
                for k1 in xfirst:xlast
                    row_first = pattern_x[1, k1]
                    row_last = pattern_x[2, k1]
                    accum = zero_work
                    if row_first != 0
                        for j1 in row_first:row_last
                            accum += table_x[j1, k1] * src[j1, j2, j3]
                        end
                    end
                    ff[k1, j2, j3] = accum
                end
            end
        end

        for j3 in 1:n
            for k2 in yfirst:ylast
                row_first = pattern_y[1, k2]
                row_last = pattern_y[2, k2]
                for k1 in xfirst:xlast
                    accum = zero_work
                    if row_first != 0
                        for j2 in row_first:row_last
                            accum += table_y[j2, k2] * ff[k1, j2, j3]
                        end
                    end
                    ff2[k1, k2, j3] = accum
                end
            end
        end

        for k3 in zfirst:zlast
            row_first = pattern_z[1, k3]
            row_last = pattern_z[2, k3]
            for k2 in yfirst:ylast
                for k1 in xfirst:xlast
                    accum = zero_work
                    if row_first != 0
                        for j3 in row_first:row_last
                            accum += table_z[j3, k3] * ff2[k1, k2, j3]
                        end
                    end
                    dest[k1, k2, k3] += weighted_scale * accum
                end
            end
        end
    end

    return pot_box
end

function apply_local!(pot, tree::BoxTree, fvals, tables::LocalTables, lists::InteractionLists, sog_deltas, sog_weights)
    tree.ndim == 3 || throw(ArgumentError("apply_local! currently expects ndim == 3"))
    size(pot) == size(fvals) || throw(DimensionMismatch("pot and fvals must have the same size"))
    size(pot, 3) == nboxes(tree) || throw(DimensionMismatch("pot/fvals third dimension must match tree boxes"))
    size(pot, 2) == tree.norder^tree.ndim || throw(DimensionMismatch("pot/fvals second dimension must equal norder^ndim"))
    length(lists.list1) == nboxes(tree) || throw(DimensionMismatch("lists.list1 must have one entry per box"))
    length(sog_deltas) == length(sog_weights) || throw(DimensionMismatch("sog_deltas and sog_weights must have the same length"))
    length(sog_deltas) == size(tables.tab, 4) || throw(DimensionMismatch("local table delta count does not match supplied SOG components"))
    size(tables.tab, 1) == tree.norder || throw(DimensionMismatch("local table order does not match tree.norder"))
    size(tables.tab, 2) == tree.norder || throw(DimensionMismatch("local table order does not match tree.norder"))

    n = tree.norder
    level_count = size(tables.tab, 5)
    work_type = promote_type(eltype(pot), eltype(fvals), eltype(tables.tab), eltype(sog_weights))
    leaf_boxes = collect(leaves(tree))
    use_fortran_hotpath = _FORTRAN_HOTPATHS_AVAILABLE[] &&
                          pot isa StridedArray{Float64,3} &&
                          fvals isa StridedArray{Float64,3} &&
                          eltype(tables.tab) === Float64
    ind_cint = use_fortran_hotpath ? Array{Cint}(tables.ind) : nothing

    Threads.@threads for ileaf in eachindex(leaf_boxes)
        ibox = leaf_boxes[ileaf]
        level_index = tree.level[ibox] + 1
        level_index <= level_count || throw(BoundsError(tables.tab, (:, :, :, :, level_index)))
        pot_box = @view pot[:, :, ibox]
        ff = use_fortran_hotpath ? nothing : Array{work_type,3}(undef, n, n, n)
        ff2 = use_fortran_hotpath ? nothing : similar(ff)

        for jbox in lists.list1[ibox]
            offset_indices = _local_offset_indices(tree, ibox, jbox)
            src_box = @view fvals[:, :, jbox]
            ixyz = use_fortran_hotpath ? Cint[offset_index - (_LOCAL_OFFSET_RADIUS + 1) for offset_index in offset_indices] : nothing

            for idelta in eachindex(sog_deltas, sog_weights)
                weight = sog_weights[idelta]
                iszero(weight) && continue

                if use_fortran_hotpath
                    tab_slice = @view tables.tab[:, :, :, idelta, level_index]
                    ind_slice = @view ind_cint[:, :, :, idelta, level_index]
                    _f_tens_prod_to_potloc!(
                        pot_box,
                        src_box,
                        weight,
                        tab_slice,
                        ind_slice,
                        ixyz,
                        tree.ndim,
                        size(src_box, 1),
                        n,
                        _LOCAL_OFFSET_RADIUS,
                    )
                else
                    pattern_x = @view tables.ind[:, :, offset_indices[1], idelta, level_index]
                    pattern_y = @view tables.ind[:, :, offset_indices[2], idelta, level_index]
                    pattern_z = @view tables.ind[:, :, offset_indices[3], idelta, level_index]

                    _apply_local_sparse_3d!(
                        pot_box,
                        src_box,
                        @view(tables.tab[:, :, offset_indices[1], idelta, level_index]),
                        pattern_x,
                        @view(tables.tab[:, :, offset_indices[2], idelta, level_index]),
                        pattern_y,
                        @view(tables.tab[:, :, offset_indices[3], idelta, level_index]),
                        pattern_z,
                        weight,
                        ff,
                        ff2,
                        n,
                    )
                end
            end
        end
    end

    return pot
end
