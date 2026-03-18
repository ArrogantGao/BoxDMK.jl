function _boxsize(tree::BoxTree, ibox::Int)
    return tree.boxsize[tree.level[ibox] + 1]
end

function _boxes_touch(tree::BoxTree, ibox::Int, jbox::Int)
    size_i = Float64(_boxsize(tree, ibox))
    size_j = Float64(_boxsize(tree, jbox))
    tol = 128 * eps(Float64) * max(size_i, size_j, 1.0)

    for d in 1:tree.ndim
        if abs(Float64(tree.centers[d, ibox]) - Float64(tree.centers[d, jbox])) > (size_i + size_j) / 2 + tol
            return false
        end
    end

    return true
end

function _push_unique!(boxes::Vector{Int}, seen::BitVector, ibox::Int)
    if !seen[ibox]
        push!(boxes, ibox)
        seen[ibox] = true
    end
    return boxes
end

function _build_list1_for_leaf(tree::BoxTree, ibox::Int)
    near = Int[]
    seen = falses(nboxes(tree))

    for jbox in tree.colleagues[ibox]
        if isleaf(tree, jbox)
            _push_unique!(near, seen, jbox)
            continue
        end

        for kbox in @view tree.children[:, jbox]
            kbox == 0 && continue
            isleaf(tree, kbox) || continue
            _boxes_touch(tree, ibox, kbox) || continue
            _push_unique!(near, seen, kbox)
        end
    end

    parent = tree.parent[ibox]
    if parent != 0
        for jbox in tree.colleagues[parent]
            isleaf(tree, jbox) || continue
            _boxes_touch(tree, ibox, jbox) || continue
            _push_unique!(near, seen, jbox)
        end
    end

    return near
end

function _build_listpw_for_box(tree::BoxTree, ibox::Int, colleague_sets::Vector{Set{Int}})
    parent = tree.parent[ibox]
    parent == 0 && return Int[]

    far = Int[]
    seen = falses(nboxes(tree))

    for pbox in tree.colleagues[parent]
        for jbox in @view tree.children[:, pbox]
            jbox == 0 && continue
            jbox == ibox && continue
            tree.level[jbox] == tree.level[ibox] || continue
            jbox in colleague_sets[ibox] && continue
            _push_unique!(far, seen, jbox)
        end
    end

    return far
end

function build_interaction_lists(tree::BoxTree)
    nboxes_total = nboxes(tree)
    list1 = [Int[] for _ in 1:nboxes_total]
    colleague_sets = [Set(colleagues) for colleagues in tree.colleagues]

    for ibox in 1:nboxes_total
        isleaf(tree, ibox) || continue
        list1[ibox] = _build_list1_for_leaf(tree, ibox)
    end

    listpw = [_build_listpw_for_box(tree, ibox, colleague_sets) for ibox in 1:nboxes_total]

    return InteractionLists(list1, listpw)
end
