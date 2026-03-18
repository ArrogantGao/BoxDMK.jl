function _check_solver_inputs(tree::BoxTree, fvals)
    nd = size(fvals, 1)
    np = npbox(tree.norder, tree.ndim)
    size(fvals, 2) == np || throw(DimensionMismatch("fvals must have size (_, $np, _)"))
    size(fvals, 3) == nboxes(tree) || throw(DimensionMismatch("fvals must have size (_, _, $(nboxes(tree)))"))
    return nd, np
end

function _density_to_proxy_leaves!(proxy_charges, tree::BoxTree, fvals, proxy::ProxyData)
    for ibox in leaves(tree)
        density_to_proxy!(@view(proxy_charges[:, :, ibox]), @view(fvals[:, :, ibox]), proxy)
    end

    return proxy_charges
end

function _apply_taylor_correction!(pot, tree::BoxTree, fvals, flvals, fl2vals, coeffs)
    c0, c1, c2 = coeffs

    for ibox in leaves(tree)
        @views pot[:, :, ibox] .+= c0 .* fvals[:, :, ibox] .+ c1 .* flvals[:, :, ibox] .+ c2 .* fl2vals[:, :, ibox]
    end

    return pot
end

function _solver_local_lists(tree::BoxTree, lists::InteractionLists)
    filtered = Vector{Vector{Int}}(undef, nboxes(tree))

    for ibox in 1:nboxes(tree)
        if !isleaf(tree, ibox)
            filtered[ibox] = Int[]
            continue
        end

        filtered[ibox] = [
            jbox for jbox in lists.list1[ibox]
            if jbox != ibox && tree.level[jbox] == tree.level[ibox]
        ]
    end

    return InteractionLists(filtered, lists.listpw)
end

function _normalize_targets(targets, tree::BoxTree)
    if targets isa AbstractVector
        length(targets) == tree.ndim || throw(DimensionMismatch("target vector must have length $(tree.ndim)"))
        return reshape(Float64.(collect(targets)), tree.ndim, 1)
    elseif targets isa AbstractMatrix
        if size(targets, 1) == tree.ndim
            return Float64.(targets)
        elseif size(targets, 2) == tree.ndim
            return permutedims(Float64.(targets))
        end
    end

    throw(DimensionMismatch("targets must be an AbstractVector of length $(tree.ndim) or an AbstractMatrix with one dimension equal to $(tree.ndim)"))
end

function _box_contains_point(tree::BoxTree, ibox::Int, point::AbstractVector{<:Real})
    halfsize = tree.boxsize[tree.level[ibox] + 1] / 2
    tol = 128 * eps(Float64) * max(tree.boxsize[1], 1.0)

    for dim in 1:tree.ndim
        abs(point[dim] - tree.centers[dim, ibox]) <= halfsize + tol || return false
    end

    return true
end

function _find_leaf_box(tree::BoxTree, point::AbstractVector{<:Real})
    best_box = 0
    best_level = -1

    for ibox in leaves(tree)
        _box_contains_point(tree, ibox, point) || continue

        if tree.level[ibox] > best_level
            best_box = ibox
            best_level = tree.level[ibox]
        end
    end

    best_box != 0 || throw(ArgumentError("target point $point lies outside the tree domain"))
    return best_box
end

function _target_local_coordinates(tree::BoxTree, ibox::Int, point::AbstractVector{<:Real})
    boxsize = tree.boxsize[tree.level[ibox] + 1]
    return [2 * (Float64(point[dim]) - Float64(tree.centers[dim, ibox])) / Float64(boxsize) for dim in 1:tree.ndim]
end

function _interpolation_rows(tree::BoxTree, local_point::AbstractVector{<:Real})
    nodes, _ = nodes_and_weights(tree.basis, tree.norder)
    return ntuple(dim -> vec(interpolation_matrix(tree.basis, nodes, [local_point[dim]])), tree.ndim)
end

function _evaluate_tensor_values(values, rows)
    n = length(first(rows))
    tensor = reshape(values, ntuple(_ -> n, length(rows))...)
    accum = zero(eltype(values))

    for idx in CartesianIndices(tensor)
        weight = one(eltype(values))

        for dim in 1:length(rows)
            weight *= rows[dim][idx[dim]]
        end

        accum += weight * tensor[idx]
    end

    return accum
end

function _evaluate_targets(pot, grad, hess, tree::BoxTree, targets)
    target_points = _normalize_targets(targets, tree)
    ntarg = size(target_points, 2)
    nd = size(pot, 1)

    target_pot = Matrix{eltype(pot)}(undef, nd, ntarg)
    target_grad = grad === nothing ? nothing : Array{eltype(grad),3}(undef, nd, tree.ndim, ntarg)
    target_hess = hess === nothing ? nothing : Array{eltype(hess),3}(undef, nd, nhess(tree.ndim), ntarg)

    for itarg in 1:ntarg
        point = @view target_points[:, itarg]
        ibox = _find_leaf_box(tree, point)
        rows = _interpolation_rows(tree, _target_local_coordinates(tree, ibox, point))

        for idensity in 1:nd
            target_pot[idensity, itarg] = _evaluate_tensor_values(@view(pot[idensity, :, ibox]), rows)

            if target_grad !== nothing
                for dim in 1:tree.ndim
                    target_grad[idensity, dim, itarg] = _evaluate_tensor_values(@view(grad[idensity, dim, :, ibox]), rows)
                end
            end

            if target_hess !== nothing
                for component in 1:nhess(tree.ndim)
                    target_hess[idensity, component, itarg] = _evaluate_tensor_values(@view(hess[idensity, component, :, ibox]), rows)
                end
            end
        end
    end

    return target_pot, target_grad, target_hess
end

function bdmk(
    tree::BoxTree,
    fvals::Array,
    kernel::AbstractKernel;
    eps = 1e-6,
    grad = false,
    hess = false,
    targets = nothing,
)
    eps_value = Float64(eps)
    eps_value > 0 || throw(ArgumentError("eps must be positive"))

    nd, np = _check_solver_inputs(tree, fvals)
    result_type = promote_type(eltype(fvals), Float64)

    sog = load_sog_nodes(kernel, tree.ndim, eps_value)
    porder = select_porder(eps_value)
    proxy = build_proxy_data(tree.basis, tree.norder, porder, tree.ndim)
    lists = build_interaction_lists(tree)
    delta_groups = group_deltas_by_level(sog, tree, eps_value)
    local_tabs = build_local_tables(
        kernel,
        tree.basis,
        tree.norder,
        tree.ndim,
        Float64.(sog.deltas),
        Float64.(tree.boxsize),
        tree.nlevels,
    )
    pw_data = setup_planewave_data(tree, proxy, eps_value; nd = nd)

    pot = zeros(result_type, nd, np, nboxes(tree))

    flvals = zeros(result_type, size(fvals))
    fl2vals = zeros(result_type, size(fvals))
    compute_laplacian!(flvals, tree, fvals, tree.basis)
    compute_bilaplacian!(fl2vals, tree, fvals, flvals, tree.basis)
    _apply_taylor_correction!(pot, tree, fvals, flvals, fl2vals, taylor_coefficients(kernel, sog))

    proxy_charges = zeros(result_type, proxy.ncbox, nd, nboxes(tree))
    _density_to_proxy_leaves!(proxy_charges, tree, fvals, proxy)
    upward_pass!(proxy_charges, tree, proxy)

    proxy_pot = zeros(result_type, proxy.ncbox, nd, nboxes(tree))
    for (_, deltas, weights) in delta_groups.normal
        boxfgt!(proxy_pot, tree, proxy_charges, deltas, weights, pw_data, lists)
    end

    for (delta, weight) in delta_groups.fat
        handle_fat_gaussian!(proxy_pot, tree, proxy_charges, delta, weight, pw_data)
    end

    downward_pass!(proxy_pot, tree, proxy)

    apply_local!(pot, tree, fvals, local_tabs, _solver_local_lists(tree, lists), sog.deltas, sog.weights)
    apply_asymptotic!(pot, tree, fvals, flvals, fl2vals, delta_groups.asymptotic)

    proxy_box_pot = zeros(result_type, nd, np, nboxes(tree))
    proxy_to_potential!(proxy_box_pot, proxy_pot, proxy)
    pot .+= proxy_box_pot

    grad_out = nothing
    if grad
        grad_out = zeros(result_type, nd, tree.ndim, np, nboxes(tree))
        compute_gradient!(grad_out, pot, tree, tree.basis)
    end

    hess_out = nothing
    if hess
        hess_out = zeros(result_type, nd, nhess(tree.ndim), np, nboxes(tree))
        compute_hessian!(hess_out, pot, tree, tree.basis)
    end

    target_pot = nothing
    target_grad = nothing
    target_hess = nothing
    if targets !== nothing
        target_pot, target_grad, target_hess = _evaluate_targets(pot, grad_out, hess_out, tree, targets)
    end

    return SolverResult(pot, grad_out, hess_out, target_pot, target_grad, target_hess)
end
