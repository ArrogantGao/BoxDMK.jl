using BoxDMK
using LinearAlgebra

const HYBRID_REFERENCE_CASE = (
    kernel = :laplace,
    ndim = 3,
    norder = 16,
    eps = 1e-6,
    outputs = :potentials_only,
)

const HYBRID_STEP_NAMES = (
    :precomp,
    :taylor,
    :upward,
    :charge_to_pw,
    :m2l,
    :pw_to_proxy_and_down,
    :local,
    :asymptotic,
    :proxy_to_pot,
)

if !isdefined(@__MODULE__, :make_problem)
    include(joinpath(@__DIR__, "julia_vs_fortran.jl"))
end

function _timings_named_tuple(values)
    return NamedTuple{HYBRID_STEP_NAMES}(Tuple(Float64.(values)))
end

function _base_report(; execute::Bool)
    return (
        case = HYBRID_REFERENCE_CASE,
        library_path = BoxDMK._resolve_fortran_library_path(),
        vendored_root = BoxDMK._vendored_fortran_root(),
        execute = execute,
        step_names = collect(HYBRID_STEP_NAMES),
    )
end

function _stage_error_report(problem, kernel)
    rhs = shifted_julia_rhs(problem.boxlen[])
    data = build_tree_fortran(
        rhs,
        kernel,
        LegendreBasis();
        ndim = 3,
        norder = Int(problem.norder[]),
        eps = problem.epstree[],
        boxlen = problem.boxlen[],
        nd = 1,
        eta = problem.eta[],
    )

    BoxDMK.reset_fortran_debug!()
    reference = bdmk_fortran(data, kernel; eps = problem.eps[])
    snapshot = BoxDMK.get_fortran_debug_snapshot()

    tree = data.tree
    fvals = data.fvals
    eps_value = Float64(problem.eps[])
    nd, np = BoxDMK._check_solver_inputs(tree, fvals)
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
    pw_data = BoxDMK._setup_normal_pw_data(tree, proxy, eps_value; nd = nd, delta_groups = delta_groups)

    pot = zeros(result_type, nd, np, BoxDMK.nboxes(tree))
    flvals = zeros(result_type, size(fvals))
    fl2vals = zeros(result_type, size(fvals))
    compute_laplacian!(flvals, tree, fvals, tree.basis)
    compute_bilaplacian!(fl2vals, tree, fvals, flvals, tree.basis)
    BoxDMK._apply_taylor_correction!(pot, tree, fvals, flvals, fl2vals, taylor_coefficients(kernel, sog))
    pot_step2 = copy(pot)

    proxy_charges = zeros(result_type, proxy.ncbox, nd, BoxDMK.nboxes(tree))
    BoxDMK._density_to_proxy_leaves!(proxy_charges, tree, fvals, proxy)
    upward_pass!(proxy_charges, tree, proxy)

    proxy_pot = zeros(result_type, proxy.ncbox, nd, BoxDMK.nboxes(tree))
    fat_tables_cache = Dict{Int, Any}()
    for (level, deltas, weights) in delta_groups.normal
        batch = _prepare_normal_pw_batch(tree, proxy_charges, deltas, weights, pw_data)
        _charge_to_pw!(tree, proxy_charges, pw_data, batch)
    end
    for (level, deltas, weights) in BoxDMK._group_fat_gaussians_by_level(tree, delta_groups.fat, eps_value)
        tables = get!(fat_tables_cache, level) do
            BoxDMK.build_fat_gaussian_tables(tree, proxy, eps_value, level)
        end
        handle_fat_gaussian!(proxy_pot, tree, proxy_charges, deltas, weights, tables)
    end
    for (_, deltas, weights) in delta_groups.normal
        batch = _prepare_normal_pw_batch(tree, proxy_charges, deltas, weights, pw_data)
        _m2l!(tree, lists, pw_data, batch)
    end
    for (_, deltas, weights) in delta_groups.normal
        batch = _prepare_normal_pw_batch(tree, proxy_charges, deltas, weights, pw_data)
        _pw_to_proxy!(proxy_pot, tree, pw_data, batch)
    end
    downward_pass!(proxy_pot, tree, proxy)

    apply_local!(pot, tree, fvals, local_tabs, BoxDMK._solver_local_lists(tree, lists), sog.deltas, sog.weights)
    pot_step7 = copy(pot)

    apply_asymptotic!(pot, tree, fvals, flvals, fl2vals, delta_groups.asymptotic)
    pot_step8 = copy(pot)

    proxy_box_pot = zeros(result_type, nd, np, BoxDMK.nboxes(tree))
    proxy_to_potential!(proxy_box_pot, proxy_pot, proxy)
    pot .+= proxy_box_pot
    pot_step9 = copy(pot)

    fortran_local = snapshot.step7_pot .- snapshot.step2_pot
    julia_local = pot_step7 .- pot_step2
    fortran_asym = snapshot.step8_pot .- snapshot.step7_pot
    julia_asym = pot_step8 .- pot_step7
    fortran_proxy = snapshot.step9_pot .- snapshot.step8_pot
    julia_proxy = pot_step9 .- pot_step8

    return (
        reference_tree_nboxes = BoxDMK.nboxes(tree),
        reference_tree_nlevels = tree.nlevels,
        stages = (
            step2 = (relerr = relerr(snapshot.step2_pot, pot_step2), maxabs = maxabsdiff(snapshot.step2_pot, pot_step2), fortran_norm = norm(snapshot.step2_pot), julia_norm = norm(pot_step2)),
            step3 = (relerr = relerr(snapshot.step3_proxycharge, proxy_charges), maxabs = maxabsdiff(snapshot.step3_proxycharge, proxy_charges), fortran_norm = norm(snapshot.step3_proxycharge), julia_norm = norm(proxy_charges)),
            step6 = (relerr = relerr(snapshot.step6_proxypotential, proxy_pot), maxabs = maxabsdiff(snapshot.step6_proxypotential, proxy_pot), fortran_norm = norm(snapshot.step6_proxypotential), julia_norm = norm(proxy_pot)),
            step7 = (relerr = relerr(snapshot.step7_pot, pot_step7), maxabs = maxabsdiff(snapshot.step7_pot, pot_step7), fortran_norm = norm(snapshot.step7_pot), julia_norm = norm(pot_step7)),
            step8 = (relerr = relerr(snapshot.step8_pot, pot_step8), maxabs = maxabsdiff(snapshot.step8_pot, pot_step8), fortran_norm = norm(snapshot.step8_pot), julia_norm = norm(pot_step8)),
            step9 = (relerr = relerr(snapshot.step9_pot, pot_step9), maxabs = maxabsdiff(snapshot.step9_pot, pot_step9), fortran_norm = norm(snapshot.step9_pot), julia_norm = norm(pot_step9)),
        ),
        increments = (
            local_increment = (relerr = relerr(fortran_local, julia_local), maxabs = maxabsdiff(fortran_local, julia_local), fortran_norm = norm(fortran_local), julia_norm = norm(julia_local)),
            asymptotic_increment = (relerr = relerr(fortran_asym, julia_asym), maxabs = maxabsdiff(fortran_asym, julia_asym), fortran_norm = norm(fortran_asym), julia_norm = norm(julia_asym)),
            proxy_increment = (relerr = relerr(fortran_proxy, julia_proxy), maxabs = maxabsdiff(fortran_proxy, julia_proxy), fortran_norm = norm(fortran_proxy), julia_norm = norm(julia_proxy)),
        ),
        final = (
            relerr = relerr(reference.pot, pot_step9),
            maxabs = maxabsdiff(reference.pot, pot_step9),
            fortran_norm = norm(reference.pot),
            julia_norm = norm(pot_step9),
        ),
    )
end

function _hybrid_candidate_report(problem, kernel, julia_targets)
    rhs = shifted_julia_rhs(problem.boxlen[])
    reference_tree = build_tree_fortran(
        rhs,
        kernel,
        LegendreBasis();
        ndim = 3,
        norder = Int(problem.norder[]),
        eps = problem.epstree[],
        boxlen = problem.boxlen[],
        nd = 1,
        eta = problem.eta[],
    )
    reference_targets = bdmk_fortran(reference_tree, kernel; eps = problem.eps[], targets = julia_targets)
    reference_result = bdmk_fortran(reference_tree, kernel; eps = problem.eps[])

    t_build = @elapsed native_tree, native_fvals = build_tree(
        rhs,
        kernel,
        LegendreBasis();
        ndim = 3,
        norder = Int(problem.norder[]),
        eps = problem.epstree[],
        boxlen = problem.boxlen[],
        nd = 1,
        eta = problem.eta[],
    )
    t_solve = @elapsed hybrid_result = bdmk_fortran(native_tree, native_fvals, kernel; eps = problem.eps[], targets = julia_targets)

    return (
        build_s = t_build,
        solve_s = t_solve,
        total_s = t_build + t_solve,
        native_nboxes = BoxDMK.nboxes(native_tree),
        native_nlevels = native_tree.nlevels,
        target_relerr = relerr(reference_targets.target_pot, hybrid_result.target_pot),
        target_maxabs = maxabsdiff(reference_targets.target_pot, hybrid_result.target_pot),
        target_reference_norm = norm(reference_targets.target_pot),
        target_hybrid_norm = norm(hybrid_result.target_pot),
        pot_reference_norm = norm(reference_result.pot),
        pot_hybrid_norm = norm(hybrid_result.pot),
    )
end

function run_hybrid_parity_reference(; execute::Bool = true)
    base = _base_report(; execute = execute)
    execute || return (; base..., status = :timing_and_final_error_only)

    problem = make_problem()
    kernel = LaplaceKernel()
    shift = fill(problem.boxlen[] / 2, 3)
    julia_targets = TARGETS_PHYSICAL .+ shift
    shifted_rhs = shifted_julia_rhs(problem.boxlen[])

    fortran_tree = build_fortran_tree(problem)
    fortran_result = solve_fortran(problem, fortran_tree)
    fortran_targets = solve_fortran(problem, fortran_tree; targets = TARGETS_PHYSICAL)

    imported_tree = unpack_fortran_tree(problem, fortran_tree)
    imported_fvals = reshape(copy(fortran_tree.fvals), Int(problem.nd[]), fortran_tree.npbox, fortran_tree.nboxes)

    warmup_relerr = warmup_julia_methods()

    t_tree_j = @elapsed begin
        native_tree, native_fvals = build_tree(
            shifted_rhs,
            kernel,
            LegendreBasis();
            ndim = 3,
            norder = Int(problem.norder[]),
            eps = problem.epstree[],
            boxlen = problem.boxlen[],
            nd = 1,
            eta = problem.eta[],
        )
        global _hybrid_native_tree = native_tree
        global _hybrid_native_fvals = native_fvals
    end

    native_tree = _hybrid_native_tree
    native_fvals = _hybrid_native_fvals
    native_pipeline = timed_julia_pipeline(native_tree, native_fvals, kernel; eps = problem.eps[])
    native_target_pot, _, _ = BoxDMK._evaluate_targets(native_pipeline.pot, nothing, nothing, native_tree, julia_targets)

    same_tree_pipeline = timed_julia_pipeline(imported_tree, imported_fvals, kernel; eps = problem.eps[])
    same_tree_target_pot, _, _ = BoxDMK._evaluate_targets(same_tree_pipeline.pot, nothing, nothing, imported_tree, TARGETS_PHYSICAL)
    stage_errors = _stage_error_report(problem, kernel)
    hybrid_candidate = _hybrid_candidate_report(problem, kernel, julia_targets)

    return (
        base...,
        status = :timing_final_error_and_stage_debug,
        warmup_api_vs_pipeline_relerr = warmup_relerr,
        tree = (
            fortran_nboxes = fortran_tree.nboxes,
            fortran_nlevels = fortran_tree.nlevels,
            julia_nboxes = BoxDMK.nboxes(native_tree),
            julia_nlevels = native_tree.nlevels,
            fortran_build_s = fortran_tree.t_mem + fortran_tree.t_build,
            julia_build_s = t_tree_j,
        ),
        native_compare = (
            solve_fortran_s = fortran_result.t_solve,
            solve_julia_s = native_pipeline.total,
            fortran_pnorm = fortran_result.pnorm,
            julia_pnorm = norm(native_pipeline.pot),
            target_relerr = relerr(fortran_targets.pote, native_target_pot),
            target_maxabs = maxabsdiff(fortran_targets.pote, native_target_pot),
            timings_fortran = _timings_named_tuple(fortran_result.tottimeinfo[1:9]),
            timings_julia = _timings_named_tuple(native_pipeline.timings),
        ),
        same_tree_compare = (
            solve_fortran_s = fortran_result.t_solve,
            solve_julia_s = same_tree_pipeline.total,
            pot_relerr = relerr(fortran_result.pot, same_tree_pipeline.pot),
            pot_maxabs = maxabsdiff(fortran_result.pot, same_tree_pipeline.pot),
            target_relerr = relerr(fortran_targets.pote, same_tree_target_pot),
            target_maxabs = maxabsdiff(fortran_targets.pote, same_tree_target_pot),
            fortran_pnorm = fortran_result.pnorm,
            julia_pnorm = norm(same_tree_pipeline.pot),
            timings_fortran = _timings_named_tuple(fortran_result.tottimeinfo[1:9]),
            timings_julia = _timings_named_tuple(same_tree_pipeline.timings),
        ),
        same_tree_stage_errors = stage_errors,
        hybrid_candidate = hybrid_candidate,
        recommended_language = (
            tree_build = :julia,
            precomp = :fortran,
            taylor = :fortran,
            upward = :fortran,
            charge_to_pw = :fortran,
            m2l = :fortran,
            pw_to_proxy_and_down = :fortran,
            local_interactions = :fortran,
            asymptotic = :fortran,
            proxy_to_pot = :fortran,
        ),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(run_hybrid_parity_reference())
end
