using BoxDMK
using LinearAlgebra
using Printf

const LIBBOXDMK = "/mnt/home/xgao1/codes/boxdmk/build/libboxdmk.so"
const STEP_NAMES = [
    "precomp",
    "taylor",
    "upward",
    "charge→PW",
    "M2L",
    "PW→proxy+down",
    "local",
    "asymptotic",
    "proxy→pot",
]

const TARGETS_PHYSICAL = [
    -0.32  -0.18   0.00   0.12   0.24  -0.28   0.30   0.08;
    -0.10   0.14  -0.22   0.20  -0.24   0.32   0.06  -0.30;
     0.18  -0.26   0.28  -0.12   0.04   0.10  -0.16   0.22;
]

const RSIG = 1e-4
const NGAUSS = 2

function analytic_rhs_scalar(x::AbstractVector{<:Real})
    ndim = 3
    rsign = (RSIG * 1.0)^(ndim / 2)
    centers = (
        (0.1, 0.02, 0.04),
        (0.03, -0.1, 0.05),
    )
    sigmas = (RSIG, RSIG / 2)
    strengths = (
        1.0 / (π * rsign),
        -0.5 / (π * rsign),
    )

    value = 0.0
    @inbounds for k in 1:NGAUSS
        rr = 0.0
        for d in 1:ndim
            delta = Float64(x[d]) - centers[k][d]
            rr += delta * delta
        end
        sigma = sigmas[k]
        value += strengths[k] * exp(-rr / sigma) * (-2.0 * ndim + 4.0 * rr / sigma) / sigma
    end

    return value
end

analytic_rhs(x::AbstractVector{<:Real}) = [analytic_rhs_scalar(x)]

function shifted_julia_rhs(boxlen::Real)
    shift = fill(Float64(boxlen) / 2, 3)
    return x -> [analytic_rhs_scalar(x .- shift)]
end

function fortran_rhs_cb(
    nd::Cint,
    xyz_ptr::Ptr{Cdouble},
    dpars_ptr::Ptr{Cdouble},
    zpars_ptr::Ptr{ComplexF64},
    ipars_ptr::Ptr{Cint},
    f_ptr::Ptr{Cdouble},
)::Cvoid
    _ = zpars_ptr
    ndim = Int(unsafe_load(ipars_ptr, 1))
    ng = Int(unsafe_load(ipars_ptr, 3))

    @inbounds for i in 1:Int(nd)
        fi = 0.0
        for k in 1:ng
            idp = (k - 1) * 5
            rr = 0.0
            for d in 1:ndim
                xk = unsafe_load(xyz_ptr, d) - unsafe_load(dpars_ptr, idp + d)
                rr += xk * xk
            end
            sigma = unsafe_load(dpars_ptr, idp + 4)
            strength = unsafe_load(dpars_ptr, idp + 5)
            fi += strength * exp(-rr / sigma) * (-2.0 * ndim + 4.0 * rr / sigma) / sigma
        end
        unsafe_store!(f_ptr, fi, i)
    end

    return nothing
end

const RHS_CFUNC = @cfunction(
    fortran_rhs_cb,
    Cvoid,
    (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{ComplexF64}, Ptr{Cint}, Ptr{Cdouble}),
)

function relerr(a, b)
    na = norm(a)
    nb = norm(b)
    scale = max(na, nb, eps(Float64))
    return norm(a .- b) / scale
end

function maxabsdiff(a, b)
    return maximum(abs.(a .- b))
end

function make_problem()
    nd = Ref{Cint}(1)
    ndim = Ref{Cint}(3)
    ikernel = Ref{Cint}(1)
    beta = Ref{Cdouble}(1.0)
    eps = Ref{Cdouble}(1e-6)

    norder = Ref{Cint}(16)
    ipoly = Ref{Cint}(0)
    iperiod = Ref{Cint}(0)
    iptype = Ref{Cint}(2)
    ifnewtree = Ref{Cint}(0)

    eta = Ref{Cdouble}(0.0)
    boxlen = Ref{Cdouble}(1.18)
    epstree = Ref{Cdouble}(eps[] * 500.0)
    zk = Ref{ComplexF64}(30.0 + 0.0im)

    dpars = zeros(Float64, 1000)
    zpars = zeros(ComplexF64, 16)
    ipars = fill(Cint(0), 256)

    rsign = (RSIG * 1.0)^(Int(ndim[]) / 2)
    ipars[1] = ndim[]
    ipars[2] = ikernel[]
    ipars[3] = NGAUSS
    ipars[5] = iperiod[]
    ipars[10] = 1

    dpars[1:5] = [0.1, 0.02, 0.04, RSIG, 1.0 / (π * rsign)]
    dpars[6:10] = [0.03, -0.1, 0.05, RSIG / 2, -0.5 / (π * rsign)]
    dpars[201] = beta[]

    return (
        nd = nd,
        ndim = ndim,
        ikernel = ikernel,
        beta = beta,
        eps = eps,
        norder = norder,
        ipoly = ipoly,
        iperiod = iperiod,
        iptype = iptype,
        ifnewtree = ifnewtree,
        eta = eta,
        boxlen = boxlen,
        epstree = epstree,
        zk = zk,
        dpars = dpars,
        zpars = zpars,
        ipars = ipars,
    )
end

function build_fortran_tree(problem)
    rintl = zeros(Float64, 201)
    nboxes = Ref{Cint}(0)
    nlevels = Ref{Cint}(0)
    ltree = Ref{Cint}(0)

    t_mem = @elapsed begin
        ccall(
            (:boxdmk_vol_tree_mem, LIBBOXDMK),
            Cvoid,
            (
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cdouble},
                Ref{ComplexF64},
                Ref{Cdouble},
                Ref{Cint},
                Ref{Cint},
                Ref{Cdouble},
                Ptr{Cvoid},
                Ref{Cint},
                Ptr{Cdouble},
                Ptr{ComplexF64},
                Ptr{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ptr{Cdouble},
            ),
            problem.ndim,
            problem.ipoly,
            problem.iperiod,
            problem.epstree,
            problem.zk,
            problem.boxlen,
            problem.norder,
            problem.iptype,
            problem.eta,
            RHS_CFUNC,
            problem.nd,
            problem.dpars,
            problem.zpars,
            problem.ipars,
            problem.ifnewtree,
            nboxes,
            nlevels,
            ltree,
            rintl,
        )
    end

    npbox = Int(problem.norder[])^Int(problem.ndim[])
    nb = Int(nboxes[])
    nl = Int(nlevels[])
    lt = Int(ltree[])

    itree = zeros(Cint, lt)
    iptr = zeros(Cint, 8)
    centers = zeros(Float64, Int(problem.ndim[]) * nb)
    boxsize = zeros(Float64, nl + 1)
    fvals = zeros(Float64, Int(problem.nd[]) * npbox * nb)

    t_build = @elapsed begin
        ccall(
            (:boxdmk_vol_tree_build, LIBBOXDMK),
            Cvoid,
            (
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cdouble},
                Ref{ComplexF64},
                Ref{Cdouble},
                Ref{Cint},
                Ref{Cint},
                Ref{Cdouble},
                Ptr{Cvoid},
                Ref{Cint},
                Ptr{Cdouble},
                Ptr{ComplexF64},
                Ptr{Cint},
                Ptr{Cdouble},
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ptr{Cint},
                Ptr{Cint},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
            ),
            problem.ndim,
            problem.ipoly,
            problem.iperiod,
            problem.epstree,
            problem.zk,
            problem.boxlen,
            problem.norder,
            problem.iptype,
            problem.eta,
            RHS_CFUNC,
            problem.nd,
            problem.dpars,
            problem.zpars,
            problem.ipars,
            rintl,
            nboxes,
            nlevels,
            ltree,
            itree,
            iptr,
            centers,
            boxsize,
            fvals,
        )
    end

    return (
        npbox = npbox,
        nboxes = nb,
        nlevels = nl,
        ltree = lt,
        rintl = rintl,
        itree = itree,
        iptr = iptr,
        centers = centers,
        boxsize = boxsize,
        fvals = fvals,
        t_mem = t_mem,
        t_build = t_build,
    )
end

function solve_fortran(problem, tree_data; targets = nothing)
    npbox = tree_data.npbox

    ifpgh = Ref{Cint}(1)
    if targets === nothing
        ntarg = Ref{Cint}(1)
        ifpghtarg = Ref{Cint}(0)
        targs = zeros(Float64, Int(problem.ndim[]) * Int(ntarg[]))
        pote = zeros(Float64, Int(problem.nd[]) * Int(ntarg[]))
        grade = zeros(Float64, max(1, Int(problem.nd[]) * Int(problem.ndim[]) * Int(ntarg[])))
        hesse = zeros(Float64, 1)
    else
        size(targets, 1) == Int(problem.ndim[]) || throw(DimensionMismatch("targets must have size ($(Int(problem.ndim[])), ntarg)"))
        ntarg = Ref{Cint}(size(targets, 2))
        ifpghtarg = Ref{Cint}(1)
        targs = vec(copy(targets))
        pote = zeros(Float64, Int(problem.nd[]) * Int(ntarg[]))
        grade = zeros(Float64, 1)
        hesse = zeros(Float64, 1)
    end

    pot = zeros(Float64, Int(problem.nd[]) * npbox * tree_data.nboxes)
    grad = zeros(Float64, 1)
    hess = zeros(Float64, 1)
    tottimeinfo = zeros(Float64, 20)

    t_solve = @elapsed begin
        ccall(
            (:boxdmk_bdmk, LIBBOXDMK),
            Cvoid,
            (
                Ref{Cint},
                Ref{Cint},
                Ref{Cdouble},
                Ref{Cint},
                Ref{Cdouble},
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ref{Cint},
                Ptr{Cint},
                Ptr{Cint},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ref{Cint},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ref{Cint},
                Ptr{Cdouble},
                Ref{Cint},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
            ),
            problem.nd,
            problem.ndim,
            problem.eps,
            problem.ikernel,
            problem.beta,
            problem.ipoly,
            problem.norder,
            Ref{Cint}(tree_data.npbox),
            Ref{Cint}(tree_data.nboxes),
            Ref{Cint}(tree_data.nlevels),
            Ref{Cint}(tree_data.ltree),
            tree_data.itree,
            tree_data.iptr,
            tree_data.centers,
            tree_data.boxsize,
            tree_data.fvals,
            ifpgh,
            pot,
            grad,
            hess,
            ntarg,
            targs,
            ifpghtarg,
            pote,
            grade,
            hesse,
            tottimeinfo,
        )
    end

    return (
        pot = reshape(pot, Int(problem.nd[]), tree_data.npbox, tree_data.nboxes),
        pote = if ifpghtarg[] >= 1
            reshape(pote, Int(problem.nd[]), Int(ntarg[]))
        else
            nothing
        end,
        tottimeinfo = tottimeinfo,
        t_solve = t_solve,
        pnorm = norm(pot),
    )
end

function unpack_fortran_tree(problem, tree_data)
    ndim = Int(problem.ndim[])
    nboxes = tree_data.nboxes
    mc = 2^ndim
    mnbors = 3^ndim
    centers_mat = reshape(copy(tree_data.centers), ndim, nboxes)

    level_ptr = Int(tree_data.iptr[2])
    parent_ptr = Int(tree_data.iptr[3])
    nchild_ptr = Int(tree_data.iptr[4])
    child_ptr = Int(tree_data.iptr[5])
    ncoll_ptr = Int(tree_data.iptr[6])
    coll_ptr = Int(tree_data.iptr[7])

    levels = Int.(tree_data.itree[level_ptr:(level_ptr + nboxes - 1)])
    parents = Int.(tree_data.itree[parent_ptr:(parent_ptr + nboxes - 1)])
    nchild = Int.(tree_data.itree[nchild_ptr:(nchild_ptr + nboxes - 1)])

    children = zeros(Int, mc, nboxes)
    for ibox in 1:nboxes
        for ic in 1:mc
            child = Int(tree_data.itree[child_ptr + (ibox - 1) * mc + ic - 1])
            child > 0 || continue

            slot = 1
            for d in 1:ndim
                if centers_mat[d, child] > centers_mat[d, ibox]
                    slot += 1 << (d - 1)
                end
            end
            children[slot, ibox] = child
        end
    end

    colleagues = Vector{Vector{Int}}(undef, nboxes)
    for ibox in 1:nboxes
        ncoll = Int(tree_data.itree[ncoll_ptr + ibox - 1])
        coll = Int[]
        for j in 1:min(ncoll, mnbors)
            entry = Int(tree_data.itree[coll_ptr + (ibox - 1) * mnbors + j - 1])
            entry > 0 && push!(coll, entry)
        end
        colleagues[ibox] = coll
    end

    return BoxDMK.BoxTree(
        ndim,
        tree_data.nlevels,
        centers_mat,
        copy(tree_data.boxsize),
        [p > 0 ? p : 0 for p in parents],
        children,
        colleagues,
        levels,
        LegendreBasis(),
        Int(problem.norder[]),
    )
end

function sample_rhs_on_tree(tree::BoxDMK.BoxTree, rhs)
    nodes, _ = BoxDMK.nodes_and_weights(tree.basis, tree.norder)
    grid = BoxDMK._reference_grid(Float64.(nodes), tree.ndim)
    nd = 1
    npbox = BoxDMK.npbox(tree.norder, tree.ndim)
    values = zeros(Float64, nd, npbox, BoxDMK.nboxes(tree))
    x = Vector{Float64}(undef, tree.ndim)

    for ibox in 1:BoxDMK.nboxes(tree)
        halfsize = tree.boxsize[tree.level[ibox] + 1] / 2
        for ipoint in 1:npbox
            @inbounds for d in 1:tree.ndim
                x[d] = tree.centers[d, ibox] + halfsize * grid[d, ipoint]
            end
            values[1, ipoint, ibox] = rhs(x)[1]
        end
    end

    return values
end

function leaf_indices(tree::BoxDMK.BoxTree)
    return collect(BoxDMK.leaves(tree))
end

function _prepare_normal_pw_batch(tree, proxy_charges, deltas, weights, pw_data)
    level = BoxDMK._batch_level(tree, deltas, pw_data.eps)
    porder = BoxDMK._infer_proxy_order(size(proxy_charges, 1), tree.ndim)
    nd = size(proxy_charges, 2)
    npw = pw_data.npw[level + 1]
    nexp_half = BoxDMK.pw_expansion_size_half(npw, tree.ndim)

    return (
        level = level,
        porder = porder,
        nd = nd,
        npw = npw,
        nexp_half = nexp_half,
        ww = pw_data.ww_1d[level + 1],
        nmax = pw_data.nmax,
        level_boxes = collect(BoxDMK.boxes_at_level(tree, level)),
        kernel_ft = BoxDMK.kernel_fourier_transform(
            deltas,
            weights,
            pw_data.pw_nodes[level + 1],
            pw_data.pw_weights[level + 1],
            tree.ndim,
        ),
        proxy_workspace = Matrix{ComplexF64}(undef, nd, porder^tree.ndim),
        pw_workspace = Matrix{ComplexF64}(undef, nd, npw^tree.ndim),
        to_pw_workspace = BoxDMK._rect_tensor_apply_workspace(ComplexF64, nd, porder, npw, tree.ndim),
        to_proxy_workspace = BoxDMK._rect_tensor_apply_workspace(ComplexF64, nd, npw, porder, tree.ndim),
        shift_vec = Vector{ComplexF64}(undef, nexp_half),
    )
end

function _charge_to_pw!(tree, proxy_charges, pw_data, batch)
    for ibox in batch.level_boxes
        pw_data.ifpwexp[ibox] || continue
        mp = BoxDMK._pw_expansion_view(pw_data, 1, ibox, batch.nexp_half, batch.nd)
        loc = BoxDMK._pw_expansion_view(pw_data, 2, ibox, batch.nexp_half, batch.nd)
        fill!(mp, 0)
        fill!(loc, 0)
        BoxDMK._proxycharge_to_pw!(
            mp,
            @view(proxy_charges[:, :, ibox]),
            pw_data.tab_coefs2pw[batch.level + 1],
            tree.ndim,
            batch.porder,
            batch.npw,
            batch.nd,
            batch.proxy_workspace,
            batch.pw_workspace,
            batch.to_pw_workspace,
        )
        loc .= mp
    end
    return nothing
end

function _m2l!(tree, lists, pw_data, batch)
    for ibox in batch.level_boxes
        pw_data.ifpwexp[ibox] || continue
        loc = BoxDMK._pw_expansion_view(pw_data, 2, ibox, batch.nexp_half, batch.nd)

        for jbox in lists.listpw[ibox]
            pw_data.ifpwexp[jbox] || continue
            tree.level[jbox] == batch.level || continue

            offset = BoxDMK._box_offset(tree, ibox, jbox, batch.level)
            BoxDMK.compute_shift_vector!(batch.shift_vec, batch.ww, offset, batch.npw, batch.nmax)
            src = BoxDMK._pw_expansion_view(pw_data, 1, jbox, batch.nexp_half, batch.nd)
            @views loc .+= src .* batch.shift_vec
        end

        @views loc .*= batch.kernel_ft
    end

    return nothing
end

function _pw_to_proxy!(proxy_pot, tree, pw_data, batch)
    for ibox in batch.level_boxes
        pw_data.ifpwexp[ibox] || continue
        loc = BoxDMK._pw_expansion_view(pw_data, 2, ibox, batch.nexp_half, batch.nd)
        BoxDMK._pw_to_proxy!(
            @view(proxy_pot[:, :, ibox]),
            loc,
            pw_data.tab_pw2pot[batch.level + 1],
            tree.ndim,
            batch.porder,
            batch.npw,
            batch.nd,
            batch.pw_workspace,
            batch.proxy_workspace,
            batch.to_proxy_workspace,
        )
    end

    return nothing
end

function timed_julia_pipeline(tree::BoxDMK.BoxTree, fvals, kernel::BoxDMK.AbstractKernel; eps::Real, verbose::Bool = false)
    eps_value = Float64(eps)
    nd, np = BoxDMK._check_solver_inputs(tree, fvals)
    nboxes = BoxDMK.nboxes(tree)
    result_type = promote_type(eltype(fvals), Float64)

    timings = zeros(Float64, 9)
    stage_norms = Dict{String, Float64}()

    verbose && println("  [pipeline] step 1/9 precomp")
    t0 = time_ns()
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
    timings[1] = (time_ns() - t0) / 1e9

    pot = zeros(result_type, nd, np, nboxes)
    flvals = zeros(result_type, size(fvals))
    fl2vals = zeros(result_type, size(fvals))

    verbose && println("  [pipeline] step 2/9 taylor")
    t0 = time_ns()
    compute_laplacian!(flvals, tree, fvals, tree.basis)
    compute_bilaplacian!(fl2vals, tree, fvals, flvals, tree.basis)
    BoxDMK._apply_taylor_correction!(pot, tree, fvals, flvals, fl2vals, taylor_coefficients(kernel, sog))
    timings[2] = (time_ns() - t0) / 1e9
    stage_norms["taylor_pot_norm"] = norm(pot)

    proxy_charges = zeros(result_type, proxy.ncbox, nd, nboxes)
    verbose && println("  [pipeline] step 3/9 upward")
    t0 = time_ns()
    BoxDMK._density_to_proxy_leaves!(proxy_charges, tree, fvals, proxy)
    upward_pass!(proxy_charges, tree, proxy)
    timings[3] = (time_ns() - t0) / 1e9
    stage_norms["proxy_charge_norm"] = norm(proxy_charges)

    proxy_pot = zeros(result_type, proxy.ncbox, nd, nboxes)

    verbose && println("  [pipeline] step 4/9 charge→PW")
    t0 = time_ns()
    for (level, deltas, weights) in delta_groups.normal
        batch = _prepare_normal_pw_batch(tree, proxy_charges, deltas, weights, pw_data)
        level == batch.level || error("normal PW batch level mismatch")
        _charge_to_pw!(tree, proxy_charges, pw_data, batch)
    end
    fat_tables_cache = Dict{Int, Any}()
    for (level, deltas, weights) in BoxDMK._group_fat_gaussians_by_level(tree, delta_groups.fat, eps_value)
        tables = get!(fat_tables_cache, level) do
            BoxDMK.build_fat_gaussian_tables(tree, proxy, eps_value, level)
        end
        handle_fat_gaussian!(proxy_pot, tree, proxy_charges, deltas, weights, tables)
    end
    timings[4] = (time_ns() - t0) / 1e9

    verbose && println("  [pipeline] step 5/9 M2L")
    t0 = time_ns()
    for (_, deltas, weights) in delta_groups.normal
        batch = _prepare_normal_pw_batch(tree, proxy_charges, deltas, weights, pw_data)
        _m2l!(tree, lists, pw_data, batch)
    end
    timings[5] = (time_ns() - t0) / 1e9

    verbose && println("  [pipeline] step 6/9 PW→proxy+down")
    t0 = time_ns()
    for (_, deltas, weights) in delta_groups.normal
        batch = _prepare_normal_pw_batch(tree, proxy_charges, deltas, weights, pw_data)
        _pw_to_proxy!(proxy_pot, tree, pw_data, batch)
    end
    downward_pass!(proxy_pot, tree, proxy)
    timings[6] = (time_ns() - t0) / 1e9
    stage_norms["proxy_pot_norm"] = norm(proxy_pot)

    verbose && println("  [pipeline] step 7/9 local")
    t0 = time_ns()
    apply_local!(pot, tree, fvals, local_tabs, BoxDMK._solver_local_lists(tree, lists), sog.deltas, sog.weights)
    timings[7] = (time_ns() - t0) / 1e9
    stage_norms["local_pot_norm"] = norm(pot)

    verbose && println("  [pipeline] step 8/9 asymptotic")
    t0 = time_ns()
    apply_asymptotic!(pot, tree, fvals, flvals, fl2vals, delta_groups.asymptotic)
    timings[8] = (time_ns() - t0) / 1e9
    stage_norms["asymptotic_pot_norm"] = norm(pot)

    verbose && println("  [pipeline] step 9/9 proxy→pot")
    t0 = time_ns()
    proxy_box_pot = zeros(result_type, nd, np, nboxes)
    proxy_to_potential!(proxy_box_pot, proxy_pot, proxy)
    pot .+= proxy_box_pot
    timings[9] = (time_ns() - t0) / 1e9
    stage_norms["final_pot_norm"] = norm(pot)

    return (
        pot = pot,
        timings = timings,
        stage_norms = stage_norms,
        total = sum(timings),
    )
end

function warmup_julia_methods()
    kernel = LaplaceKernel()
    rhs = x -> [exp(-sum(abs2, x))]
    tree, fvals = build_tree(rhs, kernel, LegendreBasis();
        ndim = 3, norder = 4, eps = 1e-2, boxlen = 1.0, nd = 1, eta = 0.0)
    public = bdmk(tree, fvals, kernel; eps = 1e-3)
    pipeline = timed_julia_pipeline(tree, fvals, kernel; eps = 1e-3)
    return relerr(public.pot, pipeline.pot)
end

function print_step_table(fortran_steps, julia_steps)
    @printf("%2s  %-16s  %12s  %12s  %10s\n", "#", "Step", "Fortran ms", "Julia ms", "J/F")
    @printf("%2s  %-16s  %12s  %12s  %10s\n", "--", "-"^16, "-"^12, "-"^12, "-"^10)
    for i in eachindex(STEP_NAMES)
        ratio = julia_steps[i] / max(fortran_steps[i], eps(Float64))
        @printf("%2d  %-16s  %12.3f  %12.3f  %10.2f\n", i, STEP_NAMES[i], 1000 * fortran_steps[i], 1000 * julia_steps[i], ratio)
    end
    @printf("%2s  %-16s  %12.3f  %12.3f  %10.2f\n", "", "sum", 1000 * sum(fortran_steps), 1000 * sum(julia_steps), sum(julia_steps) / max(sum(fortran_steps), eps(Float64)))
end

function main()
    problem = make_problem()
    kernel = LaplaceKernel()
    shift = fill(problem.boxlen[] / 2, 3)
    julia_targets = TARGETS_PHYSICAL .+ shift

    println("=" ^ 78)
    println("Julia vs Fortran BoxDMK Benchmark")
    println("=" ^ 78)
    println("Problem: Laplace 3D, nd=1, eps=1e-6, norder=16, boxlen=1.18, eta=0")
    println("Note: libboxdmk exposes per-step timings and final outputs, but not intermediate")
    println("      step arrays. This script compares step timings slot-by-slot, validates tree")
    println("      samples against the analytic RHS, and compares both solvers at shared physical")
    println("      target points. Julia builds on [0, boxlen]^3, so the analytic RHS/targets are")
    println("      translated by boxlen/2 to represent the same physical problem as the Fortran run.")
    println()

    println("[1/6] Building Fortran tree...")
    fortran_tree = build_fortran_tree(problem)
    println("[2/6] Solving with Fortran...")
    fortran_result = solve_fortran(problem, fortran_tree)
    println("[2b/6] Computing Fortran target potentials for accuracy...")
    fortran_targets = solve_fortran(problem, fortran_tree; targets = TARGETS_PHYSICAL)

    imported_tree = unpack_fortran_tree(problem, fortran_tree)
    imported_fvals = reshape(copy(fortran_tree.fvals), Int(problem.nd[]), fortran_tree.npbox, fortran_tree.nboxes)
    fortran_sampled_rhs = sample_rhs_on_tree(imported_tree, analytic_rhs)
    fvals_relerr = relerr(imported_fvals, fortran_sampled_rhs)
    fvals_maxabs = maxabsdiff(imported_fvals, fortran_sampled_rhs)

    shifted_rhs = shifted_julia_rhs(problem.boxlen[])
    println("[3/6] Warming up Julia methods on a small problem...")
    warmup_relerr = warmup_julia_methods()

    println("[4/6] Timing Julia native tree build...")
    t_tree_j = @elapsed begin
        native_tree, native_fvals = build_tree(shifted_rhs, kernel, LegendreBasis();
            ndim = 3, norder = Int(problem.norder[]), eps = problem.epstree[], boxlen = problem.boxlen[], nd = 1, eta = problem.eta[])
        global _native_tree = native_tree
        global _native_fvals = native_fvals
    end
    native_tree = _native_tree
    native_fvals = _native_fvals
    native_rhs_values = sample_rhs_on_tree(native_tree, shifted_rhs)
    native_fvals_relerr = relerr(native_fvals, native_rhs_values)
    println("[5/6] Timing Julia solve on native Julia tree...")
    pipeline = timed_julia_pipeline(native_tree, native_fvals, kernel; eps = problem.eps[], verbose = true)
    println("[6/6] Finalizing comparisons...")

    target_pot_j, _, _ = BoxDMK._evaluate_targets(pipeline.pot, nothing, nothing, native_tree, julia_targets)
    target_relerr = relerr(fortran_targets.pote, target_pot_j)
    target_maxabs = maxabsdiff(fortran_targets.pote, target_pot_j)

    println("Tree Build")
    println("-" ^ 78)
    @printf("Fortran tree mem:   %.6f s\n", fortran_tree.t_mem)
    @printf("Fortran tree build: %.6f s\n", fortran_tree.t_build)
    @printf("Fortran tree total: %.6f s\n", fortran_tree.t_mem + fortran_tree.t_build)
    @printf("Julia tree build:   %.6f s\n", t_tree_j)
    @printf("Fortran tree stats: nboxes=%d nlevels=%d leaves=%d npbox=%d\n",
        fortran_tree.nboxes, fortran_tree.nlevels, length(leaf_indices(imported_tree)), fortran_tree.npbox)
    @printf("Julia tree stats:   nboxes=%d nlevels=%d leaves=%d npbox=%d\n",
        BoxDMK.nboxes(native_tree), native_tree.nlevels, length(leaf_indices(native_tree)), BoxDMK.npbox(native_tree.norder, native_tree.ndim))
    @printf("Fortran sampled RHS relerr: %.3e (max abs %.3e)\n", fvals_relerr, fvals_maxabs)
    @printf("Julia sampled RHS relerr:   %.3e\n", native_fvals_relerr)
    @printf("Warmup API vs pipeline relerr: %.3e\n", warmup_relerr)
    println()

    println("Solve Comparison")
    println("-" ^ 78)
    @printf("Fortran solve time:           %.6f s\n", fortran_result.t_solve)
    @printf("Julia native solve time:      %.6f s\n", pipeline.total)
    @printf("Fortran pnorm:                %.6e\n", fortran_result.pnorm)
    @printf("Julia native pnorm:           %.6e\n", norm(pipeline.pot))
    @printf("Shared-target relerr:         %.3e (max abs %.3e)\n", target_relerr, target_maxabs)
    println()

    println("End-to-End Timing")
    println("-" ^ 78)
    @printf("%18s  %12s  %12s  %10s\n", "Metric", "Fortran", "Julia", "J/F")
    @printf("%18s  %12s  %12s  %10s\n", "-"^18, "-"^12, "-"^12, "-"^10)
    @printf("%18s  %12.6f  %12.6f  %10.2f\n", "Tree total (s)", fortran_tree.t_mem + fortran_tree.t_build, t_tree_j, t_tree_j / max(fortran_tree.t_mem + fortran_tree.t_build, eps(Float64)))
    @printf("%18s  %12.6f  %12.6f  %10.2f\n", "Solve (s)", fortran_result.t_solve, pipeline.total, pipeline.total / max(fortran_result.t_solve, eps(Float64)))
    println()

    println("Per-Step Timing")
    println("-" ^ 78)
    print_step_table(fortran_result.tottimeinfo[1:9], pipeline.timings)
    println()

    println("Julia Stage Norms")
    println("-" ^ 78)
    for key in ("taylor_pot_norm", "proxy_charge_norm", "proxy_pot_norm", "local_pot_norm", "asymptotic_pot_norm", "final_pot_norm")
        @printf("%-20s %.6e\n", key * ":", pipeline.stage_norms[key])
    end
    println()

    println("ABI Note")
    println("-" ^ 78)
    println("The callback is bound as `nd::Cint` by value, matching `integer(c_int), value :: nd`")
    println("in `/mnt/home/xgao1/codes/boxdmk/src/bdmk/bdmk_c_api.f90`. Using `Ptr{Cint}`")
    println("for `nd` here would not match the actual `bind(C)` signature.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
