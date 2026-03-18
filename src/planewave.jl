const _PW_TERM_TABLES = Dict(
    2 => [4, 4, 4, 4, 4, 4, 4, 6, 8, 12, 16],
    3 => [6, 6, 6, 6, 6, 6, 6, 8, 10, 16, 22],
    4 => [8, 8, 8, 8, 6, 8, 8, 10, 14, 20, 30],
    5 => [8, 8, 8, 8, 8, 8, 10, 12, 16, 24, 38],
    6 => [10, 10, 10, 10, 10, 10, 12, 14, 18, 28, 44],
    7 => [10, 12, 12, 12, 10, 12, 14, 16, 22, 32, 52],
    8 => [12, 12, 12, 12, 12, 14, 14, 18, 24, 36, 58],
    9 => [14, 14, 14, 14, 14, 14, 16, 20, 26, 40, 66],
    10 => [14, 14, 16, 16, 16, 16, 18, 22, 30, 44, 72],
    11 => [16, 16, 16, 16, 16, 18, 20, 24, 32, 50, 80],
)

function _pw_precision_bucket(eps::Real)
    eps_float = Float64(eps)
    1e-12 < eps_float <= 1e-2 || throw(ArgumentError("eps must lie in (1e-12, 1e-2] for tabulated PW term counts"))
    return clamp(floor(Int, -log10(eps_float)), 2, 11)
end

function _pw_spacing(npw::Integer, boxdim::Real)
    npw_int = Int(npw)
    npw_int > 0 || throw(ArgumentError("npw must be positive"))

    boxdim_float = Float64(boxdim)
    boxdim_float > 0 || throw(ArgumentError("boxdim must be positive"))
    return 2pi / (boxdim_float * npw_int)
end

function _scaled_legendre_rule(porder::Integer, boxdim::Real)
    porder_int = _check_basis_order(porder)
    boxdim_float = Float64(boxdim)
    half_box = boxdim_float / 2
    xs_ref, quadweights_ref = nodes_and_weights(LegendreBasis(), porder_int)
    return xs_ref .* half_box, quadweights_ref .* half_box
end

function _validate_planewave_inputs(::AbstractKernel, sog::SOGNodes)
    length(sog.deltas) == length(sog.weights) || throw(ArgumentError("SOG deltas and weights must have the same length"))
    all(>(0), sog.deltas) || throw(ArgumentError("SOG deltas must be positive"))
    return nothing
end

function get_pw_term_count(eps::Real, level::Integer)
    eps_float = Float64(eps)
    eps_float > 0 || throw(ArgumentError("eps must be positive"))

    level_int = Int(level)
    if level_int < -10
        return 2 * ceil(Int, 2 * log(10 / eps_float) / pi)
    end

    bucket = _pw_precision_bucket(eps_float)
    table = _PW_TERM_TABLES[bucket]
    index = min(level_int, 0) + 11
    return table[index]
end

function get_pw_nodes(eps::Real, level::Integer, boxdim::Real)
    npw = get_pw_term_count(eps, level)
    spacing = _pw_spacing(npw, boxdim)
    start = -(npw ÷ 2)
    stop = npw ÷ 2 - 1
    nodes = collect(Float64, start:stop) .* spacing
    weights = fill(spacing, npw)
    return nodes, weights
end

function build_pw_conversion_tables(porder::Integer, npw::Integer, nodes::AbstractVector, boxdim::Real)
    npw_int = Int(npw)
    length(nodes) == npw_int || throw(DimensionMismatch("nodes must have length $npw_int"))

    xs, quadweights = _scaled_legendre_rule(porder, boxdim)
    pw_weight = _pw_spacing(npw_int, boxdim)
    scale = sqrt(pw_weight)

    tab_coefs2pw = Matrix{ComplexF64}(undef, npw_int, length(xs))
    tab_pw2pot = Matrix{ComplexF64}(undef, npw_int, length(xs))

    for j in eachindex(xs)
        x = xs[j]
        quadweight = quadweights[j]
        for k in eachindex(nodes)
            phase = nodes[k] * x
            tab_coefs2pw[k, j] = exp(-im * phase) * scale
            tab_pw2pot[k, j] = exp(im * phase) * scale * quadweight
        end
    end

    return tab_coefs2pw, tab_pw2pot
end

function build_pw_shift_matrices(npw::Integer, nodes::AbstractVector, boxdim::Real)
    npw_int = Int(npw)
    length(nodes) == npw_int || throw(DimensionMismatch("nodes must have length $npw_int"))

    shifts = (-1:1) .* Float64(boxdim)
    wpwshift = Matrix{ComplexF64}(undef, npw_int, length(shifts))

    for (icol, shift_distance) in enumerate(shifts)
        for k in eachindex(nodes)
            wpwshift[k, icol] = exp(im * nodes[k] * shift_distance)
        end
    end

    return wpwshift
end

function _kernel_fourier_transform(
    deltas::AbstractVector,
    weights_sog::AbstractVector,
    npw::Integer,
    pw_nodes::AbstractVector,
    ndim::Integer,
)
    npw_int = Int(npw)
    length(pw_nodes) == npw_int || throw(DimensionMismatch("pw_nodes must have length $npw_int"))
    length(deltas) == length(weights_sog) || throw(DimensionMismatch("deltas and weights_sog must have the same length"))

    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))

    kernel_ft = Vector{ComplexF64}(undef, npw_int)

    for k in eachindex(pw_nodes)
        node2 = pw_nodes[k]^2
        value = 0.0

        for id in eachindex(deltas)
            delta = Float64(deltas[id])
            delta > 0 || throw(ArgumentError("deltas must be positive"))
            value += Float64(weights_sog[id]) * exp(-node2 * delta / 4) * sqrt(delta)^ndim_int
        end

        kernel_ft[k] = ComplexF64(value)
    end

    return kernel_ft
end

function kernel_fourier_transform(deltas::AbstractVector, weights_sog::AbstractVector, npw::Integer, pw_nodes::AbstractVector)
    return _kernel_fourier_transform(deltas, weights_sog, npw, pw_nodes, 3)
end

function setup_planewave_data(tree::BoxTree, proxy::ProxyData, kernel::AbstractKernel, sog::SOGNodes, eps::Real)
    _validate_planewave_inputs(kernel, sog)

    levels = 0:tree.nlevels
    npw = Vector{Int}(undef, length(levels))
    pw_nodes = Vector{Vector{Float64}}(undef, length(levels))
    pw_weights = Vector{Vector{Float64}}(undef, length(levels))
    wpwshift = Vector{Matrix{ComplexF64}}(undef, length(levels))
    tab_coefs2pw = Vector{Matrix{ComplexF64}}(undef, length(levels))
    tab_pw2pot = Vector{Matrix{ComplexF64}}(undef, length(levels))

    for level in levels
        idx = level + 1
        npw[idx] = get_pw_term_count(eps, level)
        pw_nodes[idx], pw_weights[idx] = get_pw_nodes(eps, level, tree.boxsize[idx])
        tab_coefs2pw[idx], tab_pw2pot[idx] = build_pw_conversion_tables(proxy.porder, npw[idx], pw_nodes[idx], tree.boxsize[idx])
        wpwshift[idx] = build_pw_shift_matrices(npw[idx], pw_nodes[idx], tree.boxsize[idx])
    end

    ifpwexp = fill(true, nboxes(tree))
    counts_per_box = [npw[tree.level[ibox] + 1] for ibox in 1:nboxes(tree)]
    iaddr = zeros(Int, 2, nboxes(tree))

    offset = 1
    for ibox in 1:nboxes(tree)
        if ifpwexp[ibox]
            iaddr[1, ibox] = offset
            offset += counts_per_box[ibox]
        end
    end

    for ibox in 1:nboxes(tree)
        if ifpwexp[ibox]
            iaddr[2, ibox] = offset
            offset += counts_per_box[ibox]
        end
    end

    rmlexp = zeros(ComplexF64, offset - 1)

    # Kernel-dependent Fourier factors are provided through kernel_fourier_transform;
    # PlaneWaveData stores the reusable geometry-dependent PW tables/workspace.
    return PlaneWaveData(
        rmlexp,
        iaddr,
        npw,
        pw_nodes,
        pw_weights,
        wpwshift,
        tab_coefs2pw,
        tab_pw2pot,
        ifpwexp,
    )
end
