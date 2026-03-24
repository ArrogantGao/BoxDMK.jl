const _MAX_TREE_LEVELS = 200

_lev(ilev) = ilev + 1

mutable struct _FortranTreeState
    ndim::Int
    mc::Int
    mnbors::Int
    nboxes::Int
    nlevels::Int
    nbmax::Int
    nlmax::Int
    laddr::Matrix{Int}
    ilevel::Vector{Int}
    iparent::Vector{Int}
    nchild::Vector{Int}
    ichild::Matrix{Int}
    centers::Matrix{Float64}
    boxsize::Vector{Float64}
    fvals::Array{Float64, 3}
    rintbs::Vector{Float64}
    rintl::Vector{Float64}
    rint::Float64
    iflag::Vector{Int}
    nnbors::Vector{Int}
    nbors::Matrix{Int}
end

function _ftstate_init(ndim::Int, nd::Int, npbox::Int, nbmax::Int, nlmax::Int)
    mc = 2^ndim
    mnbors = 3^ndim
    laddr = Matrix{Int}(undef, 2, nlmax + 1)
    laddr[1, :] .= 0
    laddr[2, :] .= -1
    laddr[:, _lev(0)] .= (1, 1)

    state = _FortranTreeState(
        ndim,
        mc,
        mnbors,
        1,
        0,
        nbmax,
        nlmax,
        laddr,
        zeros(Int, nbmax),
        fill(-1, nbmax),
        zeros(Int, nbmax),
        fill(-1, mc, nbmax),
        zeros(Float64, ndim, nbmax),
        zeros(Float64, nlmax + 1),
        zeros(Float64, nd, npbox, nbmax),
        zeros(Float64, nbmax),
        zeros(Float64, nlmax + 1),
        0.0,
        zeros(Int, nbmax),
        zeros(Int, nbmax),
        fill(-1, mnbors, nbmax),
    )

    state.ilevel[1] = 0
    state.iparent[1] = -1
    state.nchild[1] = 0
    state.ichild[:, 1] .= -1
    state.centers[:, 1] .= 0.0
    return state
end

function _ftstate_grow!(state::_FortranTreeState, new_nbmax::Int)
    new_nbmax <= state.nbmax && return state

    used = state.nboxes
    new_ilevel = zeros(Int, new_nbmax)
    new_iparent = fill(-1, new_nbmax)
    new_nchild = zeros(Int, new_nbmax)
    new_ichild = fill(-1, state.mc, new_nbmax)
    new_centers = zeros(Float64, state.ndim, new_nbmax)
    nd, npbox, _ = size(state.fvals)
    new_fvals = zeros(Float64, nd, npbox, new_nbmax)
    new_rintbs = zeros(Float64, new_nbmax)
    new_iflag = zeros(Int, new_nbmax)
    new_nnbors = zeros(Int, new_nbmax)
    new_nbors = fill(-1, state.mnbors, new_nbmax)

    new_ilevel[1:used] .= state.ilevel[1:used]
    new_iparent[1:used] .= state.iparent[1:used]
    new_nchild[1:used] .= state.nchild[1:used]
    new_ichild[:, 1:used] .= state.ichild[:, 1:used]
    new_centers[:, 1:used] .= state.centers[:, 1:used]
    new_fvals[:, :, 1:used] .= state.fvals[:, :, 1:used]
    new_rintbs[1:used] .= state.rintbs[1:used]
    new_iflag[1:used] .= state.iflag[1:used]
    new_nnbors[1:used] .= state.nnbors[1:used]
    new_nbors[:, 1:used] .= state.nbors[:, 1:used]

    state.nbmax = new_nbmax
    state.ilevel = new_ilevel
    state.iparent = new_iparent
    state.nchild = new_nchild
    state.ichild = new_ichild
    state.centers = new_centers
    state.fvals = new_fvals
    state.rintbs = new_rintbs
    state.iflag = new_iflag
    state.nnbors = new_nnbors
    state.nbors = new_nbors
    return state
end

function _ftstate_to_boxtree(state::_FortranTreeState, basis::AbstractBasis, norder::Int, boxlen::Real)
    nboxes = state.nboxes
    centers = copy(state.centers[:, 1:nboxes])
    centers .+= Float64(boxlen) / 2

    parent = Vector{Int}(undef, nboxes)
    children = Matrix{Int}(undef, state.mc, nboxes)
    level = copy(state.ilevel[1:nboxes])
    colleagues = Vector{Vector{Int}}(undef, nboxes)

    for ibox in 1:nboxes
        parent[ibox] = state.iparent[ibox] < 0 ? 0 : state.iparent[ibox]
        for ich in 1:state.mc
            child = state.ichild[ich, ibox]
            children[ich, ibox] = child < 0 ? 0 : child
        end
        colleagues[ibox] = state.nnbors[ibox] > 0 ? collect(state.nbors[1:state.nnbors[ibox], ibox]) : Int[]
    end

    boxsize = [Float64(boxlen) / (2.0^ilevel) for ilevel in 0:state.nlevels]
    tree = BoxTree(
        state.ndim,
        state.nlevels,
        centers,
        boxsize,
        parent,
        children,
        colleagues,
        level,
        basis,
        norder,
    )

    return tree, copy(state.fvals[:, :, 1:nboxes])
end

function _reference_grid(nodes::AbstractVector{T}, ndim::Int) where {T<:Real}
    np = length(nodes)^ndim
    grid = Matrix{T}(undef, ndim, np)
    idx = 1

    for point in Iterators.product(ntuple(_ -> nodes, ndim)...)
        @inbounds for d in 1:ndim
            grid[d, idx] = point[d]
        end
        idx += 1
    end

    return grid
end

function _reference_weights(weights::AbstractVector{T}, ndim::Int) where {T<:Real}
    np = length(weights)^ndim
    tensor_weights = Vector{Float64}(undef, np)
    idx = 1

    for point in Iterators.product(ntuple(_ -> weights, ndim)...)
        weight = 1.0
        @inbounds for d in 1:ndim
            weight *= Float64(point[d])
        end
        tensor_weights[idx] = weight
        idx += 1
    end

    return tensor_weights
end

function _sample_box(
    f,
    center::AbstractVector{T},
    boxsize::T,
    grid::AbstractMatrix{T},
    nd::Int,
    coord_shift::Real = 0.0,
) where {T<:Real}
    np = size(grid, 2)
    ndim = length(center)
    values = Matrix{Float64}(undef, nd, np)
    point = Vector{Float64}(undef, ndim)
    halfsize = Float64(boxsize) / 2
    shift = Float64(coord_shift)

    for ipoint in 1:np
        @inbounds for d in 1:ndim
            point[d] = Float64(center[d]) + halfsize * Float64(grid[d, ipoint]) + shift
        end

        raw = f(point)
        length(raw) == nd || throw(DimensionMismatch("f(x) must return a vector of length $nd"))

        @inbounds for idim in 1:nd
            values[idim, ipoint] = Float64(raw[idim])
        end
    end

    return values
end

function _modal_tail_mask(ndim::Int, norder::Int)
    npols = norder^ndim
    rmask = zeros(Float64, npols)
    morder = ndim == 1 ? norder - 1 : norder
    idx = 1
    rsum = 0.0

    for multi_index in Iterators.product(ntuple(_ -> 0:(norder - 1), ndim)...)
        degree_sum = 0
        @inbounds for d in 1:ndim
            degree_sum += (multi_index[d] + 1)^2
        end

        if degree_sum >= morder^2
            rmask[idx] = 1.0
            rsum += 1.0
        end

        idx += 1
    end

    return rmask, sqrt(rsum)
end

function _modal_tail_error(
    values,
    modal_transforms::Tuple,
    rmask::AbstractVector,
    rsum::Real,
    boxsize::Real,
    eta::Real,
    norder::Int,
    ndim::Int,
    nd::Int,
    coeffs,
    workspace,
)
    tensor_product_apply!(coeffs, modal_transforms, values, norder, ndim, nd, workspace)

    error = 0.0
    for idim in 1:nd
        tail_sq = 0.0
        @inbounds for index in eachindex(rmask)
            tail_sq += abs2(coeffs[idim, index]) * Float64(rmask[index])
        end
        error = max(error, sqrt(tail_sq))
    end

    return error * (Float64(boxsize) / 2)^Float64(eta) / Float64(rsum)
end

function _prefix_sum(values::AbstractVector{<:Integer})
    sums = Vector{Int}(undef, length(values))
    total = 0

    for i in eachindex(values)
        total += Int(values[i])
        sums[i] = total
    end

    return sums
end

function _ftstate_init_root!(
    state::_FortranTreeState,
    f,
    grid::AbstractMatrix{<:Real},
    wts2::AbstractVector{<:Real},
    boxlen::Real,
    coord_shift::Real,
)
    boxlen_value = Float64(boxlen)
    state.boxsize[_lev(0)] = boxlen_value
    state.fvals[:, :, 1] .= _sample_box(f, zeros(Float64, state.ndim), boxlen_value, grid, size(state.fvals, 1), coord_shift)

    rsc = boxlen_value^2 / state.mc
    state.rintbs[1] = 0.0
    for ipoint in eachindex(wts2)
        weighted_scale = Float64(wts2[ipoint]) * rsc
        @inbounds for idim in 1:size(state.fvals, 1)
            state.rintbs[1] += state.fvals[idim, ipoint, 1]^2 * weighted_scale
        end
    end

    state.rint = sqrt(state.rintbs[1])
    state.rintl[_lev(0)] = state.rint
    return state
end

function _ftstate_find_box_refine!(
    irefinebox::Vector{Int},
    state::_FortranTreeState,
    ifirstbox::Int,
    nbloc::Int,
    modal_transforms::Tuple,
    rmask::AbstractVector,
    rsum::Real,
    boxsize::Real,
    rsc::Real,
    eps::Real,
    eta::Real,
    norder::Int,
    nd::Int,
    coeffs,
    workspace,
)
    if 30.0 * Float64(boxsize) > 5.0
        fill!(irefinebox, 1)
        return 1
    end

    irefine = 0
    for i in 1:nbloc
        ibox = ifirstbox + i - 1
        irefinebox[i] = 0

        error = _modal_tail_error(
            @view(state.fvals[:, :, ibox]),
            modal_transforms,
            rmask,
            rsum,
            boxsize,
            eta,
            norder,
            state.ndim,
            nd,
            coeffs,
            workspace,
        )

        if error > Float64(eps) * Float64(rsc)
            irefinebox[i] = 1
            irefine = 1
        end
    end

    return irefine
end

function _ftstate_refine_boxes!(
    state::_FortranTreeState,
    irefinebox::Vector{Int},
    ifirstbox::Int,
    nbloc::Int,
    bs::Real,
    nlctr::Int,
    f,
    grid::AbstractMatrix{<:Real},
    coord_shift::Real,
)
    nbctr = state.nboxes
    isum = _prefix_sum(irefinebox)
    bsh = Float64(bs) / 2
    mc = state.mc
    bs_value = Float64(bs)
    coord_shift_value = Float64(coord_shift)
    nd = size(state.fvals, 1)

    for i in 1:nbloc
        ibox = ifirstbox + i - 1
        irefinebox[i] == 1 || continue

        nbl = nbctr + (isum[i] - 1) * mc
        state.nchild[ibox] = mc
        for j in 1:mc
            jbox = nbl + j
            child_bits = j - 1
            for k in 1:state.ndim
                sign = ((child_bits >> (k - 1)) & 0x1) == 0 ? -1.0 : 1.0
                state.centers[k, jbox] = state.centers[k, ibox] + sign * bsh
            end

            state.fvals[:, :, jbox] .= _sample_box(
                f,
                @view(state.centers[:, jbox]),
                bs_value,
                grid,
                nd,
                coord_shift_value,
            )
            state.iparent[jbox] = ibox
            state.nchild[jbox] = 0
            state.ichild[:, jbox] .= -1
            state.ichild[j, ibox] = jbox
            state.ilevel[jbox] = nlctr
        end
    end

    if nbloc > 0
        state.nboxes = nbctr + isum[end] * mc
    end
    return state
end

function _ftstate_update_rints!(
    state::_FortranTreeState,
    ifirstbox::Int,
    nbloc::Int,
    wts::AbstractVector{<:Real},
    rsc::Real,
)
    rintsq = state.rint^2
    for i in 1:nbloc
        ibox = ifirstbox + i - 1
        if state.nchild[ibox] > 0
            rintsq -= state.rintbs[ibox]
        end
    end
    rintsq = max(rintsq, 0.0)

    nd = size(state.fvals, 1)
    npbox = size(state.fvals, 2)
    rsc_value = Float64(rsc)
    for i in 1:nbloc
        ibox = ifirstbox + i - 1
        if state.nchild[ibox] > 0
            for j in 1:state.mc
                jbox = state.ichild[j, ibox]
                state.rintbs[jbox] = 0.0
                for l in 1:npbox
                    weighted_scale = Float64(wts[l]) * rsc_value
                    @inbounds for idim in 1:nd
                        state.rintbs[jbox] += state.fvals[idim, l, jbox]^2 * weighted_scale
                    end
                end
                rintsq += state.rintbs[jbox]
            end
        end
    end

    state.rint = sqrt(rintsq)
    return state
end

function _ftstate_adaptive_refine!(
    state::_FortranTreeState,
    f,
    grid::AbstractMatrix{<:Real},
    wts2::AbstractVector{<:Real},
    modal_transforms::Tuple,
    rmask::AbstractVector,
    rsum::Real,
    eps::Real,
    eta::Real,
    norder::Int,
)
    nd = size(state.fvals, 1)
    coeffs = Matrix{Float64}(undef, nd, norder^state.ndim)
    workspace = _tensor_apply_workspace(Float64, nd, norder, state.ndim)

    for ilev in 0:(state.nlmax - 1)
        ifirstbox = state.laddr[1, _lev(ilev)]
        ilastbox = state.laddr[2, _lev(ilev)]
        nbloc = ilastbox - ifirstbox + 1
        irefinebox = zeros(Int, nbloc)
        rsc = sqrt(inv(state.boxsize[_lev(0)]^state.ndim)) * state.rintl[_lev(ilev)]
        irefine = _ftstate_find_box_refine!(
            irefinebox,
            state,
            ifirstbox,
            nbloc,
            modal_transforms,
            rmask,
            rsum,
            state.boxsize[_lev(ilev)],
            rsc,
            eps,
            eta,
            norder,
            nd,
            coeffs,
            workspace,
        )

        if irefine == 0
            state.nlevels = ilev
            return state
        end

        nbadd = count(==(1), irefinebox) * state.mc
        if state.nboxes + nbadd > state.nbmax
            _ftstate_grow!(state, state.nboxes + nbadd)
        end

        state.boxsize[_lev(ilev + 1)] = state.boxsize[_lev(ilev)] / 2
        state.laddr[1, _lev(ilev + 1)] = state.nboxes + 1

        _ftstate_refine_boxes!(
            state,
            irefinebox,
            ifirstbox,
            nbloc,
            state.boxsize[_lev(ilev + 1)],
            ilev + 1,
            f,
            grid,
            state.boxsize[_lev(0)] / 2,
        )

        _ftstate_update_rints!(
            state,
            ifirstbox,
            nbloc,
            wts2,
            state.boxsize[_lev(ilev + 1)]^state.ndim / state.mc,
        )
        state.rintl[_lev(ilev + 1)] = state.rint
        state.laddr[2, _lev(ilev + 1)] = state.nboxes
    end

    state.nlevels = state.nlmax
    return state
end

function _ftstate_computecoll!(state::_FortranTreeState)
    state.nnbors[1:state.nboxes] .= 0
    state.nbors[:, 1:state.nboxes] .= -1
    state.nnbors[1] = 1
    state.nbors[1, 1] = 1

    for ilev in 1:state.nlevels
        _ftstate_fill_colleagues!(state, state.laddr[1, _lev(ilev)], state.laddr[2, _lev(ilev)], ilev)
    end

    return state
end

function _prefix_count_positive(values::AbstractVector{<:Integer})
    sums = Vector{Int}(undef, length(values))
    total = 0

    for i in eachindex(values)
        Int(values[i]) > 0 && (total += 1)
        sums[i] = total
    end

    return sums
end

function _ftstate_fill_colleagues!(state::_FortranTreeState, ifirstbox::Int, ilastbox::Int, ilev::Int)
    for ibox in ifirstbox:ilastbox
        state.nnbors[ibox] = 0
        state.nbors[:, ibox] .= -1
        dad = state.iparent[ibox]
        for i in 1:state.nnbors[dad]
            jbox = state.nbors[i, dad]
            for j in 1:state.mc
                kbox = state.ichild[j, jbox]
                kbox > 0 || continue

                ifnbor = true
                for k in 1:state.ndim
                    dis = abs(state.centers[k, kbox] - state.centers[k, ibox])
                    if dis > 1.05 * state.boxsize[_lev(ilev)]
                        ifnbor = false
                        break
                    end
                end

                if ifnbor
                    state.nnbors[ibox] += 1
                    state.nbors[state.nnbors[ibox], ibox] = kbox
                end
            end
        end
    end

    return state
end

function _ftstate_refine_boxes_flag!(
    state::_FortranTreeState,
    ifirstbox::Int,
    nbloc::Int,
    bs::Real,
    nlctr::Int,
    f,
    grid::AbstractMatrix{<:Real},
    coord_shift::Real,
)
    nbloc <= 0 && return state

    ilastbox = ifirstbox + nbloc - 1
    nbctr = state.nboxes
    isum = _prefix_count_positive(@view(state.iflag[ifirstbox:ilastbox]))
    bsh = Float64(bs) / 2
    bs_value = Float64(bs)
    coord_shift_value = Float64(coord_shift)
    nd = size(state.fvals, 1)

    for ibox in ifirstbox:ilastbox
        state.iflag[ibox] > 0 || continue

        state.nchild[ibox] = state.mc
        nbl = nbctr + (isum[ibox - ifirstbox + 1] - 1) * state.mc
        for j in 1:state.mc
            jbox = nbl + j
            child_bits = j - 1
            for k in 1:state.ndim
                sign = ((child_bits >> (k - 1)) & 0x1) == 0 ? -1.0 : 1.0
                state.centers[k, jbox] = state.centers[k, ibox] + sign * bsh
            end

            state.fvals[:, :, jbox] .= _sample_box(
                f,
                @view(state.centers[:, jbox]),
                bs_value,
                grid,
                nd,
                coord_shift_value,
            )
            state.iparent[jbox] = ibox
            state.nchild[jbox] = 0
            state.ichild[:, jbox] .= -1
            state.ichild[j, ibox] = jbox
            state.ilevel[jbox] = nlctr + 1
            state.iflag[jbox] = state.iflag[ibox] == 1 ? 3 : 0
        end
    end

    state.nboxes = nbctr + isum[end] * state.mc
    return state
end

function _ftstate_reorg!(state::_FortranTreeState, laddrtail::Matrix{Int})
    nboxes = state.nboxes
    tladdr = copy(state.laddr)
    tilevel = copy(state.ilevel[1:nboxes])
    tiparent = copy(state.iparent[1:nboxes])
    tnchild = copy(state.nchild[1:nboxes])
    tichild = copy(state.ichild[:, 1:nboxes])
    tiflag = copy(state.iflag[1:nboxes])
    tfvals = copy(state.fvals[:, :, 1:nboxes])
    tcenters = copy(state.centers[:, 1:nboxes])
    iboxtocurbox = zeros(Int, nboxes)

    for ilev in 0:1
        for ibox in state.laddr[1, _lev(ilev)]:state.laddr[2, _lev(ilev)]
            iboxtocurbox[ibox] = ibox
        end
    end

    curbox = state.laddr[1, _lev(2)]
    for ilev in 2:state.nlevels
        state.laddr[1, _lev(ilev)] = curbox
        for ibox in tladdr[1, _lev(ilev)]:tladdr[2, _lev(ilev)]
            state.ilevel[curbox] = tilevel[ibox]
            state.nchild[curbox] = tnchild[ibox]
            state.centers[:, curbox] .= tcenters[:, ibox]
            state.fvals[:, :, curbox] .= tfvals[:, :, ibox]
            state.iflag[curbox] = tiflag[ibox]
            iboxtocurbox[ibox] = curbox
            curbox += 1
        end
        for ibox in laddrtail[1, _lev(ilev)]:laddrtail[2, _lev(ilev)]
            state.ilevel[curbox] = tilevel[ibox]
            state.nchild[curbox] = tnchild[ibox]
            state.centers[:, curbox] .= tcenters[:, ibox]
            state.fvals[:, :, curbox] .= tfvals[:, :, ibox]
            state.iflag[curbox] = tiflag[ibox]
            iboxtocurbox[ibox] = curbox
            curbox += 1
        end
        state.laddr[2, _lev(ilev)] = curbox - 1
    end

    for ibox in 1:nboxes
        newbox = iboxtocurbox[ibox]
        state.iparent[newbox] = tiparent[ibox] < 0 ? -1 : iboxtocurbox[tiparent[ibox]]
        for j in 1:state.mc
            child = tichild[j, ibox]
            state.ichild[j, newbox] = child < 0 ? -1 : iboxtocurbox[child]
        end
    end

    return state
end

function _ftstate_vol_updateflags!(state::_FortranTreeState, curlev::Int, laddr::Matrix{Int})
    distest = 1.05 * (state.boxsize[_lev(curlev)] + state.boxsize[_lev(curlev + 1)]) / 2

    for ibox in laddr[1, _lev(curlev)]:laddr[2, _lev(curlev)]
        state.iflag[ibox] == 3 || continue
        state.iflag[ibox] = 0

        needs_refine = false
        for i in 1:state.nnbors[ibox]
            jbox = state.nbors[i, ibox]
            for j in 1:state.mc
                kbox = state.ichild[j, jbox]
                if kbox > 0 && state.nchild[kbox] > 0
                    ict = 0
                    for k in 1:state.ndim
                        dis = state.centers[k, kbox] - state.centers[k, ibox]
                        abs(dis) <= distest && (ict += 1)
                    end
                    if ict == state.ndim
                        state.iflag[ibox] = 1
                        needs_refine = true
                        break
                    end
                end
            end
            needs_refine && break
        end
    end

    return state
end

function _ftstate_fix_lr!(
    state::_FortranTreeState,
    f,
    grid::AbstractMatrix{<:Real},
    coord_shift::Real,
)
    state.nlevels < 2 && return state

    if 2 * state.mc * state.nboxes > state.nbmax
        _ftstate_grow!(state, 2 * state.mc * state.nboxes)
    end

    state.iflag[1:state.nboxes] .= 0

    for ilev in state.nlevels:-1:2
        distest = 1.05 * (state.boxsize[_lev(ilev - 1)] + state.boxsize[_lev(ilev - 2)]) / 2
        for ibox in state.laddr[1, _lev(ilev)]:state.laddr[2, _lev(ilev)]
            idad = state.iparent[ibox]
            igranddad = state.iparent[idad]
            for i in 1:state.nnbors[igranddad]
                jbox = state.nbors[i, igranddad]
                if state.nchild[jbox] == 0 && state.iflag[jbox] == 0
                    ict = 0
                    for k in 1:state.ndim
                        dis = state.centers[k, jbox] - state.centers[k, idad]
                        abs(dis) <= distest && (ict += 1)
                    end
                    if ict == state.ndim
                        state.iflag[jbox] = 1
                    end
                end
            end
        end
    end

    for ilev in state.nlevels:-1:1
        distest = 1.05 * (state.boxsize[_lev(ilev)] + state.boxsize[_lev(ilev - 1)]) / 2
        for ibox in state.laddr[1, _lev(ilev)]:state.laddr[2, _lev(ilev)]
            if state.iflag[ibox] == 1 || state.iflag[ibox] == 2
                idad = state.iparent[ibox]
                for i in 1:state.nnbors[idad]
                    jbox = state.nbors[i, idad]
                    if state.nchild[jbox] == 0 && state.iflag[jbox] == 0
                        ict = 0
                        for k in 1:state.ndim
                            dis = state.centers[k, jbox] - state.centers[k, ibox]
                            abs(dis) <= distest && (ict += 1)
                        end
                        if ict == state.ndim
                            state.iflag[jbox] = 2
                        end
                    end
                end
            end
        end
    end

    laddrtail = Matrix{Int}(undef, 2, state.nlmax + 1)
    laddrtail[1, :] .= 0
    laddrtail[2, :] .= -1

    for ilev in 1:(state.nlevels - 2)
        laddrtail[1, _lev(ilev + 1)] = state.nboxes + 1
        nbloc = state.laddr[2, _lev(ilev)] - state.laddr[1, _lev(ilev)] + 1
        _ftstate_refine_boxes_flag!(
            state,
            state.laddr[1, _lev(ilev)],
            nbloc,
            state.boxsize[_lev(ilev + 1)],
            ilev,
            f,
            grid,
            coord_shift,
        )
        laddrtail[2, _lev(ilev + 1)] = state.nboxes
    end

    _ftstate_reorg!(state, laddrtail)
    _ftstate_computecoll!(state)

    for ibox in 1:state.nboxes
        state.iflag[ibox] = state.iflag[ibox] == 3 ? 3 : 0
    end

    laddrtail[1, :] .= 0
    laddrtail[2, :] .= -1

    for ilev in 2:(state.nlevels - 2)
        _ftstate_vol_updateflags!(state, ilev, state.laddr)
        _ftstate_vol_updateflags!(state, ilev, laddrtail)

        laddrtail[1, _lev(ilev + 1)] = state.nboxes + 1

        nbloc = state.laddr[2, _lev(ilev)] - state.laddr[1, _lev(ilev)] + 1
        _ftstate_refine_boxes_flag!(
            state,
            state.laddr[1, _lev(ilev)],
            nbloc,
            state.boxsize[_lev(ilev + 1)],
            ilev,
            f,
            grid,
            coord_shift,
        )

        nbloc_tail = laddrtail[2, _lev(ilev)] - laddrtail[1, _lev(ilev)] + 1
        _ftstate_refine_boxes_flag!(
            state,
            laddrtail[1, _lev(ilev)],
            nbloc_tail,
            state.boxsize[_lev(ilev + 1)],
            ilev,
            f,
            grid,
            coord_shift,
        )

        laddrtail[2, _lev(ilev + 1)] = state.nboxes
        _ftstate_fill_colleagues!(state, laddrtail[1, _lev(ilev + 1)], laddrtail[2, _lev(ilev + 1)], ilev + 1)
    end

    _ftstate_reorg!(state, laddrtail)
    _ftstate_computecoll!(state)
    return state
end

function build_tree(
    f,
    kernel::AbstractKernel,
    basis::AbstractBasis;
    ndim::Integer = 3,
    norder::Integer = 6,
    eps::Real = 1e-6,
    boxlen::Real = 1.0,
    nd::Integer = 1,
    dpars = nothing,
    eta::Real = 1.0,
)
    _ = dpars
    ndim_int = Int(ndim)
    ndim_int > 0 || throw(ArgumentError("ndim must be positive"))
    nd_int = Int(nd)
    nd_int > 0 || throw(ArgumentError("nd must be positive"))
    norder_int = _check_basis_order(norder)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    boxlen > 0 || throw(ArgumentError("boxlen must be positive"))
    eps_value = Float64(eps)
    boxlen_value = Float64(boxlen)
    eta_value = Float64(eta)

    mc = 2^ndim_int
    npbox = norder_int^ndim_int

    nodes, weights = nodes_and_weights(basis, norder_int)
    # Grid on [-1,1]^d — _sample_box handles the /2 scaling internally via halfsize
    grid = _reference_grid(Float64.(nodes), ndim_int)
    # Tensor-product weights on [-1,1]^d for update_rints
    wts2 = _reference_weights(Float64.(weights), ndim_int)

    modal_1d = forward_transform(basis, norder_int)
    modal_transforms = ntuple(_ -> modal_1d, ndim_int)
    rmask, rsum = _modal_tail_mask(ndim_int, norder_int)

    coord_shift = boxlen_value / 2

    # Initialize state
    nbmax = 10_000
    state = _ftstate_init(ndim_int, nd_int, npbox, nbmax, _MAX_TREE_LEVELS)

    # Initialize root box and rint
    _ftstate_init_root!(state, f, grid, wts2, boxlen_value, coord_shift)

    # Adaptive refinement
    _ftstate_adaptive_refine!(state, f, grid, wts2, modal_transforms, rmask, rsum, eps_value, eta_value, norder_int)

    # Compute colleagues (unconditionally)
    _ftstate_computecoll!(state)

    # Level restriction (only if nlevels >= 2)
    if state.nlevels >= 2
        _ftstate_fix_lr!(state, f, grid, coord_shift)
    end

    # Materialize into BoxTree + fvals
    return _ftstate_to_boxtree(state, basis, norder_int, boxlen_value)
end
