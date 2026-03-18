# Kernels
abstract type AbstractKernel end
struct LaplaceKernel <: AbstractKernel end
struct YukawaKernel{T<:Real} <: AbstractKernel
    beta::T
end
struct SqrtLaplaceKernel <: AbstractKernel end

# Polynomial bases
abstract type AbstractBasis end
struct LegendreBasis <: AbstractBasis end
struct ChebyshevBasis <: AbstractBasis end

# Adaptive tree (geometry only — no function data)
struct BoxTree{T<:Real, B<:AbstractBasis}
    ndim::Int
    nlevels::Int
    centers::Matrix{T}           # (ndim, nboxes)
    boxsize::Vector{T}           # one per level (0:nlevels)
    parent::Vector{Int}
    children::Matrix{Int}        # (2^ndim, nboxes); 0 = no child
    colleagues::Vector{Vector{Int}}
    level::Vector{Int}
    basis::B
    norder::Int
end

# Interaction lists (distinct from colleagues)
struct InteractionLists
    list1::Vector{Vector{Int}}    # direct (local/near-field) interaction list
    listpw::Vector{Vector{Int}}   # plane wave (far-field) interaction list
end

# SOG data
struct SOGNodes{T<:Real}
    weights::Vector{T}
    deltas::Vector{T}
    r0::T                         # cutoff radius for SOG approximation accuracy
end

# Proxy charge/potential subsystem
struct ProxyData{T<:Real}
    porder::Int                    # proxy polynomial order (depends on eps)
    ncbox::Int                     # porder^ndim
    den2pc_mat::Matrix{T}         # (porder, norder) density-to-proxy-charge
    poteval_mat::Matrix{T}        # (norder, porder) proxy-potential-to-potential
    p2c_transmat::Array{T,4}      # (porder, porder, ndim, 2^ndim)
    c2p_transmat::Array{T,4}      # (porder, porder, ndim, 2^ndim)
end

# Local interaction precomputed data (per-level, per-delta structure)
struct LocalTables{T<:Real}
    tab::Array{T,5}               # (norder, norder, nloctab2, ndeltas, 0:nlevels)
    tabx::Array{T,5}              # gradient tables (same shape)
    tabxx::Array{T,5}             # hessian tables (same shape)
    ind::Array{Int,5}             # (2, norder+1, nloctab2, ndeltas, 0:nlevels) sparse ranges
end

# Plane wave expansion workspace
struct PlaneWaveData{T<:Real}
    rmlexp::Vector{ComplexF64}     # multipole/local expansion storage
    iaddr::Matrix{Int}             # (2, nboxes) pointers into rmlexp
    npw::Vector{Int}               # PW term count per level
    pw_nodes::Vector{Vector{T}}    # PW quadrature nodes per level
    pw_weights::Vector{Vector{T}}  # PW quadrature weights per level
    wpwshift::Vector{Matrix{ComplexF64}}  # PW shift matrices per level
    tab_coefs2pw::Vector{Matrix{ComplexF64}}  # coeffs-to-PW per level
    tab_pw2pot::Vector{Matrix{ComplexF64}}    # PW-to-potential per level
    ifpwexp::Vector{Bool}          # which boxes need PW processing
    eps::T                         # requested precision used for PW setup
end

# Solver output
struct SolverResult{T<:Real}
    pot::Array{T,3}                              # (nd, npbox, nboxes)
    grad::Union{Nothing, Array{T,4}}             # (nd, ndim, npbox, nboxes)
    hess::Union{Nothing, Array{T,4}}             # (nd, nhess, npbox, nboxes)
    target_pot::Union{Nothing, Matrix{T}}        # (nd, ntarg)
    target_grad::Union{Nothing, Array{T,3}}      # (nd, ndim, ntarg)
    target_hess::Union{Nothing, Array{T,3}}      # (nd, nhess, ntarg)
end
