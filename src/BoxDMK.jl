module BoxDMK

using LinearAlgebra
using JLD2

function build_tree end
function bdmk end

include("types.jl")
include("utils.jl")
include("sog.jl")
if isfile(joinpath(@__DIR__, "proxy.jl"))
    include("proxy.jl")
end
if isfile(joinpath(@__DIR__, "passes.jl"))
    include("passes.jl")
end
include("kernels.jl")
include("basis.jl")
include("tensor.jl")
if isfile(joinpath(@__DIR__, "tree.jl"))
    include("tree.jl")
end
if isfile(joinpath(@__DIR__, "tree_data.jl"))
    include("tree_data.jl")
end
include("derivatives.jl")
include("interaction_lists.jl")
include("local_tables.jl")
include("local.jl")

export LaplaceKernel, YukawaKernel, SqrtLaplaceKernel
export LegendreBasis, ChebyshevBasis
export build_tree, build_interaction_lists, bdmk
export load_sog_nodes
export taylor_coefficients, taylor_coefficients_grad, taylor_coefficients_hess
export select_porder, build_proxy_data, density_to_proxy!, proxy_to_potential!
export compute_laplacian!, compute_bilaplacian!, compute_gradient_density!
export compute_hessian_density!, eval_asymptotic!, apply_asymptotic!
export compute_gradient!, compute_hessian!
export upward_pass!, downward_pass!
export build_local_tables
export apply_local!

end
