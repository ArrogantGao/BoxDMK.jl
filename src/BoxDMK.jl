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
include("kernels.jl")
include("basis.jl")
include("tensor.jl")
if isfile(joinpath(@__DIR__, "tree.jl"))
    include("tree.jl")
end

export LaplaceKernel, YukawaKernel, SqrtLaplaceKernel
export LegendreBasis, ChebyshevBasis
export build_tree, bdmk
export load_sog_nodes
export taylor_coefficients, taylor_coefficients_grad, taylor_coefficients_hess
export select_porder, build_proxy_data, density_to_proxy!, proxy_to_potential!

end
