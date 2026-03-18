module BoxDMK

using LinearAlgebra
using JLD2

include("types.jl")
include("utils.jl")

function build_tree end
function bdmk end

export LaplaceKernel, YukawaKernel, SqrtLaplaceKernel
export LegendreBasis, ChebyshevBasis
export build_tree, bdmk

end
