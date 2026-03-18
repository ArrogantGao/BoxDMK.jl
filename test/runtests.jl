using BoxDMK
using Test

@testset "BoxDMK.jl" begin
    include("test_types.jl")
    include("test_basis.jl")
    include("test_sog.jl")
    if isfile(joinpath(@__DIR__, "test_kernels.jl")) &&
       isdefined(BoxDMK, :taylor_coefficients) &&
       length(methods(BoxDMK.taylor_coefficients)) > 0
        include("test_kernels.jl")
    end
    include("test_tensor.jl")
    if isfile(joinpath(@__DIR__, "..", "src", "proxy.jl")) &&
       isdefined(BoxDMK, :select_porder) &&
       length(methods(BoxDMK.select_porder)) > 0
        include("test_proxy.jl")
    end
    if isfile(joinpath(@__DIR__, "..", "src", "tree.jl")) && length(methods(BoxDMK.build_tree)) > 0
        include("test_tree.jl")
    end
end
