using BoxDMK
using Test
using LinearAlgebra

@testset "BoxDMK.jl" begin
    include("test_module_surface.jl")
    include("test_types.jl")
    include("test_basis.jl")
    include("test_tensor.jl")
    include("test_tree.jl")
    include("test_sog.jl")
    include("test_kernels.jl")
    include("test_proxy.jl")
    include("test_tree_data.jl")
    include("test_local.jl")
    include("test_planewave.jl")
    include("test_passes.jl")
    include("test_boxfgt.jl")
    include("test_derivatives.jl")
    include("test_solver.jl")
    include("test_cross_validation.jl")
    include("test_fortran_wrapper.jl")
    include("test_hybrid_parity.jl")
end
