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
    if isfile(joinpath(@__DIR__, "..", "src", "tree_data.jl")) &&
       isdefined(BoxDMK, :compute_laplacian!) &&
       length(methods(BoxDMK.compute_laplacian!)) > 0
        include("test_tree_data.jl")
    end
    if isfile(joinpath(@__DIR__, "..", "src", "local_tables.jl")) &&
       isfile(joinpath(@__DIR__, "..", "src", "local.jl")) &&
       isdefined(BoxDMK, :build_local_tables) &&
       length(methods(BoxDMK.build_local_tables)) > 0 &&
       isdefined(BoxDMK, :apply_local!) &&
       length(methods(BoxDMK.apply_local!)) > 0
        include("test_local.jl")
    end
    if isfile(joinpath(@__DIR__, "..", "src", "planewave.jl")) &&
       isfile(joinpath(@__DIR__, "test_planewave.jl")) &&
       isdefined(BoxDMK, :get_pw_term_count) &&
       length(methods(BoxDMK.get_pw_term_count)) > 0
        include("test_planewave.jl")
    end
    if isfile(joinpath(@__DIR__, "..", "src", "boxfgt.jl")) &&
       isdefined(BoxDMK, :boxfgt!) &&
       length(methods(BoxDMK.boxfgt!)) > 0
        include("test_boxfgt.jl")
    end
    if isfile(joinpath(@__DIR__, "..", "src", "tree.jl")) && length(methods(BoxDMK.build_tree)) > 0
        include("test_derivatives.jl")
    end
    include("test_passes.jl")
    include("test_solver.jl")
end
