using BoxDMK
using Test

@testset "BoxDMK.jl" begin
    include("test_types.jl")
    include("test_basis.jl")
    include("test_sog.jl")
    include("test_tensor.jl")
end
