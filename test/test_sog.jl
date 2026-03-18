using BoxDMK
using Test

const SOG_DATA_DIR = joinpath(dirname(@__DIR__), "data", "sog")

@testset "SOG Loading" begin
    @test isdefined(BoxDMK, :load_sog_nodes)
    @test isdir(SOG_DATA_DIR)
    @test isfile(joinpath(SOG_DATA_DIR, "laplace_3d.jld2"))
    @test isfile(joinpath(SOG_DATA_DIR, "yukawa_3d.jld2"))
    @test isfile(joinpath(SOG_DATA_DIR, "sqrtlaplace_3d.jld2"))

    if isdefined(BoxDMK, :load_sog_nodes)
        for kernel in (LaplaceKernel(), YukawaKernel(1.0), SqrtLaplaceKernel())
            sog = BoxDMK.load_sog_nodes(kernel, 3, 1e-6)
            @test length(sog.weights) == length(sog.deltas)
            @test length(sog.weights) > 0
            @test all(sog.weights .> 0)
            @test all(sog.deltas .> 0)
            @test sog.r0 > 0
        end

        @test length(BoxDMK.load_sog_nodes(LaplaceKernel(), 3, 1e-6).weights) == 48
        @test length(BoxDMK.load_sog_nodes(LaplaceKernel(), 3, 1e-7).weights) == 70
        @test length(BoxDMK.load_sog_nodes(SqrtLaplaceKernel(), 3, 1e-6).weights) == 52

        @test_throws ArgumentError BoxDMK.load_sog_nodes(LaplaceKernel(), 2, 1e-6)
        @test_throws ArgumentError BoxDMK.load_sog_nodes(LaplaceKernel(), 3, 0.0)
    end
end
