using Test

include("run_benchmarks.jl")

@testset "run_benchmarks helpers" begin
    @test relative_l2_error([2.0, 4.0], [1.0, 2.0]) ≈ 1.0
    @test relative_l2_error(zeros(3), zeros(3)) == 0.0

    targets = generate_target_points(50; seed = 1234)
    @test size(targets) == (3, 50)
    @test all((targets .>= 0.1) .& (targets .<= 0.9))
    @test generate_target_points(5; seed = 1234) == generate_target_points(5; seed = 1234)

    sample = """
    0   1.0300932530563919E-005   8.0362517189154995E-006
    Reallocating
    BENCH_FORTRAN tree_build_s=   1.193000 solve_s=   1.177000 total_s=   2.370000 eps=  1.00000E-06 pnorm=  1.49608E+08 nboxes=     1129 nlevels=        6
    """
    parsed = parse_fortran_benchmark_output(sample)
    @test parsed.tree_build_s ≈ 1.193
    @test parsed.solve_s ≈ 1.177
    @test parsed.total_s ≈ 2.37
    @test parsed.nboxes == 1129
    @test parsed.nlevels == 6
end
