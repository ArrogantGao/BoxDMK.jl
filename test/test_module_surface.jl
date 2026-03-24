using BoxDMK
using Test

@testset "Module Surface" begin
    @test :LaplaceKernel in names(BoxDMK)
    @test :YukawaKernel in names(BoxDMK)
    @test :SqrtLaplaceKernel in names(BoxDMK)
    @test :LegendreBasis in names(BoxDMK)
    @test :ChebyshevBasis in names(BoxDMK)
    @test :build_tree in names(BoxDMK)
    @test :bdmk in names(BoxDMK)
    @test :build_tree_fortran in names(BoxDMK)
    @test :bdmk_fortran in names(BoxDMK)

    @test isdefined(BoxDMK, :SolverResult)
    @test isdefined(BoxDMK, :FortranTreeData)
    @test isdefined(BoxDMK, :reset_fortran_debug!)
    @test isdefined(BoxDMK, :get_fortran_debug_snapshot)
    @test isdefined(BoxDMK, :_resolve_fortran_library_path)
    @test isdefined(BoxDMK, :_resolve_fortran_solve_library_path)
end
