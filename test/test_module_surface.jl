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
    @test isdefined(BoxDMK, :_require_fortran_solve_library!)

    solve_path = BoxDMK._require_fortran_solve_library!()
    @test solve_path == BoxDMK._resolve_fortran_solve_library_path(; must_exist = true)
    @test isfile(solve_path)

    missing_path = joinpath(mktempdir(), "libboxdmk_hot.so")
    err = try
        BoxDMK._require_fortran_solve_library!(; path = missing_path)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(missing_path, sprint(showerror, err))
    @test occursin("julia --project deps/build_fortran_ref.jl", sprint(showerror, err))
end
