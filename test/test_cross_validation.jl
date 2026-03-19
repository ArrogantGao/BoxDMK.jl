using BoxDMK
using Test

const _CROSS_VALIDATION_CENTER = [0.5, 0.5, 0.5]
const _CROSS_VALIDATION_RSIG = 0.01

_cross_validation_source(x) = [exp(-sum((x .- _CROSS_VALIDATION_CENTER) .^ 2) / _CROSS_VALIDATION_RSIG^2)]

function _run_cross_validation_case(kernel::BoxDMK.AbstractKernel, eps_val::Float64)
    tree, fvals = build_tree(
        _cross_validation_source,
        kernel,
        LegendreBasis();
        ndim = 3,
        norder = 8,
        eps = eps_val,
        boxlen = 1.0,
        nd = 1,
    )

    result = bdmk(tree, fvals, kernel; eps = eps_val)
    return tree, fvals, result
end

@testset "Cross-Validation" begin
    laplace_results = Dict{Float64, Tuple{Any, Any, BoxDMK.SolverResult}}()

    for kernel in (LaplaceKernel(), YukawaKernel(1.0), SqrtLaplaceKernel())
        @testset "$(typeof(kernel))" begin
            eps_values = kernel isa LaplaceKernel ? (1e-3, 1e-6) : (1e-3,)

            for eps_val in eps_values
                tree, fvals, result = _run_cross_validation_case(kernel, eps_val)

                @test size(result.pot) == size(fvals)
                @test all(isfinite.(result.pot))
                @test maximum(abs.(result.pot)) > 0

                if kernel isa LaplaceKernel
                    laplace_results[eps_val] = (tree, fvals, result)
                else
                    @test result.target_pot === nothing
                end
            end
        end
    end

    @testset "Convergence" begin
        _, _, coarse = laplace_results[1e-3]
        _, _, fine = laplace_results[1e-6]

        max1 = maximum(abs.(coarse.pot))
        max2 = maximum(abs.(fine.pot))
        @test abs(max1 - max2) / max(max1, max2) < 0.25
    end
end
