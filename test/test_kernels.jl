using BoxDMK
using Test

@testset "Taylor Coefficients" begin
    for kernel in (LaplaceKernel(), YukawaKernel(1.0), SqrtLaplaceKernel())
        sog = BoxDMK.load_sog_nodes(kernel, 3, 1e-6)
        c0, c1, c2 = BoxDMK.taylor_coefficients(kernel, sog)
        gc0, gc1 = BoxDMK.taylor_coefficients_grad(kernel, sog)
        hc0, hc1 = BoxDMK.taylor_coefficients_hess(kernel, sog)

        @test all(isfinite, (c0, c1, c2, gc0, gc1, hc0, hc1))
        @test abs(c0) < 1.0
        @test (gc0, gc1) == (c0, c1)
        @test (hc0, hc1) == (c0, c1)
    end
end
