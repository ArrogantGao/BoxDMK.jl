using BoxDMK
using LinearAlgebra
using Test

@testset "LegendreBasis" begin
    basis = LegendreBasis()

    for n in (4, 8, 16)
        x, w = BoxDMK.nodes_and_weights(basis, n)

        @test all((-1 .< x) .& (x .< 1))
        @test sum(w) ≈ 2.0 atol = 1e-14
        @test sum(w .* x .^ (2n - 1)) ≈ 0.0 atol = 1e-12
        @test sum(w .* x .^ (2n - 2)) ≈ 2 / (2n - 1) atol = 1e-12

        D = BoxDMK.derivative_matrix(basis, n)
        vals_x2 = x .^ 2
        @test D * vals_x2 ≈ 2 .* x atol = 1e-12

        D2 = BoxDMK.second_derivative_matrix(basis, n)
        @test D2 * vals_x2 ≈ fill(2.0, n) atol = 1e-12

        U = BoxDMK.forward_transform(basis, n)
        V = BoxDMK.inverse_transform(basis, n)
        @test U * V ≈ Matrix{Float64}(I, n, n) atol = 1e-13
    end
end

@testset "ChebyshevBasis" begin
    basis = ChebyshevBasis()

    for n in (4, 8, 16)
        x, w = BoxDMK.nodes_and_weights(basis, n)

        @test all((-1 .< x) .& (x .< 1))
        @test sum(w) ≈ 2.0 atol = 1e-14

        D = BoxDMK.derivative_matrix(basis, n)
        @test D * (x .^ 2) ≈ 2 .* x atol = 1e-12

        D2 = BoxDMK.second_derivative_matrix(basis, n)
        @test D2 * (x .^ 2) ≈ fill(2.0, n) atol = 1e-10

        U = BoxDMK.forward_transform(basis, n)
        V = BoxDMK.inverse_transform(basis, n)
        @test U * V ≈ Matrix{Float64}(I, n, n) atol = 1e-13
    end
end
