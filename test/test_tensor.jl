using BoxDMK
using LinearAlgebra
using Test

@testset "Tensor Products 3D" begin
    n = 4
    ndim = 3
    nd = 2
    I_mat = Matrix{Float64}(I, n, n)
    vals = rand(nd, n^3)
    out = similar(vals)

    BoxDMK.tensor_product_apply!(out, (I_mat, I_mat, I_mat), vals, n, ndim, nd)
    @test out ≈ vals

    A = rand(n, n)
    B = rand(n, n)
    C = rand(n, n)
    K = kron(C, kron(B, A))

    BoxDMK.tensor_product_apply!(out, (A, B, C), vals, n, ndim, nd)
    @test out ≈ vals * K' atol = 1e-11
end

@testset "P2C/C2P Transforms" begin
    basis = LegendreBasis()
    p2c = BoxDMK.p2c_transform(basis, 6, 3)
    c2p = BoxDMK.c2p_transform(basis, 6, 3)

    @test size(p2c) == (6, 6, 3, 8)
    @test size(c2p) == (6, 6, 3, 8)

    for ic in 1:8, d in 1:3
        @test c2p[:, :, d, ic] ≈ transpose(p2c[:, :, d, ic])
    end
end
