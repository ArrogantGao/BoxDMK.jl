using BoxDMK
using Test

maxabsdiff(a, b) = maximum(abs.(a .- b))

function box_coordinates(tree::BoxDMK.BoxTree, ibox::Integer)
    nodes, _ = BoxDMK.nodes_and_weights(tree.basis, tree.norder)
    np = length(nodes)^tree.ndim
    coords = Matrix{Float64}(undef, tree.ndim, np)
    halfsize = tree.boxsize[tree.level[ibox] + 1] / 2
    idx = 1

    for point in Iterators.product(ntuple(_ -> nodes, tree.ndim)...)
        for d in 1:tree.ndim
            coords[d, idx] = tree.centers[d, ibox] + halfsize * point[d]
        end
        idx += 1
    end

    return coords
end

@testset "Tree Data Transforms" begin
    basis = LegendreBasis()

    f_quad(x) = [sum(x .^ 2)]
    tree_quad, fvals_quad = build_tree(
        f_quad,
        LaplaceKernel(),
        basis;
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    flvals_quad = similar(fvals_quad)
    BoxDMK.compute_laplacian!(flvals_quad, tree_quad, fvals_quad, basis)

    for ibox in BoxDMK.leaves(tree_quad)
        @test maxabsdiff(flvals_quad[1, :, ibox], fill(6.0, size(flvals_quad, 2))) <= 1e-10
    end

    gvals = Array{Float64}(undef, 1, tree_quad.ndim, size(fvals_quad, 2), size(fvals_quad, 3))
    BoxDMK.compute_gradient_density!(gvals, tree_quad, fvals_quad, basis)

    hvals = Array{Float64}(undef, 1, BoxDMK.nhess(tree_quad.ndim), size(fvals_quad, 2), size(fvals_quad, 3))
    BoxDMK.compute_hessian_density!(hvals, tree_quad, fvals_quad, basis)

    for ibox in BoxDMK.leaves(tree_quad)
        coords = box_coordinates(tree_quad, ibox)
        @test maxabsdiff(gvals[1, 1, :, ibox], 2 .* coords[1, :]) <= 1e-10
        @test maxabsdiff(gvals[1, 2, :, ibox], 2 .* coords[2, :]) <= 1e-10
        @test maxabsdiff(gvals[1, 3, :, ibox], 2 .* coords[3, :]) <= 1e-10

        @test maxabsdiff(hvals[1, 1, :, ibox], fill(2.0, size(hvals, 3))) <= 1e-10
        @test maxabsdiff(hvals[1, 2, :, ibox], fill(2.0, size(hvals, 3))) <= 1e-10
        @test maxabsdiff(hvals[1, 3, :, ibox], fill(2.0, size(hvals, 3))) <= 1e-10
        @test maxabsdiff(hvals[1, 4, :, ibox], fill(0.0, size(hvals, 3))) <= 1e-10
        @test maxabsdiff(hvals[1, 5, :, ibox], fill(0.0, size(hvals, 3))) <= 1e-10
        @test maxabsdiff(hvals[1, 6, :, ibox], fill(0.0, size(hvals, 3))) <= 1e-10
    end

    f_quartic(x) = [sum(x .^ 4)]
    tree_quartic, fvals_quartic = build_tree(
        f_quartic,
        LaplaceKernel(),
        basis;
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    flvals_quartic = similar(fvals_quartic)
    fl2vals_quartic = similar(fvals_quartic)
    BoxDMK.compute_laplacian!(flvals_quartic, tree_quartic, fvals_quartic, basis)
    BoxDMK.compute_bilaplacian!(fl2vals_quartic, tree_quartic, fvals_quartic, flvals_quartic, basis)

    for ibox in BoxDMK.leaves(tree_quartic)
        @test maxabsdiff(fl2vals_quartic[1, :, ibox], fill(72.0, size(fl2vals_quartic, 2))) <= 2e-6
    end

    delta = 1e-3
    weight = 0.75
    pot = zeros(size(fvals_quartic))
    BoxDMK.eval_asymptotic!(pot, tree_quartic, fvals_quartic, flvals_quartic, fl2vals_quartic, delta, weight)

    scale = weight * (sqrt(pi * delta))^tree_quartic.ndim
    for ibox in BoxDMK.leaves(tree_quartic)
        expected = scale .* (
            fvals_quartic[1, :, ibox] .+
            (delta / 4) .* flvals_quartic[1, :, ibox] .+
            (delta^2 / 32) .* fl2vals_quartic[1, :, ibox]
        )
        @test maxabsdiff(pot[1, :, ibox], expected) <= 1e-12
    end

    pot2 = zeros(size(fvals_quartic))
    asymptotic_deltas = [(delta = 1e-3, weight = 0.75), (delta = 2e-3, weight = -0.125)]
    BoxDMK.apply_asymptotic!(pot2, tree_quartic, fvals_quartic, flvals_quartic, fl2vals_quartic, asymptotic_deltas)

    expected2 = zero(pot2)
    for component in asymptotic_deltas
        scale = component.weight * (sqrt(pi * component.delta))^tree_quartic.ndim
        for ibox in BoxDMK.leaves(tree_quartic)
            expected2[1, :, ibox] .+= scale .* (
                fvals_quartic[1, :, ibox] .+
                (component.delta / 4) .* flvals_quartic[1, :, ibox] .+
                (component.delta^2 / 32) .* fl2vals_quartic[1, :, ibox]
            )
        end
    end

    @test pot2 ≈ expected2 atol = 1e-12
end
