using BoxDMK
using Test

@testset "Tree Construction" begin
    f_smooth(x) = [sin(pi * x[1]) * cos(pi * x[2]) * exp(x[3])]
    tree, fvals = build_tree(
        f_smooth,
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    @test tree.nlevels >= 0
    @test size(tree.centers, 2) > 0
    @test all(tree.level .>= 0)
    @test size(fvals) == (1, 6^3, size(tree.centers, 2))

    f_sharp(x) = [exp(-1000 * sum((x .- 0.5) .^ 2))]
    tree2, fvals2 = build_tree(
        f_sharp,
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )

    @test tree2.nlevels > tree.nlevels
    @test size(fvals2) == (1, 6^3, size(tree2.centers, 2))

    for ibox in 1:size(tree2.centers, 2)
        for jbox in tree2.colleagues[ibox]
            @test abs(tree2.level[ibox] - tree2.level[jbox]) <= 1
        end
    end
end
