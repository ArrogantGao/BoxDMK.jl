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

@testset "Interaction Lists" begin
    f(x) = [exp(-100 * sum((x .- 0.5) .^ 2))]
    tree, _ = build_tree(
        f,
        LaplaceKernel(),
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = 1e-6,
        boxlen = 1.0,
        nd = 1,
    )
    lists = BoxDMK.build_interaction_lists(tree)
    nboxes = size(tree.centers, 2)
    @test length(lists.list1) == nboxes
    @test length(lists.listpw) == nboxes

    function touches(tree, ibox, jbox)
        size_i = tree.boxsize[tree.level[ibox] + 1]
        size_j = tree.boxsize[tree.level[jbox] + 1]
        tol = 128 * eps(Float64) * max(size_i, size_j, 1.0)

        for d in 1:tree.ndim
            if abs(tree.centers[d, ibox] - tree.centers[d, jbox]) > (size_i + size_j) / 2 + tol
                return false
            end
        end

        return true
    end

    for ibox in 1:nboxes
        @test length(unique(lists.list1[ibox])) == length(lists.list1[ibox])
        @test length(unique(lists.listpw[ibox])) == length(lists.listpw[ibox])

        if BoxDMK.isleaf(tree, ibox)
            for jbox in lists.list1[ibox]
                @test BoxDMK.isleaf(tree, jbox)
                @test touches(tree, ibox, jbox)
            end
        else
            @test isempty(lists.list1[ibox])
        end

        for jbox in lists.listpw[ibox]
            @test tree.level[ibox] == tree.level[jbox]
            @test !(jbox in tree.colleagues[ibox])
            @test tree.parent[ibox] > 0
            @test tree.parent[jbox] in tree.colleagues[tree.parent[ibox]]
        end
    end
end
