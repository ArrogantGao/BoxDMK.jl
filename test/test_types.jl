using BoxDMK
using Test

@testset "Task 1 types and helpers" begin
    @test :LaplaceKernel in names(BoxDMK)
    @test :YukawaKernel in names(BoxDMK)
    @test :SqrtLaplaceKernel in names(BoxDMK)
    @test :LegendreBasis in names(BoxDMK)
    @test :ChebyshevBasis in names(BoxDMK)
    @test :build_tree in names(BoxDMK)
    @test :bdmk in names(BoxDMK)
    @test isdefined(BoxDMK, :BoxTree)

    if isdefined(BoxDMK, :BoxTree)
        tree = BoxDMK.BoxTree(
            3,
            0,
            reshape([0.0, 0.0, 0.0], 3, 1),
            [1.0],
            [0],
            zeros(Int, 8, 1),
            [Int[]],
            [0],
            BoxDMK.LegendreBasis(),
            4,
        )

        @test fieldnames(BoxDMK.BoxTree) == (
            :ndim,
            :nlevels,
            :centers,
            :boxsize,
            :parent,
            :children,
            :colleagues,
            :level,
            :basis,
            :norder,
        )

        @test BoxDMK.nboxes(tree) == 1
        @test BoxDMK.nleaves(tree) == 1
        @test BoxDMK.isleaf(tree, 1)
        @test collect(BoxDMK.leaves(tree)) == [1]
        @test BoxDMK.npbox(4, 3) == 64
        @test BoxDMK.nhess(3) == 6
        @test BoxDMK.boxes_at_level(tree, 0) == [1]

        yukawa = BoxDMK.YukawaKernel(2.0)
        @test yukawa.beta == 2.0
    end
end
