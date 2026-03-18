using BoxDMK
using Test

function _local_two_level_tree()
    ndim = 3
    nchildren = 2^ndim
    nboxes_total = nchildren + 1

    centers = zeros(Float64, ndim, nboxes_total)
    centers[:, 1] .= 0.5

    for child in 1:nchildren
        bits = child - 1
        for d in 1:ndim
            centers[d, child + 1] = 0.5 + (((bits >> (d - 1)) & 0x1) == 0 ? -0.25 : 0.25)
        end
    end

    parent = zeros(Int, nboxes_total)
    parent[2:end] .= 1

    children = zeros(Int, nchildren, nboxes_total)
    children[:, 1] .= 2:(nchildren + 1)

    colleagues = Vector{Vector{Int}}(undef, nboxes_total)
    colleagues[1] = [1]
    for ibox in 2:nboxes_total
        colleagues[ibox] = collect(2:nboxes_total)
    end

    return BoxDMK.BoxTree(
        ndim,
        1,
        centers,
        [1.0, 0.5],
        parent,
        children,
        colleagues,
        vcat(0, ones(Int, nchildren)),
        LegendreBasis(),
        6,
    )
end

@testset "Local Tables" begin
    basis = LegendreBasis()
    norder = 6
    ndim = 3
    nlevels = 3
    deltas = [0.01, 0.001]
    boxsizes = [1.0, 0.5, 0.25, 0.125]

    tables = BoxDMK.build_local_tables(
        LaplaceKernel(),
        basis,
        norder,
        ndim,
        deltas,
        boxsizes,
        nlevels,
    )

    @test size(tables.tab) == (norder, norder, 3, length(deltas), nlevels + 1)
    @test size(tables.tabx) == size(tables.tab)
    @test size(tables.tabxx) == size(tables.tab)
    @test size(tables.ind) == (2, norder + 1, 3, length(deltas), nlevels + 1)

    @test all(isfinite, tables.tab)
    @test all(isfinite, tables.tabx)
    @test all(isfinite, tables.tabxx)

    for ilevel in 1:(nlevels + 1), idelta in eachindex(deltas), ioffset in 1:3
        for j in 1:norder
            first_nonzero = tables.ind[1, j, ioffset, idelta, ilevel]
            last_nonzero = tables.ind[2, j, ioffset, idelta, ilevel]
            @test (first_nonzero == 0 && last_nonzero == -1) || (1 <= first_nonzero <= last_nonzero <= norder)
        end

        first_active = tables.ind[1, norder + 1, ioffset, idelta, ilevel]
        last_active = tables.ind[2, norder + 1, ioffset, idelta, ilevel]
        @test (first_active == 0 && last_active == -1) || (1 <= first_active <= last_active <= norder)
    end

    for ilevel in 1:(nlevels + 1), idelta in eachindex(deltas)
        constant_response = vec(sum(@view(tables.tab[:, :, 2, idelta, ilevel]); dims = 1))
        @test all(constant_response .> 0)
    end
end

@testset "Local Application" begin
    tree = _local_two_level_tree()
    lists = BoxDMK.build_interaction_lists(tree)
    norder = tree.norder
    npbox = norder^tree.ndim
    deltas = [0.01, 0.005]
    weights = [0.7, 0.3]

    tables = BoxDMK.build_local_tables(
        LaplaceKernel(),
        tree.basis,
        tree.norder,
        tree.ndim,
        deltas,
        tree.boxsize,
        tree.nlevels,
    )

    fvals = ones(Float64, 1, npbox, BoxDMK.nboxes(tree))
    pot = zeros(Float64, size(fvals))

    BoxDMK.apply_local!(pot, tree, fvals, tables, lists, deltas, weights)

    leaf_boxes = collect(BoxDMK.leaves(tree))
    @test !isempty(leaf_boxes)
    @test all(sum(@view(pot[1, :, ibox])) > 0 for ibox in leaf_boxes)
    @test all(all(@view(pot[1, :, ibox]) .>= 0) for ibox in leaf_boxes)
end
