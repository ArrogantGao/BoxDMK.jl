using BoxDMK
using Test

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
