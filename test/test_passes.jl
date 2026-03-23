using BoxDMK
using Test

function _two_level_tree()
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

    return BoxDMK.BoxTree(
        ndim,
        1,
        centers,
        [1.0, 0.5],
        parent,
        children,
        [Int[] for _ in 1:nboxes_total],
        vcat(0, ones(Int, nchildren)),
        LegendreBasis(),
        6,
    )
end

function _apply_child_transform(transmat, proxy, data, child, ndim)
    mats = ntuple(d -> view(transmat, :, :, d, child), ndim)
    out = similar(data)
    BoxDMK.tensor_product_apply!(out, mats, data, proxy.porder, ndim, size(data, 1))
    return out
end

@testset "Upward/Downward Passes" begin
    tree = _two_level_tree()
    proxy = BoxDMK.build_proxy_data(LegendreBasis(), tree.norder, 4, tree.ndim)
    ncbox = proxy.ncbox
    nd = 2
    nboxes_total = BoxDMK.nboxes(tree)

    @testset "Upward pass accumulates child proxy charges into the root" begin
        proxy_charges = zeros(Float64, ncbox, nd, nboxes_total)
        proxy_charges[:, :, 1] .= reshape(collect(1:(ncbox * nd)), ncbox, nd) ./ 10

        expected_root = copy(@view(proxy_charges[:, :, 1]))

        for child in 1:size(tree.children, 1)
            ibox = tree.children[child, 1]
            proxy_charges[:, :, ibox] .= reshape(collect(1:(ncbox * nd)) .+ 1000 * child, ncbox, nd)

            transformed = _apply_child_transform(
                proxy.c2p_transmat,
                proxy,
                Matrix(transpose(@view(proxy_charges[:, :, ibox]))),
                child,
                tree.ndim,
            )
            expected_root .+= transpose(transformed)
        end

        charges_before = copy(proxy_charges)
        BoxDMK.upward_pass!(proxy_charges, tree, proxy)

        @test proxy_charges[:, :, 1] ≈ expected_root
        @test proxy_charges[:, :, 2:end] == charges_before[:, :, 2:end]
    end

    @testset "Downward pass distributes root proxy potential to children" begin
        proxy_pot = zeros(Float64, ncbox, nd, nboxes_total)
        proxy_pot[:, :, 1] .= reshape(collect(1:(ncbox * nd)) .+ 5000, ncbox, nd) ./ 7

        expected_children = [
            reshape(collect(1:(ncbox * nd)) .+ 200 * child, ncbox, nd) ./ 13
            for child in 1:size(tree.children, 1)
        ]

        for child in 1:size(tree.children, 1)
            ibox = tree.children[child, 1]
            proxy_pot[:, :, ibox] .= expected_children[child]

            transformed = _apply_child_transform(
                proxy.p2c_transmat,
                proxy,
                Matrix(transpose(@view(proxy_pot[:, :, 1]))),
                child,
                tree.ndim,
            )
            expected_children[child] .+= transpose(transformed)
        end

        pot_before = copy(proxy_pot)
        BoxDMK.downward_pass!(proxy_pot, tree, proxy)

        @test proxy_pot[:, :, 1] == pot_before[:, :, 1]

        for child in 1:size(tree.children, 1)
            ibox = tree.children[child, 1]
            @test proxy_pot[:, :, ibox] ≈ expected_children[child]
        end
    end

    @testset "Tensor product workspace matches default path" begin
        data = reshape(collect(1.0:(ncbox * nd)), nd, ncbox)
        mats = ntuple(d -> view(proxy.p2c_transmat, :, :, d, 1), tree.ndim)
        out = similar(data)
        out_ws = similar(data)
        workspace = BoxDMK._tensor_apply_workspace(Float64, nd, proxy.porder, tree.ndim)

        BoxDMK.tensor_product_apply!(out, mats, data, proxy.porder, tree.ndim, nd)
        BoxDMK.tensor_product_apply!(out_ws, mats, data, proxy.porder, tree.ndim, nd, workspace)

        @test out_ws ≈ out
    end
end

@testset "Fortran Pass Hotpath Wrapper" begin
    if BoxDMK._FORTRAN_HOTPATHS_AVAILABLE[]
        tree = _two_level_tree()
        proxy = BoxDMK.build_proxy_data(LegendreBasis(), tree.norder, 4, tree.ndim)
        child = 3
        mats = BoxDMK._child_transfer_mats(proxy.c2p_transmat, tree.ndim, child)
        umat_nd = Array{Float64}(undef, proxy.porder, proxy.porder, tree.ndim)
        for d in 1:tree.ndim
            umat_nd[:, :, d] .= mats[d]
        end

        fin = collect(1.0:proxy.ncbox)
        expected = zeros(Float64, 1, proxy.ncbox)
        BoxDMK.tensor_product_apply!(expected, mats, reshape(fin, 1, :), proxy.porder, tree.ndim, 1)

        fout = fill(-7.0, proxy.ncbox)
        BoxDMK._f_tens_prod_trans!(fout, fin, umat_nd, tree.ndim, proxy.porder, proxy.porder, 0)
        @test fout ≈ vec(expected) atol = 1e-12 rtol = 1e-12

        accum = fill(3.5, proxy.ncbox)
        BoxDMK._f_tens_prod_trans!(accum, fin, umat_nd, tree.ndim, proxy.porder, proxy.porder, 1)
        @test accum ≈ vec(expected) .+ 3.5 atol = 1e-12 rtol = 1e-12
    else
        @test_skip "Fortran hotpaths unavailable"
    end
end
