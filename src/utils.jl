nboxes(tree::BoxTree) = size(tree.centers, 2)

isleaf(tree::BoxTree, ibox::Integer) = all(iszero, @view tree.children[:, ibox])

nleaves(tree::BoxTree) = count(ibox -> isleaf(tree, ibox), 1:nboxes(tree))

leaves(tree::BoxTree) = (ibox for ibox in 1:nboxes(tree) if isleaf(tree, ibox))

npbox(norder::Integer, ndim::Integer) = norder^ndim

nhess(ndim::Integer) = ndim * (ndim + 1) ÷ 2

boxes_at_level(tree::BoxTree, level::Integer) = findall(==(level), tree.level)
