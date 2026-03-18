const SOG_DATA_DIR = joinpath(dirname(@__DIR__), "data", "sog")
const SOG_TABLE_CACHE = Dict{String, Any}()

function _sog_filename(kernel::LaplaceKernel, ndim::Int)
    ndim == 3 || throw(ArgumentError("SOG tables are only available for 3D kernels"))
    return "laplace_3d.jld2"
end

function _sog_filename(kernel::YukawaKernel, ndim::Int)
    ndim == 3 || throw(ArgumentError("SOG tables are only available for 3D kernels"))
    return "yukawa_3d.jld2"
end

function _sog_filename(kernel::SqrtLaplaceKernel, ndim::Int)
    ndim == 3 || throw(ArgumentError("SOG tables are only available for 3D kernels"))
    return "sqrtlaplace_3d.jld2"
end

function _load_sog_table(path::String)
    return get!(SOG_TABLE_CACHE, path) do
        isfile(path) || throw(ArgumentError("missing SOG data file: $path"))
        JLD2.load(path, "tables")
    end
end

function _select_precision_key(tables, eps::Float64)
    levels = sort([(parse(Float64, key), key) for key in keys(tables)])
    isempty(levels) && throw(ArgumentError("SOG table is empty"))

    eligible = filter(entry -> entry[1] <= eps, levels)
    selected = isempty(eligible) ? first(levels) : last(eligible)
    return selected[2]
end

function load_sog_nodes(kernel::AbstractKernel, ndim::Int, eps::Float64)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    isfinite(eps) || throw(ArgumentError("eps must be finite"))

    filename = _sog_filename(kernel, ndim)
    path = joinpath(SOG_DATA_DIR, filename)
    tables = _load_sog_table(path)
    key = _select_precision_key(tables, eps)
    entry = tables[key]

    return SOGNodes(copy(entry.weights), copy(entry.deltas), entry.r0)
end
