const _FORTRAN_DEBUG_SYMBOL_CACHE = Dict{Symbol, Bool}()

function _has_fortran_debug_symbol(symbol::Symbol)
    return get!(_FORTRAN_DEBUG_SYMBOL_CACHE, symbol) do
        handle = Libdl.dlopen(_fortran_solve_libboxdmk_path())
        ptr = Libdl.dlsym_e(handle, symbol)
        Libdl.dlclose(handle)
        ptr != C_NULL
    end
end

function reset_fortran_debug!()
    _has_fortran_debug_symbol(:boxdmk_debug_reset) || return nothing
    ccall((:boxdmk_debug_reset, _fortran_solve_libboxdmk_path()), Cvoid, ())
    return nothing
end

function _fortran_debug_meta()
    _has_fortran_debug_symbol(:boxdmk_debug_get_meta) || return (
        nd = 0,
        npbox = 0,
        ncbox = 0,
        nboxes = 0,
        has_step2 = false,
        has_step3 = false,
        has_step6 = false,
        has_step7 = false,
        has_step8 = false,
        has_step9 = false,
    )

    nd = Ref{Cint}(0)
    npbox = Ref{Cint}(0)
    ncbox = Ref{Cint}(0)
    nboxes = Ref{Cint}(0)
    has_step2 = Ref{Cint}(0)
    has_step3 = Ref{Cint}(0)
    has_step6 = Ref{Cint}(0)
    has_step7 = Ref{Cint}(0)
    has_step8 = Ref{Cint}(0)
    has_step9 = Ref{Cint}(0)

    ccall(
        (:boxdmk_debug_get_meta, _fortran_solve_libboxdmk_path()),
        Cvoid,
        (
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
            Ref{Cint},
        ),
        nd,
        npbox,
        ncbox,
        nboxes,
        has_step2,
        has_step3,
        has_step6,
        has_step7,
        has_step8,
        has_step9,
    )

    return (
        nd = Int(nd[]),
        npbox = Int(npbox[]),
        ncbox = Int(ncbox[]),
        nboxes = Int(nboxes[]),
        has_step2 = has_step2[] != 0,
        has_step3 = has_step3[] != 0,
        has_step6 = has_step6[] != 0,
        has_step7 = has_step7[] != 0,
        has_step8 = has_step8[] != 0,
        has_step9 = has_step9[] != 0,
    )
end

function _copy_fortran_debug_array(dims::NTuple{3, Int}, symbol::Val{:boxdmk_debug_copy_step2_pot})
    length = prod(dims)
    data = Vector{Float64}(undef, length)
    ccall((:boxdmk_debug_copy_step2_pot, _fortran_solve_libboxdmk_path()), Cvoid, (Ptr{Cdouble}, Cint), data, Cint(length))
    return reshape(data, dims...)
end

function _copy_fortran_debug_array(dims::NTuple{3, Int}, symbol::Val{:boxdmk_debug_copy_step3_proxycharge})
    length = prod(dims)
    data = Vector{Float64}(undef, length)
    ccall((:boxdmk_debug_copy_step3_proxycharge, _fortran_solve_libboxdmk_path()), Cvoid, (Ptr{Cdouble}, Cint), data, Cint(length))
    return reshape(data, dims...)
end

function _copy_fortran_debug_array(dims::NTuple{3, Int}, symbol::Val{:boxdmk_debug_copy_step6_proxypotential})
    length = prod(dims)
    data = Vector{Float64}(undef, length)
    ccall((:boxdmk_debug_copy_step6_proxypotential, _fortran_solve_libboxdmk_path()), Cvoid, (Ptr{Cdouble}, Cint), data, Cint(length))
    return reshape(data, dims...)
end

function _copy_fortran_debug_array(dims::NTuple{3, Int}, symbol::Val{:boxdmk_debug_copy_step7_pot})
    length = prod(dims)
    data = Vector{Float64}(undef, length)
    ccall((:boxdmk_debug_copy_step7_pot, _fortran_solve_libboxdmk_path()), Cvoid, (Ptr{Cdouble}, Cint), data, Cint(length))
    return reshape(data, dims...)
end

function _copy_fortran_debug_array(dims::NTuple{3, Int}, symbol::Val{:boxdmk_debug_copy_step8_pot})
    length = prod(dims)
    data = Vector{Float64}(undef, length)
    ccall((:boxdmk_debug_copy_step8_pot, _fortran_solve_libboxdmk_path()), Cvoid, (Ptr{Cdouble}, Cint), data, Cint(length))
    return reshape(data, dims...)
end

function _copy_fortran_debug_array(dims::NTuple{3, Int}, symbol::Val{:boxdmk_debug_copy_step9_pot})
    length = prod(dims)
    data = Vector{Float64}(undef, length)
    ccall((:boxdmk_debug_copy_step9_pot, _fortran_solve_libboxdmk_path()), Cvoid, (Ptr{Cdouble}, Cint), data, Cint(length))
    return reshape(data, dims...)
end

function get_fortran_debug_snapshot()
    meta = _fortran_debug_meta()
    meta.nd > 0 || meta.nboxes > 0 || meta.npbox > 0 || meta.ncbox > 0 ||
        return (
            nd = 0,
            npbox = 0,
            ncbox = 0,
            nboxes = 0,
            step2_pot = nothing,
            step3_proxycharge = nothing,
            step6_proxypotential = nothing,
            step7_pot = nothing,
            step8_pot = nothing,
            step9_pot = nothing,
        )

    pot_dims = (meta.nd, meta.npbox, meta.nboxes)
    proxy_dims = (meta.ncbox, meta.nd, meta.nboxes)

    return (
        nd = meta.nd,
        npbox = meta.npbox,
        ncbox = meta.ncbox,
        nboxes = meta.nboxes,
        step2_pot = meta.has_step2 ? _copy_fortran_debug_array(pot_dims, Val(:boxdmk_debug_copy_step2_pot)) : nothing,
        step3_proxycharge = meta.has_step3 ? _copy_fortran_debug_array(proxy_dims, Val(:boxdmk_debug_copy_step3_proxycharge)) : nothing,
        step6_proxypotential = meta.has_step6 ? _copy_fortran_debug_array(proxy_dims, Val(:boxdmk_debug_copy_step6_proxypotential)) : nothing,
        step7_pot = meta.has_step7 ? _copy_fortran_debug_array(pot_dims, Val(:boxdmk_debug_copy_step7_pot)) : nothing,
        step8_pot = meta.has_step8 ? _copy_fortran_debug_array(pot_dims, Val(:boxdmk_debug_copy_step8_pot)) : nothing,
        step9_pot = meta.has_step9 ? _copy_fortran_debug_array(pot_dims, Val(:boxdmk_debug_copy_step9_pot)) : nothing,
    )
end
