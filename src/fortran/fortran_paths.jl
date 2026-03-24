const _EXTERNAL_FORTRAN_LIBBOXDMK_PATH = normpath("/mnt/home/xgao1/codes/boxdmk/build/libboxdmk.so")

_vendored_fortran_root() = normpath(joinpath(_PACKAGE_ROOT, "deps", "boxdmk_fortran"))
_vendored_fortran_build_root() = normpath(joinpath(_PACKAGE_ROOT, "deps", "usr"))
_vendored_fortran_libdir() = joinpath(_vendored_fortran_build_root(), "lib")

function _fortran_library_candidates(; debug::Bool = false)
    basename = debug ? "libboxdmk_debug.so" : "libboxdmk.so"

    candidates = String[
        normpath(joinpath(_vendored_fortran_libdir(), basename)),
    ]

    if !debug
        push!(candidates, _EXTERNAL_FORTRAN_LIBBOXDMK_PATH)
    end

    return candidates
end

function _fortran_solve_library_candidates(; debug::Bool = false)
    debug && return _fortran_library_candidates(; debug = true)

    return String[
        normpath(joinpath(_vendored_fortran_libdir(), "libboxdmk_hot.so")),
        _EXTERNAL_FORTRAN_LIBBOXDMK_PATH,
        normpath(joinpath(_vendored_fortran_libdir(), "libboxdmk.so")),
    ]
end

function _resolve_fortran_library_path(; debug::Bool = false, must_exist::Bool = false)
    candidates = _fortran_library_candidates(; debug = debug)

    for candidate in candidates
        isfile(candidate) && return candidate
    end

    must_exist && throw(ArgumentError("Fortran library not found. Checked: $(join(candidates, ", "))"))
    return first(candidates)
end

function _resolve_fortran_solve_library_path(; debug::Bool = false, must_exist::Bool = false)
    candidates = _fortran_solve_library_candidates(; debug = debug)

    for candidate in candidates
        isfile(candidate) && return candidate
    end

    must_exist && throw(ArgumentError("Fortran solve library not found. Checked: $(join(candidates, ", "))"))
    return first(candidates)
end
