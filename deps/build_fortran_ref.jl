using Dates

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const VENDORED_ROOT = normpath(joinpath(@__DIR__, "boxdmk_fortran"))
const CALLBACK_BUILD_ROOT = normpath(joinpath(@__DIR__, "boxdmk_fortran", "build_callback"))
const HOT_BUILD_ROOT = normpath(joinpath(@__DIR__, "boxdmk_fortran", "build_hot"))
const OUTPUT_ROOT = normpath(joinpath(@__DIR__, "usr", "lib"))
const OUTPUT_LIB = normpath(joinpath(OUTPUT_ROOT, "libboxdmk.so"))
const OUTPUT_HOT_LIB = normpath(joinpath(OUTPUT_ROOT, "libboxdmk_hot.so"))

function require_tool(name::AbstractString)
    path = Sys.which(name)
    path === nothing && error("Required build tool not found in PATH: $name")
    return path
end

function run_checked(cmd::Cmd)
    println(">> ", cmd)
    run(cmd)
end

function built_library_path()
    return normpath(joinpath(CALLBACK_BUILD_ROOT, "libboxdmk.so"))
end

function built_hot_library_path()
    return normpath(joinpath(HOT_BUILD_ROOT, "libboxdmk.so"))
end

function main()
    require_tool("cmake")

    isdir(VENDORED_ROOT) || error("Vendored Fortran root not found at $VENDORED_ROOT")
    isfile(joinpath(VENDORED_ROOT, "CMakeLists.txt")) || error("Vendored CMakeLists.txt missing from $VENDORED_ROOT")

    mkpath(CALLBACK_BUILD_ROOT)
    mkpath(HOT_BUILD_ROOT)
    mkpath(OUTPUT_ROOT)

    # The callback-heavy tree build path is not stable with OpenMP enabled.
    # Build a callback-safe library and a separate solve/hotpath library.
    callback_configure_cmd = `cmake -S $VENDORED_ROOT -B $CALLBACK_BUILD_ROOT -DBOXDMK_ENABLE_OPENMP=OFF -DBOXDMK_USE_MKL=OFF`
    callback_build_cmd = `cmake --build $CALLBACK_BUILD_ROOT --target boxdmk -j`
    hot_configure_cmd = `cmake -S $VENDORED_ROOT -B $HOT_BUILD_ROOT -DBOXDMK_ENABLE_OPENMP=ON -DBOXDMK_USE_MKL=OFF`
    hot_build_cmd = `cmake --build $HOT_BUILD_ROOT --target boxdmk -j`

    run_checked(callback_configure_cmd)
    run_checked(callback_build_cmd)
    run_checked(hot_configure_cmd)
    run_checked(hot_build_cmd)

    libpath = built_library_path()
    hot_libpath = built_hot_library_path()
    isfile(libpath) || error("Expected built library not found at $libpath")
    isfile(hot_libpath) || error("Expected built hot library not found at $hot_libpath")
    cp(libpath, OUTPUT_LIB; force = true)
    cp(hot_libpath, OUTPUT_HOT_LIB; force = true)

    println("Built vendored Fortran library at $OUTPUT_LIB")
    println("Built vendored Fortran hot library at $OUTPUT_HOT_LIB")
    println("Build timestamp: $(Dates.now())")
    return (callback = OUTPUT_LIB, hot = OUTPUT_HOT_LIB)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
