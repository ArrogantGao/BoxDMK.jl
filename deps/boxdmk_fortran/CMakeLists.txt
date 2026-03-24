cmake_minimum_required(VERSION 3.18)

project(boxdmk LANGUAGES Fortran)

option(BOXDMK_USE_MKL "Try to use Intel MKL first; fall back to OpenBLAS" OFF)
option(BOXDMK_ENABLE_OPENMP "Enable OpenMP for Fortran when available" ON)

set(BOXDMK_CORE_SOURCES
  src/common/specialfunctions/besseljs3d.f
  src/common/specialfunctions/hank103.f
  src/common/specialfunctions/legeexps.f
  src/common/specialfunctions/chebexps.f
  src/common/prini_new.f
  src/common/fmmcommon2d.f
  src/common/lapack_f77.f
  src/common/cumsum.f
  src/common/hkrand.f
  src/common/dlaran.f
  src/common/voltab2d.f
  src/common/voltab3d.f
  src/common/polytens.f
  src/common/tree_data_routs.f
  src/common/tensor_prod_routs.f
  src/common/pts_tree.f
  src/common/tree_routs.f
  src/common/tree_vol_coeffs.f
  src/common/dmk_routs.f
  src/bdmk/sogapproximation/get_sognodes.f
  src/bdmk/sogapproximation/l2dsognodes.f
  src/bdmk/sogapproximation/l3dsognodes.f
  src/bdmk/sogapproximation/sl3dsognodes.f
  src/bdmk/sogapproximation/y2dsognodes.f
  src/bdmk/sogapproximation/y3dsognodes.f
  src/bdmk/bdmk_local_tables.f
  src/bdmk/bdmk_local.f
  src/bdmk/bdmk_pwterms.f
  src/bdmk/bdmk_pwrouts.f
  src/bdmk/boxfgt_md.f
  src/bdmk/bdmk_debug_store.f90
  src/bdmk/bdmk4.f
)

set(BOXDMK_C_API_SOURCES
  src/bdmk/bdmk_c_api.f90
)

add_library(boxdmk SHARED ${BOXDMK_CORE_SOURCES} ${BOXDMK_C_API_SOURCES})
set_target_properties(boxdmk PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME boxdmk
)

add_executable(int2-bdmk test/bdmk/testbdmk.f ${BOXDMK_CORE_SOURCES})
add_executable(bench-fortran-same-problem benchmark/bench_fortran_same_problem.f90)

# Match the current Makefile tuning for GNU/Intel compilers.
target_compile_options(boxdmk PRIVATE
  $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-O3;-std=legacy;-funroll-loops;-march=native>
  $<$<COMPILE_LANG_AND_ID:Fortran,Intel>:-O3;-funroll-loops>
  $<$<COMPILE_LANG_AND_ID:Fortran,IntelLLVM>:-O3;-funroll-loops>
)

target_compile_options(int2-bdmk PRIVATE
  $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-O3;-std=legacy;-funroll-loops;-march=native>
  $<$<COMPILE_LANG_AND_ID:Fortran,Intel>:-O3;-funroll-loops>
  $<$<COMPILE_LANG_AND_ID:Fortran,IntelLLVM>:-O3;-funroll-loops>
)

target_compile_options(bench-fortran-same-problem PRIVATE
  $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-O3;-std=legacy;-funroll-loops;-march=native>
  $<$<COMPILE_LANG_AND_ID:Fortran,Intel>:-O3;-funroll-loops>
  $<$<COMPILE_LANG_AND_ID:Fortran,IntelLLVM>:-O3;-funroll-loops>
)

set(_blas_ok FALSE)
if(BOXDMK_USE_MKL)
  set(BLA_VENDOR Intel10_64lp_seq)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    set(_blas_ok TRUE)
    message(STATUS "Found BLAS with MKL preference: ${BLAS_LIBRARIES}")
  else()
    message(WARNING "BOXDMK_USE_MKL=ON but MKL BLAS was not found; falling back to OpenBLAS")
    unset(BLAS_LIBRARIES CACHE)
    unset(BLAS_LIBRARIES)
    unset(BLAS_FOUND)
  endif()
endif()

if(NOT _blas_ok)
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)
  message(STATUS "Using OpenBLAS/BLAS libs: ${BLAS_LIBRARIES}")
endif()

target_link_libraries(boxdmk PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(int2-bdmk PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(bench-fortran-same-problem PRIVATE boxdmk ${BLAS_LIBRARIES})

if(BOXDMK_ENABLE_OPENMP)
  find_package(OpenMP QUIET COMPONENTS Fortran)
  if(OpenMP_Fortran_FOUND)
    target_link_libraries(boxdmk PRIVATE OpenMP::OpenMP_Fortran)
    target_link_libraries(int2-bdmk PRIVATE OpenMP::OpenMP_Fortran)
    target_link_libraries(bench-fortran-same-problem PRIVATE OpenMP::OpenMP_Fortran)
  endif()
endif()

enable_testing()
add_test(NAME testbdmk COMMAND int2-bdmk)
