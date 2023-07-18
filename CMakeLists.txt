cmake_minimum_required(VERSION 3.12)
project(rwkv)

enable_language(Fortran)

if(NOT DEFINED ENV{FC})
    set(ENV{FC} gfortran)
endif()

message(STATUS "Fortran compiler: $ENV{FC}")

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=f2018 -fall-intrinsics -ffree-line-length-none -O3 -ffast-math -funroll-loops -fopenmp -lpthread -flto -fprefetch-loop-arrays -march=native")

set(LIB_SOURCES
        src/modules/mod_essentials.f90
        src/modules/mod_real_precision.f90
        src/modules/mod_arr_ops_broadcasting.f90
        src/modules/mod_stats.f90
        src/modules/mod_functions.f90
        src/modules/mod_state.f90
        src/modules/mod_linear.f90
        src/modules/mod_layer_norm.f90
        src/modules/mod_channel_mix.f90
        src/modules/mod_time_mix.f90
        src/modules/mod_token_shift.f90
        src/modules/mod_rwkv_layer.f90
        src/modules/mod_rwkv_lm.f90
        src/modules/mod_trie_tokenizer.f90
        src/modules/mod_prompt_utils.f90
        src/modules/mod_generation.f90
        )

set(MAIN_SOURCES
        src/main.f90
        )

set(BLAS_LIBRARY "" CACHE STRING "External BLAS library to use")

message(STATUS "BLAS library: ${BLAS_LIBRARY}")

if(BLAS_LIBRARY STREQUAL "Accelerate")
    message(STATUS "Using Accelerate library")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fexternal-blas")
elseif(BLAS_LIBRARY STREQUAL "OpenBLAS")
    message(STATUS "Using OpenBLAS library")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fexternal-blas")
    # Specify the directory where OpenBLAS library is located.
    # Modify this according to your actual OpenBLAS installation path.
    link_directories(/opt/homebrew/opt/openblas/lib)
elseif(BLAS_LIBRARY STREQUAL "" OR BLAS_LIBRARY STREQUAL "Fortran")
    message(STATUS "No external BLAS library specified. Relying on compiler's native capabilities (slow).")
else()
    message(FATAL_ERROR "Invalid BLAS library choice. Available options (case sensitive): Accelerate, OpenBLAS")
endif()

add_library(${PROJECT_NAME}_lib ${LIB_SOURCES})
add_executable(${PROJECT_NAME} ${MAIN_SOURCES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)

if(BLAS_LIBRARY STREQUAL "Accelerate")
    target_link_options(${PROJECT_NAME}_lib PUBLIC -framework accelerate)
elseif(BLAS_LIBRARY STREQUAL "OpenBLAS")
    target_link_libraries(${PROJECT_NAME}_lib openblas)
endif()
