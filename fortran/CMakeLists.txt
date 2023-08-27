cmake_minimum_required(VERSION 3.12)
project(rwkv)

enable_language(Fortran)
cmake_policy(SET CMP0076 NEW)

if(NOT DEFINED ENV{FC})
  set(ENV{FC} gfortran)
endif()

message(STATUS "Fortran compiler: $ENV{FC}")

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -Wextra -Wpedantic -Wno-compare-reals -std=f2018 -fall-intrinsics -ffree-line-length-none")
set(CMAKE_Fortran_FLAGS_DEBUG "-g")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ffast-math -funroll-loops -fopenmp -lpthread -flto -fprefetch-loop-arrays -fno-protect-parens -fno-semantic-interposition")

find_package(BLAS)

if(BLAS_FOUND)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fexternal-blas")
else()
  message(WARNING "BLAS not found. Relying on compiler's native capabilities (slow).")
endif()

add_library(rwkv STATIC)

target_sources(
  rwkv
  PRIVATE
    modules/mod_arr_ops_broadcasting.f90
    modules/mod_channel_mix.f90
    modules/mod_essentials.f90
    modules/mod_functions.f90
    modules/mod_generation.f90
    modules/mod_layer_norm.f90
    modules/mod_linear.f90
    modules/mod_prompt_utils.f90
    modules/mod_real_precision.f90
    modules/mod_rwkv_layer.f90
    modules/mod_rwkv_lm.f90
    modules/mod_state.f90
    modules/mod_stats.f90
    modules/mod_time_mix.f90
    modules/mod_token_shift.f90
    modules/mod_trie_tokenizer.f90
  PUBLIC
    lib_rwkv.f90
)

add_executable(rwkv-cli main.f90)

target_link_libraries(rwkv ${BLAS_LIBRARIES})
target_link_libraries(rwkv-cli rwkv ${BLAS_LIBRARIES})

if(BLA_VENDOR STREQUAL "Apple")
    target_link_options(rwkv PUBLIC -framework accelerate)
    target_link_options(rwkv-cli PUBLIC -framework accelerate)
endif()
