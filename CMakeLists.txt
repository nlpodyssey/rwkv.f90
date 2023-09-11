cmake_minimum_required(VERSION 3.12)
project(rwkv)

enable_language(Fortran)
cmake_policy(SET CMP0076 NEW)

if(NOT CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER gfortran)
endif()

message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER}")

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -Wextra -Wpedantic -Wno-compare-reals -std=f2018 -fall-intrinsics -ffree-line-length-none")
set(CMAKE_Fortran_FLAGS_DEBUG "-g")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-loops -fopenmp -lpthread -flto -fprefetch-loop-arrays -fno-protect-parens -fno-semantic-interposition")

add_library(rwkv STATIC)
add_executable(rwkv-cli main.f90)

# BLAS Configuration
option(USE_BLAS "Use a BLAS library for matrix operations" ON)
if(USE_BLAS)
  find_package(BLAS)

  if(BLAS_FOUND AND BLAS_LIBRARIES)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fexternal-blas")
    target_link_libraries(rwkv ${BLAS_LIBRARIES})
    target_link_libraries(rwkv-cli rwkv ${BLAS_LIBRARIES})

    if(BLAS_LIBRARIES MATCHES "Accelerate")
      target_link_options(rwkv PUBLIC -framework accelerate)
      target_link_options(rwkv-cli PUBLIC -framework accelerate)
    endif()
  else()
    message(WARNING "BLAS not found.")
    target_link_libraries(rwkv-cli rwkv)
  endif()
else()
  message(STATUS "BLAS not requested.")
  target_link_libraries(rwkv-cli rwkv)
endif()

# Source files
target_sources(
        rwkv
        PRIVATE
        modules/mod_arr_ops_broadcasting.f90
        modules/mod_channel_mix.f90
        modules/mod_command_arguments.f90
        modules/mod_essentials.f90
        modules/mod_functions.f90
        modules/mod_generation.f90
        modules/mod_inference.f90
        modules/mod_layer_norm.f90
        modules/mod_linear.f90
        modules/mod_prompt_utils.f90
        modules/mod_readline.f90
        modules/mod_real_precision.f90
        modules/mod_rwkv_layer.f90
        modules/mod_rwkv_lm.f90
        modules/mod_state.f90
        modules/mod_stats.f90
        modules/mod_time_mix.f90
        modules/mod_timer.f90
        modules/mod_token_shift.f90
        modules/mod_trie_tokenizer.f90
        PUBLIC
        lib_rwkv.f90
)
