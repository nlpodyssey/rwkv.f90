cmake_minimum_required(VERSION 3.12)
project(rwkv)

enable_language(Fortran)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=f2018 -fall-intrinsics -ffree-line-length-none -O3 -ffast-math -funroll-loops -fopenmp -fexternal-blas")

link_directories(/opt/homebrew/opt/openblas/lib) # Set OpenBLAS library path

set(LIB_SOURCES
        src/modules/mod_essentials.f90
        src/modules/mod_real_precision.f90
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

add_library(${PROJECT_NAME}_lib ${LIB_SOURCES})

target_link_libraries(${PROJECT_NAME}_lib openblas) # Set OpenBLAS library path

set(MAIN_SOURCES
        src/main.f90
        )

add_executable(${PROJECT_NAME} ${MAIN_SOURCES})

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)
