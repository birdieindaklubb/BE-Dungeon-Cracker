cmake_minimum_required(VERSION 3.20)

project(pe16015_dungeon_cracker LANGUAGES CXX)

add_executable(pe16015_dungeon_cracker
    src/dungeon_observation.cpp
    src/main.cpp
)

target_compile_features(pe16015_dungeon_cracker PRIVATE cxx_std_20)

if (MSVC)
    target_compile_options(pe16015_dungeon_cracker PRIVATE /W4 /permissive-)
else()
    target_compile_options(pe16015_dungeon_cracker PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Keep the data-parallel scanner isolated from the portable executable.  The
# main program dispatches only after a runtime AVX2 capability check.
include(CheckCXXCompilerFlag)
if (MSVC)
    check_cxx_compiler_flag("/arch:AVX2" PE16015_HAS_AVX2_FLAG)
    set(PE16015_AVX2_FLAG "/arch:AVX2")
else()
    check_cxx_compiler_flag("-mavx2" PE16015_HAS_AVX2_FLAG)
    set(PE16015_AVX2_FLAG "-mavx2")
endif()

if (PE16015_HAS_AVX2_FLAG)
    add_library(pe16015_dungeon_cracker_avx2 OBJECT src/avx2.cpp)
    target_compile_features(pe16015_dungeon_cracker_avx2 PRIVATE cxx_std_20)
    target_compile_options(pe16015_dungeon_cracker_avx2 PRIVATE ${PE16015_AVX2_FLAG})
    if (MSVC)
        target_compile_options(pe16015_dungeon_cracker_avx2 PRIVATE /W4 /permissive-)
    else()
        target_compile_options(pe16015_dungeon_cracker_avx2 PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_sources(pe16015_dungeon_cracker PRIVATE
        $<TARGET_OBJECTS:pe16015_dungeon_cracker_avx2>)
    target_compile_definitions(pe16015_dungeon_cracker PRIVATE PE16015_HAVE_AVX2=1)
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(pe16015_dungeon_cracker PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(pe16015_dungeon_cracker PRIVATE PE16015_HAS_OPENMP=1)
    if (TARGET pe16015_dungeon_cracker_avx2)
        target_link_libraries(pe16015_dungeon_cracker_avx2 PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
