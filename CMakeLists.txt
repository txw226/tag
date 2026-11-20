cmake_minimum_required(VERSION 3.26)
project(tag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Local builds tune for this CPU; CI passes -DTAG_NATIVE=OFF for portable binaries.
option(TAG_NATIVE "Compile with -march=native" ON)
option(TAG_BUILD_BENCHMARKS "Build the nanobench microbenchmarks (fetches nanobench)" OFF)

# Flags shared by every target. Release already gives -O3. No -ffast-math: it breaks NaN checks
# and can stop the Kepler iteration converging.
add_library(tag_flags INTERFACE)
target_compile_options(tag_flags INTERFACE -fno-math-errno $<$<BOOL:${TAG_NATIVE}>:-march=native>)

# The C++ core: plain C++ types only, no Python.
add_library(tag_core STATIC cpp/src/propagation/kepler.cpp cpp/src/propagation/two_body.cpp
                            cpp/src/propagation/elements.cpp)
target_include_directories(tag_core PUBLIC cpp/include)
target_link_libraries(tag_core PUBLIC tag_flags)
set_target_properties(tag_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link-time optimization, so core functions can inline into the bindings and benchmarks.
include(CheckIPOSupported)
check_ipo_supported(RESULT tag_ipo_supported OUTPUT tag_ipo_output)
set_target_properties(tag_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${tag_ipo_supported})

# The Python extension, built only by scikit-build-core (uv sync / uv build).
if(SKBUILD)
  find_package(Python 3.14 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)
  # NOMINSIZE: optimize the bindings for speed (nanobind defaults to -Os).
  nanobind_add_module(_propagation NOMINSIZE cpp/bindings/propagation.cpp)
  target_link_libraries(_propagation PRIVATE tag_core)
  set_target_properties(_propagation PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${tag_ipo_supported})
  install(TARGETS _propagation LIBRARY DESTINATION tag/propagation)
endif()

if(TAG_BUILD_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(nanobench)
  add_executable(tag_bench cpp/bench/main.cpp cpp/bench/kepler_bench.cpp cpp/bench/two_body_bench.cpp)
  target_link_libraries(tag_bench PRIVATE tag_core nanobench)
  set_target_properties(tag_bench PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${tag_ipo_supported})
endif()
