cmake_minimum_required(VERSION 3.22)
project(liyab VERSION 0.1.0 LANGUAGES C CXX)

if(APPLE)
  enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --------------------------------------------------------------------------
# Target architecture
# --------------------------------------------------------------------------
set(LIYAB_ARM64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" OR ANDROID_ABI STREQUAL "arm64-v8a"
   OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
  set(LIYAB_ARM64 ON)
endif()

# --------------------------------------------------------------------------
# Feature flags
# --------------------------------------------------------------------------
set(_liyab_android OFF)
if(ANDROID)
  set(_liyab_android ON)
endif()

option(LIYAB_USE_NEON "ARM NEON SIMD kernels (arm64 only)" ${LIYAB_ARM64})
option(LIYAB_ARM_DOTPROD "Target ARMv8.2-A dot-product (SDOT) for the INT8/INT4 kernels" ${LIYAB_ARM64})
option(LIYAB_USE_METAL "Apple GPU backend (Metal)" ${APPLE})
option(LIYAB_USE_VULKAN "Vulkan runtime probe (Adreno / Mali / Xclipse)" ${_liyab_android})
option(LIYAB_USE_QNN "Qualcomm QNN HTP runtime probe" ${_liyab_android})
option(LIYAB_USE_NEUROPILOT "MediaTek NeuroPilot runtime probe" ${_liyab_android})
option(LIYAB_BUILD_SHARED "Build libliyab as a shared library" ON)
option(LIYAB_BUILD_CLI "Build the liyab-cli tool" ON)
option(LIYAB_ENABLE_EXPERIMENTAL "Experimental modules: early exit, io_uring direct I/O, head pruning" OFF)

set(_liyab_top_level OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(_liyab_top_level ON)
endif()
option(LIYAB_BUILD_TESTS "Build unit tests" ${_liyab_top_level})

if(LIYAB_USE_METAL AND NOT APPLE)
  message(FATAL_ERROR "LIYAB_USE_METAL requires an Apple platform")
endif()
if(LIYAB_USE_NEON AND NOT LIYAB_ARM64)
  message(STATUS "liyab: LIYAB_USE_NEON ignored on ${CMAKE_SYSTEM_PROCESSOR}")
  set(LIYAB_USE_NEON OFF)
endif()

# --------------------------------------------------------------------------
# Core library (object library shared by libliyab and the tests)
# --------------------------------------------------------------------------
add_library(liyab_core OBJECT
  src/core/types.cpp
  src/core/log.cpp
  src/core/quant.cpp
  src/core/quant_ext.cpp
  src/core/quant_lowbit.cpp
  src/core/thread_pool.cpp
  src/core/mmap_loader.cpp
  src/core/device_detect.cpp
  src/core/power_manager.cpp
  src/core/kv_cache.cpp
  src/core/triple_buffer_loader.cpp
  src/core/tokenizer.cpp
  src/core/sampling.cpp
  src/core/transformer.cpp
  src/core/expert_store.cpp
  src/core/direct_io.cpp
  src/core/speculative_decoder.cpp
  src/core/engine.cpp
  src/backends/cpu/cpu_backend.cpp
  src/backends/qnn/qnn_backend.cpp
  src/backends/neuropilot/neuropilot_backend.cpp
  src/backends/vulkan/vulkan_backend.cpp
  src/c_api/liyab_c_api.cpp
)
if(LIYAB_ENABLE_EXPERIMENTAL)
  target_sources(liyab_core PRIVATE
    src/experimental/early_exit.cpp
    src/experimental/egls.cpp
    src/experimental/jit_unpacker.cpp
    src/experimental/tdss.cpp
    src/experimental/kv_dedup.cpp
    src/experimental/head_pruner.cpp
    src/experimental/io_uring_loader.cpp)
  target_compile_definitions(liyab_core PUBLIC LIYAB_ENABLE_EXPERIMENTAL=1)
endif()
if(LIYAB_USE_METAL)
  target_sources(liyab_core PRIVATE src/backends/metal/metal_backend.mm)
  set_source_files_properties(src/backends/metal/metal_backend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  target_sources(liyab_core PRIVATE src/backends/metal/metal_unavailable.cpp)
endif()

target_include_directories(liyab_core
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${PROJECT_SOURCE_DIR}/src)

foreach(flag NEON METAL VULKAN QNN NEUROPILOT)
  if(LIYAB_USE_${flag})
    target_compile_definitions(liyab_core PUBLIC LIYAB_USE_${flag}=1)
  endif()
endforeach()

# Warnings. -ffast-math is deliberately never used: the fp16 conversions and
# softmax rely on IEEE semantics.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(liyab_core PRIVATE -Wall -Wextra -Wno-unused-parameter
    $<$<CONFIG:Release>:-ffunction-sections -fdata-sections>)
endif()

# ARM SIMD. On AArch64 (arm64-v8a, Apple arm64) NEON/ASIMD is part of the base
# ISA: there is no -mfpu=neon switch (that exists only for 32-bit ARMv7), and
# the compiler defines __ARM_NEON by itself. What we do configure:
#  * LIYAB_USE_NEON=1   selects the hand-written intrinsics paths.
#  * -march=armv8.2-a+dotprod+fp16 unlocks SDOT (4x int8 MACs per lane) used
#    by the Q4_0/Q8_0 kernels. Every Snapdragon 8 Gen 1+, Dimensity 9000+,
#    Tensor and Apple A13+/M1+ core supports it. Apple's default macOS arm64
#    target (apple-m1) already includes it, so only mobile targets get the flag.
if(LIYAB_USE_NEON AND LIYAB_ARM_DOTPROD AND (ANDROID OR IOS OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  target_compile_options(liyab_core PRIVATE -march=armv8.2-a+dotprod+fp16)
endif()

find_package(Threads REQUIRED)
target_link_libraries(liyab_core PUBLIC Threads::Threads)
if(ANDROID)
  target_link_libraries(liyab_core PUBLIC log dl)
elseif(UNIX AND NOT APPLE)
  target_link_libraries(liyab_core PUBLIC dl)
endif()
if(APPLE)
  # Foundation/objc: thermal state via NSProcessInfo; Metal: GPU backend.
  target_link_libraries(liyab_core PUBLIC "-framework Foundation" objc)
  if(LIYAB_USE_METAL)
    target_link_libraries(liyab_core PUBLIC "-framework Metal")
  endif()
endif()

if(LIYAB_USE_VULKAN)
  # Compute shaders: one SPIR-V module per weight type, embedded at build time.
  file(GLOB _liyab_glslc_hints "${ANDROID_NDK}/shader-tools/*")
  find_program(LIYAB_GLSLC glslc HINTS ${_liyab_glslc_hints} $ENV{VULKAN_SDK}/bin)
  if(NOT LIYAB_GLSLC)
    message(FATAL_ERROR "LIYAB_USE_VULKAN=ON needs glslc (NDK shader-tools or the Vulkan SDK)")
  endif()
  set(_liyab_shader_src ${PROJECT_SOURCE_DIR}/src/backends/vulkan/shaders/matvec.comp)
  set(_liyab_spv_names)
  set(_liyab_spv_files)
  foreach(type F32 F16 Q4_0 Q4_1 Q8_0)
    string(TOLOWER ${type} name)
    set(spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/matvec_${name}.spv)
    add_custom_command(OUTPUT ${spv}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${LIYAB_GLSLC} -fshader-stage=compute --target-env=vulkan1.1 -O -DTYPE_${type} ${_liyab_shader_src} -o ${spv}
      DEPENDS ${_liyab_shader_src} VERBATIM)
    list(APPEND _liyab_spv_names ${name})
    list(APPEND _liyab_spv_files ${spv})
  endforeach()
  # Native-layout kernels (weights read in place from shared memory).
  set(_liyab_native_src ${PROJECT_SOURCE_DIR}/src/backends/vulkan/shaders/matvec_native.comp)
  foreach(type F32 F16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 IQ4_NL MXFP4 Q4_K Q5_K Q6_K Q2_K Q3_K IQ4_XS IQ2_XXS IQ2_XS IQ2_S
               IQ3_XXS IQ3_S IQ1_S IQ1_M TQ2_0)
    string(TOLOWER ${type} name)
    set(spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/native_${name}.spv)
    add_custom_command(OUTPUT ${spv}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${LIYAB_GLSLC} -fshader-stage=compute --target-env=vulkan1.1 -O -DTYPE_${type} ${_liyab_native_src} -o ${spv}
      DEPENDS ${_liyab_native_src} VERBATIM)
    list(APPEND _liyab_spv_names native_${name})
    list(APPEND _liyab_spv_files ${spv})
  endforeach()
  set(_liyab_shader_header ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_shaders.h)
  add_custom_command(OUTPUT ${_liyab_shader_header}
    COMMAND ${CMAKE_COMMAND} -DOUT=${_liyab_shader_header} "-DNAMES=${_liyab_spv_names}" "-DINPUTS=${_liyab_spv_files}"
            -P ${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake
    DEPENDS ${_liyab_spv_files} ${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake VERBATIM)
  target_sources(liyab_core PRIVATE ${_liyab_shader_header})
  target_include_directories(liyab_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

  # Headers only: the loader is opened with dlopen at runtime. The NDK ships
  # <vulkan/vulkan.h> in its sysroot; elsewhere use the Vulkan-Headers package.
  include(CheckIncludeFileCXX)
  check_include_file_cxx(vulkan/vulkan.h LIYAB_HAVE_VULKAN_H)
  if(NOT LIYAB_HAVE_VULKAN_H)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
      target_include_directories(liyab_core PRIVATE ${Vulkan_INCLUDE_DIRS})
    else()
      message(FATAL_ERROR "LIYAB_USE_VULKAN=ON but <vulkan/vulkan.h> was not found")
    endif()
  endif()
endif()

# --------------------------------------------------------------------------
# libliyab
# --------------------------------------------------------------------------
if(LIYAB_BUILD_SHARED)
  add_library(liyab SHARED)
else()
  add_library(liyab STATIC)
endif()
# Linking an OBJECT library pulls its objects into libliyab (and only there)
# plus its usage requirements (include dirs, definitions, system libraries).
target_link_libraries(liyab PUBLIC liyab_core)
set_target_properties(liyab PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

if(LIYAB_BUILD_SHARED)
  # Only LIYAB_API / LIYAB_C_API symbols are exported.
  set_target_properties(liyab_core PROPERTIES CXX_VISIBILITY_PRESET hidden OBJCXX_VISIBILITY_PRESET hidden
                                              VISIBILITY_INLINES_HIDDEN ON)
  if(ANDROID)
    # 16 KB page alignment is required for Android 15+ devices with 16 KB pages.
    target_link_options(liyab PRIVATE -Wl,-z,max-page-size=16384 -Wl,--gc-sections -Wl,--exclude-libs,ALL)
  elseif(UNIX AND NOT APPLE)
    target_link_options(liyab PRIVATE -Wl,--gc-sections)
  elseif(APPLE)
    target_link_options(liyab PRIVATE -Wl,-dead_strip)
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS liyab LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/liyab DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# --------------------------------------------------------------------------
# CLI and tests
# --------------------------------------------------------------------------
if(LIYAB_BUILD_CLI)
  add_executable(liyab-cli tools/liyab_cli.cpp)
  target_link_libraries(liyab-cli PRIVATE liyab)
  if(ANDROID)
    target_link_options(liyab-cli PRIVATE -Wl,-z,max-page-size=16384)
  endif()
endif()

if(LIYAB_BUILD_TESTS)
  enable_testing()
  set(_liyab_tests test_mmap test_device_detect test_engine test_kv_cache test_backends)
  if(LIYAB_ENABLE_EXPERIMENTAL)
    list(APPEND _liyab_tests test_experimental)
  endif()
  foreach(name ${_liyab_tests})
    add_executable(${name} tests/${name}.cpp tests/test_model.cpp)
    # Tests use internal headers, so they link the object library directly.
    target_link_libraries(${name} PRIVATE liyab_core)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
    target_compile_definitions(${name} PRIVATE LIYAB_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
  # Kernel throughput benchmark (internal headers, like the tests; not a test).
  add_executable(liyab-bench tools/liyab_bench.cpp)
  target_link_libraries(liyab-bench PRIVATE liyab_core)
  target_include_directories(liyab-bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()
