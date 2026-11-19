cmake_minimum_required(VERSION 3.21)
project(nvn_standalone VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- CPM (dependency manager) ----
set(CPM_DOWNLOAD_VERSION 0.40.8)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# ---- Dependencies ----
CPMAddPackage(
    NAME VulkanHeaders
    GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers
    VERSION 1.4.345
)

CPMAddPackage(
    NAME VulkanMemoryAllocator
    GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    VERSION 3.3.0
    OPTIONS "VMA_BUILD_DOCUMENTATION OFF" "VMA_BUILD_SAMPLES OFF"
)

CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 12.1.0
)

CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.17.0
    OPTIONS "BUILD_GMOCK ON" "INSTALL_GTEST OFF"
)

# ---- Build options ----
option(NVN_ENABLE_VULKAN_TESTS "Enable tests that require a real Vulkan device" OFF)

# ---- Stub library (replaces Eglantine internals) ----
add_library(nvn_stubs STATIC
    src/stubs/common_stubs.cpp
    src/stubs/vulkan_device_stub.cpp
    src/stubs/vulkan_wrapper_real.cpp
    src/stubs/nvn_types_stub.cpp
    src/stubs/sink_stub.cpp
    src/stubs/vma_impl.cpp
)
target_include_directories(nvn_stubs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(nvn_stubs PUBLIC
    Vulkan::Headers
    fmt::fmt
    GPUOpen::VulkanMemoryAllocator
)
# VMA implementation is compiled once, in src/stubs/vma_impl.cpp

if(WIN32)
    target_compile_definitions(nvn_stubs PUBLIC
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# ---- NvnClean library (the actual code under test) ----
add_library(nvn_clean STATIC
    src/nvn_clean/nv_sync_manager.cpp
    src/nvn_clean/nv_mempool.cpp
    src/nvn_clean/nv_pass_manager.cpp
    src/nvn_clean/nv_command_recorder.cpp
    src/nvn_clean/nv_bindless.cpp
)
target_include_directories(nvn_clean PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(nvn_clean PUBLIC nvn_stubs)

# ---- Tests ----
enable_testing()

add_executable(nvn_tests
    tests/test_main.cpp
    tests/test_sync_manager.cpp
    tests/test_mempool.cpp
    tests/test_alignment.cpp
)
target_link_libraries(nvn_tests PRIVATE
    nvn_clean
    GTest::gtest
    GTest::gmock
)
target_include_directories(nvn_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(nvn_tests)

# ---- Real-GPU integration tests + benchmarks ----
# NVN_ENABLE_VULKAN_TESTS=ON  -> nvn_vulkan_tests (needs a Vulkan 1.3 GPU; uses validation layer)
# NVN_ENABLE_BENCHMARKS=ON    -> nvn_bench (Google Benchmark; CPU + GPU timings of SyncManager)
option(NVN_ENABLE_BENCHMARKS "Build the nvn_bench benchmark executable" OFF)

if(NVN_ENABLE_VULKAN_TESTS OR NVN_ENABLE_BENCHMARKS)
    add_library(nvn_vk_test_context STATIC tests/vk_test_context.cpp)
    target_link_libraries(nvn_vk_test_context PUBLIC nvn_clean)
    target_include_directories(nvn_vk_test_context PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

if(NVN_ENABLE_VULKAN_TESTS)
    add_executable(nvn_vulkan_tests
        tests/test_vulkan_integration.cpp
    )
    target_link_libraries(nvn_vulkan_tests PRIVATE
        nvn_vk_test_context
        GTest::gtest_main
    )
    target_compile_definitions(nvn_vulkan_tests PRIVATE NVN_HAS_VULKAN_DEVICE)
    gtest_discover_tests(nvn_vulkan_tests DISCOVERY_MODE PRE_TEST)
endif()

if(NVN_ENABLE_BENCHMARKS)
    CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.9.4
        OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_GTEST_TESTS OFF"
                "BENCHMARK_ENABLE_INSTALL OFF"
    )
    add_executable(nvn_bench
        bench/bench_sync_manager.cpp
    )
    target_link_libraries(nvn_bench PRIVATE
        nvn_vk_test_context
        benchmark::benchmark
        benchmark::benchmark_main
    )
endif()
