cmake_minimum_required(VERSION 3.20)
project(see-through-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# seethrough (STATIC) is linked into two SHARED libraries (seethrough_c,
# see_through_c) -- GNU ld enforces PIC for anything going into a shared
# object, unlike MSVC/Windows where this was never an issue locally.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# diffusers UNet state-dict names exceed ggml's default 64-char cap
add_compile_definitions(GGML_MAX_NAME=128)

# We only ship cpu/vulkan backends and drive threading ourselves via
# PipelineConfig::threads / ggml's own thread pool -- ggml's OpenMP option
# defaults ON and, on MinGW Windows builds, makes every binary depend on
# libgomp-1.dll at runtime. That DLL isn't part of the MinGW toolchain's
# redistributable set our CI packages, so release binaries fail to even
# start on a machine without MSYS2/MinGW installed (STATUS_DLL_NOT_FOUND).
# Force it off so we don't link against OpenMP at all.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

# Same problem as GGML_OPENMP above, one layer down: MinGW-w64 gcc/clang
# dynamically links libgcc_s_seh-1.dll, libstdc++-6.dll and
# libwinpthread-1.dll by default. None of those ship in the release zips
# our CI packages (they're MinGW/MSYS2 toolchain files, not build
# outputs), so every exe/DLL we produce fails to launch with
# STATUS_DLL_NOT_FOUND on a plain Windows machine. Statically link the
# toolchain runtime instead so the binaries are self-contained. No effect
# on MSVC builds (which never dynamically link these in the first place).
if (WIN32 AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

# ggml defaults BUILD_SHARED_LIBS to ON everywhere except MinGW, which
# splits it into ggml-base/ggml-cpu/ggml(.dll/.so/.dylib) that see-through
# dynamically loads at runtime. CI's packaging step only copies the
# top-level exe/DLL targets it knows the names of, never these -- so the
# non-MinGW release zips (MSVC Windows/Vulkan, Linux, macOS) are missing a
# hard runtime dependency and fail to launch at all outside the build tree.
# Build ggml statically everywhere so every produced binary is
# self-contained; matches the MinGW static-runtime fix above in spirit.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

# psd_sdk's own CMakeLists.txt requires a cmake_minimum_required too old for
# current CMake; compile its sources directly instead of add_subdirectory
# (read-only usage: PsdExport*/PsdNativeFile writer bits aren't needed for
# our upstream-parity reader, but bringing the whole psd_source list in is
# simplest and matches how the upstream project itself builds it)
set(PSD_SDK_DIR third_party/psd_sdk/src/Psd)
file(GLOB PSD_SDK_SOURCES ${PSD_SDK_DIR}/*.cpp ${PSD_SDK_DIR}/*.c)
if (WIN32)
    list(FILTER PSD_SDK_SOURCES EXCLUDE REGEX ".*NativeFile_(Mac|Linux)\\.cpp$")
else()
    list(FILTER PSD_SDK_SOURCES EXCLUDE REGEX ".*NativeFile\\.cpp$")
endif()
add_library(Psd STATIC ${PSD_SDK_SOURCES})
target_include_directories(Psd PUBLIC third_party/psd_sdk/src)

add_library(seethrough STATIC
    src/ops.cpp
    src/vae.cpp
    src/clip.cpp
    src/unet_frame.cpp
    src/scheduler.cpp
    src/lama.cpp
    src/image_utils.cpp
    src/postproc.cpp
    src/pipeline.cpp
)
target_link_libraries(seethrough PUBLIC ggml)
# -DGGML_CPU=OFF is a supported configuration: the pipeline is GPU-only
# (src/pipeline.cpp rejects --device cpu outright), so the CPU backend is
# only needed by the code that uses it as a *reference* to check GPU math
# against. Propagate the option as a define so those call sites -- and the
# targets built around them below -- can compile out instead of failing to
# link against a ggml-cpu that was never built.
if (GGML_CPU)
    target_compile_definitions(seethrough PUBLIC SEETHROUGH_HAVE_CPU_BACKEND)
endif()
target_include_directories(seethrough PUBLIC src third_party/stb)
target_include_directories(seethrough PRIVATE third_party/ggml/src)

add_executable(see-through src/see_through.cpp src/psd_write.cpp)
target_link_libraries(see-through PRIVATE seethrough Psd)

# Lean witness testing (st_witness_check*): runs the same GEMM on CPU and on
# the GPU backend and compares them, so the CPU backend isn't a fallback here
# but the ground truth being checked against. Nothing to verify without it --
# skip the whole target rather than ship one that can't do its job.
if (GGML_CPU)
    add_library(seethrough_c SHARED src/seethrough_capi.cpp src/gemm_witness.cpp)
    target_link_libraries(seethrough_c PRIVATE seethrough)
    target_compile_definitions(seethrough_c PRIVATE ST_BUILD_DLL)
    # ggml's own Q4_0 reference quantizer, for the quantization witness cases
    # (private header: block_q4_0 layout + quantize_row_q4_0_ref)
    target_include_directories(seethrough_c PRIVATE third_party/ggml/src)
endif()

# separate from seethrough_c above: that one is scoped to Lean witness
# testing (st_witness_check*), this one drives the real production
# pipeline (image bytes in, PNG layers out) for embedding in a
# server or other host process
add_library(see_through_c SHARED src/see_through_capi.cpp)
target_link_libraries(see_through_c PRIVATE seethrough)
target_compile_definitions(see_through_c PRIVATE ST2_BUILD_DLL)

set(SEETHROUGH_TESTS test_trans_vae test_sd_vae test_trans_vae_full test_clip test_vae_encode test_vae_encode_160 test_vae_encode_tiled test_unet_blocks test_unet_forward test_unet_forward_160 test_unet_forward_head_real test_scheduler test_layerdiff_e2e test_layerdiff_e2e_160 test_ddim test_marigold_e2e test_lama test_metal_nan test_vae_decode_metal)

# Same reasoning as seethrough_c: these three name ggml_backend_cpu_init
# directly and can't be made backend-agnostic -- the precision tests exist to
# diff GPU output against CPU output, and graph_mem measures allocation
# against the CPU buffer type.
if (GGML_CPU)
    list(APPEND SEETHROUGH_TESTS graph_mem test_linear_precision test_attn_precision)
endif()

foreach(t ${SEETHROUGH_TESTS})
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE seethrough)
endforeach()

add_executable(psd_layers tests/psd_layers.cpp)
target_link_libraries(psd_layers PRIVATE Psd)
target_include_directories(psd_layers PRIVATE src third_party/stb)
target_sources(psd_layers PRIVATE src/image_utils.cpp)

