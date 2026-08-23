# ggml-rocket — RK3588 NPU backend for ggml (separate project; links the
# standalone rocket-userspace driver library).
#
# Two build modes:
#
# 1. STANDALONE VALIDATION (default) — build the backend as a static lib against a
#    fresh upstream ggml clone and run test-rocket-matmul (rocket vs CPU backend).
#      git clone https://github.com/ggml-org/ggml ../ggml      # if not present
#      cmake -S . -B build [-DGGML_DIR=/path/to/ggml]
#      cmake --build build -j && sudo ./build/test-rocket-matmul
#
# 2. DROP-IN DL BACKEND (-DGGML_ROCKET_DL=ON) — build libggml-rocket.so as a
#    GGML_BACKEND_DL module that an EXISTING whisper.cpp OR llama.cpp build loads
#    at runtime (no host source patch). See README "Drop-in use".
#      # First build the host app with shared, DL-capable ggml, e.g. llama.cpp:
#      #   cmake -S llama.cpp -B llama.cpp/build \
#      #         -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON && cmake --build llama.cpp/build -j
#      cmake -S . -B build-dl -DGGML_ROCKET_DL=ON -DHOST_DIR=/abs/path/to/llama.cpp
#      cmake --build build-dl -j
#      GGML_BACKEND_PATH=$PWD/build-dl/libggml-rocket.so \
#        llama.cpp/build/bin/llama-cli -m model-f16.gguf -p "..." -ngl 0
cmake_minimum_required(VERSION 3.16)
project(ggml-rocket CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # driver static lib links into our .so
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(GGML_ROCKET_DL "Build libggml-rocket.so as a drop-in GGML_BACKEND_DL backend for an existing (whisper.cpp) ggml build" OFF)

# Heavy per-op diagnostics (the CPU-recompute VERIFY / CPU_FORWARD, the per-op TRACE, and
# the prepacked-vs-mt AB / DUMP scaffolding) compile out of the production .so by default;
# turn this ON to include them (then arm at runtime with ROCKET_VERIFY / ROCKET_CPU_FORWARD
# / ROCKET_TRACE / ROCKET_AB / ROCKET_DUMP). The lightweight ROCKET_MM_PROFILE timing and
# ROCKET_DEBUG trace lines are always available (cached-getenv, near-zero when unset).
option(GGML_ROCKET_DIAGNOSTICS "Compile in the heavy per-op recompute/trace diagnostics (off in production)" OFF)

# Portability: internalize the glibc 2.38 __isoc23_* symbols so libggml-rocket.so loads on
# an older-glibc runtime (e.g. a Debian-bookworm container) even when built on a newer host.
# Off by default -- a normal build is byte-unchanged. Declared here rather than inside the
# DL branch so it appears in `cmake -LH` either way; it only has an EFFECT in DL mode,
# because the shim exists for the shipped .so and a static standalone build has no loader
# to satisfy. See glibc_compat.c.
option(GGML_ROCKET_PORTABLE_GLIBC
       "Link a shim so libggml-rocket.so loads on glibc < 2.38 (e.g. an older container). DL mode only" OFF)

# --- the rk3588 driver library (dependency, both modes) ---
# Prefer an installed rocketnpu CMake package (cmake --install of rocket-userspace, which
# exports rocketnpu::rocketnpu); otherwise build it from a sibling checkout. Override
# the source location with -DROCKETNPU_DIR=/path/to/rocket-userspace.
#
# The source fallback builds on a non-aarch64 host too (e.g. x86_64 CI): the driver's
# NEON/FP16 code is guarded by __aarch64__/__ARM_NEON, so an off-target build compiles
# the portable scalar paths. The NPU-touching tests SKIP (exit 77) without /dev/accel,
# but the build + the metadata-only placement gate run anywhere -- so the backend stays
# CI-checkable off-device.
find_package(rocketnpu QUIET CONFIG)
if(rocketnpu_FOUND)
    message(STATUS "ggml-rocket: using installed rocketnpu package")
else()
    if(NOT DEFINED ROCKETNPU_DIR)
        set(ROCKETNPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rocket-userspace CACHE PATH "Path to a rocket-userspace checkout")
    endif()
    if(NOT EXISTS ${ROCKETNPU_DIR}/CMakeLists.txt)
        message(FATAL_ERROR
            "rocketnpu package not found and no checkout at ROCKETNPU_DIR=${ROCKETNPU_DIR}. "
            "Install rocket-userspace (cmake --install) or set -DROCKETNPU_DIR=/path/to/rocket-userspace.")
    endif()
    message(STATUS "ggml-rocket: building rocketnpu from ${ROCKETNPU_DIR}")
    set(ROCKETNPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)   # don't build driver tests here
    add_subdirectory(${ROCKETNPU_DIR} ${CMAKE_BINARY_DIR}/rocketnpu)
endif()

if(GGML_ROCKET_DL)
    # =======================================================================
    # DROP-IN DL MODE: link the EXISTING shared ggml from a HOST app's build
    # (whisper.cpp OR llama.cpp — both vendor ggml/ with the same layout).
    # The point is a SINGLE ggml instance: our .so dynamically links the same
    # libggml-base.so that the host binary already loaded, so the backend we
    # register lands in the host's registry (no duplicate ggml state).
    #
    # Point us at the host checkout with ONE of (checked in order):
    #   -DHOST_DIR=/abs/path     generic (whisper.cpp or llama.cpp)
    #   -DLLAMA_DIR=/abs/path    llama.cpp
    #   -DWHISPER_DIR=/abs/path  whisper.cpp (legacy alias)
    # =======================================================================
    if(DEFINED HOST_DIR)
        set(_host_dir ${HOST_DIR})
    elseif(DEFINED LLAMA_DIR)
        set(_host_dir ${LLAMA_DIR})
    elseif(DEFINED WHISPER_DIR)
        set(_host_dir ${WHISPER_DIR})
    else()
        message(FATAL_ERROR
            "GGML_ROCKET_DL requires -DHOST_DIR=/abs/path/to/{whisper.cpp|llama.cpp} "
            "(its bundled ggml/ provides headers + the shared libs to link). "
            "Legacy: -DLLAMA_DIR / -DWHISPER_DIR also accepted.")
    endif()
    message(STATUS "ggml-rocket DL: host app = ${_host_dir}")
    set(_ggml_inc ${_host_dir}/ggml/include)
    set(_ggml_src ${_host_dir}/ggml/src)              # private headers
    if(NOT EXISTS ${_ggml_src}/ggml-backend-impl.h)
        message(FATAL_ERROR "ggml private headers not found under ${_ggml_src}; is the host dir correct?")
    endif()

    # Where the host build put the shared ggml libs (recent ggml: <build>/bin).
    if(NOT DEFINED GGML_LIB_DIR)
        set(GGML_LIB_DIR ${_host_dir}/build/bin CACHE PATH "dir holding prebuilt libggml-base.so etc.")
    endif()
    set(_lib_hints ${GGML_LIB_DIR}
        ${_host_dir}/build/bin ${_host_dir}/build/src
        ${_host_dir}/build/ggml/src ${_host_dir}/build/lib)
    find_library(GGML_BASE_LIB NAMES ggml-base HINTS ${_lib_hints} REQUIRED)
    find_library(GGML_LIB      NAMES ggml      HINTS ${_lib_hints} REQUIRED)
    find_library(GGML_CPU_LIB  NAMES ggml-cpu  HINTS ${_lib_hints})   # cpu buffer-type symbols
    message(STATUS "ggml-rocket DL: base=${GGML_BASE_LIB} ggml=${GGML_LIB} cpu=${GGML_CPU_LIB}")

    add_library(ggml-rocket SHARED ggml-rocket.cpp)
    set_target_properties(ggml-rocket PROPERTIES PREFIX "lib" OUTPUT_NAME ggml-rocket)
    # GGML_BACKEND_DL selects the dlopen entry-point path; ggml_backend_init keeps
    # default visibility (no -fvisibility=hidden) so the loader can dlsym it.
    target_compile_definitions(ggml-rocket PRIVATE GGML_BACKEND_DL)
    target_include_directories(ggml-rocket PRIVATE ${_ggml_inc} ${_ggml_src})
    target_link_libraries(ggml-rocket PRIVATE
        ${GGML_LIB} ${GGML_BASE_LIB} rocketnpu::rocketnpu)
    if(GGML_CPU_LIB)
        target_link_libraries(ggml-rocket PRIVATE ${GGML_CPU_LIB})
    endif()

    # The glibc shim itself (option declared at top level). It must be C11 so its own
    # strtol/scanf aren't redirected back into the symbols it is replacing.
    if(GGML_ROCKET_PORTABLE_GLIBC)
        target_sources(ggml-rocket PRIVATE glibc_compat.c)
        set_source_files_properties(glibc_compat.c PROPERTIES COMPILE_OPTIONS "-std=gnu11")
        message(STATUS "ggml-rocket: PORTABLE_GLIBC on -> libggml-rocket.so floors at glibc 2.34")
    endif()
else()
    # =======================================================================
    # STANDALONE VALIDATION MODE: build against a fresh ggml clone (static).
    # =======================================================================
    if(NOT DEFINED GGML_DIR)
        set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ggml CACHE PATH "Path to a ggml checkout")
    endif()
    if(NOT EXISTS ${GGML_DIR}/CMakeLists.txt)
        message(FATAL_ERROR "ggml not found at GGML_DIR=${GGML_DIR}. Clone it: git clone https://github.com/ggml-org/ggml ${GGML_DIR}")
    endif()
    set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml)

    add_library(ggml-rocket STATIC ggml-rocket.cpp)
    # ggml-impl.h and ggml-backend-impl.h are private headers under ggml/src.
    target_include_directories(ggml-rocket PRIVATE
        ${GGML_DIR}/include
        ${GGML_DIR}/src)
    target_link_libraries(ggml-rocket PUBLIC ggml ggml-cpu rocketnpu::rocketnpu)

    # --- standalone validation tests ---
    enable_testing()
    add_executable(test-rocket-matmul test-rocket-matmul.cpp)
    target_include_directories(test-rocket-matmul PRIVATE ${GGML_DIR}/include)
    target_link_libraries(test-rocket-matmul PRIVATE ggml-rocket ggml ggml-cpu)
    target_compile_options(test-rocket-matmul PRIVATE -Wall -Wextra)
    add_test(NAME test-rocket-matmul COMMAND test-rocket-matmul)
    set_tests_properties(test-rocket-matmul PROPERTIES SKIP_RETURN_CODE 77)  # no NPU -> SKIP

    # MoE MUL_MAT_ID (routed-expert FFN) handler vs the CPU backend: both b-forms,
    # balanced/imbalanced routing, pad rows, and the F16 / MXFP4 / Q4_K weight paths.
    add_executable(test-rocket-moe test-rocket-moe.cpp)
    target_include_directories(test-rocket-moe PRIVATE ${GGML_DIR}/include)
    target_link_libraries(test-rocket-moe PRIVATE ggml-rocket ggml ggml-cpu)
    target_compile_options(test-rocket-moe PRIVATE -Wall -Wextra)
    add_test(NAME test-rocket-moe COMMAND test-rocket-moe)
    set_tests_properties(test-rocket-moe PROPERTIES SKIP_RETURN_CODE 77)  # no NPU -> SKIP

    # Native int4 (W4A4) matmul quality vs an fp16 golden (cosine), Gemma-scale
    # shapes. Forces ROCKET_INT4=1; the fast proxy for in-model int4 quality.
    add_executable(test-rocket-int4 test-rocket-int4.cpp)
    target_include_directories(test-rocket-int4 PRIVATE ${GGML_DIR}/include)
    target_link_libraries(test-rocket-int4 PRIVATE ggml-rocket ggml ggml-cpu)
    target_compile_options(test-rocket-int4 PRIVATE -Wall -Wextra)
    add_test(NAME test-rocket-int4 COMMAND test-rocket-int4)
    set_tests_properties(test-rocket-int4 PROPERTIES SKIP_RETURN_CODE 77)  # no NPU -> SKIP

    # Native BF16-weight guard: supports_op rejects batched-src1 bf16 (which the
    # f16/f32 fallthrough cannot decode), and a directly-run batched bf16 matmul
    # decodes the weights correctly. Sets ROCKET_BF16=1 itself; SKIPs off-device.
    add_executable(test-rocket-bf16 test-rocket-bf16.cpp)
    target_include_directories(test-rocket-bf16 PRIVATE ${GGML_DIR}/include)
    target_link_libraries(test-rocket-bf16 PRIVATE ggml-rocket ggml ggml-cpu)
    target_compile_options(test-rocket-bf16 PRIVATE -Wall -Wextra)
    add_test(NAME test-rocket-bf16 COMMAND test-rocket-bf16)
    set_tests_properties(test-rocket-bf16 PROPERTIES SKIP_RETURN_CODE 77)  # no NPU -> SKIP

    # Placement-gate guard: the device supports_op / offload_op contract is pure metadata
    # (shape/type only, no NPU handle), so this runs anywhere — no skip. Pins exactly which
    # MUL_MATs the scheduler may offload vs must leave on the CPU (shape contract, static
    # leaf weights, MUL_MAT-only). Closes the placement-rejection gap in test-rocket-matmul.
    add_executable(test-rocket-placement test-rocket-placement.cpp)
    target_include_directories(test-rocket-placement PRIVATE ${GGML_DIR}/include)
    target_link_libraries(test-rocket-placement PRIVATE ggml-rocket ggml ggml-cpu)
    target_compile_options(test-rocket-placement PRIVATE -Wall -Wextra)
    add_test(NAME test-rocket-placement COMMAND test-rocket-placement)

    # Pure-C NEON Hadamard-rotation gate (no ggml dep; libm only). On a non-aarch64 host
    # it exits 77 (SKIP) because the NEON path is not exercised there — the real gate
    # runs on the device. Wired as a CTest target so it can't rot.
    add_executable(test-hadamard-neon test-hadamard-neon.c)
    target_link_libraries(test-hadamard-neon PRIVATE m)
    target_compile_options(test-hadamard-neon PRIVATE -Wall -Wextra)
    add_test(NAME test-hadamard-neon COMMAND test-hadamard-neon)
    set_tests_properties(test-hadamard-neon PROPERTIES SKIP_RETURN_CODE 77)

    # RK3576 W8A8 correctness gate: the route with the most bespoke arithmetic in the
    # backend (the calibration bootstrap, the frozen per-column output scale, the host
    # K-split) against the CPU backend. SKIPs off-device AND on any part that is not an
    # RK3576, so it is safe to leave in the default ctest run on either board.
    add_executable(test-rk3576-w8a8 test-rk3576-w8a8.cpp)
    target_include_directories(test-rk3576-w8a8 PRIVATE ${GGML_DIR}/include)
    target_link_libraries(test-rk3576-w8a8 PRIVATE ggml-rocket ggml ggml-cpu)
    target_compile_options(test-rk3576-w8a8 PRIVATE -Wall -Wextra)
    add_test(NAME test-rk3576-w8a8 COMMAND test-rk3576-w8a8)
    set_tests_properties(test-rk3576-w8a8 PROPERTIES SKIP_RETURN_CODE 77)  # no NPU / not RK3576

    # RK3576 NPU-vs-CPU GEMM BENCHMARK (not a correctness gate, and not in ctest):
    # a throughput A/B against ggml F32, ggml Q8_0 and OpenBLAS. Off by default for two
    # reasons -- it needs OpenBLAS, and its own header records that the first run took a
    # board down hard enough to need a physical power cycle. Turn it on deliberately:
    #   cmake -B build -DGGML_ROCKET_RK3576_BENCH=ON
    option(GGML_ROCKET_RK3576_BENCH "Build the RK3576 NPU-vs-CPU GEMM benchmark (needs OpenBLAS)" OFF)
    if(GGML_ROCKET_RK3576_BENCH)
        find_library(OPENBLAS_LIB NAMES openblas REQUIRED)
        add_executable(test-rk3576-sgemm-ab test-rk3576-sgemm-ab.cpp)
        target_include_directories(test-rk3576-sgemm-ab PRIVATE ${GGML_DIR}/include)
        target_link_libraries(test-rk3576-sgemm-ab PRIVATE ggml-rocket ggml ggml-cpu ${OPENBLAS_LIB})
        target_compile_options(test-rk3576-sgemm-ab PRIVATE -Wall -Wextra)
        message(STATUS "ggml-rocket: RK3576 GEMM benchmark ON (openblas=${OPENBLAS_LIB})")
    endif()

    # Pure-CPU Hadamard W8A8 characterization gate (no ggml, no NPU — portable scalar
    # math that runs anywhere). README-advertised; wire it as a CTest target so the
    # hand-built g++ line can't bit-rot. Parity with test-hadamard-neon, minus the
    # aarch64 skip (this one has no NEON path to gate on).
    add_executable(test-hadamard-quant test-hadamard-quant.cpp)
    target_link_libraries(test-hadamard-quant PRIVATE m)
    target_compile_options(test-hadamard-quant PRIVATE -Wall -Wextra)
    add_test(NAME test-hadamard-quant COMMAND test-hadamard-quant)
endif()

# Warnings posture: -Wall -Wextra on our backend target (both DL and standalone
# modes). Not -Werror (the vendored ggml subbuild is a separate target and unaffected).
target_compile_options(ggml-rocket PRIVATE -Wall -Wextra)

# Heavy per-op diagnostics compile out of the .so unless explicitly requested (see the
# GGML_ROCKET_DIAGNOSTICS option above). Applies in both DL and standalone modes.
if(GGML_ROCKET_DIAGNOSTICS)
    target_compile_definitions(ggml-rocket PRIVATE ROCKET_DIAGNOSTICS)
endif()

# The backend convert kernels (rocket_pack_activations / rocket_unpack_output) use NEON
# FP16 vector intrinsics gated on __ARM_FEATURE_FP16_VECTOR_ARITHMETIC. Default aarch64
# gcc targets armv8-a, which does NOT define that macro -> the kernels silently fall back
# to scalar. RK3588 cores (A76 + A55) are all armv8.2-a+fp16, so by default request that
# baseline. This BAKES IN an armv8.2-a+fp16 assumption: the resulting .so emits FP16 vector
# ops and will SIGILL on an older aarch64 core that lacks them. The library targets RK3588,
# where this always holds; set -DGGML_ROCKET_NATIVE_FP16=OFF to build a portable
# (scalar-convert) .so for a different aarch64 target.
option(GGML_ROCKET_NATIVE_FP16 "Build the convert kernels for the armv8.2-a+fp16 (RK3588) baseline; OFF = portable scalar" ON)
if(GGML_ROCKET_NATIVE_FP16 AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=armv8.2-a+fp16" GGML_ROCKET_HAS_ARMV82_FP16)
    if(GGML_ROCKET_HAS_ARMV82_FP16)
        target_compile_options(ggml-rocket PRIVATE -march=armv8.2-a+fp16)
    else()
        message(WARNING "ggml-rocket: compiler rejects -march=armv8.2-a+fp16; "
            "the convert kernels fall back to scalar (slower, still correct).")
    endif()
endif()
