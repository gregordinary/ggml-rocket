// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * Declares the ggml backend for Rockchip NPUs (validated on the RK3588), which offloads
 * prefill work through the mainline `rocket` DRM-accel driver and `librocketnpu`. The backend
 * claims `GGML_OP_MUL_MAT`, `GGML_OP_FLASH_ATTN_EXT` and `GGML_OP_MUL_MAT_ID`, and the CPU
 * backend computes every op it declines, as with ggml's BLAS backend.
 *
 * A host loads the backend through `ggml_backend_rocket_reg()` or `GGML_BACKEND_PATH`, and
 * `ggml_backend_rocket_init()` creates one instance. The other functions configure an instance
 * or report what it computed.
 */
#ifndef GGML_ROCKET_H
#define GGML_ROCKET_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Creates a backend instance. It uses the host's CPU buffer type, so weights stay in host
 * buffers, and the instance opens its NPU workers at its first offload. */
GGML_BACKEND_API ggml_backend_t ggml_backend_rocket_init(void);

/* Returns true if `backend` is an instance of this backend. */
GGML_BACKEND_API bool ggml_backend_is_rocket(ggml_backend_t backend);

/* Sets the number of NPU workers, each with its own device file descriptor, that an offloaded
 * op is spread across. The default is 5, or `ROCKET_N_THREADS` where it is set. A value outside
 * 1-8 is clamped, with a warning.
 *
 * The caller must call it before the first offload. The workers are created at the first
 * offload, so a later call changes the stored value and warns, and the running workers keep
 * their count. */
GGML_BACKEND_API void ggml_backend_rocket_set_n_threads(ggml_backend_t backend, int n_threads);

/* Reports how many MoE experts the native-quant route holds resident and how many it streams.
 *
 * Writes the number of experts converted to resident int8 codes to `*n_resident`, and the
 * number streamed on the fp16 route to `*n_streamed`. Either pointer can be NULL. Both counts
 * are cumulative over the backend's life and count distinct experts, not calls.
 *
 * The native-quant route converts a quantized expert to int8 codes once and holds them in NPU
 * buffers. An expert that does not fit the RAM budget or the NPU's IOVA window is streamed. A
 * streamed expert is dequantized to fp16 on every call, which is correct and slower. The
 * resident share explains a MoE model's prefill speed, and this function is the only place it
 * is reported outside the teardown log.
 *
 * With `ROCKET_MOE` unset, the pre-flight reserves a whole expert stack before the backend
 * claims its op. A stack that does not fit stays on the CPU, and `*n_streamed` stays 0. A
 * nonzero value then means a limit the pre-flight cannot see before the conversion. Either the
 * IOVA window is exhausted, or a second live backend's own resident copy of a placed stack did
 * not fit the shared budget. Under `ROCKET_MOE=1` nothing is reserved, and streamed experts
 * are expected. */
GGML_BACKEND_API void ggml_backend_rocket_moe_stats(ggml_backend_t backend,
                                                    long * n_resident, long * n_streamed);

/* Report which route computed the backend's ops, so that a test can assert it.
 *
 * `ggml_backend_rocket_route_ops()` returns the number of ops the named route completed. The
 * `MUL_MAT` routes are:
 *
 *     "mt"              the per-call fp16 route
 *     "prepacked"       resident fp16 weights
 *     "prepacked-q"     a quantized weight held resident as fp16
 *     "bf16"            the bf16 route
 *     "int8", "int8r"   int8, per call and resident
 *     "i8-76"           the RK3576 route
 *     "int4", "int4r"   int4, per call and resident
 *     "fused"           a fused QKV or gate-up group, per call
 *     "fused-resident"  a fused group with resident weights
 *
 * `"moe"` counts a `MUL_MAT_ID` the expert handler computed, and `"fa"` a `FLASH_ATTN_EXT` the
 * NPU computed. A NULL `route` returns the total over every route.
 *
 * `ggml_backend_rocket_cpu_fallbacks()` returns the number of slices, MoE experts and attention
 * ops that the backend computed on the CPU instead. A fallback follows a failed NPU job, or an
 * attention op whose range fp16 cannot hold. The CPU reference of the `ROCKET_MOE_COSINE`
 * check is not counted. An `"mt"` or `"moe"` op counts once in its route however many of its
 * slices or experts fell back, so a test reads both counts. Both are cumulative over the
 * backend's life.
 *
 * Caution: a matmul fallback computes in fp64, as a test's reference does. A test that compares
 * only the output passes on a device that computed nothing. */
GGML_BACKEND_API long ggml_backend_rocket_route_ops(ggml_backend_t backend, const char * route);
GGML_BACKEND_API long ggml_backend_rocket_cpu_fallbacks(ggml_backend_t backend);

/* Returns the backend's registry entry, which ggml's loader and device enumeration use. */
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rocket_reg(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_ROCKET_H */
