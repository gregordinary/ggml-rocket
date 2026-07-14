// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * ggml-rocket.h — ggml backend for the RK3588 NPU via the mainline "rocket"
 * DRM-accel driver. Offloads GGML_OP_MUL_MAT and GGML_OP_FLASH_ATTN_EXT (prefill
 * attention) to the NPU through the standalone rocket-userspace driver library
 * (librocketnpu) — plus GGML_OP_MUL_MAT_ID (MoE) when ROCKET_MOE is set; everything
 * else falls back to the CPU backend, BLAS-style.
 *
 * Separate project from the driver: this links rocketnpu::rocketnpu and builds
 * against a fresh upstream ggml checkout. Modeled on ggml's own BLAS backend.
 */
#ifndef GGML_ROCKET_H
#define GGML_ROCKET_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Create a rocket backend instance (host buffers; mul_mat on the NPU). */
GGML_BACKEND_API ggml_backend_t ggml_backend_rocket_init(void);

/* True if `backend` is a rocket backend. */
GGML_BACKEND_API bool ggml_backend_is_rocket(ggml_backend_t backend);

/* Number of NPU worker threads/fds used for the multicore fan-out (default 5 =
 * 3 cores + 2 to fill pack/read idle bubbles). Call before/independent of graph compute. */
GGML_BACKEND_API void ggml_backend_rocket_set_n_threads(ggml_backend_t backend, int n_threads);

/* Native-quant MoE expert residency, for diagnostics and for the gate.
 *
 * A quantized MoE expert is ingested ONCE into int8 codes that stay resident in NPU BOs,
 * which is what removes the per-micro-batch host dequant. Residency is ADMISSION-ONLY: an
 * expert that does not fit the budget or the NPU's IOVA window streams on the fp16 dequant
 * route instead, correctly but at the streaming cost. So "how much of the model went
 * resident" is the number that explains the speed, and it is not otherwise observable.
 *
 * Writes the count of experts ingested to *n_resident and the count that fell back to
 * streaming to *n_streamed (either pointer may be NULL). Both are cumulative over the
 * backend's life and count DISTINCT experts, not calls. */
GGML_BACKEND_API void ggml_backend_rocket_moe_stats(ggml_backend_t backend,
                                                    long * n_resident, long * n_streamed);

/* Backend registry entry point (for ggml_backend_load / device enumeration). */
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rocket_reg(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_ROCKET_H */
