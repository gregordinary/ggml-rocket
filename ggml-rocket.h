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

/* Backend registry entry point (for ggml_backend_load / device enumeration). */
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rocket_reg(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_ROCKET_H */
