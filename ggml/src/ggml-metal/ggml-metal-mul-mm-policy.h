// TASK-066.7 / 066.9 Phase A2 — pure Metal mul_mm vs mul_mv dispatch policy.
//
// Shared by ggml-metal-ops.cpp and CPU unit tests/test-metal-mul-mm-policy.cpp.
// No Metal runtime / ObjC — safe to include from host unit tests.
//
// Rule (matches ggml_metal_op_mul_mat): use mul_mm when ne11 > mm_min
// (TQ / other special cases stay in ops.cpp).
//
// History:
//   upstream-like: mm_min=8  → N=8 on mul_mv
//   066.7 KEEP:    mm_min=7  → N>=8 on mul_mm; rem∈{2..7} still on mul_mv (A2 hole)
//   066.9 A2:      mm_min=1  → N>=2 on mul_mm; closes last-ubatch rem cross-kernel
//   066.9 B:       mm_min=0  → N>=1 on mul_mm; closes MTP verify N=1 vs {2,3,4}
//                  (decode tax ~2.45x tg; MTP speed sense eroded — see 066.15)
//   066.15 probe:  mm_min=1  → restore decode speed; accept MTP verify cross-kernel
//                  hazard pending quality A/B (MTP vs nomtp). Rem A2 lock retained.

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GGML_METAL_MUL_MM_MIN_DEFAULT
#define GGML_METAL_MUL_MM_MIN_DEFAULT 1
#endif

/** True when the Metal mul_mat path selects mul_mm for this ne11 under mm_min. */
static inline bool ggml_metal_mul_mm_ne11_uses_mm(int64_t ne11, int mm_min) {
    return ne11 > (int64_t) mm_min;
}

/**
 * MTP / speculative verify class: N=1 decode vs verify width N in {2,3,4}.
 * True when serial and verify disagree on mul_mm vs mul_mv under mm_min.
 *
 * Example (066.9 Phase B): mm_min=1 → N=1 mv, N=2..4 mm → hazard.
 * With mm_min=0 → all on mm → no hazard. Raising min to hide is forbidden.
 */
static inline bool ggml_metal_mtp_verify_cross_kernel_hazard(int64_t verify_n, int mm_min) {
    if (verify_n < 2) {
        return false;
    }
    const bool serial_mm = ggml_metal_mul_mm_ne11_uses_mm(/*ne11=*/1, mm_min);
    const bool verify_mm = ggml_metal_mul_mm_ne11_uses_mm(verify_n, mm_min);
    return serial_mm != verify_mm;
}

/**
 * Size of the last prefill ubatch when splitting n_tokens into chunks of at
 * most n_ubatch (same remainder rule as llama_batch_allocr::split_simple).
 * Returns 0 if inputs are invalid.
 */
static inline int32_t ggml_metal_ubatch_last_chunk_ne11(int32_t n_tokens, int32_t n_ubatch) {
    if (n_tokens <= 0 || n_ubatch <= 0) {
        return 0;
    }
    if (n_tokens <= n_ubatch) {
        return n_tokens;
    }
    const int32_t rem = n_tokens % n_ubatch;
    return rem == 0 ? n_ubatch : rem;
}

/**
 * Prefill with full chunks of size n_ubatch plus a remainder rem:
 * true if full chunks and rem disagree on mul_mm vs mul_mv under mm_min.
 *
 * Example (066.9 A2): n_tokens=1708, n_ubatch=8, mm_min=7 → rem=4 → hazard.
 * With mm_min=1 → no hazard.
 */
static inline bool ggml_metal_ubatch_rem_cross_kernel_hazard(
        int32_t n_tokens,
        int32_t n_ubatch,
        int     mm_min) {
    if (n_tokens <= 0 || n_ubatch <= 0) {
        return false;
    }
    if (n_tokens <= n_ubatch) {
        return false;
    }
    const int32_t rem = n_tokens % n_ubatch;
    if (rem == 0) {
        return false;
    }
    const bool full_mm = ggml_metal_mul_mm_ne11_uses_mm((int64_t) n_ubatch, mm_min);
    const bool rem_mm  = ggml_metal_mul_mm_ne11_uses_mm((int64_t) rem, mm_min);
    return full_mm != rem_mm;
}

#ifdef __cplusplus
}
#endif
