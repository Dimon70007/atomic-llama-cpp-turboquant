// TASK-069 — optional CUDA product-knob lock for fork e2e localizers.
// When TASK069_SUITE_PROFILE=cuda, required TASK069_* env knobs must be set
// (exported from tests/fixtures/task069/cuda-mtp.reference.yaml via task069_suite_knobs.py).
// Unset profile → caller keeps Metal TASK-066 lock (no silent CUDA substitution).

#pragma once

#include "common.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace task069_suite_lock {

inline bool parse_kv_type(const char * name, ggml_type & out) {
    if (!name || !name[0]) {
        return false;
    }
    if (std::strcmp(name, "q8_0") == 0) {
        out = GGML_TYPE_Q8_0;
        return true;
    }
    if (std::strcmp(name, "turbo3") == 0) {
        out = GGML_TYPE_TURBO3_0;
        return true;
    }
    if (std::strcmp(name, "turbo4") == 0) {
        out = GGML_TYPE_TURBO4_0;
        return true;
    }
    if (std::strcmp(name, "f16") == 0) {
        out = GGML_TYPE_F16;
        return true;
    }
    return false;
}

inline const char * require_env(const char * key) {
    const char * v = std::getenv(key);
    if (!v || !v[0]) {
        fprintf(stderr, "FAIL: missing required %s (TASK069_SUITE_PROFILE=cuda; refusing silent default)\n", key);
        std::exit(2);
    }
    return v;
}

inline int require_int_env(const char * key) {
    const char * v = require_env(key);
    char * end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || n < 0) {
        fprintf(stderr, "FAIL: %s must be a non-negative integer (got %s)\n", key, v);
        std::exit(2);
    }
    return (int) n;
}

// Returns true if CUDA suite lock applied; false if profile unset (Metal path).
inline bool apply_cuda_if_requested(common_params & params, bool lock_batch_ubatch) {
    const char * profile = std::getenv("TASK069_SUITE_PROFILE");
    if (!profile || !profile[0]) {
        return false;
    }
    if (std::strcmp(profile, "cuda") != 0) {
        fprintf(stderr, "FAIL: TASK069_SUITE_PROFILE must be 'cuda' (got %s)\n", profile);
        std::exit(2);
    }

    params.n_gpu_layers = require_int_env("TASK069_NGL");
    if (lock_batch_ubatch) {
        params.n_batch = require_int_env("TASK069_BATCH");
        params.n_ubatch = require_int_env("TASK069_UBATCH");
    }
    params.cache_ram_mib = require_int_env("TASK069_CRAM");
    params.cpuparams.n_threads = require_int_env("TASK069_THREADS");
    params.cpuparams_batch.n_threads = require_int_env("TASK069_THREADS_BATCH");
    params.n_parallel = require_int_env("TASK069_PARALLEL");
    params.main_gpu = require_int_env("TASK069_MAIN_GPU");
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    ggml_type ctk = GGML_TYPE_COUNT;
    ggml_type ctv = GGML_TYPE_COUNT;
    if (!parse_kv_type(require_env("TASK069_CTK"), ctk) || !parse_kv_type(require_env("TASK069_CTV"), ctv)) {
        fprintf(stderr, "FAIL: TASK069_CTK/CTV unsupported type\n");
        std::exit(2);
    }
    params.cache_type_k = ctk;
    params.cache_type_v = ctv;

    const char * ts = require_env("TASK069_TENSOR_SPLIT");
    {
        std::memset(params.tensor_split, 0, sizeof(params.tensor_split));
        int i = 0;
        const char * p = ts;
        while (*p && i < (int) (sizeof(params.tensor_split) / sizeof(params.tensor_split[0]))) {
            char * end = nullptr;
            const float f = std::strtof(p, &end);
            if (end == p) {
                fprintf(stderr, "FAIL: TASK069_TENSOR_SPLIT parse error near '%s'\n", p);
                std::exit(2);
            }
            params.tensor_split[i++] = f;
            p = end;
            if (*p == ',') {
                ++p;
            } else if (*p != '\0') {
                fprintf(stderr, "FAIL: TASK069_TENSOR_SPLIT unexpected char '%c'\n", *p);
                std::exit(2);
            }
        }
        if (i < 2) {
            fprintf(stderr, "FAIL: TASK069_TENSOR_SPLIT needs >=2 fractions (got %s)\n", ts);
            std::exit(2);
        }
    }

    fprintf(stderr,
            "task069_cuda_suite_lock: ngl=%d b=%d ub=%d ctk=%s ctv=%s fa=on cram=%d t=%d tb=%d np=%d "
            "mg=%d ts=%s ctx=%d\n",
            params.n_gpu_layers, params.n_batch, params.n_ubatch,
            ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
            params.cache_ram_mib, params.cpuparams.n_threads, params.cpuparams_batch.n_threads,
            params.n_parallel, params.main_gpu, ts, (int) params.n_ctx);
    return true;
}

} // namespace task069_suite_lock
