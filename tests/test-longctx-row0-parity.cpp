// TASK-066 P5 residual localizer — long-KV FA row0 parity (hypothesis: FA still
// diverges for N>1 at product KV lengths ≫ 4096 after metal-fa-nsg-pin).
//
// TDD gate for patch C on FA long-KV: RED → candidate Metal/CUDA FA fix;
// GREEN on a host means this surface needs no new patch *on that backend* (still run CUDA).
//
// Contract: product FA KV q8_0/turbo4; KV ∈ {8192, 32768, 65536}; N ∈ {2,3,4,5,8};
// row0(batch N) bit-identical to serial N=1. Also early-cross causal at long KV.
// Host-skip (exit 77) when no Metal backend.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t kHead = 128;
constexpr int64_t kHeads = 1;

struct fa_run {
    std::vector<float> out;
};

std::vector<uint8_t> quantize_rows(ggml_type type, const std::vector<float> & src_f32, int64_t ne0) {
    GGML_ASSERT(src_f32.size() % size_t(ne0) == 0);
    const int64_t nrows = int64_t(src_f32.size() / size_t(ne0));
    const size_t row_bytes = ggml_row_size(type, ne0);
    std::vector<uint8_t> out(size_t(nrows) * row_bytes, 0);

    if (type == GGML_TYPE_TURBO4_0) {
        constexpr uint8_t turbo4_centroid_index = 10;
        constexpr uint8_t turbo4_indices = turbo4_centroid_index | (turbo4_centroid_index << 4);
        const int64_t block_size = ggml_blck_size(GGML_TYPE_TURBO4_0);
        const size_t block_bytes = ggml_type_size(GGML_TYPE_TURBO4_0);
        GGML_ASSERT(ne0 % block_size == 0);
        const float norm_f32 = 1.0f;
        ggml_fp16_t norm_f16;
        ggml_fp32_to_fp16_row(&norm_f32, &norm_f16, 1);
        for (int64_t r = 0; r < nrows; ++r) {
            uint8_t * row = out.data() + size_t(r) * row_bytes;
            for (int64_t ib = 0; ib < ne0 / block_size; ++ib) {
                uint8_t * block = row + size_t(ib) * block_bytes;
                memcpy(block, &norm_f16, sizeof(norm_f16));
                memset(block + sizeof(norm_f16), turbo4_indices, size_t(block_size / 2));
            }
        }
        return out;
    }

    std::vector<float> imatrix(size_t(ne0), 1.0f);
    const size_t written = ggml_quantize_chunk(
            type, src_f32.data(), out.data(), /*start=*/0, nrows, ne0, imatrix.data());
    GGML_ASSERT(written == out.size());
    return out;
}

fa_run run_fa(
        ggml_backend_t backend,
        int64_t nq,
        int64_t nkv,
        const std::vector<float> & q_f32,
        const std::vector<float> & k_f32,
        const std::vector<float> & v_f32,
        bool causal_mask,
        ggml_type type_k,
        ggml_type type_v,
        // When >= 0: q_pos for iq is q_pos0_base+iq (keep ne11 equal across decode vs verify).
        // When < 0: default causal q_pos = (nkv - nq) + iq.
        int64_t q_pos0_base = -1) {
    // Large KV: tensor metadata only in ctx; device buffers via backend alloc.
    const size_t mem = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(64, false) + 64 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, kHead, nq, kHeads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), type_k, kHead, nkv, kHeads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), type_v, kHead, nkv, kHeads, 1);
    ggml_tensor * m = nullptr;
    if (causal_mask) {
        m = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, nkv, nq, 1, 1);
    }

    ggml_tensor * out = ggml_flash_attn_ext(
            ctx.get(), q, k, v, m, 1.0f / std::sqrt(float(kHead)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);

    GGML_ASSERT(q_f32.size() == size_t(kHead * nq * kHeads));
    GGML_ASSERT(k_f32.size() == size_t(kHead * nkv * kHeads));
    GGML_ASSERT(v_f32.size() == size_t(kHead * nkv * kHeads));

    ggml_backend_tensor_set(q, q_f32.data(), 0, q_f32.size() * sizeof(float));

    const std::vector<uint8_t> k_q = quantize_rows(type_k, k_f32, kHead);
    const std::vector<uint8_t> v_q = quantize_rows(type_v, v_f32, kHead);
    ggml_backend_tensor_set(k, k_q.data(), 0, k_q.size());
    ggml_backend_tensor_set(v, v_q.data(), 0, v_q.size());

    if (m) {
        std::vector<ggml_fp16_t> mask(size_t(nkv * nq));
        for (int64_t iq = 0; iq < nq; ++iq) {
            const int64_t q_pos = (q_pos0_base >= 0) ? (q_pos0_base + iq) : ((nkv - nq) + iq);
            for (int64_t ik = 0; ik < nkv; ++ik) {
                const bool visible = ik <= q_pos;
                mask[size_t(iq * nkv + ik)] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    fa_run result;
    result.out.resize(size_t(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.out.data(), 0, result.out.size() * sizeof(float));
    return result;
}

bool row0_bit_identical(const std::vector<float> & a, const std::vector<float> & b_full, int64_t /*nq_b*/) {
    GGML_ASSERT(a.size() == size_t(kHead * kHeads));
    GGML_ASSERT(b_full.size() >= a.size());
    return memcmp(a.data(), b_full.data(), a.size() * sizeof(float)) == 0;
}

double max_abs_diff_row0(const std::vector<float> & a, const std::vector<float> & b_full) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(double(a[i]) - double(b_full[i])));
    }
    return m;
}

void fill_sin(std::vector<float> & dst, float phase) {
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] = 0.17f * std::sin(float(i) * 0.013f + phase);
    }
}

bool check_c1_long(
        ggml_backend_t backend,
        int64_t kv,
        const std::vector<int64_t> & widths,
        ggml_type type_k,
        ggml_type type_v,
        int & fails) {
    printf("LONG-C1 row0 parity @ kv=%lld K=%s V=%s\n",
           (long long) kv, ggml_type_name(type_k), ggml_type_name(type_v));
    std::vector<float> k(static_cast<size_t>(kHead * kv), 0.0f);
    std::vector<float> v(static_cast<size_t>(kHead * kv), 0.0f);
    fill_sin(k, 0.11f);
    fill_sin(v, 0.73f);

    bool ok_all = true;
    for (int64_t n : widths) {
        std::vector<float> q1(static_cast<size_t>(kHead), 0.0f);
        fill_sin(q1, 0.31f);
        std::vector<float> qn(static_cast<size_t>(kHead * n), 0.0f);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t d = 0; d < kHead; ++d) {
                qn[size_t(i * kHead + d)] = (i == 0)
                    ? q1[size_t(d)]
                    : 0.17f * std::sin(float(i * kHead + d) * 0.019f + 1.7f);
            }
        }

        const fa_run serial = run_fa(backend, 1, kv, q1, k, v, false, type_k, type_v);
        const fa_run batch  = run_fa(backend, n, kv, qn, k, v, false, type_k, type_v);
        const bool exact = row0_bit_identical(serial.out, batch.out, n);
        const double mad = max_abs_diff_row0(serial.out, batch.out);
        printf("  N=%lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

bool check_early_cross_long(
        ggml_backend_t backend,
        int64_t decode_kv,
        int64_t n,
        ggml_type type_k,
        ggml_type type_v,
        int & fails) {
    // Same-ne11+mask: both sides use verify_kv so Metal nsg(ne11) matches; mask hides future.
    const int64_t verify_kv = decode_kv + (n - 1);
    const int64_t q_pos0 = decode_kv - 1;
    printf("LONG-C2 early-cross decode_kv=%lld verify_kv=%lld N=%lld K=%s V=%s (same-ne11+mask)\n",
           (long long) decode_kv, (long long) verify_kv, (long long) n,
           ggml_type_name(type_k), ggml_type_name(type_v));

    std::vector<float> k(static_cast<size_t>(kHead * verify_kv), 0.0f);
    std::vector<float> v(static_cast<size_t>(kHead * verify_kv), 0.0f);
    fill_sin(k, 0.41f);
    fill_sin(v, 0.91f);

    std::vector<float> q1(static_cast<size_t>(kHead), 0.0f);
    fill_sin(q1, 0.55f);
    std::vector<float> qn(static_cast<size_t>(kHead * n), 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t d = 0; d < kHead; ++d) {
            qn[size_t(i * kHead + d)] = (i == 0)
                ? q1[size_t(d)]
                : 0.17f * std::sin(float(i * kHead + d) * 0.023f + 2.1f);
        }
    }

    const fa_run decode = run_fa(
            backend, 1, verify_kv, q1, k, v, true, type_k, type_v, /*q_pos0_base=*/q_pos0);
    const fa_run verify = run_fa(
            backend, n, verify_kv, qn, k, v, true, type_k, type_v, /*q_pos0_base=*/q_pos0);
    const bool exact = row0_bit_identical(decode.out, verify.out, n);
    const double mad = max_abs_diff_row0(decode.out, verify.out);
    printf("  exact=%d max_abs_diff=%.6g %s\n", exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
    if (!exact) {
        ++fails;
        return false;
    }
    return true;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t chosen = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (name && std::string(name).find("Metal") != std::string::npos) {
            chosen = dev;
            break;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && chosen == nullptr) {
            chosen = dev;
        }
    }
    if (!chosen) {
        printf("host-skip: no Metal backend (TASK-066 long-ctx FA localizer)\n");
        return 77;
    }
    ggml_backend_ptr backend_holder(ggml_backend_dev_init(chosen, nullptr));
    if (!backend_holder) {
        printf("host-skip: Metal backend init failed\n");
        return 77;
    }
    ggml_backend_t backend = backend_holder.get();
    printf("backend=%s device=%s\n", ggml_backend_name(backend), ggml_backend_dev_description(chosen));
    printf("HYPOTHESIS: FA long-KV row0 after A+B (metal-fa-nsg-pin + mul_mv_ext disable)\n");
    printf("TDD: RED → patch C candidate on this backend; GREEN → no new FA patch here (still run CUDA)\n");
    printf("KV types: product q8_0/turbo4\n");

    int fails = 0;
    const std::vector<int64_t> widths = {2, 3, 4, 5, 8};
    const ggml_type type_k = GGML_TYPE_Q8_0;
    const ggml_type type_v = GGML_TYPE_TURBO4_0;

    for (int64_t kv : {8192, 32768, 65536}) {
        check_c1_long(backend, kv, widths, type_k, type_v, fails);
    }

    // Speculative verify crosses KV earlier than decode (skoulik class) at long ctx.
    check_early_cross_long(backend, 8192, 2, type_k, type_v, fails);
    check_early_cross_long(backend, 8192, 3, type_k, type_v, fails);
    check_early_cross_long(backend, 32768, 2, type_k, type_v, fails);
    check_early_cross_long(backend, 65536, 2, type_k, type_v, fails);

    if (fails != 0) {
        printf("RESULT FAIL fails=%d (RED → FA long-KV patch-C candidate on this backend)\n", fails);
        return 1;
    }
    printf("RESULT PASS (GREEN on this backend without new FA patch; re-run on CUDA / next localizer)\n");
    return 0;
}
