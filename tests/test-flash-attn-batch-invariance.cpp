// TASK-066 / llama.cpp#25618 — Metal FA flash_attn_ext_vec batch invariance.
// Applicable to Metal patch A (FA nsg pin):
//   C1 row0 parity, C2 nsg boundary, C4 product KV quants (q8_0/turbo4),
//   C5 N/A (FA pin does not touch mul_mat_id), C6 via separate smoke script.
// Host-skip (exit 77) when no Metal backend is registered.

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
constexpr int32_t kNwg = 32;
constexpr int32_t kNcpsg = 32; // OP_FLASH_ATTN_EXT_VEC_NCPSG

int64_t nsg_for_ne11(int64_t ne11) {
    int64_t nsg = 1;
    while (2 * kNwg * nsg * kNcpsg < ne11 && nsg < 4) {
        nsg *= 2;
    }
    return nsg;
}

struct fa_run {
    std::vector<float> out; // layout: [d, nq, hq, 1]
};

std::vector<uint8_t> quantize_rows(ggml_type type, const std::vector<float> & src_f32, int64_t ne0) {
    GGML_ASSERT(src_f32.size() % size_t(ne0) == 0);
    const int64_t nrows = int64_t(src_f32.size() / size_t(ne0));
    const size_t row_bytes = ggml_row_size(type, ne0);
    std::vector<uint8_t> out(size_t(nrows) * row_bytes, 0);

    if (type == GGML_TYPE_F16) {
        for (int64_t r = 0; r < nrows; ++r) {
            ggml_fp32_to_fp16_row(
                    src_f32.data() + size_t(r * ne0),
                    reinterpret_cast<ggml_fp16_t *>(out.data() + size_t(r) * row_bytes),
                    ne0);
        }
        return out;
    }

    if (type == GGML_TYPE_TURBO4_0) {
        // Deterministic constant-centroid packing (same idea as test_flash_attn_ext_turbo4_vec).
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

    // Generic path (Q8_0, etc.)
    std::vector<float> imatrix(size_t(ne0), 1.0f);
    const size_t written = ggml_quantize_chunk(
            type,
            src_f32.data(),
            out.data(),
            /*start=*/0,
            nrows,
            ne0,
            imatrix.data());
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
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(32, false),
        /*.mem_base   =*/ nullptr,
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

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    fa_run result;
    result.out.resize(size_t(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.out.data(), 0, result.out.size() * sizeof(float));
    return result;
}

bool row0_bit_identical(const std::vector<float> & a, const std::vector<float> & b_full, int64_t nq_b) {
    GGML_ASSERT(a.size() == size_t(kHead * kHeads));
    GGML_ASSERT(b_full.size() == size_t(kHead * nq_b * kHeads));
    return memcmp(a.data(), b_full.data(), a.size() * sizeof(float)) == 0;
}

double max_abs_diff_row0(const std::vector<float> & a, const std::vector<float> & b_full, int64_t nq_b) {
    GGML_ASSERT(a.size() == size_t(kHead * kHeads));
    GGML_ASSERT(b_full.size() == size_t(kHead * nq_b * kHeads));
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

bool check_c1(
        ggml_backend_t backend,
        int64_t kv,
        const std::vector<int64_t> & widths,
        ggml_type type_k,
        ggml_type type_v,
        const char * label,
        int & fails) {
    printf("C1 row0 parity [%s] @ kv=%lld K=%s V=%s\n",
           label, (long long) kv, ggml_type_name(type_k), ggml_type_name(type_v));
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
        const double mad = max_abs_diff_row0(serial.out, batch.out, n);
        printf("  N=%lld nsg_formula(kv)=%lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, (long long) nsg_for_ne11(kv), exact ? 1 : 0, mad,
               exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

// C7 — multi-step ubatch vs one-shot (product C2 class at FA-op level).
// n_total queries causal over KV=n_total: one FA(nq=n_total) vs floor(n_total/ub)
// steps of FA(nq=ub) with growing KV. Bit-exact required (no ε).
// Pre-fix (legacy ne01<20 vec): n=16/ub=8 both-vec often GREEN; n=32/ub=8 vec vs
// tile RED. After TASK-066 always-tile: both must be GREEN.
bool check_c7_multistep_ubatch(
        ggml_backend_t backend,
        int64_t n_total,
        int64_t ub,
        ggml_type type_k,
        ggml_type type_v,
        const char * label,
        int & fails) {
    GGML_ASSERT(n_total > 0 && ub > 0 && (n_total % ub) == 0);
    printf("C7 multistep-ubatch [%s] n=%lld ub=%lld K=%s V=%s\n",
           label, (long long) n_total, (long long) ub,
           ggml_type_name(type_k), ggml_type_name(type_v));

    std::vector<float> k(static_cast<size_t>(kHead * n_total), 0.0f);
    std::vector<float> v(static_cast<size_t>(kHead * n_total), 0.0f);
    std::vector<float> q(static_cast<size_t>(kHead * n_total), 0.0f);
    fill_sin(k, 0.21f);
    fill_sin(v, 0.67f);
    fill_sin(q, 0.43f);

    const fa_run oneshot = run_fa(
            backend, n_total, n_total, q, k, v, /*causal_mask=*/true, type_k, type_v);

    std::vector<float> multi(static_cast<size_t>(kHead * n_total * kHeads), 0.0f);
    for (int64_t step = 0; step < n_total / ub; ++step) {
        const int64_t q0 = step * ub;
        const int64_t nkv = q0 + ub;
        std::vector<float> q_step(static_cast<size_t>(kHead * ub), 0.0f);
        std::vector<float> k_step(static_cast<size_t>(kHead * nkv), 0.0f);
        std::vector<float> v_step(static_cast<size_t>(kHead * nkv), 0.0f);
        for (int64_t i = 0; i < ub; ++i) {
            for (int64_t d = 0; d < kHead; ++d) {
                q_step[size_t(i * kHead + d)] = q[size_t((q0 + i) * kHead + d)];
            }
        }
        for (int64_t i = 0; i < nkv; ++i) {
            for (int64_t d = 0; d < kHead; ++d) {
                k_step[size_t(i * kHead + d)] = k[size_t(i * kHead + d)];
                v_step[size_t(i * kHead + d)] = v[size_t(i * kHead + d)];
            }
        }
        const fa_run step_out = run_fa(
                backend, ub, nkv, q_step, k_step, v_step, /*causal_mask=*/true,
                type_k, type_v, /*q_pos0_base=*/q0);
        GGML_ASSERT(step_out.out.size() == size_t(kHead * ub * kHeads));
        memcpy(
                multi.data() + size_t(q0 * kHead * kHeads),
                step_out.out.data(),
                step_out.out.size() * sizeof(float));
    }

    bool exact = oneshot.out.size() == multi.size() &&
            memcmp(oneshot.out.data(), multi.data(), oneshot.out.size() * sizeof(float)) == 0;
    double mad = 0.0;
    const size_t n = std::min(oneshot.out.size(), multi.size());
    for (size_t i = 0; i < n; ++i) {
        mad = std::max(mad, std::abs(double(oneshot.out[i]) - double(multi[i])));
    }
    printf("  exact=%d mad=%.6g %s\n", exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
    if (!exact) {
        ++fails;
        return false;
    }
    return true;
}

bool check_c2_boundary(
        ggml_backend_t backend,
        int64_t decode_kv,
        int64_t n,
        ggml_type type_k,
        ggml_type type_v,
        const char * label,
        int & fails) {
    // Speculative verify grows KV earlier than plain decode. Keep ne11 identical
    // (verify_kv) on both arms and mask decode row0 to decode_kv-1 — otherwise
    // different ne11 alone yields ~1 ULP even when nsg is pinned.
    const int64_t verify_kv = decode_kv + (n - 1);
    const int64_t q_pos0 = decode_kv - 1;
    printf("C2 early-cross [%s] decode_kv=%lld verify_kv=%lld N=%lld K=%s V=%s nsg_formula %lld→%lld (same-ne11+mask)\n",
           label, (long long) decode_kv, (long long) verify_kv, (long long) n,
           ggml_type_name(type_k), ggml_type_name(type_v),
           (long long) nsg_for_ne11(decode_kv), (long long) nsg_for_ne11(verify_kv));

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
    const double mad = max_abs_diff_row0(decode.out, verify.out, n);
    printf("  exact=%d max_abs_diff=%.6g %s\n", exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
    if (!exact) {
        ++fails;
        return false;
    }
    return true;
}

void run_pair_suite(
        ggml_backend_t backend,
        ggml_type type_k,
        ggml_type type_v,
        const char * label,
        const std::vector<int64_t> & widths,
        bool full_c1_boundaries,
        int & fails) {
    if (full_c1_boundaries) {
        for (int64_t kv : {2047, 2048, 2049, 4095, 4096, 4097}) {
            check_c1(backend, kv, widths, type_k, type_v, label, fails);
        }
    } else {
        // Product-quant C4: hit both nsg thresholds without re-running the full F16 matrix.
        for (int64_t kv : {2048, 2049, 4096, 4097}) {
            check_c1(backend, kv, widths, type_k, type_v, label, fails);
        }
    }
    check_c2_boundary(backend, 2048, 2, type_k, type_v, label, fails);
    check_c2_boundary(backend, 2047, 3, type_k, type_v, label, fails);
    check_c2_boundary(backend, 4096, 2, type_k, type_v, label, fails);
    check_c2_boundary(backend, 4095, 3, type_k, type_v, label, fails);
    // C7: product-class multi-step ubatch (H6 n=16 / H7=C2 n=32).
    check_c7_multistep_ubatch(backend, 16, 8, type_k, type_v, label, fails);
    check_c7_multistep_ubatch(backend, 32, 8, type_k, type_v, label, fails);
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
        printf("host-skip: no Metal backend (TASK-066 FA corners)\n");
        return 77;
    }
    ggml_backend_ptr backend_holder(ggml_backend_dev_init(chosen, nullptr));
    if (!backend_holder) {
        printf("host-skip: Metal backend init failed\n");
        return 77;
    }
    ggml_backend_t backend = backend_holder.get();
    printf("backend=%s device=%s\n", ggml_backend_name(backend), ggml_backend_dev_description(chosen));

    printf("APPLICABLE patchA(FA nsg): C1,C2,C4(product KV); C5=N/A(mul_mat_id untouched); "
           "C6=see run-task066-fa-nsg-corners.sh\n");
    printf("C5 N/A: metal-fa-nsg-pin only changes flash_attn_ext_vec nsg; MoE mul_mv_id is ST-2.1b\n");
    printf("C4 note: IQ4_XS is target *weight* matmul (not FA KV); product FA KV is q8_0/turbo4\n");

    GGML_ASSERT(nsg_for_ne11(2047) == 1);
    GGML_ASSERT(nsg_for_ne11(2048) == 1);
    GGML_ASSERT(nsg_for_ne11(2049) == 2);
    GGML_ASSERT(nsg_for_ne11(4095) == 2);
    GGML_ASSERT(nsg_for_ne11(4096) == 2);
    GGML_ASSERT(nsg_for_ne11(4097) == 4);
    printf("C2 formula thresholds OK (2048/4096)\n");

    int fails = 0;
    const std::vector<int64_t> widths = {2, 3, 4, 5, 8};

    // C1+C2 baseline types (F16/F16) — full boundary matrix.
    run_pair_suite(backend, GGML_TYPE_F16, GGML_TYPE_F16, "f16/f16", widths, /*full*/ true, fails);

    // C4 product MTP Metal profile: quant_k=q8_0, quant_v=turbo4.
    run_pair_suite(
            backend, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0, "product-q8_0/turbo4", widths,
            /*full*/ false, fails);

    // C4 sibling: symmetric q8_0 KV (drafter/aux / control profiles).
    run_pair_suite(
            backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "q8_0/q8_0", widths, /*full*/ false, fails);

    if (fails != 0) {
        printf("RESULT FAIL fails=%d (C1/C2/C4/C7)\n", fails);
        return 1;
    }
    printf("RESULT PASS (C1+C2+C4+C7; C5=N/A logged; C6 external)\n");
    return 0;
}
