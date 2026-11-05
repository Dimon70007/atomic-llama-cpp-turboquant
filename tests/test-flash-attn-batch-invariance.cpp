// TASK-066 / llama.cpp#25618 — Metal FA flash_attn_ext_vec batch invariance.
// C1: row0(nb=N) == nb=1 at the same KV length.
// C2: decode at kv=L vs verify row0 at kv=L+(N-1) (early nsg boundary cross).
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

fa_run run_fa(
        ggml_backend_t backend,
        int64_t nq,
        int64_t nkv,
        const std::vector<float> & q_f32,
        const std::vector<float> & k_f32,
        const std::vector<float> & v_f32,
        bool causal_mask) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(32, false),
        /*.mem_base   =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, kHead, nq, kHeads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, kHead, nkv, kHeads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, kHead, nkv, kHeads, 1);
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

    std::vector<ggml_fp16_t> k_f16(k_f32.size());
    std::vector<ggml_fp16_t> v_f16(v_f32.size());
    ggml_fp32_to_fp16_row(k_f32.data(), k_f16.data(), k_f32.size());
    ggml_fp32_to_fp16_row(v_f32.data(), v_f16.data(), v_f32.size());
    ggml_backend_tensor_set(k, k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));

    if (m) {
        std::vector<ggml_fp16_t> mask(size_t(nkv * nq));
        for (int64_t iq = 0; iq < nq; ++iq) {
            for (int64_t ik = 0; ik < nkv; ++ik) {
                // Causal relative to the *last* query occupying slots [nkv-nq, nkv).
                const int64_t q_pos = (nkv - nq) + iq;
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

bool check_c1(ggml_backend_t backend, int64_t kv, const std::vector<int64_t> & widths, int & fails) {
    printf("C1 row0 parity @ kv=%lld\n", (long long) kv);
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

        // No causal mask: row0 must see the same full KV as N=1 (pure kernel parity).
        const fa_run serial = run_fa(backend, 1, kv, q1, k, v, /*causal_mask=*/false);
        const fa_run batch  = run_fa(backend, n, kv, qn, k, v, /*causal_mask=*/false);
        const bool exact = row0_bit_identical(serial.out, batch.out, n);
        const double mad = max_abs_diff_row0(serial.out, batch.out, n);
        printf("  N=%lld nsg(kv)=%lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, (long long) nsg_for_ne11(kv), exact ? 1 : 0, mad,
               exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

bool check_c2_boundary(ggml_backend_t backend, int64_t decode_kv, int64_t n, int & fails) {
    const int64_t verify_kv = decode_kv + (n - 1);
    const int64_t nsg_dec = nsg_for_ne11(decode_kv);
    const int64_t nsg_ver = nsg_for_ne11(verify_kv);
    printf("C2 early-cross decode_kv=%lld verify_kv=%lld N=%lld nsg %lld→%lld\n",
           (long long) decode_kv, (long long) verify_kv, (long long) n,
           (long long) nsg_dec, (long long) nsg_ver);

    std::vector<float> k(static_cast<size_t>(kHead * verify_kv), 0.0f);
    std::vector<float> v(static_cast<size_t>(kHead * verify_kv), 0.0f);
    fill_sin(k, 0.41f);
    fill_sin(v, 0.91f);

    std::vector<float> k_dec(k.begin(), k.begin() + size_t(kHead * decode_kv));
    std::vector<float> v_dec(v.begin(), v.begin() + size_t(kHead * decode_kv));

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

    const fa_run decode = run_fa(backend, 1, decode_kv, q1, k_dec, v_dec, true);
    const fa_run verify = run_fa(backend, n, verify_kv, qn, k, v, true);
    const bool exact = row0_bit_identical(decode.out, verify.out, n);
    const double mad = max_abs_diff_row0(decode.out, verify.out, n);
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
            // Prefer an explicit Metal name; fall back to first GPU on Darwin.
            chosen = dev;
        }
    }
    if (!chosen) {
        printf("host-skip: no Metal backend (C1/C2 FA fixtures)\n");
        return 77;
    }
    ggml_backend_ptr backend_holder(ggml_backend_dev_init(chosen, nullptr));
    if (!backend_holder) {
        printf("host-skip: Metal backend init failed\n");
        return 77;
    }
    ggml_backend_t backend = backend_holder.get();
    printf("backend=%s device=%s\n", ggml_backend_name(backend), ggml_backend_dev_description(chosen));

    // Formula gate (C2 thresholds from ST-1 / skoulik).
    GGML_ASSERT(nsg_for_ne11(2047) == 1);
    GGML_ASSERT(nsg_for_ne11(2048) == 1);
    GGML_ASSERT(nsg_for_ne11(2049) == 2);
    GGML_ASSERT(nsg_for_ne11(4095) == 2);
    GGML_ASSERT(nsg_for_ne11(4096) == 2);
    GGML_ASSERT(nsg_for_ne11(4097) == 4);
    printf("C2 formula thresholds OK (2048/4096)\n");

    int fails = 0;
    const std::vector<int64_t> widths = {2, 3, 4, 5, 8};

    // C1 at each boundary KV (same ne11 → same nsg; still must be row0-exact).
    for (int64_t kv : {2047, 2048, 2049, 4095, 4096, 4097}) {
        check_c1(backend, kv, widths, fails);
    }

    // C2: verify batch crosses nsg threshold earlier than decode.
    // decode_kv=2048 (nsg=1), N=2 → verify_kv=2049 (nsg=2) before pin.
    check_c2_boundary(backend, 2048, 2, fails);
    check_c2_boundary(backend, 2047, 3, fails);
    check_c2_boundary(backend, 4096, 2, fails);
    check_c2_boundary(backend, 4095, 3, fails);

    if (fails != 0) {
        printf("RESULT FAIL fails=%d\n", fails);
        return 1;
    }
    printf("RESULT PASS\n");
    return 0;
}
