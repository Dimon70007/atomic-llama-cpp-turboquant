// TASK-066 ST-3.2d — MoE mul_mv_id row0 localizer (P5 residual after FA long-KV GREEN).
//
// C5 (test-mul-mat-batch-invariance) only covered Q8_0, 4 experts / top-2, uniform ids.
// Product Gemma-4-26B-A4B is IQ4_XS, 128 experts / top-8 (+ shared MLP outside this op).
// Hypothesis: IQ4_XS mul_mv_id with Gemma-like used-expert count and heterogeneous routing
// across verify tokens (N=2..8) can break bit-exact row0 vs serial N=1.
//
// TDD: RED → patch C candidate on MoE surface; GREEN → MoE micro-op not the residual
// (next: logits / first-diff instrumentation). Host-skip 77 without Metal/GPU.

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

struct moe_shape {
    int64_t k;               // ne00 (inner) — must be IQ4_XS-aligned (multiple of 256)
    int64_t m;               // ne01 (rows)
    int64_t n_experts;       // ne02
    int64_t n_expert_used;   // ne20
};

std::vector<uint8_t> quantize_weights(ggml_type type, const std::vector<float> & src_f32, int64_t ne0) {
    GGML_ASSERT(src_f32.size() % size_t(ne0) == 0);
    const int64_t nrows = int64_t(src_f32.size() / size_t(ne0));
    const size_t row_bytes = ggml_row_size(type, ne0);
    std::vector<uint8_t> out(size_t(nrows) * row_bytes, 0);
    std::vector<float> imatrix(size_t(ne0), 1.0f);
    const size_t written = ggml_quantize_chunk(
            type, src_f32.data(), out.data(), /*start=*/0, nrows, ne0, imatrix.data());
    GGML_ASSERT(written == out.size());
    return out;
}

void fill_sin(std::vector<float> & v, float phase) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.13f * std::sin(float(i) * 0.017f + phase);
    }
}

std::vector<float> run_mul_mat_id(
        ggml_backend_t backend,
        ggml_type wtype,
        const moe_shape & sh,
        int64_t n_tokens,
        const std::vector<uint8_t> & w_bytes,
        const std::vector<float> & act_f32,
        const std::vector<int32_t> & ids) {
    // weights: [k, m, n_experts]; acts: [k, n_expert_used, n_tokens]; ids: [n_expert_used, n_tokens]
    // Use no_alloc + backend buffers — product-ish K=2816 × experts can exceed ctx RAM.
    const size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(128, false) + 64 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_holder(ggml_init(params));
    ggml_context * ctx = ctx_holder.get();
    GGML_ASSERT(ctx);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, wtype, sh.k, sh.m, sh.n_experts);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.k, sh.n_expert_used, n_tokens);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, sh.n_expert_used, n_tokens);
    ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, id);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    GGML_ASSERT(buf);
    ggml_backend_tensor_set(w, w_bytes.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, act_f32.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(id, ids.data(), 0, ggml_nbytes(id));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(gf, y);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> out(size_t(ggml_nelements(y)), 0.0f);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    return out;
}

void fill_uniform_ids(std::vector<int32_t> & ids, int64_t n_expert_used, int64_t n_tokens) {
    // Every token uses experts [0 .. n_expert_used).
    ids.assign(size_t(n_expert_used * n_tokens), 0);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert_used; ++e) {
            ids[size_t(e + t * n_expert_used)] = int32_t(e);
        }
    }
}

void fill_hetero_ids(
        std::vector<int32_t> & ids,
        int64_t n_experts,
        int64_t n_expert_used,
        int64_t n_tokens) {
    // Token 0: experts [0 .. n_expert_used). Other tokens: rotated distinct sets.
    // Speculative verify packs draft tokens with different routers — row0 must stay stable.
    ids.assign(size_t(n_expert_used * n_tokens), 0);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert_used; ++e) {
            const int64_t expert = (t == 0)
                ? e
                : (e + t * 3) % n_experts;
            ids[size_t(e + t * n_expert_used)] = int32_t(expert);
        }
    }
}

bool compare_row0(
        const std::vector<float> & serial,
        const std::vector<float> & batch,
        const moe_shape & sh,
        int64_t /*n_tokens*/,
        bool & exact,
        double & mad) {
    // out [m, n_expert_used, n_tokens]; token0 plane at t=0
    GGML_ASSERT(serial.size() == size_t(sh.m * sh.n_expert_used));
    GGML_ASSERT(batch.size() >= serial.size());
    exact = true;
    mad = 0.0;
    for (int64_t e = 0; e < sh.n_expert_used; ++e) {
        for (int64_t r = 0; r < sh.m; ++r) {
            const size_t si = size_t(r + e * sh.m);
            const float a = serial[si];
            const float b = batch[si];
            mad = std::max(mad, double(std::fabs(a - b)));
            if (memcmp(&a, &b, sizeof(float)) != 0) {
                exact = false;
            }
        }
    }
    return exact;
}

bool check_moe_case(
        ggml_backend_t backend,
        ggml_type wtype,
        const char * label,
        const moe_shape & sh,
        bool hetero_ids,
        const std::vector<int64_t> & widths,
        int & fails) {
    printf("MOE [%s] type=%s k=%lld m=%lld experts=%lld used=%lld ids=%s\n",
           label, ggml_type_name(wtype),
           (long long) sh.k, (long long) sh.m,
           (long long) sh.n_experts, (long long) sh.n_expert_used,
           hetero_ids ? "hetero" : "uniform");

    GGML_ASSERT(sh.k % 256 == 0); // IQ4_XS / Q8_0 block-friendly
    GGML_ASSERT(sh.n_expert_used <= sh.n_experts);

    std::vector<float> w_f32(size_t(sh.k * sh.m * sh.n_experts), 0.0f);
    fill_sin(w_f32, 0.44f);
    const std::vector<uint8_t> w_q = quantize_weights(wtype, w_f32, sh.k);

    bool ok_all = true;
    for (int64_t n : widths) {
        std::vector<int32_t> ids;
        if (hetero_ids) {
            fill_hetero_ids(ids, sh.n_experts, sh.n_expert_used, n);
        } else {
            fill_uniform_ids(ids, sh.n_expert_used, n);
        }

        std::vector<float> x1(size_t(sh.k * sh.n_expert_used), 0.0f);
        fill_sin(x1, 0.61f);
        std::vector<float> xn(size_t(sh.k * sh.n_expert_used * n), 0.0f);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t e = 0; e < sh.n_expert_used; ++e) {
                for (int64_t i = 0; i < sh.k; ++i) {
                    const size_t dst = size_t(((t * sh.n_expert_used) + e) * sh.k + i);
                    const size_t src = size_t((e * sh.k) + i);
                    xn[dst] = (t == 0)
                        ? x1[src]
                        : 0.09f * std::sin(float(dst) * 0.031f + 2.0f);
                }
            }
        }

        std::vector<int32_t> ids1(ids.begin(), ids.begin() + size_t(sh.n_expert_used));
        const auto serial = run_mul_mat_id(backend, wtype, sh, 1, w_q, x1, ids1);
        const auto batch  = run_mul_mat_id(backend, wtype, sh, n, w_q, xn, ids);

        bool exact = false;
        double mad = 0.0;
        compare_row0(serial, batch, sh, n, exact, mad);
        printf("  ne21=N=%lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
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
        printf("host-skip: no Metal backend (TASK-066 MoE mul_mv_id localizer)\n");
        return 77;
    }
    ggml_backend_ptr backend_holder(ggml_backend_dev_init(chosen, nullptr));
    if (!backend_holder) {
        printf("host-skip: Metal backend init failed\n");
        return 77;
    }
    ggml_backend_t backend = backend_holder.get();
    printf("backend=%s device=%s\n", ggml_backend_name(backend), ggml_backend_dev_description(chosen));
    printf("HYPOTHESIS: IQ4_XS mul_mv_id row0 under Gemma-like top-8 + hetero routing (P5 residual)\n");
    printf("TDD: RED → MoE patch-C candidate; GREEN → next logits/first-diff localizer\n");
    printf("NOTE: stays on mul_mv_id (ne21 < 32); mul_mm_id threshold is 32 for non-TQ\n");

    int fails = 0;
    const std::vector<int64_t> widths = {2, 3, 4, 5, 8};

    // MOE-C1: product quant + top-8, still small K — extends C5 which skipped IQ4_XS id path.
    {
        moe_shape sh{/*k=*/256, /*m=*/64, /*n_experts=*/32, /*n_expert_used=*/8};
        check_moe_case(backend, GGML_TYPE_IQ4_XS, "c1-iq4xs-top8-uniform", sh, false, widths, fails);
    }

    // MOE-C2: same + heterogeneous expert ids across draft tokens (verify-batch realism).
    {
        moe_shape sh{/*k=*/256, /*m=*/64, /*n_experts=*/32, /*n_expert_used=*/8};
        check_moe_case(backend, GGML_TYPE_IQ4_XS, "c2-iq4xs-top8-hetero", sh, true, widths, fails);
    }

    // MOE-C3: Gemma-ish inner dim (embed 2816), subset of expert rows, top-8, hetero.
    {
        moe_shape sh{/*k=*/2816, /*m=*/128, /*n_experts=*/32, /*n_expert_used=*/8};
        check_moe_case(backend, GGML_TYPE_IQ4_XS, "c3-iq4xs-k2816-hetero", sh, true, widths, fails);
    }

    // MOE-C4: Q8_0 control with hetero (C5 was Q8_0 uniform only).
    {
        moe_shape sh{/*k=*/256, /*m=*/64, /*n_experts=*/32, /*n_expert_used=*/8};
        check_moe_case(backend, GGML_TYPE_Q8_0, "c4-q8_0-top8-hetero", sh, true, widths, fails);
    }

    // MOE-C5: closer to full expert count (128) but thinner rows — stress id map / tgpig.z packing.
    {
        moe_shape sh{/*k=*/256, /*m=*/64, /*n_experts=*/128, /*n_expert_used=*/8};
        check_moe_case(backend, GGML_TYPE_IQ4_XS, "c5-iq4xs-e128-hetero", sh, true, widths, fails);
    }

    if (fails != 0) {
        printf("RESULT FAIL fails=%d (RED → MoE mul_mv_id patch-C candidate on this backend)\n", fails);
        return 1;
    }
    printf("RESULT PASS (GREEN on this backend for MoE micro-op; next logits/first-diff)\n");
    return 0;
}
