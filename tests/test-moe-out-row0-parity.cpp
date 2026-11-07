// TASK-066 ST-3.2f-ter-loc - MoE ffn_moe_out dual-topology row0 localizer.
//
// Bisect N=64 (A+B): first value RED at ffn_moe_out-0 under dual topology.
// Patch C: build_moe_ffn always uses view+ADD for n_expert_used>1 (no decode-only
// permute+sum_rows). This localizer mirrors that unified reduce for N=1 and N>=2
// and asserts bit-exact row0 (C1). Pre-patch dual-topology RED stamp:
//   logs/task066-batch-invariance-moe-out-20261106T002308Z
// Host-skip 77 without Metal/GPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct moe_out_shape {
    int64_t n_embd;        // Gemma-4-26B-A4B: 2816
    int64_t n_expert_used; // top-8
};

void fill_sin(std::vector<float> & v, float phase) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.17f * std::sin(float(i) * 0.019f + phase);
    }
}

bool compare_exact(const std::vector<float> & a, const std::vector<float> & b, double & mad) {
    GGML_ASSERT(a.size() == b.size());
    bool exact = true;
    mad = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        mad = std::max(mad, double(std::fabs(a[i] - b[i])));
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            exact = false;
        }
    }
    return exact;
}

// Decode path (n_tokens==1): permute -> cont -> sum_rows -> reshape_2d
std::vector<float> reduce_permute_sum(
        ggml_backend_t backend,
        const moe_out_shape & sh,
        const std::vector<float> & weighted /* [n_embd, n_expert_used, 1] */) {
    GGML_ASSERT(weighted.size() == size_t(sh.n_embd * sh.n_expert_used));

    const size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false) + 64 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_holder(ggml_init(params));
    ggml_context * ctx = ctx_holder.get();
    GGML_ASSERT(ctx);

    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.n_embd, sh.n_expert_used, 1);
    ggml_set_name(experts, "ffn_moe_weighted");

    ggml_tensor * experts_pe = ggml_cont(ctx, ggml_permute(ctx, experts, 1, 0, 2, 3));
    ggml_set_name(experts_pe, "ffn_moe_weighted (permuted)");
    ggml_tensor * summed = ggml_sum_rows(ctx, experts_pe);
    ggml_tensor * moe_out = ggml_reshape_2d(ctx, summed, sh.n_embd, 1);
    ggml_set_name(moe_out, "ffn_moe_out");

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    GGML_ASSERT(buf);
    ggml_backend_tensor_set(experts, weighted.data(), 0, ggml_nbytes(experts));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, moe_out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> out(size_t(sh.n_embd), 0.0f);
    ggml_backend_tensor_get(moe_out, out.data(), 0, ggml_nbytes(moe_out));
    return out;
}

// Prefill / verify path (n_tokens>=2): view_2d per expert + ADD chain
std::vector<float> reduce_view_add(
        ggml_backend_t backend,
        const moe_out_shape & sh,
        int64_t n_tokens,
        const std::vector<float> & weighted /* [n_embd, n_expert_used, n_tokens] */) {
    GGML_ASSERT(n_tokens >= 1);
    GGML_ASSERT(weighted.size() == size_t(sh.n_embd * sh.n_expert_used * n_tokens));

    const size_t mem = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false) + 64 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_holder(ggml_init(params));
    ggml_context * ctx = ctx_holder.get();
    GGML_ASSERT(ctx);

    ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.n_embd, sh.n_expert_used, n_tokens);
    ggml_set_name(experts, "ffn_moe_weighted");

    std::vector<ggml_tensor *> cur_experts(size_t(sh.n_expert_used), nullptr);
    for (int64_t i = 0; i < sh.n_expert_used; ++i) {
        cur_experts[size_t(i)] = ggml_view_2d(
                ctx, experts, sh.n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);
        ggml_set_name(cur_experts[size_t(i)], "ffn_moe_weighted (view)");
    }
    ggml_tensor * moe_out = cur_experts[0];
    for (int64_t i = 1; i < sh.n_expert_used; ++i) {
        moe_out = ggml_add(ctx, moe_out, cur_experts[size_t(i)]);
    }
    ggml_set_name(moe_out, "ffn_moe_out");

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    GGML_ASSERT(buf);
    ggml_backend_tensor_set(experts, weighted.data(), 0, ggml_nbytes(experts));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    for (int64_t i = 0; i < sh.n_expert_used; ++i) {
        ggml_build_forward_expand(gf, cur_experts[size_t(i)]);
    }
    ggml_build_forward_expand(gf, moe_out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> full(size_t(sh.n_embd * n_tokens), 0.0f);
    ggml_backend_tensor_get(moe_out, full.data(), 0, ggml_nbytes(moe_out));
    return std::vector<float>(full.begin(), full.begin() + size_t(sh.n_embd));
}

bool check_case(
        ggml_backend_t backend,
        const char * label,
        const moe_out_shape & sh,
        int64_t n_tokens,
        int & fails) {
    printf("MOE-OUT [%s] n_embd=%lld n_used=%lld N=%lld\n",
           label, (long long) sh.n_embd, (long long) sh.n_expert_used, (long long) n_tokens);

    std::vector<float> w1(size_t(sh.n_embd * sh.n_expert_used), 0.0f);
    fill_sin(w1, 0.71f);

    std::vector<float> wn(size_t(sh.n_embd * sh.n_expert_used * n_tokens), 0.0f);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < sh.n_expert_used; ++e) {
            for (int64_t i = 0; i < sh.n_embd; ++i) {
                const size_t dst = size_t(((t * sh.n_expert_used) + e) * sh.n_embd + i);
                const size_t src = size_t(e * sh.n_embd + i);
                wn[dst] = (t == 0)
                    ? w1[src]
                    : 0.11f * std::sin(float(dst) * 0.029f + 1.7f);
            }
        }
    }

    // Unified path (patch C): view+ADD for both serial N=1 and verify N>=2.
    const auto serial = reduce_view_add(backend, sh, 1, w1);
    const auto batch0 = reduce_view_add(backend, sh, n_tokens, wn);

    double mad = 0.0;
    const bool exact = compare_exact(serial, batch0, mad);
    printf("  exact=%d max_abs_diff=%.6g %s\n", exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
    if (!exact) {
        ++fails;
        return false;
    }
    return true;
}

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

struct weighted_out {
    std::vector<float> moe_out;
};

weighted_out run_weighted_then_reduce(
        ggml_backend_t backend,
        ggml_type wtype,
        int64_t k,
        int64_t n_embd,
        int64_t n_experts,
        int64_t n_expert_used,
        int64_t n_tokens,
        bool use_permute_path,
        const std::vector<uint8_t> & w_bytes,
        const std::vector<float> & act_f32,
        const std::vector<int32_t> & ids,
        const std::vector<float> & gate_w /* [1, n_expert_used, n_tokens] */) {
    const size_t mem = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false) + 256 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_holder(ggml_init(params));
    ggml_context * ctx = ctx_holder.get();
    GGML_ASSERT(ctx);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, wtype, k, n_embd, n_experts);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_expert_used, n_tokens);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * gw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_expert_used, n_tokens);

    ggml_tensor * experts = ggml_mul_mat_id(ctx, w, x, id);
    ggml_set_name(experts, "ffn_moe_down");
    ggml_tensor * weighted = ggml_mul(ctx, experts, gw);
    ggml_set_name(weighted, "ffn_moe_weighted");

    ggml_tensor * moe_out = nullptr;
    std::vector<ggml_tensor *> views;
    if (use_permute_path) {
        // Legacy dual-topology path (pre-patch C). Kept for optional RED repro;
        // production + default localizer use view+ADD only.
        GGML_ASSERT(n_tokens == 1);
        ggml_tensor * pe = ggml_cont(ctx, ggml_permute(ctx, weighted, 1, 0, 2, 3));
        ggml_set_name(pe, "ffn_moe_weighted (permuted)");
        ggml_tensor * summed = ggml_sum_rows(ctx, pe);
        moe_out = ggml_reshape_2d(ctx, summed, n_embd, 1);
    } else {
        GGML_ASSERT(n_tokens >= 1);
        views.resize(size_t(n_expert_used), nullptr);
        for (int64_t i = 0; i < n_expert_used; ++i) {
            views[size_t(i)] = ggml_view_2d(
                    ctx, weighted, n_embd, n_tokens, weighted->nb[2], i * weighted->nb[1]);
            ggml_set_name(views[size_t(i)], "ffn_moe_weighted (view)");
        }
        moe_out = views[0];
        for (int64_t i = 1; i < n_expert_used; ++i) {
            moe_out = ggml_add(ctx, moe_out, views[size_t(i)]);
        }
    }
    ggml_set_name(moe_out, "ffn_moe_out");

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    GGML_ASSERT(buf);
    ggml_backend_tensor_set(w, w_bytes.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, act_f32.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(id, ids.data(), 0, ggml_nbytes(id));
    ggml_backend_tensor_set(gw, gate_w.data(), 0, ggml_nbytes(gw));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    if (!use_permute_path) {
        for (int64_t i = 0; i < n_expert_used; ++i) {
            ggml_build_forward_expand(gf, views[size_t(i)]);
        }
    }
    ggml_build_forward_expand(gf, moe_out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    weighted_out out;
    if (use_permute_path) {
        out.moe_out.resize(size_t(n_embd), 0.0f);
        ggml_backend_tensor_get(moe_out, out.moe_out.data(), 0, ggml_nbytes(moe_out));
    } else {
        std::vector<float> full(size_t(n_embd * n_tokens), 0.0f);
        ggml_backend_tensor_get(moe_out, full.data(), 0, ggml_nbytes(moe_out));
        out.moe_out.assign(full.begin(), full.begin() + size_t(n_embd));
    }
    return out;
}

bool check_weighted_case(
        ggml_backend_t backend,
        const char * label,
        ggml_type wtype,
        int64_t k,
        int64_t n_embd,
        int64_t n_experts,
        int64_t n_expert_used,
        int64_t n_tokens,
        int & fails) {
    printf("MOE-OUT-W [%s] type=%s k=%lld n_embd=%lld experts=%lld used=%lld N=%lld\n",
           label, ggml_type_name(wtype),
           (long long) k, (long long) n_embd,
           (long long) n_experts, (long long) n_expert_used, (long long) n_tokens);

    GGML_ASSERT(k % 256 == 0);
    GGML_ASSERT(n_tokens >= 2);

    std::vector<float> w_f32(size_t(k * n_embd * n_experts), 0.0f);
    fill_sin(w_f32, 0.44f);
    const auto w_q = quantize_weights(wtype, w_f32, k);

    std::vector<int32_t> ids(size_t(n_expert_used * n_tokens), 0);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert_used; ++e) {
            ids[size_t(e + t * n_expert_used)] = int32_t((e + t * 3) % n_experts);
        }
    }
    for (int64_t e = 0; e < n_expert_used; ++e) {
        ids[size_t(e)] = int32_t(e);
    }

    std::vector<float> x1(size_t(k * n_expert_used), 0.0f);
    fill_sin(x1, 0.61f);
    std::vector<float> xn(size_t(k * n_expert_used * n_tokens), 0.0f);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert_used; ++e) {
            for (int64_t i = 0; i < k; ++i) {
                const size_t dst = size_t(((t * n_expert_used) + e) * k + i);
                const size_t src = size_t(e * k + i);
                xn[dst] = (t == 0) ? x1[src] : 0.09f * std::sin(float(dst) * 0.031f + 2.0f);
            }
        }
    }

    std::vector<float> gw1(size_t(n_expert_used), 0.0f);
    for (int64_t e = 0; e < n_expert_used; ++e) {
        gw1[size_t(e)] = 0.05f + 0.1f * float(e);
    }
    std::vector<float> gwn(size_t(n_expert_used * n_tokens), 0.0f);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert_used; ++e) {
            gwn[size_t(e + t * n_expert_used)] = (t == 0)
                ? gw1[size_t(e)]
                : 0.07f * float((e + t) % 7);
        }
    }

    std::vector<int32_t> ids1(ids.begin(), ids.begin() + size_t(n_expert_used));
    // Unified path (patch C): view+ADD for N=1 and N>=2 (no permute+sum_rows).
    const auto serial = run_weighted_then_reduce(
            backend, wtype, k, n_embd, n_experts, n_expert_used, 1,
            /*use_permute_path=*/false, w_q, x1, ids1, gw1);
    const auto batch = run_weighted_then_reduce(
            backend, wtype, k, n_embd, n_experts, n_expert_used, n_tokens,
            /*use_permute_path=*/false, w_q, xn, ids, gwn);

    double mad = 0.0;
    const bool exact = compare_exact(serial.moe_out, batch.moe_out, mad);
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
        printf("host-skip: no Metal backend (TASK-066 MoE ffn_moe_out localizer)\n");
        return 77;
    }
    ggml_backend_ptr backend_holder(ggml_backend_dev_init(chosen, nullptr));
    if (!backend_holder) {
        printf("host-skip: Metal backend init failed\n");
        return 77;
    }
    ggml_backend_t backend = backend_holder.get();
    printf("backend=%s device=%s\n", ggml_backend_name(backend), ggml_backend_dev_description(chosen));
    printf("CONTRACT: unified MoE reduce (view+ADD) row0 bit-exact N=1 vs N>=2 (TASK-066 patch C)\n");
    printf("PRE-PATCH RED: dual permute vs view (stamp moe-out-20261106T002308Z); post-patch expect GREEN\n");
    printf("SOURCE: llama-graph build_moe_ffn; bisect FIRST_DIFF=ffn_moe_out-0\n");

    int fails = 0;

    // Optional: re-prove pre-patch dual-topology RED (does not gate KEEP; diagnostic only).
    if (const char * dual = std::getenv("TASK066_MOE_OUT_DUAL")) {
        if (dual[0] == '1') {
            printf("MODE=DUAL_TOPOLOGY (legacy permute N=1 vs view N>=2; expect RED without bit-identity)\n");
            moe_out_shape sh{/*n_embd=*/2816, /*n_expert_used=*/8};
            std::vector<float> w1(size_t(sh.n_embd * sh.n_expert_used), 0.0f);
            fill_sin(w1, 0.71f);
            std::vector<float> wn(size_t(sh.n_embd * sh.n_expert_used * 2), 0.0f);
            for (int64_t e = 0; e < sh.n_expert_used; ++e) {
                for (int64_t i = 0; i < sh.n_embd; ++i) {
                    const size_t src = size_t(e * sh.n_embd + i);
                    wn[src] = w1[src];
                    wn[size_t((1 * sh.n_expert_used + e) * sh.n_embd + i)] =
                        0.11f * std::sin(float(src) * 0.029f + 1.7f);
                }
            }
            const auto serial = reduce_permute_sum(backend, sh, w1);
            const auto batch0 = reduce_view_add(backend, sh, 2, wn);
            double mad = 0.0;
            const bool exact = compare_exact(serial, batch0, mad);
            printf("DUAL exact=%d mad=%.6g %s\n", exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
            return exact ? 0 : 1;
        }
    }

    {
        moe_out_shape sh{/*n_embd=*/2816, /*n_expert_used=*/8};
        check_case(backend, "c1-f32-reduce-embd2816-top8-N2", sh, 2, fails);
    }
    {
        moe_out_shape sh{/*n_embd=*/2816, /*n_expert_used=*/8};
        check_case(backend, "c2-f32-reduce-embd2816-top8-N8", sh, 8, fails);
    }
    {
        moe_out_shape sh{/*n_embd=*/256, /*n_expert_used=*/8};
        check_case(backend, "c3-f32-reduce-embd256-top8-N2", sh, 2, fails);
    }
    {
        check_weighted_case(
                backend, "c4-iq4xs-weighted-reduce-N2",
                GGML_TYPE_IQ4_XS, /*k=*/256, /*n_embd=*/128,
                /*n_experts=*/32, /*n_expert_used=*/8, /*n_tokens=*/2, fails);
    }
    {
        check_weighted_case(
                backend, "c5-iq4xs-k2816-embd2816-N2",
                GGML_TYPE_IQ4_XS, /*k=*/2816, /*n_embd=*/2816,
                /*n_experts=*/32, /*n_expert_used=*/8, /*n_tokens=*/2, fails);
    }

    if (fails != 0) {
        printf("RESULT FAIL fails=%d (RED - unified view+ADD row0 still diverges)\n", fails);
        return 1;
    }
    printf("RESULT PASS (GREEN - MoE ffn_moe_out unified reduce row0 bit-exact)\n");
    return 0;
}
