// TASK-066 ST-3.2d — MoE router→soft_max→argsort_top_k→mul_mat_id row0 localizer.
//
// Prior MoE micro-op (fixed expert ids) is GREEN on Metal. Real Gemma MoE computes
// expert ids from router logits per token. Hypothesis: N>1 verify batches flip row0
// expert selection and/or MoE out even when mul_mat_id(fixed ids) is exact.
//
// Self-contained: build & run inside this repo only (no external project deps).
// Env:
//   TASK066_ROUTER_STAGE=logits|probs|ids|moe  — stop after stage (bisection; default moe)
// TDD: RED → patch C candidate; GREEN → live first-diff dump next (product harness elsewhere).
// Host-skip 77 without Metal/GPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int64_t kK = 256;
constexpr int64_t kM = 64;
constexpr int64_t kNExperts = 32;
constexpr int64_t kNExpertUsed = 8;

enum class stage_t { logits, probs, ids, moe };

stage_t parse_stage() {
    const char * s = std::getenv("TASK066_ROUTER_STAGE");
    if (!s || !*s) {
        return stage_t::moe;
    }
    if (std::strcmp(s, "logits") == 0) return stage_t::logits;
    if (std::strcmp(s, "probs") == 0)  return stage_t::probs;
    if (std::strcmp(s, "ids") == 0)    return stage_t::ids;
    if (std::strcmp(s, "moe") == 0)    return stage_t::moe;
    fprintf(stderr, "unknown TASK066_ROUTER_STAGE=%s (use logits|probs|ids|moe)\n", s);
    std::exit(2);
}

struct chain_out {
    std::vector<float> router_logits;
    std::vector<float> router_probs;
    std::vector<int32_t> selected;
    std::vector<float> moe_out;
    bool has_probs = false;
    bool has_ids = false;
    bool has_moe = false;
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

void fill_near_tie_router(std::vector<float> & w_f32) {
    GGML_ASSERT(w_f32.size() == size_t(kK * kNExperts));
    for (int64_t e = 0; e < kNExperts; ++e) {
        for (int64_t i = 0; i < kK; ++i) {
            w_f32[size_t(e * kK + i)] =
                0.11f * std::sin(float(i) * 0.019f + 0.5f)
                + 1.0e-4f * float(e) * std::cos(float(i) * 0.007f + 0.2f);
        }
    }
}

chain_out run_router_moe_chain(
        ggml_backend_t backend,
        ggml_type wtype,
        int64_t n_tokens,
        const std::vector<uint8_t> & w_router_q,
        const std::vector<uint8_t> & w_expert_q,
        const std::vector<float> & act_f32,
        stage_t stage) {
    const size_t mem = ggml_tensor_overhead() * 96 + ggml_graph_overhead_custom(256, false) + 64 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_holder(ggml_init(params));
    ggml_context * ctx = ctx_holder.get();
    GGML_ASSERT(ctx);

    ggml_tensor * w_r = ggml_new_tensor_2d(ctx, wtype, kK, kNExperts);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, n_tokens);
    ggml_tensor * logits = ggml_mul_mat(ctx, w_r, x);

    ggml_tensor * leaf = logits;
    ggml_tensor * probs = nullptr;
    ggml_tensor * selected = nullptr;
    ggml_tensor * selected_c = nullptr;
    ggml_tensor * moe = nullptr;

    if (stage != stage_t::logits) {
        probs = ggml_soft_max(ctx, logits);
        leaf = probs;
    }
    if (stage == stage_t::ids || stage == stage_t::moe) {
        // Production MoE uses argsort_top_k (view). Contiguous copy avoids VIEW-as-ids edge cases in the harness.
        selected = ggml_argsort_top_k(ctx, probs, int(kNExpertUsed));
        selected_c = ggml_cont(ctx, selected);
        leaf = selected_c;
    }
    if (stage == stage_t::moe) {
        ggml_tensor * w_e = ggml_new_tensor_3d(ctx, wtype, kK, kM, kNExperts);
        ggml_tensor * x_e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, kNExpertUsed, n_tokens);
        moe = ggml_mul_mat_id(ctx, w_e, x_e, selected_c);
        leaf = moe;
    }

    printf("    graph_alloc n_tokens=%lld stage=%d ...\n", (long long) n_tokens, int(stage));
    fflush(stdout);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    GGML_ASSERT(buf);

    ggml_backend_tensor_set(w_r, w_router_q.data(), 0, ggml_nbytes(w_r));
    ggml_backend_tensor_set(x, act_f32.data(), 0, ggml_nbytes(x));

    if (stage == stage_t::moe) {
        // w_e / x_e are src of moe
        ggml_tensor * w_e = moe->src[0];
        ggml_tensor * x_e = moe->src[1];
        ggml_backend_tensor_set(w_e, w_expert_q.data(), 0, ggml_nbytes(w_e));
        std::vector<float> x_e_host(size_t(kK * kNExpertUsed * n_tokens), 0.0f);
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t e = 0; e < kNExpertUsed; ++e) {
                for (int64_t i = 0; i < kK; ++i) {
                    x_e_host[size_t(((t * kNExpertUsed) + e) * kK + i)] = act_f32[size_t(t * kK + i)];
                }
            }
        }
        ggml_backend_tensor_set(x_e, x_e_host.data(), 0, ggml_nbytes(x_e));
    }

    printf("    graph_compute ...\n");
    fflush(stdout);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gf, leaf);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    printf("    graph_done\n");
    fflush(stdout);

    chain_out out;
    out.router_logits.resize(size_t(ggml_nelements(logits)));
    ggml_backend_tensor_get(logits, out.router_logits.data(), 0, ggml_nbytes(logits));

    if (probs) {
        out.has_probs = true;
        out.router_probs.resize(size_t(ggml_nelements(probs)));
        ggml_backend_tensor_get(probs, out.router_probs.data(), 0, ggml_nbytes(probs));
    }
    if (selected_c) {
        out.has_ids = true;
        out.selected.resize(size_t(ggml_nelements(selected_c)));
        ggml_backend_tensor_get(selected_c, out.selected.data(), 0, ggml_nbytes(selected_c));
    }
    if (moe) {
        out.has_moe = true;
        out.moe_out.resize(size_t(ggml_nelements(moe)));
        ggml_backend_tensor_get(moe, out.moe_out.data(), 0, ggml_nbytes(moe));
    }
    return out;
}

bool memcmp_prefix(const void * a, const void * b, size_t nbytes) {
    return memcmp(a, b, nbytes) == 0;
}

bool check_chain(
        ggml_backend_t backend,
        ggml_type wtype,
        const char * label,
        bool near_tie,
        const std::vector<int64_t> & widths,
        stage_t stage,
        int & fails) {
    printf("ROUTER-CHAIN [%s] type=%s near_tie=%d experts=%lld used=%lld stage=%d\n",
           label, ggml_type_name(wtype), near_tie ? 1 : 0,
           (long long) kNExperts, (long long) kNExpertUsed, int(stage));
    fflush(stdout);

    std::vector<float> w_router_f32(size_t(kK * kNExperts), 0.0f);
    if (near_tie) {
        fill_near_tie_router(w_router_f32);
    } else {
        fill_sin(w_router_f32, 0.33f);
    }
    const std::vector<uint8_t> w_router_q = quantize_weights(wtype, w_router_f32, kK);

    std::vector<float> w_expert_f32(size_t(kK * kM * kNExperts), 0.0f);
    fill_sin(w_expert_f32, 0.77f);
    const std::vector<uint8_t> w_expert_q = quantize_weights(wtype, w_expert_f32, kK);

    bool ok_all = true;
    for (int64_t n : widths) {
        std::vector<float> x1(size_t(kK), 0.0f);
        fill_sin(x1, 0.91f);
        std::vector<float> xn(size_t(kK * n), 0.0f);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t i = 0; i < kK; ++i) {
                xn[size_t(t * kK + i)] = (t == 0)
                    ? x1[size_t(i)]
                    : 0.09f * std::sin(float(t * kK + i) * 0.029f + 1.4f);
            }
        }

        const chain_out serial = run_router_moe_chain(
                backend, wtype, 1, w_router_q, w_expert_q, x1, stage);
        const chain_out batch = run_router_moe_chain(
                backend, wtype, n, w_router_q, w_expert_q, xn, stage);

        const bool logits_ok = memcmp_prefix(
                serial.router_logits.data(), batch.router_logits.data(),
                size_t(kNExperts) * sizeof(float));
        bool probs_ok = true;
        bool ids_ok = true;
        bool moe_ok = true;

        if (serial.has_probs) {
            probs_ok = memcmp_prefix(
                    serial.router_probs.data(), batch.router_probs.data(),
                    size_t(kNExperts) * sizeof(float));
        }
        if (serial.has_ids) {
            ids_ok = memcmp_prefix(
                    serial.selected.data(), batch.selected.data(),
                    size_t(kNExpertUsed) * sizeof(int32_t));
        }
        if (serial.has_moe) {
            moe_ok = memcmp_prefix(
                    serial.moe_out.data(), batch.moe_out.data(),
                    size_t(kM * kNExpertUsed) * sizeof(float));
        }

        double mad_logits = 0.0;
        for (int64_t e = 0; e < kNExperts; ++e) {
            mad_logits = std::max(
                    mad_logits,
                    double(std::fabs(serial.router_logits[size_t(e)] - batch.router_logits[size_t(e)])));
        }

        const bool exact = logits_ok && probs_ok && ids_ok && moe_ok;
        printf("  N=%lld logits=%d probs=%d ids=%d moe=%d mad_logits=%.6g %s\n",
               (long long) n,
               logits_ok ? 1 : 0, probs_ok ? 1 : 0, ids_ok ? 1 : 0, moe_ok ? 1 : 0,
               mad_logits, exact ? "PASS" : "FAIL");
        fflush(stdout);
        if (!ids_ok && serial.has_ids) {
            printf("    serial_ids:");
            for (int64_t e = 0; e < kNExpertUsed; ++e) {
                printf(" %d", serial.selected[size_t(e)]);
            }
            printf("\n    batch_ids: ");
            for (int64_t e = 0; e < kNExpertUsed; ++e) {
                printf(" %d", batch.selected[size_t(e)]);
            }
            printf("\n");
        }
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

} // namespace

int main() {
    const stage_t stage = parse_stage();

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
        printf("host-skip: no Metal backend (TASK-066 MoE router/logits localizer)\n");
        return 77;
    }
    ggml_backend_ptr backend_holder(ggml_backend_dev_init(chosen, nullptr));
    if (!backend_holder) {
        printf("host-skip: Metal backend init failed\n");
        return 77;
    }
    ggml_backend_t backend = backend_holder.get();
    printf("backend=%s device=%s\n", ggml_backend_name(backend), ggml_backend_dev_description(chosen));
    printf("HYPOTHESIS: router soft_max/argsort_top_k flips row0 experts under N>1\n");
    printf("TDD: RED → logits/router patch-C candidate; GREEN → live first-diff dump next\n");
    printf("TASK066_ROUTER_STAGE=%d (0=logits 1=probs 2=ids 3=moe)\n", int(stage));
    fflush(stdout);

    int fails = 0;
    const std::vector<int64_t> widths = {2, 3, 4, 5, 8};

    check_chain(backend, GGML_TYPE_IQ4_XS, "rt1-iq4xs-smooth", false, widths, stage, fails);
    check_chain(backend, GGML_TYPE_IQ4_XS, "rt2-iq4xs-neartie", true, widths, stage, fails);
    check_chain(backend, GGML_TYPE_Q8_0, "rt3-q8_0-neartie", true, widths, stage, fails);

    if (fails != 0) {
        printf("RESULT FAIL fails=%d (RED → MoE router/logits patch-C candidate)\n", fails);
        return 1;
    }
    printf("RESULT PASS (GREEN router chain stage; escalate stage or live first-diff)\n");
    return 0;
}
