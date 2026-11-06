// TASK-066 / llama.cpp#25618 — Metal mul_mat / mul_mat_id batch invariance.
// Applicable to Metal patch B (disable mul_mv_ext for ne11 in 2..8):
//   C1 row0 parity, C4 Q8_0 (+ IQ4_XS control), C5 mul_mat_id verify widths,
//   C6 via separate smoke script (test-backend-ops).
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

constexpr int64_t kK = 256;   // ne00 % 128 == 0 → eligible for mul_mv_ext
constexpr int64_t kM = 64;    // output rows

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

bool row0_bit_identical(const std::vector<float> & serial, const std::vector<float> & batch, int64_t n_cols) {
    GGML_ASSERT(serial.size() == size_t(kM));
    GGML_ASSERT(batch.size() == size_t(kM * n_cols));
    for (int64_t r = 0; r < kM; ++r) {
        const float a = serial[size_t(r)];
        const float b = batch[size_t(r)]; // column-major? ggml mul_mat out is [m, n] → element (i,j) at i + j*m
        // out ne = [m, n] contiguous: index = i + j * ne0 = r + 0 * kM for col0
        if (memcmp(&a, &b, sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

double max_abs_diff_row0(const std::vector<float> & serial, const std::vector<float> & batch, int64_t /*n_cols*/) {
    double mad = 0.0;
    for (int64_t r = 0; r < kM; ++r) {
        mad = std::max(mad, double(std::fabs(serial[size_t(r)] - batch[size_t(r)])));
    }
    return mad;
}

size_t mul_mat_scratch_bytes(int64_t n) {
    // acts + outs for large ne11 (product ladder band up to 1024)
    const size_t acts = size_t(kK) * size_t(n) * sizeof(float);
    const size_t outs = size_t(kM) * size_t(n) * sizeof(float);
    return 2 * 1024 * 1024 + acts + outs;
}

std::vector<float> run_mul_mat(
        ggml_backend_t backend,
        ggml_type wtype,
        int64_t n,
        const std::vector<uint8_t> & w_bytes,
        const std::vector<float> & act_f32) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(128, false) +
                             mul_mat_scratch_bytes(n),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_holder(ggml_init(params));
    ggml_context * ctx = ctx_holder.get();

    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, kK, kM);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, n);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    ggml_backend_tensor_set(w, w_bytes.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, act_f32.data(), 0, ggml_nbytes(x));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> out(size_t(ggml_nelements(y)), 0.0f);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    return out;
}

std::vector<float> run_mul_mat_id(
        ggml_backend_t backend,
        ggml_type wtype,
        int64_t n_tokens,
        int64_t n_experts,
        int64_t n_expert_used,
        const std::vector<uint8_t> & w_bytes,
        const std::vector<float> & act_f32,
        const std::vector<int32_t> & ids) {
    // weights: [k, m, n_experts]; acts: [k, n_expert_used, n_tokens]; ids: [n_expert_used, n_tokens]
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(128, false) +
                             mul_mat_scratch_bytes(n_tokens) * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_holder(ggml_init(params));
    ggml_context * ctx = ctx_holder.get();

    ggml_tensor * w = ggml_new_tensor_3d(ctx, wtype, kK, kM, n_experts);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, n_expert_used, n_tokens);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, id);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    ggml_backend_tensor_set(w, w_bytes.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, act_f32.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(id, ids.data(), 0, ggml_nbytes(id));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> out(size_t(ggml_nelements(y)), 0.0f);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    return out;
}

bool check_c1_c4_mul_mat(
        ggml_backend_t backend,
        ggml_type wtype,
        const char * label,
        const std::vector<int64_t> & widths,
        int & fails) {
    printf("C1/C4 MUL_MAT [%s] type=%s k=%lld m=%lld\n", label, ggml_type_name(wtype),
           (long long) kK, (long long) kM);

    std::vector<float> w_f32(size_t(kK * kM), 0.0f);
    fill_sin(w_f32, 0.31f);
    const std::vector<uint8_t> w_q = quantize_weights(wtype, w_f32, kK);

    bool ok_all = true;
    for (int64_t n : widths) {
        std::vector<float> x1(size_t(kK), 0.0f);
        fill_sin(x1, 0.71f);
        std::vector<float> xn(size_t(kK * n), 0.0f);
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < kK; ++i) {
                xn[size_t(j * kK + i)] = (j == 0)
                    ? x1[size_t(i)]
                    : 0.11f * std::sin(float(j * kK + i) * 0.029f + 1.3f);
            }
        }

        const auto serial = run_mul_mat(backend, wtype, 1, w_q, x1);
        const auto batch  = run_mul_mat(backend, wtype, n, w_q, xn);
        const bool exact = row0_bit_identical(serial, batch, n);
        const double mad = max_abs_diff_row0(serial, batch, n);
        printf("  N=%lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

// Residual / product-ladder class: row0(N) bit-exact to row0(golden_n=8), not N=1.
// Covers ubatch-class drift {9,16,32,256,512,1024} vs S8 (TASK-066.6 / mul-mat-residual).
bool check_c1_c4_mul_mat_vs_golden(
        ggml_backend_t backend,
        ggml_type wtype,
        const char * label,
        int64_t golden_n,
        const std::vector<int64_t> & widths,
        int & fails) {
    printf("C1/C4 MUL_MAT vs golden N=%lld [%s] type=%s k=%lld m=%lld\n",
           (long long) golden_n, label, ggml_type_name(wtype), (long long) kK, (long long) kM);

    std::vector<float> w_f32(size_t(kK * kM), 0.0f);
    fill_sin(w_f32, 0.31f);
    const std::vector<uint8_t> w_q = quantize_weights(wtype, w_f32, kK);

    std::vector<float> x_row0(size_t(kK), 0.0f);
    fill_sin(x_row0, 0.71f);

    auto fill_batch = [&](int64_t n) {
        std::vector<float> xn(size_t(kK * n), 0.0f);
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < kK; ++i) {
                xn[size_t(j * kK + i)] = (j == 0)
                    ? x_row0[size_t(i)]
                    : 0.11f * std::sin(float(j * kK + i) * 0.029f + 1.3f);
            }
        }
        return xn;
    };

    const auto golden = run_mul_mat(backend, wtype, golden_n, w_q, fill_batch(golden_n));
    // row0 of golden is first kM floats (same layout as serial N=1 in row0_bit_identical)
    std::vector<float> golden_row0(golden.begin(), golden.begin() + size_t(kM));

    bool ok_all = true;
    for (int64_t n : widths) {
        const auto batch = run_mul_mat(backend, wtype, n, w_q, fill_batch(n));
        const bool exact = row0_bit_identical(golden_row0, batch, n);
        const double mad = max_abs_diff_row0(golden_row0, batch, n);
        printf("  N=%lld vs N=%lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, (long long) golden_n, exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

bool check_c5_mul_mat_id(
        ggml_backend_t backend,
        ggml_type wtype,
        const char * label,
        const std::vector<int64_t> & widths,
        int & fails) {
    constexpr int64_t n_experts = 4;
    constexpr int64_t n_expert_used = 2;
    printf("C5 MUL_MAT_ID [%s] type=%s experts=%lld used=%lld\n", label, ggml_type_name(wtype),
           (long long) n_experts, (long long) n_expert_used);

    std::vector<float> w_f32(size_t(kK * kM * n_experts), 0.0f);
    fill_sin(w_f32, 0.44f);
    const std::vector<uint8_t> w_q = quantize_weights(wtype, w_f32, kK);

    bool ok_all = true;
    for (int64_t n : widths) {
        // ids: for each token, experts [0,1]
        std::vector<int32_t> ids(size_t(n_expert_used * n), 0);
        for (int64_t t = 0; t < n; ++t) {
            ids[size_t(0 + t * n_expert_used)] = 0;
            ids[size_t(1 + t * n_expert_used)] = 1;
        }

        std::vector<float> x1(size_t(kK * n_expert_used * 1), 0.0f);
        fill_sin(x1, 0.61f);
        std::vector<float> xn(size_t(kK * n_expert_used * n), 0.0f);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t e = 0; e < n_expert_used; ++e) {
                for (int64_t i = 0; i < kK; ++i) {
                    const size_t dst = size_t(((t * n_expert_used) + e) * kK + i);
                    const size_t src = size_t((e * kK) + i);
                    xn[dst] = (t == 0)
                        ? x1[src]
                        : 0.09f * std::sin(float(dst) * 0.031f + 2.0f);
                }
            }
        }

        const auto serial = run_mul_mat_id(backend, wtype, 1, n_experts, n_expert_used, w_q, x1, ids);
        // For N>1 reuse same id pattern length
        std::vector<int32_t> ids_n = ids;
        const auto batch = run_mul_mat_id(backend, wtype, n, n_experts, n_expert_used, w_q, xn, ids_n);

        // out shape for mul_mat_id: [m, n_expert_used, n_tokens]
        GGML_ASSERT(serial.size() == size_t(kM * n_expert_used));
        GGML_ASSERT(batch.size() == size_t(kM * n_expert_used * n));
        bool exact = true;
        double mad = 0.0;
        for (int64_t e = 0; e < n_expert_used; ++e) {
            for (int64_t r = 0; r < kM; ++r) {
                const size_t si = size_t(r + e * kM);
                const size_t bi = size_t(r + e * kM); // token 0 plane
                const float a = serial[si];
                const float b = batch[bi];
                mad = std::max(mad, double(std::fabs(a - b)));
                if (memcmp(&a, &b, sizeof(float)) != 0) {
                    exact = false;
                }
            }
        }
        printf("  ne21=N=%lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

bool check_c5_mul_mat_id_vs_golden(
        ggml_backend_t backend,
        ggml_type wtype,
        const char * label,
        int64_t golden_n,
        const std::vector<int64_t> & widths,
        int & fails) {
    constexpr int64_t n_experts = 4;
    constexpr int64_t n_expert_used = 2;
    printf("C5 MUL_MAT_ID vs golden ne21=%lld [%s] type=%s experts=%lld used=%lld\n",
           (long long) golden_n, label, ggml_type_name(wtype),
           (long long) n_experts, (long long) n_expert_used);

    std::vector<float> w_f32(size_t(kK * kM * n_experts), 0.0f);
    fill_sin(w_f32, 0.44f);
    const std::vector<uint8_t> w_q = quantize_weights(wtype, w_f32, kK);

    auto make_ids = [&](int64_t n) {
        std::vector<int32_t> ids(size_t(n_expert_used * n), 0);
        for (int64_t t = 0; t < n; ++t) {
            ids[size_t(0 + t * n_expert_used)] = 0;
            ids[size_t(1 + t * n_expert_used)] = 1;
        }
        return ids;
    };

    std::vector<float> x_tok0(size_t(kK * n_expert_used), 0.0f);
    fill_sin(x_tok0, 0.61f);

    auto fill_acts = [&](int64_t n) {
        std::vector<float> xn(size_t(kK * n_expert_used * n), 0.0f);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t e = 0; e < n_expert_used; ++e) {
                for (int64_t i = 0; i < kK; ++i) {
                    const size_t dst = size_t(((t * n_expert_used) + e) * kK + i);
                    const size_t src = size_t((e * kK) + i);
                    xn[dst] = (t == 0)
                        ? x_tok0[src]
                        : 0.09f * std::sin(float(dst) * 0.031f + 2.0f);
                }
            }
        }
        return xn;
    };

    const auto golden = run_mul_mat_id(
            backend, wtype, golden_n, n_experts, n_expert_used, w_q, fill_acts(golden_n), make_ids(golden_n));
    GGML_ASSERT(golden.size() >= size_t(kM * n_expert_used));
    std::vector<float> golden_tok0(golden.begin(), golden.begin() + size_t(kM * n_expert_used));

    bool ok_all = true;
    for (int64_t n : widths) {
        const auto batch = run_mul_mat_id(
                backend, wtype, n, n_experts, n_expert_used, w_q, fill_acts(n), make_ids(n));
        GGML_ASSERT(batch.size() == size_t(kM * n_expert_used * n));
        bool exact = true;
        double mad = 0.0;
        for (int64_t e = 0; e < n_expert_used; ++e) {
            for (int64_t r = 0; r < kM; ++r) {
                const size_t si = size_t(r + e * kM);
                const size_t bi = size_t(r + e * kM); // token 0
                const float a = golden_tok0[si];
                const float b = batch[bi];
                mad = std::max(mad, double(std::fabs(a - b)));
                if (memcmp(&a, &b, sizeof(float)) != 0) {
                    exact = false;
                }
            }
        }
        printf("  ne21=N=%lld vs %lld exact=%d max_abs_diff=%.6g %s\n",
               (long long) n, (long long) golden_n, exact ? 1 : 0, mad, exact ? "PASS" : "FAIL");
        if (!exact) {
            ok_all = false;
            ++fails;
        }
    }
    return ok_all;
}

} // namespace

int main(int argc, char ** argv) {
    bool residual = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--residual") {
            residual = true;
        }
    }

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
        printf("host-skip: no Metal backend (TASK-066 mul_mat corners)\n");
        return 77;
    }
    ggml_backend_ptr backend_holder(ggml_backend_dev_init(chosen, nullptr));
    if (!backend_holder) {
        printf("host-skip: Metal backend init failed\n");
        return 77;
    }
    ggml_backend_t backend = backend_holder.get();
    printf("backend=%s device=%s\n", ggml_backend_name(backend), ggml_backend_dev_description(chosen));

    int fails = 0;

    if (residual) {
        // B5 + product ladder class: row0(N) vs golden N=8 (ubatch S8).
        // Widths: former residual {9,16,32} plus {256,512,1024} (S8 vs S1024 track).
        printf("APPLICABLE residual: ne11 in {9,16,32,256,512,1024} row0 bit-exact vs golden N=8 "
               "(Q8_0 + IQ4_XS; MoE ne21 same set)\n");
        constexpr int64_t golden_n = 8;
        const std::vector<int64_t> widths = {9, 16, 32, 256, 512, 1024};
        check_c1_c4_mul_mat_vs_golden(backend, GGML_TYPE_Q8_0, "q8_0-residual-vs8", golden_n, widths, fails);
        check_c1_c4_mul_mat_vs_golden(backend, GGML_TYPE_IQ4_XS, "iq4_xs-residual-vs8", golden_n, widths, fails);
        check_c5_mul_mat_id_vs_golden(backend, GGML_TYPE_Q8_0, "q8_0-moe-residual-vs8", golden_n, widths, fails);
        if (fails != 0) {
            printf("RESULT FAIL fails=%d (residual vs N=8; see TASK-066-mul-mat-batch-inventory.md)\n", fails);
            return 1;
        }
        printf("RESULT PASS (residual ne11={9,16,32,256,512,1024} vs N=8)\n");
        return 0;
    }

    printf("APPLICABLE patchB(mul_mv_ext): C1,C4(Q8_0),C5(mul_mat_id); IQ4_XS=control; C6=external\n");
    const std::vector<int64_t> widths = {2, 3, 4, 5, 8};

    // C4 product-relevant: Q8_0 hits mul_mv_ext for ne11 in 2..8 (skoulik class).
    check_c1_c4_mul_mat(backend, GGML_TYPE_Q8_0, "q8_0-ext-hot", widths, fails);

    // Control: IQ4_XS stays on mul_mv for all ne11 — should already be bit-exact.
    check_c1_c4_mul_mat(backend, GGML_TYPE_IQ4_XS, "iq4_xs-control", widths, fails);

    // C5 MoE: verify widths stay on mul_mv_id (ne21 < 32).
    check_c5_mul_mat_id(backend, GGML_TYPE_Q8_0, "q8_0-moe", widths, fails);

    if (fails != 0) {
        printf("RESULT FAIL fails=%d\n", fails);
        return 1;
    }
    printf("RESULT PASS (C1+C4+C5; C6 external)\n");
    return 0;
}
