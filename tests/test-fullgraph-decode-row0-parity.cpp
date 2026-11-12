// TASK-066 ST-3.2f / 3.2f-ter — full-graph decode row0 batch-invariance localizer.
//
// Micro localizers (FA long-KV / MoE / router) stayed GREEN after Metal A+B KEEP, while
// live P5 + first-diff logits still MISMATCH (near-tie argmax cascade under MTP verify
// width N>1). This test reproduces that residual *class* at the llama graph:
//
//   After the same prefix, decode token T as:
//     serial:  batch N=1
//     verify:  batch N=2 with dummy draft token in slot 1
//   Require bit-exact logits for row0 (slot 0). Any ULP diff → RED (patch-C candidate).
//
// ST-3.2f-ter bisect (env TASK066_BISECT=1):
//   During the final decode only, capture F32/F16 graph tensors via cb_eval and compare
//   token-0 slices serial vs verify (batch dim may be ne[1] or ne[2]).
//   Match by **stable tensor name** (skip ggml anon node_*); print FIRST_DIFF_TENSOR / op / mad.
//   Optional TASK066_BISECT_FILTER=regex (ECMA) to ask only matching names (e.g. Qcur).
//
// Host requirements (fail-closed):
//   -m / path via common_params (product IQ4_XS GGUF)
//   optional -f / --file prompt; else synthesize a long ASCII prefix
//   Metal GPU preferred (-ngl); exit 77 if no GPU backend when TASK066_REQUIRE_GPU=1
//
// Product-knob lock:
//   Default: Metal TASK-066 / gemma4_26b_a4b_mtp_metal.yaml (b=2048 ub=512 ctv=turbo4 …).
//   CUDA TASK-069: set TASK069_SUITE_PROFILE=cuda + TASK069_* from
//     tests/fixtures/task069/cuda-mtp.reference.yaml (Metal-parity b=2048 ub=512 ctv=turbo4 …).
//     V100 product archive: cuda-mtp-v100-product.reference.yaml (b=8192 ub=2048 ctv=turbo3).
//   Only context_size (-c / TASK066_CTX) may differ for speed; draft n_max/p_min
//   N/A (no draft GGUF in this localizer). Do not hand-wave other knobs via EXTRA_ARGS.
//
// TDD: must FAIL on current A+B before any patch C. Do not "fix" by loosening to argmax-only.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "task069-suite-lock.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace {

struct logits_cmp {
    bool     exact = false;
    double   max_abs_diff = 0.0;
    int64_t  first_diff_i = -1;
    int32_t  argmax_a = -1;
    int32_t  argmax_b = -1;
};

logits_cmp compare_logits(const float * a, const float * b, int32_t n_vocab) {
    logits_cmp out;
    out.exact = true;
    float best_a = -INFINITY;
    float best_b = -INFINITY;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float da = a[i];
        const float db = b[i];
        if (da > best_a) {
            best_a = da;
            out.argmax_a = i;
        }
        if (db > best_b) {
            best_b = db;
            out.argmax_b = i;
        }
        uint32_t ua = 0;
        uint32_t ub = 0;
        static_assert(sizeof(float) == sizeof(uint32_t), "float size");
        std::memcpy(&ua, &da, sizeof(ua));
        std::memcpy(&ub, &db, sizeof(ub));
        const double ad = std::fabs(double(da) - double(db));
        if (ad > out.max_abs_diff) {
            out.max_abs_diff = ad;
        }
        if (ua != ub) {
            out.exact = false;
            if (out.first_diff_i < 0) {
                out.first_diff_i = i;
            }
        }
    }
    return out;
}

struct tensor_snap {
    std::string name;
    std::string op;
    int64_t     ne[4] = {0, 0, 0, 0};
    size_t      n_row0 = 0;
    uint64_t    hash = 0;
    double      mad_vs_peer = 0.0;  // filled when comparing
    std::vector<float> row0;       // kept for first-diff mad
};

struct capture_state {
    bool active = false;
    bool keep_row0 = true;
    std::vector<std::regex> filters;
    std::vector<tensor_snap> snaps;
    std::vector<uint8_t> scratch;
};

uint64_t fnv1a_f32(const float * data, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &data[i], sizeof(bits));
        for (int b = 0; b < 4; ++b) {
            h ^= (bits >> (8 * b)) & 0xffu;
            h *= 1099511628211ull;
        }
    }
    return h;
}

bool name_matches(const capture_state & st, const char * name) {
    if (st.filters.empty()) {
        return true;
    }
    if (!name || !name[0]) {
        return false;
    }
    for (const auto & re : st.filters) {
        if (std::regex_search(name, re)) {
            return true;
        }
    }
    return false;
}

void extract_flat(const ggml_tensor * t, capture_state & st, tensor_snap & snap) {
    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * base = nullptr;
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        base = static_cast<const uint8_t *>(t->data);
    } else {
        st.scratch.resize(nbytes);
        ggml_backend_tensor_get(t, st.scratch.data(), 0, nbytes);
        base = st.scratch.data();
    }

    snap.row0.clear();
    // Flatten in ggml index order; compare later slices token/batch dim 0.
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t off =
                            size_t(i3) * t->nb[3] + size_t(i2) * t->nb[2] +
                            size_t(i1) * t->nb[1] + size_t(i0) * t->nb[0];
                    float v = 0.0f;
                    if (t->type == GGML_TYPE_F32) {
                        std::memcpy(&v, base + off, sizeof(float));
                    } else if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_t h;
                        std::memcpy(&h, base + off, sizeof(h));
                        v = ggml_fp16_to_fp32(h);
                    } else {
                        return;
                    }
                    snap.row0.push_back(v);
                }
            }
        }
    }
    snap.n_row0 = snap.row0.size();
    snap.hash = fnv1a_f32(snap.row0.data(), snap.n_row0);
    if (!st.keep_row0) {
        snap.row0.clear();
        snap.row0.shrink_to_fit();
    }
}

bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<capture_state *>(user_data);
    if (!st->active) {
        return false;
    }
    if (!t || t->name[0] == '\0') {
        return false;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        return false;
    }
    if (!name_matches(*st, t->name)) {
        return false;
    }
    if (ask) {
        return true;
    }

    tensor_snap snap;
    snap.name = t->name;
    snap.op = ggml_op_desc(t);
    for (int i = 0; i < 4; ++i) {
        snap.ne[i] = t->ne[i];
    }
    extract_flat(t, *st, snap);
    if (snap.n_row0 == 0) {
        return true;
    }
    st->snaps.push_back(std::move(snap));
    return true;
}

bool decode_prefix(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t n_prefix) {
    const int32_t n_batch = std::max(1, (int32_t) llama_n_batch(ctx));
    int32_t i = 0;
    while (i < n_prefix) {
        const int32_t n = std::min(n_batch, n_prefix - i);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int32_t j = 0; j < n; ++j) {
            const bool want_logits = (i + j + 1 == n_prefix);
            common_batch_add(batch, tokens[size_t(i + j)], i + j, { 0 }, want_logits);
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL: prefix decode rc=%d at i=%d n=%d\n", rc, i, n);
            return false;
        }
        i += n;
    }
    return true;
}

struct decode_result {
    std::vector<float> logits;
    std::vector<tensor_snap> snaps;
};

decode_result run_serial_row0(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & tokens,
        int32_t n_prefix,
        capture_state * cap) {
    decode_result out;
    if (cap) {
        params.cb_eval = capture_cb;
        params.cb_eval_user_data = cap;
        params.warmup = false;
    }
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    if (cap) {
        cparams.cb_eval = capture_cb;
        cparams.cb_eval_user_data = cap;
    }
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx) {
        fprintf(stderr, "FAIL: serial ctx init\n");
        return out;
    }
    if (!decode_prefix(ctx.get(), tokens, n_prefix)) {
        return out;
    }
    const llama_token tok = tokens[size_t(n_prefix)];
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, n_prefix, { 0 }, true);
    if (cap) {
        cap->active = true;
        cap->snaps.clear();
    }
    if (llama_decode(ctx.get(), batch) != 0) {
        fprintf(stderr, "FAIL: serial N=1 decode\n");
        if (cap) {
            cap->active = false;
        }
        llama_batch_free(batch);
        return out;
    }
    if (cap) {
        cap->active = false;
        out.snaps = cap->snaps;
    }
    llama_batch_free(batch);
    const float * logits = llama_get_logits_ith(ctx.get(), 0);
    if (!logits) {
        fprintf(stderr, "FAIL: serial logits null\n");
        return out;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    out.logits.assign(logits, logits + n_vocab);
    return out;
}

decode_result run_verify_row0(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & tokens,
        int32_t n_prefix,
        llama_token draft_tok,
        capture_state * cap) {
    decode_result out;
    if (cap) {
        params.cb_eval = capture_cb;
        params.cb_eval_user_data = cap;
        params.warmup = false;
    }
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    if (cap) {
        cparams.cb_eval = capture_cb;
        cparams.cb_eval_user_data = cap;
    }
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx) {
        fprintf(stderr, "FAIL: verify ctx init\n");
        return out;
    }
    if (!decode_prefix(ctx.get(), tokens, n_prefix)) {
        return out;
    }
    const llama_token tok = tokens[size_t(n_prefix)];
    llama_batch batch = llama_batch_init(2, 0, 1);
    common_batch_add(batch, tok, n_prefix, { 0 }, true);
    common_batch_add(batch, draft_tok, n_prefix + 1, { 0 }, false);
    if (cap) {
        cap->active = true;
        cap->snaps.clear();
    }
    if (llama_decode(ctx.get(), batch) != 0) {
        fprintf(stderr, "FAIL: verify N=2 decode\n");
        if (cap) {
            cap->active = false;
        }
        llama_batch_free(batch);
        return out;
    }
    if (cap) {
        cap->active = false;
        out.snaps = cap->snaps;
    }
    llama_batch_free(batch);
    const float * logits = llama_get_logits_ith(ctx.get(), 0);
    if (!logits) {
        fprintf(stderr, "FAIL: verify logits null (batch idx 0)\n");
        return out;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    out.logits.assign(logits, logits + n_vocab);
    return out;
}

std::string synthesize_prompt(size_t min_chars) {
    static const char kChunk[] =
            "War and Peace chapter fixture for TASK-066 full-graph row0 parity. "
            "Pierre Bezukhov and Dolokhov duel near-tie logits under MTP verify N>1. ";
    std::string out;
    out.reserve(min_chars + 128);
    while (out.size() < min_chars) {
        out += kChunk;
    }
    return out;
}

double mad_vectors(const std::vector<float> & a, const std::vector<float> & b) {
    const size_t n = std::min(a.size(), b.size());
    double mad = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mad = std::max(mad, std::fabs(double(a[i]) - double(b[i])));
    }
    if (a.size() != b.size()) {
        mad = std::max(mad, 1.0);
    }
    return mad;
}

int find_batch_dim(const tensor_snap & a, const tensor_snap & b) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            return d;
        }
    }
    return -1;
}

double mad_slice0(const tensor_snap & a, const tensor_snap & b) {
    if (a.row0.empty() || b.row0.empty()) {
        return -1.0;  // hash-only
    }
    bool same_shape = true;
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            same_shape = false;
            break;
        }
    }
    if (same_shape) {
        return mad_vectors(a.row0, b.row0);
    }
    const int bd = find_batch_dim(a, b);
    if (bd < 0) {
        return 1.0e30;
    }
    auto take0 = [](const tensor_snap & s, int dim, std::vector<float> & out) {
        out.clear();
        const int64_t * ne = s.ne;
        size_t idx = 0;
        for (int64_t i3 = 0; i3 < ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                        const int64_t coords[4] = {i0, i1, i2, i3};
                        if (coords[dim] == 0) {
                            out.push_back(s.row0[idx]);
                        }
                        ++idx;
                    }
                }
            }
        }
    };
    std::vector<float> sa;
    std::vector<float> sb;
    take0(a, bd, sa);
    take0(b, bd, sb);
    return mad_vectors(sa, sb);
}

bool is_anon_tensor_name(const std::string & name) {
    // ggml auto names: "node_123" — not stable across N=1 vs N=2 graph shapes.
    return name.rfind("node_", 0) == 0;
}

void report_bisect(const std::vector<tensor_snap> & serial, const std::vector<tensor_snap> & verify) {
    printf("BISECT: serial_tensors=%zu verify_tensors=%zu\n", serial.size(), verify.size());
    printf("BISECT_NOTE: match by stable tensor name (skip anon node_*); N=1/N=2 graphs differ in count\n");

    // Build FIFO queues per name for verify (duplicates like norm-0 occur).
    std::map<std::string, std::vector<size_t>> verify_by_name;
    for (size_t i = 0; i < verify.size(); ++i) {
        if (is_anon_tensor_name(verify[i].name)) {
            continue;
        }
        verify_by_name[verify[i].name].push_back(i);
    }
    std::map<std::string, size_t> verify_cursor;

    size_t named_serial = 0;
    size_t named_compared = 0;
    size_t named_missing = 0;
    size_t named_occ_mismatch = 0;
    std::vector<std::string> prior_exact;
    std::string first_missing;
    std::string first_value_diff;
    bool reported_value = false;

    const size_t head = std::min<size_t>(serial.size(), 12);
    if (head > 0) {
        printf("BISECT_HEAD:");
        for (size_t i = 0; i < head; ++i) {
            printf(" %s/%s", serial[i].name.c_str(), serial[i].op.c_str());
        }
        printf("\n");
    }

    auto print_verify_candidates = [&](const std::string & name) {
        std::string stem = name;
        const size_t paren = stem.find(" (");
        if (paren != std::string::npos) {
            stem = stem.substr(0, paren);
        }
        printf("  verify_name_candidates:");
        int n_cand = 0;
        for (const auto & kv : verify_by_name) {
            if (kv.first == stem || kv.first.rfind(stem, 0) == 0) {
                printf(" %s(x%zu)", kv.first.c_str(), kv.second.size());
                if (++n_cand >= 12) {
                    break;
                }
            }
        }
        if (n_cand == 0) {
            printf(" <none>");
        }
        printf("\n");
    };

    for (size_t si = 0; si < serial.size(); ++si) {
        const auto & a = serial[si];
        if (is_anon_tensor_name(a.name)) {
            continue;
        }
        named_serial++;

        auto it = verify_by_name.find(a.name);
        if (it == verify_by_name.end()) {
            named_missing++;
            if (first_missing.empty()) {
                first_missing = a.name;
                printf("FIRST_MISSING_NAMED=%s\n", a.name.c_str());
                printf("  op=%s ne_serial=[%lld,%lld,%lld,%lld]\n",
                       a.op.c_str(),
                       (long long) a.ne[0], (long long) a.ne[1],
                       (long long) a.ne[2], (long long) a.ne[3]);
                print_verify_candidates(a.name);
                if (!prior_exact.empty()) {
                    printf("  prior_exact_named:");
                    const size_t ctx_n = std::min<size_t>(prior_exact.size(), 8);
                    for (size_t j = prior_exact.size() - ctx_n; j < prior_exact.size(); ++j) {
                        printf(" %s", prior_exact[j].c_str());
                    }
                    printf("\n");
                }
            }
            continue;  // keep scanning for first exact-name value divergence
        }

        size_t & cur = verify_cursor[a.name];
        if (cur >= it->second.size()) {
            named_occ_mismatch++;
            if (first_missing.empty()) {
                first_missing = a.name + "#occ";
                printf("FIRST_MISSING_NAMED=%s (occurrence beyond verify count=%zu)\n",
                       a.name.c_str(), it->second.size());
            }
            continue;
        }
        const size_t vi = it->second[cur++];
        const auto & b = verify[vi];
        named_compared++;

        const int bdim = find_batch_dim(a, b);
        const bool shape_batch_only = (bdim >= 0);
        bool values_equal = false;
        double mad = -1.0;
        if (!a.row0.empty() && !b.row0.empty()) {
            mad = mad_slice0(a, b);
            values_equal = (mad == 0.0);
        } else if (!shape_batch_only) {
            values_equal = (a.hash == b.hash && a.n_row0 == b.n_row0);
        } else {
            values_equal = true;
            mad = -2.0;
        }

        if (!values_equal && !reported_value) {
            reported_value = true;
            first_value_diff = a.name;
            printf("FIRST_DIFF_TENSOR=%s\n", a.name.c_str());
            printf("  op=%s ne_serial=[%lld,%lld,%lld,%lld] ne_verify=[%lld,%lld,%lld,%lld]\n",
                   a.op.c_str(),
                   (long long) a.ne[0], (long long) a.ne[1],
                   (long long) a.ne[2], (long long) a.ne[3],
                   (long long) b.ne[0], (long long) b.ne[1],
                   (long long) b.ne[2], (long long) b.ne[3]);
            printf("  n_flat=%zu hash_serial=%016llx hash_verify=%016llx mad=%.8g batch_dim=%d\n",
                   a.n_row0,
                   (unsigned long long) a.hash, (unsigned long long) b.hash, mad, bdim);
            printf("  serial_idx=%zu verify_idx=%zu\n", si, vi);
            printf("CLASS=row0-value-divergence\n");
            if (!prior_exact.empty()) {
                printf("  prior_exact_named:");
                const size_t ctx_n = std::min<size_t>(prior_exact.size(), 8);
                for (size_t j = prior_exact.size() - ctx_n; j < prior_exact.size(); ++j) {
                    printf(" %s", prior_exact[j].c_str());
                }
                printf("\n");
            }
            // Still finish counts below; value hit is the KEEP gate for patch C.
            break;
        }
        prior_exact.push_back(a.name);
    }

    printf("BISECT_NAMED: serial_named=%zu compared=%zu missing=%zu occ_mismatch=%zu\n",
           named_serial, named_compared, named_missing, named_occ_mismatch);
    if (serial.size() != verify.size()) {
        printf("NOTE: anon/total count serial=%zu verify=%zu (expected under N=2)\n",
               serial.size(), verify.size());
    }
    if (!first_missing.empty()) {
        printf("FIRST_MISSING_NAMED_SUMMARY=%s\n", first_missing.c_str());
    }
    if (reported_value) {
        printf("FIRST_DIFF_TENSOR_SUMMARY=%s\n", first_value_diff.c_str());
        return;
    }
    if (!first_missing.empty()) {
        printf("FIRST_DIFF_TENSOR=%s\n", first_missing.c_str());
        printf("CLASS=named-tensor-missing-on-verify\n");
        return;
    }
    printf("FIRST_DIFF_TENSOR=<none-in-named-set>\n");
    printf("CLASS=logits-only-or-anon-only-residual\n");
}

}  // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 1;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }

    if (params.model.path.empty()) {
        fprintf(stderr,
                "FAIL: missing required -m MODEL (product GGUF); "
                "refusing silent model substitution\n");
        return 2;
    }

    if (params.n_ctx < 4) {
        fprintf(stderr, "FAIL: -c / n_ctx must be set (>=4); refusing silent ctx\n");
        return 2;
    }
    if (!task069_suite_lock::apply_cuda_if_requested(params, /*lock_batch_ubatch=*/true)) {
        // Metal TASK-066 product lock (unchanged).
        params.n_gpu_layers = 999;
        params.n_batch = 2048;
        params.n_ubatch = 512;
        params.cache_type_k = GGML_TYPE_Q8_0;
        params.cache_type_v = GGML_TYPE_TURBO4_0;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.cache_ram_mib = 4096;
        params.cpuparams.n_threads = 4;
        params.cpuparams_batch.n_threads = 4;
        params.n_parallel = 1;
        fprintf(stderr,
                "product_lock: ngl=%d b=%d ub=%d ctk=%s ctv=%s fa=on cram=%d t=%d tb=%d np=%d ctx=%d\n",
                params.n_gpu_layers, params.n_batch, params.n_ubatch,
                ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
                params.cache_ram_mib, params.cpuparams.n_threads,
                params.cpuparams_batch.n_threads, params.n_parallel, (int) params.n_ctx);
    }

    const bool do_bisect = []() {
        const char * e = std::getenv("TASK066_BISECT");
        return e && e[0] == '1';
    }();

    capture_state cap;
    if (do_bisect) {
        if (const char * filt = std::getenv("TASK066_BISECT_FILTER"); filt && filt[0]) {
            try {
                cap.filters.emplace_back(filt, std::regex::ECMAScript);
            } catch (const std::regex_error & e) {
                fprintf(stderr, "FAIL: TASK066_BISECT_FILTER invalid regex: %s\n", e.what());
                return 2;
            }
            fprintf(stderr, "bisect_filter=%s\n", filt);
        }
        if (const char * ho = std::getenv("TASK066_BISECT_HASH_ONLY"); ho && ho[0] == '1') {
            cap.keep_row0 = false;
            fprintf(stderr, "bisect_hash_only=1\n");
        }
        fprintf(stderr, "bisect=1 (cb_eval row0 capture on final decode)\n");
    }

    ggml_backend_load_all();
    llama_backend_init();

    const bool require_gpu = []() {
        const char * e = std::getenv("TASK066_REQUIRE_GPU");
        return e && e[0] == '1';
    }();
    if (require_gpu) {
        bool has_gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                has_gpu = true;
                break;
            }
        }
        if (!has_gpu) {
            fprintf(stderr, "HOST_SKIP: TASK066_REQUIRE_GPU=1 but no GPU backend\n");
            return 77;
        }
    }

    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) {
        fprintf(stderr, "FAIL: model load\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::string prompt = params.prompt;
    if (prompt.empty()) {
        const char * n_chars_env = std::getenv("TASK066_PROMPT_CHARS");
        size_t min_chars = 16000;
        if (n_chars_env && n_chars_env[0]) {
            min_chars = size_t(std::strtoull(n_chars_env, nullptr, 10));
            if (min_chars == 0) {
                fprintf(stderr, "FAIL: TASK066_PROMPT_CHARS empty/zero\n");
                return 2;
            }
        }
        prompt = synthesize_prompt(min_chars);
        fprintf(stderr, "note: synthesized prompt chars=%zu (set -p for live first-diff text)\n",
                prompt.size());
    }

    std::vector<llama_token> tokens;
    tokens.resize(prompt.size() + 16);
    const int32_t n_tok = llama_tokenize(
            vocab, prompt.c_str(), (int32_t) prompt.size(),
            tokens.data(), (int32_t) tokens.size(),
            /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) {
        tokens.resize(size_t(-n_tok));
        const int32_t n2 = llama_tokenize(
                vocab, prompt.c_str(), (int32_t) prompt.size(),
                tokens.data(), (int32_t) tokens.size(),
                true, true);
        if (n2 < 0) {
            fprintf(stderr, "FAIL: tokenize\n");
            return 1;
        }
        tokens.resize(size_t(n2));
    } else {
        tokens.resize(size_t(n_tok));
    }

    if (tokens.size() < 4) {
        fprintf(stderr, "FAIL: prompt too short (%zu toks)\n", tokens.size());
        return 2;
    }

    if (const char * lim = std::getenv("TASK066_N_TOKENS"); lim && lim[0]) {
        const size_t n_keep = size_t(std::strtoull(lim, nullptr, 10));
        if (n_keep < 4) {
            fprintf(stderr, "FAIL: TASK066_N_TOKENS must be >= 4\n");
            return 2;
        }
        if (n_keep < tokens.size()) {
            tokens.resize(n_keep);
        }
    }

    const int32_t n_prefix = (int32_t) tokens.size() - 2;
    const llama_token draft_tok = tokens.back();

    printf("FULLGRAPH-C1 decode row0 bit-exact N=1 vs N=2\n");
    printf("  model=%s\n", params.model.path.c_str());
    printf("  n_tokens=%zu n_prefix=%d n_vocab=%d ngl=%d bisect=%d\n",
           tokens.size(), n_prefix, n_vocab, params.n_gpu_layers, do_bisect ? 1 : 0);
    printf("HYPOTHESIS: full-graph logits diverge under verify width (first-diff/P5 class)\n");

    capture_state * cap_ptr = do_bisect ? &cap : nullptr;
    const decode_result serial = run_serial_row0(model, params, tokens, n_prefix, cap_ptr);
    if (serial.logits.empty()) {
        return 1;
    }
    const decode_result verify = run_verify_row0(model, params, tokens, n_prefix, draft_tok, cap_ptr);
    if (verify.logits.empty()) {
        return 1;
    }

    if (do_bisect) {
        report_bisect(serial.snaps, verify.snaps);
    }

    const logits_cmp cmp = compare_logits(serial.logits.data(), verify.logits.data(), n_vocab);
    printf("  exact=%d max_abs_diff=%.8g first_diff_i=%lld argmax_serial=%d argmax_verify=%d\n",
           cmp.exact ? 1 : 0, cmp.max_abs_diff, (long long) cmp.first_diff_i,
           cmp.argmax_a, cmp.argmax_b);

    if (!cmp.exact) {
        printf("RESULT FAIL (RED) — full-graph row0 logits not bit-exact; patch-C TDD gate open\n");
        return 1;
    }
    if (cmp.argmax_a != cmp.argmax_b) {
        printf("RESULT FAIL (RED) — argmax mismatch with claimed exact logits\n");
        return 1;
    }
    printf("RESULT PASS (GREEN) — full-graph row0 bit-exact on this prompt/host\n");
    printf("note: GREEN here does not waive live first-diff/P5; try longer --prompt-file\n");
    return 0;
}
