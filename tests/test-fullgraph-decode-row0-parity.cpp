// TASK-066 ST-3.2f — full-graph decode row0 batch-invariance localizer (patch-C TDD RED).
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
// Host requirements (fail-closed):
//   -m / path via common_params (product IQ4_XS GGUF)
//   optional -p / --prompt-file; else synthesize a long ASCII prefix
//   Metal GPU preferred (-ngl); exit 77 if no GPU backend when TASK066_REQUIRE_GPU=1
//
// TDD: must FAIL on current A+B before any patch C. Do not "fix" by loosening to argmax-only.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        // Bit-exact: compare raw bits (catches 1 ULP near-ties that flip argmax later).
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

bool decode_prefix(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t n_prefix) {
    // Chunked prefill; logits only on the last prefix token when n_prefix == tokens used.
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

std::vector<float> run_serial_row0(
        llama_model * model,
        const common_params & params,
        const std::vector<llama_token> & tokens,
        int32_t n_prefix) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx) {
        fprintf(stderr, "FAIL: serial ctx init\n");
        return {};
    }
    if (!decode_prefix(ctx.get(), tokens, n_prefix)) {
        return {};
    }
    // Decode next token alone (N=1).
    const llama_token tok = tokens[size_t(n_prefix)];
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, n_prefix, { 0 }, true);
    if (llama_decode(ctx.get(), batch) != 0) {
        fprintf(stderr, "FAIL: serial N=1 decode\n");
        llama_batch_free(batch);
        return {};
    }
    llama_batch_free(batch);
    const float * logits = llama_get_logits_ith(ctx.get(), 0);
    if (!logits) {
        fprintf(stderr, "FAIL: serial logits null\n");
        return {};
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return std::vector<float>(logits, logits + n_vocab);
}

std::vector<float> run_verify_row0(
        llama_model * model,
        const common_params & params,
        const std::vector<llama_token> & tokens,
        int32_t n_prefix,
        llama_token draft_tok) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx) {
        fprintf(stderr, "FAIL: verify ctx init\n");
        return {};
    }
    if (!decode_prefix(ctx.get(), tokens, n_prefix)) {
        return {};
    }
    // Speculative-verify width: target + draft in one graph (N=2). Row0 must match serial.
    const llama_token tok = tokens[size_t(n_prefix)];
    llama_batch batch = llama_batch_init(2, 0, 1);
    common_batch_add(batch, tok, n_prefix, { 0 }, true);
    common_batch_add(batch, draft_tok, n_prefix + 1, { 0 }, true);
    if (llama_decode(ctx.get(), batch) != 0) {
        fprintf(stderr, "FAIL: verify N=2 decode\n");
        llama_batch_free(batch);
        return {};
    }
    llama_batch_free(batch);
    const float * logits = llama_get_logits_ith(ctx.get(), 0);
    if (!logits) {
        fprintf(stderr, "FAIL: verify logits null\n");
        return {};
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return std::vector<float>(logits, logits + n_vocab);
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

}  // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 1;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.model.path.empty()) {
        fprintf(stderr,
                "FAIL: missing required -m MODEL (product GGUF); "
                "refusing silent model substitution\n");
        return 2;
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
        // ~4k tokens worth of ASCII when tokenizer ≈4 chars/tok — enough to stress
        // long-KV paths without requiring the full Voyna 100k prompt on every run.
        // Override with -p / --prompt-file for first-diff residual reproduction.
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

    // Need prefix + target + room for draft position.
    if (tokens.size() < 4) {
        fprintf(stderr, "FAIL: prompt too short (%zu toks)\n", tokens.size());
        return 2;
    }

    // Optional truncate for faster bisect: TASK066_N_TOKENS=N keeps first N tokens.
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

    const int32_t n_prefix = (int32_t) tokens.size() - 2;  // last-1 = target, last = draft filler
    const llama_token draft_tok = tokens.back();

    printf("FULLGRAPH-C1 decode row0 bit-exact N=1 vs N=2\n");
    printf("  model=%s\n", params.model.path.c_str());
    printf("  n_tokens=%zu n_prefix=%d n_vocab=%d ngl=%d\n",
           tokens.size(), n_prefix, n_vocab, params.n_gpu_layers);
    printf("HYPOTHESIS: full-graph logits diverge under verify width (first-diff/P5 class)\n");

    const std::vector<float> serial = run_serial_row0(model, params, tokens, n_prefix);
    if (serial.empty()) {
        return 1;
    }
    const std::vector<float> verify = run_verify_row0(model, params, tokens, n_prefix, draft_tok);
    if (verify.empty()) {
        return 1;
    }

    const logits_cmp cmp = compare_logits(serial.data(), verify.data(), n_vocab);
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
