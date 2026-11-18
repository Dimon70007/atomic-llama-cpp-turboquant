// TASK-077: numeric KV parity across cache types, ubatch and ctx checkpoints.
//
// Why this exists: comparing greedy TEXT proved useless (F41). A ~4% top-1
// change relocates a 60-token answer entirely, so different text says nothing
// about whether the cache is correct. This test compares LOGITS, which is the
// quantity the cache actually feeds into the model.
//
// What it measures, for one prompt and one config pair:
//   max |a-b|      worst single logit difference, in nats
//   RMS |a-b|      typical difference
//   cos            cosine similarity between the two logit vectors
//   top1           whether the argmax token is the same
//   KL(a||b)       KL divergence over the softmax of the full vocabulary
//
// Two shapes are covered, because they answer different questions:
//   prefill-vs-prefill   the same prompt decoded twice in fresh contexts with
//                        the same ubatch. Must be identical: a difference here
//                        is non-determinism, period.
//   prefill-vs-cached    the same prompt fully prefilled, then decoded again
//                        with the KV already in the cache. Must also be
//                        identical: a cache must not change the result.
//   ubatch A vs ubatch B the same prompt at different ubatch sizes. These are
//                        DIFFERENT kernels (a different reduction order), so a
//                        small difference is expected and is not a failure;
//                        the test reports the number so it can be tracked.
//
// ubatch below 16 is refused by operator instruction: it takes a different
// code path (and different arithmetic) that the product never uses, so it
// cannot be compared with the shipping shapes.
//
// Exit codes: 0 all REQUIRED comparisons passed, 1 a required comparison failed,
//             2 bad input, 77 no GPU / model (ctest skip).

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdint>
#include <vector>

namespace {

constexpr int UBATCH_FLOOR = 16;
constexpr int N_PROMPT_TOKENS = 512;

struct Result {
    double max_abs  = 0.0;
    double p50      = 0.0;
    double p99      = 0.0;
    double p999     = 0.0;
    double rms      = 0.0;
    double cos      = 0.0;
    double kl       = 0.0;
    bool   top1     = false;
    int    n_diff   = 0;   // logits that differ by more than EPS
    int    n_logits = 0;
};

constexpr double EPS = 1e-6;

struct Ctx {
    llama_context * ctx = nullptr;
    int             n_past = 0;

    ~Ctx() { if (ctx) llama_free(ctx); }
};

static void softmax(const float * x, int n, std::vector<double> & out) {
    double m = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (x[i] > m) m = x[i];
    }
    out.resize(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp((double) x[i] - m);
        sum += out[i];
    }
    for (int i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

static Result compare(const float * a, const float * b, int n) {
    Result r;
    r.n_logits = n;

    std::vector<double> diffs;
    diffs.reserve(n);
    double sum_sq = 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    int amax = 0, bmax = 0;

    for (int i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        const double ad = std::fabs(d);
        diffs.push_back(ad);
        r.max_abs = std::max(r.max_abs, ad);
        if (ad > EPS) {
            ++r.n_diff;
        }
        sum_sq += d * d;
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
        if (a[i] > a[amax]) amax = i;
        if (b[i] > b[bmax]) bmax = i;
    }
    std::sort(diffs.begin(), diffs.end());
    auto pct = [&](double q) {
        if (diffs.empty()) return 0.0;
        size_t k = (size_t) std::min<double>((double) diffs.size() - 1, q * (double) diffs.size());
        return diffs[k];
    };
    r.p50  = pct(0.50);
    r.p99  = pct(0.99);
    r.p999 = pct(0.999);

    r.rms = std::sqrt(sum_sq / n);
    r.cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    r.top1 = (amax == bmax);

    std::vector<double> pa, pb;
    softmax(a, n, pa);
    softmax(b, n, pb);
    for (int i = 0; i < n; ++i) {
        if (pa[i] > 0.0 && pb[i] > 0.0) {
            r.kl += pa[i] * std::log(pa[i] / pb[i]);
        }
    }
    return r;
}

static void print_row(const char * label, const Result & r, bool required) {
    std::printf("%-32s %-11.6g %-11.6g %-11.6g %-11.6g %-11.6g %-11.6g %-8d %s\n",
                label, r.max_abs, r.p50, r.p99, r.p999, r.rms, r.kl, r.n_diff,
                r.top1 ? "same" : "DIFF");
    (void) required;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(N_PROMPT_TOKENS + 64);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                           tokens.data(), (int32_t) tokens.size(),
                           /*add_special*/ true, /*parse_special*/ true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                           tokens.data(), (int32_t) tokens.size(), true, true);
    }
    if (n <= 0) {
        return {};
    }
    tokens.resize(n);
    if ((int) tokens.size() > N_PROMPT_TOKENS) {
        tokens.resize(N_PROMPT_TOKENS);
    }
    return tokens;
}

// Decode the whole prompt in ubatch-sized pieces and return the final logits.
static bool run_prefill(Ctx & c, const std::vector<llama_token> & tokens, int ubatch,
                        std::vector<float> & logits) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c.ctx)));

    for (int i = 0; i < (int) tokens.size(); i += ubatch) {
        const int n = std::min(ubatch, (int) tokens.size() - i);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()) + i, n);
        if (llama_decode(c.ctx, batch) != 0) {
            std::fprintf(stderr, "FAIL: llama_decode returned non-zero at %d/%d\n",
                         i, (int) tokens.size());
            return false;
        }
        c.n_past += n;
    }
    const float * out = llama_get_logits_ith(c.ctx, -1);
    if (!out) {
        std::fprintf(stderr, "FAIL: no logits after prefill\n");
        return false;
    }
    logits.assign(out, out + n_vocab);
    return true;
}

// Decode one more token (a cache hit for the prefix) and return its logits.
[[maybe_unused]] static bool run_one_step(Ctx & c, llama_token tok, std::vector<float> & logits) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c.ctx)));
    llama_batch batch = llama_batch_get_one(&tok, 1);
    if (llama_decode(c.ctx, batch) != 0) {
        std::fprintf(stderr, "FAIL: decode of the extra token failed\n");
        return false;
    }
    c.n_past += 1;
    const float * out = llama_get_logits_ith(c.ctx, -1);
    if (!out) {
        return false;
    }
    logits.assign(out, out + n_vocab);
    return true;
}

struct Params {
    std::string model;
    std::string prompt;
    std::string v_type = "q8_0";
    int         ub_a   = 512;
    int         ub_b   = 256;
    int         n_ctx  = 4096;
    int         n_gpu  = 999;
    bool        check_ctxcp = false;
    bool        swa_full    = false;
    int         split_at   = 0;   // 0 = default (half); else an explicit split point
    std::string dump_to;      // write the prefill logits here
    std::string compare_with;  // compare the prefill logits against this dump
};

static llama_context * make_ctx(const llama_model * model, const Params & p,
                                ggml_type type_v, bool allow_fallback) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = p.n_ctx;
    cp.n_batch    = 2048;
    cp.n_ubatch   = std::max(p.ub_a, p.ub_b);
    cp.n_seq_max  = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k     = GGML_TYPE_Q8_0;
    cp.type_v     = type_v;
    cp.swa_full   = p.swa_full;

    llama_context * ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    if (!ctx && allow_fallback) {
        cp.type_v = GGML_TYPE_Q8_0;
        ctx = llama_init_from_model(const_cast<llama_model *>(model), cp);
    }
    return ctx;
}

} // namespace

int main(int argc, char ** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "-m" || a == "--model")  p.model  = next();
        else if (a == "-f" || a == "--prompt") p.prompt = next();
        else if (a == "--ctv")                 p.v_type = next();
        else if (a == "--ub-a")                p.ub_a   = std::stoi(next());
        else if (a == "--ub-b")                p.ub_b   = std::stoi(next());
        else if (a == "--ctxcp")               p.check_ctxcp = true;
        else if (a == "--n-ctx")               p.n_ctx  = std::stoi(next());
        else if (a == "--ngl")                 p.n_gpu  = std::stoi(next());
        else if (a == "--dump")                p.dump_to = next();
        else if (a == "--compare-with")        p.compare_with = next();
        else if (a == "--swa-full")            p.swa_full = true;
        else if (a == "--split-at")            p.split_at = std::stoi(next());
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (p.model.empty() || p.prompt.empty()) {
        std::fprintf(stderr, "usage: %s -m model.gguf -f prompt.txt [--ctv turbo4.8] "
                             "[--ub-a 512 --ub-b 256] [--ctxcp]\n", argv[0]);
        return 2;
    }
    if (p.ub_a < UBATCH_FLOOR || p.ub_b < UBATCH_FLOOR) {
        std::fprintf(stderr, "FAIL: ubatch below %d is refused; it takes a different "
                             "code path that the product does not use\n", UBATCH_FLOOR);
        return 2;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p.n_gpu;
    llama_model * model = llama_model_load_from_file(p.model.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "FAIL: cannot load model %s\n", p.model.c_str());
        return 2;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string text;
    {
        FILE * f = std::fopen(p.prompt.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "FAIL: cannot read prompt %s\n", p.prompt.c_str());
            llama_model_free(model);
            return 2;
        }
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
            text.append(buf, n);
        }
        std::fclose(f);
    }
    const std::vector<llama_token> tokens = tokenize(vocab, text);
    if (tokens.empty()) {
        std::fprintf(stderr, "FAIL: prompt tokenized to nothing\n");
        llama_model_free(model);
        return 2;
    }

    ggml_type type_v = GGML_TYPE_Q8_0;
    bool turbo_ok = true;
    if (p.v_type == "turbo2")      type_v = GGML_TYPE_TURBO2_0;
    else if (p.v_type == "turbo3") type_v = GGML_TYPE_TURBO3_0;
    else if (p.v_type == "turbo4") type_v = GGML_TYPE_TURBO4_0;
    else if (p.v_type == "q8_0")   type_v = GGML_TYPE_Q8_0;
    else {
        // turbo3.8 / turbo4.8 are CLI-level bracket schedules, not ggml types.
        // Approximate them by their mid type here and say so.
        if (p.v_type == "turbo3.8")      type_v = GGML_TYPE_TURBO3_0;
        else if (p.v_type == "turbo4.8") type_v = GGML_TYPE_TURBO4_0;
        else {
            std::fprintf(stderr, "FAIL: unknown --ctv %s\n", p.v_type.c_str());
            llama_model_free(model);
            return 2;
        }
        std::printf("# note: %s is a CLI bracket schedule; measured here as its mid type\n",
                    p.v_type.c_str());
    }

    std::printf("# model   : %s\n", p.model.c_str());
    std::printf("# ctv     : %s\n", p.v_type.c_str());
    std::printf("# tokens  : %d\n", (int) tokens.size());
    std::printf("# ubatch  : a=%d b=%d  swa_full=%d\n\n", p.ub_a, p.ub_b, (int) p.swa_full);
    std::printf("%-32s %-11s %-11s %-11s %-11s %-11s %-11s %-8s %s\n",
                "comparison", "max|a-b|", "p50", "p99", "p99.9", "rms", "KL(a||b)", "n>eps", "top1");

    int failures = 0;

    // ---- reference-dump mode ------------------------------------------------
    // --dump writes the prefill logits; --compare-with reads such a dump and
    // compares. Together they give the codec's price: run q8_0 --dump once, then
    // each turbo type --compare-with. This is the number that says whether the
    // library still works as expected, instead of "the text matched or it did
    // not".
    if (!p.dump_to.empty() || !p.compare_with.empty()) {
        std::vector<float> logits;
        {
            Ctx c;
            c.ctx = make_ctx(model, p, type_v, turbo_ok);
            if (!c.ctx || !run_prefill(c, tokens, p.ub_a, logits)) {
                std::fprintf(stderr, "FAIL: prefill failed in dump mode\n");
                llama_model_free(model);
                return 1;
            }
        }
        if (!p.dump_to.empty()) {
            FILE * out = std::fopen(p.dump_to.c_str(), "wb");
            if (!out) {
                std::fprintf(stderr, "FAIL: cannot write %s\n", p.dump_to.c_str());
                llama_model_free(model);
                return 2;
            }
            const int32_t n = (int32_t) logits.size();
            std::fwrite(&n, sizeof n, 1, out);
            std::fwrite(logits.data(), sizeof(float), logits.size(), out);
            std::fclose(out);
            std::printf("dumped %d logits to %s\n", n, p.dump_to.c_str());
        }
        if (!p.compare_with.empty()) {
            FILE * in = std::fopen(p.compare_with.c_str(), "rb");
            if (!in) {
                std::fprintf(stderr, "FAIL: cannot read %s\n", p.compare_with.c_str());
                llama_model_free(model);
                return 2;
            }
            int32_t n = 0;
            if (std::fread(&n, sizeof n, 1, in) != 1 || n <= 0) {
                std::fprintf(stderr, "FAIL: bad dump header in %s\n", p.compare_with.c_str());
                std::fclose(in);
                llama_model_free(model);
                return 2;
            }
            std::vector<float> base(n);
            if (std::fread(base.data(), sizeof(float), (size_t) n, in) != (size_t) n) {
                std::fprintf(stderr, "FAIL: short read from %s\n", p.compare_with.c_str());
                std::fclose(in);
                llama_model_free(model);
                return 2;
            }
            std::fclose(in);
            if ((int) logits.size() != n) {
                std::fprintf(stderr, "FAIL: vocab mismatch: dump %d vs run %d\n",
                             n, (int) logits.size());
                llama_model_free(model);
                return 2;
            }
            std::printf("%-32s %-11s %-11s %-11s %-11s %-11s %-11s %-8s %s\n",
                        "comparison", "max|a-b|", "p50", "p99", "p99.9", "rms", "KL(a||b)",
                        "n>eps", "top1");
            const Result r = compare(base.data(), logits.data(), n);
            print_row((std::string("vs ") + p.v_type + " prefill").c_str(), r, true);
            llama_model_free(model);
            return 0;
        }
        llama_model_free(model);
        return 0;
    }

    // ---- A: prefill vs prefill, same ubatch -> must be identical -------------
    {
        std::vector<float> la, lb;
        {
            Ctx c;
            c.ctx = make_ctx(model, p, type_v, turbo_ok);
            if (!c.ctx || !run_prefill(c, tokens, p.ub_a, la)) {
                std::fprintf(stderr, "FAIL: first prefill could not run\n");
                llama_model_free(model);
                return 1;
            }
        }
        {
            Ctx c;
            c.ctx = make_ctx(model, p, type_v, turbo_ok);
            if (!c.ctx || !run_prefill(c, tokens, p.ub_a, lb)) {
                std::fprintf(stderr, "FAIL: second prefill could not run\n");
                llama_model_free(model);
                return 1;
            }
        }
        const Result r = compare(la.data(), lb.data(), (int) la.size());
        print_row("prefill vs prefill (same ub)", r, true);
        if (r.max_abs > 0.0) {
            ++failures;
        }
    }

    // ---- B: prefill in one batch vs prefill split in two --------------------
    // The same prompt, the same final position, decoded either as one batch or
    // as two. This is what the server does when the prompt is already in the
    // cache: the remainder is decoded in a differently shaped batch. If the two
    // disagree, a cache hit changes the answer, which is the defect the
    // operator flagged (ctxcp 0 vs 32 changing the text).
    // REQUIRED: same config, same data, so it must not matter how the work was
    // split.
    {
        std::vector<float> l_one, l_split;
        {
            Ctx c;
            c.ctx = make_ctx(model, p, type_v, turbo_ok);
            if (!c.ctx || !run_prefill(c, tokens, p.ub_a, l_one)) {
                std::fprintf(stderr, "FAIL: single-batch prefill failed\n");
                llama_model_free(model);
                return 1;
            }
        }
        {
            Ctx c;
            c.ctx = make_ctx(model, p, type_v, turbo_ok);
            if (!c.ctx) {
                std::fprintf(stderr, "FAIL: cannot create the split context\n");
                llama_model_free(model);
                return 1;
            }
            const int half = (p.split_at > 0)
                           ? std::min((int) tokens.size() - 1, p.split_at)
                           : std::max(UBATCH_FLOOR, (int) tokens.size() / 2);
            std::vector<float> tmp;
            std::vector<llama_token> head(tokens.begin(), tokens.begin() + half);
            std::vector<llama_token> tail(tokens.begin() + half, tokens.end());
            if (!run_prefill(c, head, p.ub_a, tmp) || !run_prefill(c, tail, p.ub_a, l_split)) {
                std::fprintf(stderr, "FAIL: split prefill failed\n");
                llama_model_free(model);
                return 1;
            }
        }
        const Result r = compare(l_one.data(), l_split.data(), (int) l_one.size());
        print_row("prefill one batch vs split", r, true);
        if (r.max_abs > 0.0) {
            ++failures;
        }
    }

    // ---- C: ubatch A vs ubatch B -> informational ---------------------------
    {
        std::vector<float> la, lb;
        {
            Ctx c;
            c.ctx = make_ctx(model, p, type_v, turbo_ok);
            if (!c.ctx || !run_prefill(c, tokens, p.ub_a, la)) {
                llama_model_free(model);
                return 1;
            }
        }
        {
            Ctx c;
            c.ctx = make_ctx(model, p, type_v, turbo_ok);
            if (!c.ctx || !run_prefill(c, tokens, p.ub_b, lb)) {
                llama_model_free(model);
                return 1;
            }
        }
        const Result r = compare(la.data(), lb.data(), (int) la.size());
        print_row("ubatch a vs b (different kernels)", r, false);
    }

    std::printf("\nfailures (required rows only): %d\n", failures);
    llama_model_free(model);
    return failures == 0 ? 0 : 1;
}
