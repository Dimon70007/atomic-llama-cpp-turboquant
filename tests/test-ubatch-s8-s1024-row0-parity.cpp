// TASK-066.5 / 066.6 — product ubatch S8 vs S1024 greedy token-stream parity (TDD RED gate).
//
// Ladder confirm class (pre-A2 DIFF@text under MUL_MM_MIN=7). This autotest covers:
//   C1  raw tokenize + greedy
//   C1j jinja chat tokens + greedy (066.9 Phase A2)
//
//   Same prompt + product KV/FA knobs; greedy (argmax) decode for N tokens with
//     A: -b max(512,8)=512  -ub 8
//     B: -b 1024            -ub 1024
//   Require identical generated token ids. First mismatch → RED (not ε).
//
// Root-cause diagnostics (no PASS change; evidence only):
//   TASK066_DIAG=localize     — ubatch step / threshold matrix (product KV)
//   TASK066_DIAG=fa-ablation  — C2 (n=32 ub8 vs ub32) under FA/KV variants:
//       product turbo4+FA-on; f16+FA-on; f16+FA-off (true FA-off — turbo forces FA)
//   TASK066_DIAG=c3-bisect    — FIRST_DIFF_TENSOR on last ubatch graph:
//       n=1280 control (expect none) + n=1344 (expect first named drift);
//       cb_eval only on final decode; store last-token F32/F16 slices only.
//       Optional TASK066_BISECT_FILTER=regex; TASK066_BISECT_HASH_ONLY=1.
//   TASK066_DIAG=c3-thresh    — logits-only exact matrix around n_kv 256-pad
//       boundary (1280 GREEN lock vs 1288..1536).
//   TASK066_FA=off requires TASK066_CTV=<non-turbo> (e.g. f16); else fail-closed.
//   FA-off / non-turbo runs are LOCALIZATION ONLY — product path stays FA-on+turbo4.
//
// Fail-closed: -m MODEL, -c CTX, -f PROMPT; TASK066_N_PREDICT (default 64).
// Runner: tests/run-task066-ubatch-s8-s1024-corners.sh
//
// Phase A2 (066.9): after raw C1 GREEN, C1j applies model jinja chat template
// (user turn + generation prompt, enable_thinking=false == ladder --reasoning off)
// then greedy S8 vs S1024. Ladder `012625Z` DIFF@22 is this class — raw C1 alone
// does not cover it. TASK066_DIAG=c1j runs only C1j (evidence / fast loop).

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

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
    bool    exact = false;
    double  mad = 0.0;
    int32_t argmax_a = -1;
    int32_t argmax_b = -1;
    int64_t first_diff_i = -1;
};

struct gen_result {
    std::vector<llama_token> tokens;
    bool                     ok = false;
};

struct logits_result {
    std::vector<float> logits;
    bool               ok = false;
};

int32_t argmax_f32(const float * logits, int32_t n_vocab) {
    int32_t best_i = 0;
    float   best_v = -INFINITY;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    return best_i;
}

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
        const double ad = std::fabs(double(da) - double(db));
        if (ad > out.mad) {
            out.mad = ad;
        }
        uint32_t ua = 0;
        uint32_t ub = 0;
        std::memcpy(&ua, &da, sizeof(ua));
        std::memcpy(&ub, &db, sizeof(ub));
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
            fprintf(stderr, "FAIL: prefix decode rc=%d at i=%d n=%d (n_batch=%d n_ubatch=%d)\n",
                    rc, i, n, (int) llama_n_batch(ctx), (int) llama_n_ubatch(ctx));
            return false;
        }
        i += n;
    }
    return true;
}

logits_result run_prefill_logits(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & prompt_tokens,
        int32_t n_prefix,
        int32_t n_batch,
        int32_t n_ubatch) {
    logits_result out;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.warmup = false;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx) {
        fprintf(stderr, "FAIL: ctx init b=%d ub=%d\n", n_batch, n_ubatch);
        return out;
    }
    if (n_prefix < 2 || n_prefix > (int32_t) prompt_tokens.size()) {
        fprintf(stderr, "FAIL: bad n_prefix=%d (prompt=%zu)\n", n_prefix, prompt_tokens.size());
        return out;
    }
    if (!decode_prefix(ctx.get(), prompt_tokens, n_prefix)) {
        return out;
    }
    const float * logits = llama_get_logits(ctx.get());
    if (!logits) {
        fprintf(stderr, "FAIL: logits null after prefill b=%d ub=%d\n", n_batch, n_ubatch);
        return out;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    out.logits.assign(logits, logits + n_vocab);
    out.ok = true;
    return out;
}

gen_result run_greedy_ubatch(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & prompt_tokens,
        int32_t n_predict,
        int32_t n_batch,
        int32_t n_ubatch) {
    gen_result out;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.warmup = false;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx) {
        fprintf(stderr, "FAIL: ctx init b=%d ub=%d\n", n_batch, n_ubatch);
        return out;
    }

    const int32_t n_prefix = (int32_t) prompt_tokens.size();
    if (n_prefix < 2) {
        fprintf(stderr, "FAIL: prompt too short\n");
        return out;
    }
    if (!decode_prefix(ctx.get(), prompt_tokens, n_prefix)) {
        return out;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    out.tokens.reserve(size_t(n_predict));

    int32_t pos = n_prefix;
    for (int32_t t = 0; t < n_predict; ++t) {
        const float * logits = llama_get_logits(ctx.get());
        if (!logits) {
            fprintf(stderr, "FAIL: logits null at gen step %d (b=%d ub=%d)\n", t, n_batch, n_ubatch);
            return out;
        }
        const llama_token tok = argmax_f32(logits, n_vocab);
        out.tokens.push_back(tok);

        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, tok, pos, { 0 }, true);
        const int rc = llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL: gen decode rc=%d at step %d (b=%d ub=%d)\n",
                    rc, t, n_batch, n_ubatch);
            return out;
        }
        ++pos;
    }

    out.ok = true;
    return out;
}

int32_t first_token_diff(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return (int32_t) i;
        }
    }
    if (a.size() != b.size()) {
        return (int32_t) n;
    }
    return -1;
}

// Ladder product path: jinja chat template around the fixture user text.
// Matches llama-cli --jinja --conversation --reasoning off (enable_thinking=false).
bool tokenize_jinja_user_prompt(
        llama_model * model,
        const llama_vocab * vocab,
        const std::string & user_text,
        std::vector<llama_token> & out_tokens,
        std::string & out_prompt) {
    out_tokens.clear();
    out_prompt.clear();
    auto tmpls = common_chat_templates_init(model, /*chat_template_override=*/"");
    if (!tmpls) {
        fprintf(stderr, "FAIL: common_chat_templates_init\n");
        return false;
    }
    common_chat_msg user;
    user.role = "user";
    user.content = user_text;

    common_chat_templates_inputs inputs;
    inputs.messages = {user};
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = false;  // ladder: --reasoning off
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;

    const common_chat_params chat = common_chat_templates_apply(tmpls.get(), inputs);
    if (chat.prompt.empty()) {
        fprintf(stderr, "FAIL: jinja chat template produced empty prompt\n");
        return false;
    }
    out_prompt = chat.prompt;

    out_tokens.resize(out_prompt.size() + 16);
    int32_t n_tok = llama_tokenize(
            vocab, out_prompt.c_str(), (int32_t) out_prompt.size(),
            out_tokens.data(), (int32_t) out_tokens.size(),
            /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) {
        out_tokens.resize(size_t(-n_tok));
        n_tok = llama_tokenize(
                vocab, out_prompt.c_str(), (int32_t) out_prompt.size(),
                out_tokens.data(), (int32_t) out_tokens.size(), true, true);
        if (n_tok < 0) {
            fprintf(stderr, "FAIL: jinja prompt tokenize\n");
            return false;
        }
    }
    out_tokens.resize(size_t(n_tok));
    return !out_tokens.empty();
}

void print_gen_prefix(const char * label, const llama_vocab * vocab, const std::vector<llama_token> & toks, int32_t n) {
    const int32_t lim = std::min(n, (int32_t) toks.size());
    std::string s;
    for (int32_t i = 0; i < lim; ++i) {
        char buf[256];
        const int n_piece = llama_token_to_piece(vocab, toks[size_t(i)], buf, (int) sizeof(buf), 0, true);
        if (n_piece > 0) {
            s.append(buf, size_t(n_piece));
        }
    }
    printf("  %s first_%d_toks_text=%s\n", label, lim, s.c_str());
}

// C1j — A2 gate: greedy on jinja-wrapped prompt (ladder class).
int run_c1j_gate(
        llama_model * model,
        const llama_vocab * vocab,
        common_params params,
        const std::string & user_text,
        int32_t n_predict,
        int32_t b_a,
        int32_t ub_a,
        int32_t b_b,
        int32_t ub_b) {
    std::vector<llama_token> jtokens;
    std::string jprompt;
    if (!tokenize_jinja_user_prompt(model, vocab, user_text, jtokens, jprompt)) {
        return 1;
    }
    printf("UBATCH-S8-S1024-C1j jinja-chat greedy token-stream bit-exact (066.9 Phase A2)\n");
    printf("  model=%s\n", params.model.path.c_str());
    printf("  n_prompt_tokens_jinja=%zu n_predict=%d (raw fixture was separate C1)\n",
           jtokens.size(), n_predict);
    printf("  jinja_prompt_chars=%zu head=%.80s\n", jprompt.size(), jprompt.c_str());
    printf("HYPOTHESIS: after chat template, greedy ids diverge S8 vs S1024 (ladder DIFF@22)\n");

    const int64_t need_ctx = (int64_t) jtokens.size() + (int64_t) n_predict + 8;
    if ((int64_t) params.n_ctx < need_ctx) {
        fprintf(stderr,
                "FAIL: n_ctx=%d too small for jinja prompt=%zu + n_predict=%d (need >= %lld)\n",
                (int) params.n_ctx, jtokens.size(), n_predict, (long long) need_ctx);
        return 2;
    }

    const gen_result a = run_greedy_ubatch(model, params, jtokens, n_predict, b_a, ub_a);
    if (!a.ok) {
        return 1;
    }
    const gen_result b = run_greedy_ubatch(model, params, jtokens, n_predict, b_b, ub_b);
    if (!b.ok) {
        return 1;
    }

    const int32_t diff_i = first_token_diff(a.tokens, b.tokens);
    printf("  gen_len_a=%zu gen_len_b=%zu first_token_diff=%d\n",
           a.tokens.size(), b.tokens.size(), diff_i);
    print_gen_prefix("C1j_S8", vocab, a.tokens, 16);
    print_gen_prefix("C1j_S1024", vocab, b.tokens, 16);
    if (diff_i >= 0 && diff_i < (int32_t) a.tokens.size() && diff_i < (int32_t) b.tokens.size()) {
        printf("  tok_a=%d tok_b=%d\n", a.tokens[size_t(diff_i)], b.tokens[size_t(diff_i)]);
    }

    if (diff_i >= 0) {
        printf("RESULT FAIL (RED) — C1j jinja-chat greedy not bit-exact S8 vs S1024; "
               "066.9 Phase A2 TDD gate open\n");
        return 1;
    }

    printf("RESULT PASS (GREEN) — C1j jinja-chat greedy S8 vs S1024 identical for n_predict=%d\n",
           n_predict);
    return 0;
}

void print_cmp(const char * label, const logits_cmp & c) {
    printf("DIAG %s: exact=%d mad=%.8g first_diff_i=%lld argmax_a=%d argmax_b=%d\n",
           label, c.exact ? 1 : 0, c.mad, (long long) c.first_diff_i, c.argmax_a, c.argmax_b);
}

// --- C3 bisect (066.5): last-ubatch cb_eval + last-token named F32/F16 slices ---

struct tensor_snap {
    std::string name;
    std::string op;
    int64_t     ne[4] = {0, 0, 0, 0};
    int         token_dim = -1;
    size_t      n_row0 = 0;
    uint64_t    hash = 0;
    std::vector<float> row0;  // last-token slice (or empty if hash-only)
};

struct capture_state {
    bool active = false;
    bool keep_row0 = true;
    int32_t n_tokens = 0;  // final ubatch token count; used to pick token dim
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

bool is_anon_tensor_name(const std::string & name) {
    return name.rfind("node_", 0) == 0;
}

int find_token_dim(const ggml_tensor * t, int32_t n_tokens) {
    if (!t || n_tokens < 1) {
        return -1;
    }
    // Prefer highest dim among 1..3 that equals n_tokens. Q/K/V reshape is
    // typically [head_dim, n_head, n_tokens]; when n_tokens == n_head (e.g. 8)
    // the lowest match would wrongly pick the head axis.
    for (int d = 3; d >= 1; --d) {
        if (t->ne[d] == (int64_t) n_tokens) {
            return d;
        }
    }
    if (t->ne[0] == (int64_t) n_tokens) {
        return 0;
    }
    return -1;
}

float read_elem(const uint8_t * base, const ggml_tensor * t, size_t off) {
    if (t->type == GGML_TYPE_F32) {
        float v = 0.0f;
        std::memcpy(&v, base + off, sizeof(v));
        return v;
    }
    ggml_fp16_t h;
    std::memcpy(&h, base + off, sizeof(h));
    return ggml_fp16_to_fp32(h);
}

void extract_last_token(const ggml_tensor * t, capture_state & st, tensor_snap & snap) {
    snap.row0.clear();
    snap.token_dim = find_token_dim(t, st.n_tokens);
    if (snap.token_dim < 0) {
        // No token dim — skip bulky full capture (memory / time).
        return;
    }

    const int td = snap.token_dim;
    const int64_t last = t->ne[td] - 1;
    const size_t es = ggml_type_size(t->type);
    if (es == 0) {
        return;
    }

    // Host path: read in-place. Device path: pull only last-token elements
    // (never full n=1344 tensors — those can be multi-GB on MoE FA).
    const bool host = t->buffer && ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * host_base = host ? static_cast<const uint8_t *>(t->data) : nullptr;

    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const int64_t coords[4] = {i0, i1, i2, i3};
                    if (coords[td] != last) {
                        continue;
                    }
                    const size_t off =
                            size_t(i3) * t->nb[3] + size_t(i2) * t->nb[2] +
                            size_t(i1) * t->nb[1] + size_t(i0) * t->nb[0];
                    float v = 0.0f;
                    if (host) {
                        v = read_elem(host_base, t, off);
                    } else {
                        uint8_t tmp[8] = {0};
                        ggml_backend_tensor_get(t, tmp, off, es);
                        v = read_elem(tmp, t, 0);
                    }
                    snap.row0.push_back(v);
                }
            }
        }
    }
    snap.n_row0 = snap.row0.size();
    if (snap.n_row0 == 0) {
        return;
    }
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
    if (is_anon_tensor_name(t->name)) {
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
    // Internal ubatch: each micro-batch rebuilds the graph starting at embd.
    // Keep only the last micro-batch's named tensors; refresh n_tokens from embd.
    if (std::strcmp(t->name, "embd") == 0) {
        if (!st->snaps.empty()) {
            st->snaps.clear();
        }
        // embd layout [n_embd, n_tokens, ...]
        if (t->ne[1] > 0) {
            st->n_tokens = (int32_t) t->ne[1];
        }
    }
    extract_last_token(t, *st, snap);
    if (snap.n_row0 == 0) {
        return true;
    }
    st->snaps.push_back(std::move(snap));
    return true;
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

bool slices_bit_exact(const tensor_snap & a, const tensor_snap & b, double & mad_out) {
    mad_out = -1.0;
    if (!a.row0.empty() && !b.row0.empty()) {
        mad_out = mad_vectors(a.row0, b.row0);
        if (a.row0.size() != b.row0.size()) {
            return false;
        }
        for (size_t i = 0; i < a.row0.size(); ++i) {
            uint32_t ua = 0;
            uint32_t ub = 0;
            std::memcpy(&ua, &a.row0[i], sizeof(ua));
            std::memcpy(&ub, &b.row0[i], sizeof(ub));
            if (ua != ub) {
                return false;
            }
        }
        return true;
    }
    return a.hash == b.hash && a.n_row0 == b.n_row0;
}

void report_c3_bisect(const char * label,
                      const std::vector<tensor_snap> & multi,
                      const std::vector<tensor_snap> & oneshot) {
    printf("BISECT[%s]: multi_tensors=%zu oneshot_tensors=%zu\n",
           label, multi.size(), oneshot.size());
    printf("BISECT_NOTE: last-token slices of named tensors; skip anon node_*; "
           "multi=ub8 oneshot=ub=n\n");

    std::map<std::string, std::vector<size_t>> oneshot_by_name;
    for (size_t i = 0; i < oneshot.size(); ++i) {
        oneshot_by_name[oneshot[i].name].push_back(i);
    }
    std::map<std::string, size_t> oneshot_cursor;

    size_t named_multi = 0;
    size_t named_compared = 0;
    size_t named_missing = 0;
    size_t named_occ_mismatch = 0;
    std::vector<std::string> prior_exact;
    std::string first_missing;
    std::string first_value_diff;
    bool reported_value = false;

    const size_t head = std::min<size_t>(multi.size(), 12);
    if (head > 0) {
        printf("BISECT_HEAD:");
        for (size_t i = 0; i < head; ++i) {
            printf(" %s/%s", multi[i].name.c_str(), multi[i].op.c_str());
        }
        printf("\n");
    }

    for (size_t si = 0; si < multi.size(); ++si) {
        const auto & a = multi[si];
        named_multi++;

        auto it = oneshot_by_name.find(a.name);
        if (it == oneshot_by_name.end()) {
            named_missing++;
            if (first_missing.empty()) {
                first_missing = a.name;
                printf("FIRST_MISSING_NAMED=%s\n", a.name.c_str());
                printf("  op=%s ne_multi=[%lld,%lld,%lld,%lld] token_dim=%d\n",
                       a.op.c_str(),
                       (long long) a.ne[0], (long long) a.ne[1],
                       (long long) a.ne[2], (long long) a.ne[3], a.token_dim);
            }
            continue;
        }

        size_t & cur = oneshot_cursor[a.name];
        if (cur >= it->second.size()) {
            named_occ_mismatch++;
            if (first_missing.empty()) {
                first_missing = a.name + "#occ";
                printf("FIRST_MISSING_NAMED=%s (occurrence beyond oneshot count=%zu)\n",
                       a.name.c_str(), it->second.size());
            }
            continue;
        }
        const size_t vi = it->second[cur++];
        const auto & b = oneshot[vi];
        named_compared++;

        double mad = -1.0;
        const bool values_equal = slices_bit_exact(a, b, mad);

        if (!values_equal && !reported_value) {
            reported_value = true;
            first_value_diff = a.name;
            printf("FIRST_DIFF_TENSOR=%s\n", a.name.c_str());
            printf("  op=%s ne_multi=[%lld,%lld,%lld,%lld] ne_oneshot=[%lld,%lld,%lld,%lld]\n",
                   a.op.c_str(),
                   (long long) a.ne[0], (long long) a.ne[1],
                   (long long) a.ne[2], (long long) a.ne[3],
                   (long long) b.ne[0], (long long) b.ne[1],
                   (long long) b.ne[2], (long long) b.ne[3]);
            printf("  n_flat=%zu/%zu hash_multi=%016llx hash_oneshot=%016llx mad=%.8g "
                   "token_dim_m=%d token_dim_o=%d\n",
                   a.n_row0, b.n_row0,
                   (unsigned long long) a.hash, (unsigned long long) b.hash, mad,
                   a.token_dim, b.token_dim);
            printf("  multi_idx=%zu oneshot_idx=%zu\n", si, vi);
            printf("CLASS=last-token-value-divergence\n");
            if (!prior_exact.empty()) {
                printf("  prior_exact_named:");
                const size_t ctx_n = std::min<size_t>(prior_exact.size(), 8);
                for (size_t j = prior_exact.size() - ctx_n; j < prior_exact.size(); ++j) {
                    printf(" %s", prior_exact[j].c_str());
                }
                printf("\n");
            }
            break;
        }
        prior_exact.push_back(a.name);
    }

    printf("BISECT_NAMED: multi_named=%zu compared=%zu missing=%zu occ_mismatch=%zu\n",
           named_multi, named_compared, named_missing, named_occ_mismatch);
    if (!first_missing.empty()) {
        printf("FIRST_MISSING_NAMED_SUMMARY=%s\n", first_missing.c_str());
    }
    if (reported_value) {
        printf("FIRST_DIFF_TENSOR_SUMMARY=%s\n", first_value_diff.c_str());
        return;
    }
    if (!first_missing.empty()) {
        printf("FIRST_DIFF_TENSOR=%s\n", first_missing.c_str());
        printf("CLASS=named-tensor-missing-on-oneshot\n");
        return;
    }
    printf("FIRST_DIFF_TENSOR=<none-in-named-set>\n");
    printf("CLASS=logits-only-or-anon-only-residual\n");
}

struct bisect_prefill_result {
    std::vector<float> logits;
    std::vector<tensor_snap> snaps;
    bool ok = false;
};

bool decode_prefix_last_capture(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int32_t n_prefix,
        capture_state * cap) {
    const int32_t n_batch = std::max(1, (int32_t) llama_n_batch(ctx));
    int32_t i = 0;
    while (i < n_prefix) {
        const int32_t n = std::min(n_batch, n_prefix - i);
        const bool is_last = (i + n >= n_prefix);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int32_t j = 0; j < n; ++j) {
            const bool want_logits = (i + j + 1 == n_prefix);
            common_batch_add(batch, tokens[size_t(i + j)], i + j, { 0 }, want_logits);
        }
        if (cap && is_last) {
            cap->active = true;
            // Internal ubatch width (not outer llama_batch n) — last-token dim.
            cap->n_tokens = std::max(1, (int32_t) llama_n_ubatch(ctx));
            cap->snaps.clear();
        }
        const int rc = llama_decode(ctx, batch);
        if (cap && is_last) {
            cap->active = false;
        }
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL: prefix decode rc=%d at i=%d n=%d (n_batch=%d n_ubatch=%d)\n",
                    rc, i, n, (int) llama_n_batch(ctx), (int) llama_n_ubatch(ctx));
            return false;
        }
        i += n;
    }
    return true;
}

bisect_prefill_result run_prefill_bisect(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & prompt_tokens,
        int32_t n_prefix,
        int32_t n_batch,
        int32_t n_ubatch,
        capture_state * cap) {
    bisect_prefill_result out;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.warmup = false;
    if (cap) {
        params.cb_eval = capture_cb;
        params.cb_eval_user_data = cap;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    if (cap) {
        cparams.cb_eval = capture_cb;
        cparams.cb_eval_user_data = cap;
    }
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx) {
        fprintf(stderr, "FAIL: ctx init b=%d ub=%d\n", n_batch, n_ubatch);
        return out;
    }
    if (n_prefix < 2 || n_prefix > (int32_t) prompt_tokens.size()) {
        fprintf(stderr, "FAIL: bad n_prefix=%d (prompt=%zu)\n", n_prefix, prompt_tokens.size());
        return out;
    }
    if (!decode_prefix_last_capture(ctx.get(), prompt_tokens, n_prefix, cap)) {
        return out;
    }
    if (cap) {
        out.snaps = cap->snaps;
    }
    const float * logits = llama_get_logits(ctx.get());
    if (!logits) {
        fprintf(stderr, "FAIL: logits null after prefill b=%d ub=%d\n", n_batch, n_ubatch);
        return out;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    out.logits.assign(logits, logits + n_vocab);
    out.ok = true;
    return out;
}

bool run_c3_thresh(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & tokens) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    // 1280 = 5*256 (pad-stable); 1288 first step into pad-to-1536 bucket.
    const int32_t ns[] = {1280, 1288, 1312, 1344, 1408, 1536};
    printf("UBATCH-S8-S1024-C3-THRESH (logits-only; n_kv pad hypothesis)\n");
    printf("  model=%s n_prompt_tokens=%zu\n", params.model.path.c_str(), tokens.size());
    for (int32_t n : ns) {
        if (n > (int32_t) tokens.size()) {
            printf("DIAG C3_THRESH_n%d: SKIP (prompt short)\n", n);
            continue;
        }
        const logits_result a = run_prefill_logits(model, params, tokens, n, n, 8);
        const logits_result b = run_prefill_logits(model, params, tokens, n, n, n);
        if (!a.ok || !b.ok) {
            printf("DIAG C3_THRESH_n%d: FAIL_RUN a.ok=%d b.ok=%d\n", n, a.ok ? 1 : 0, b.ok ? 1 : 0);
            return false;
        }
        const logits_cmp c = compare_logits(a.logits.data(), b.logits.data(), n_vocab);
        char label[64];
        snprintf(label, sizeof(label), "C3_THRESH_n%d", n);
        print_cmp(label, c);
    }
    printf("C3_THRESH_DONE: n=1280 should stay exact; first RED names pad-bucket edge\n");
    return true;
}

bool run_c3_bisect(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & tokens) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    constexpr int32_t n_ok = 1280;
    constexpr int32_t n_bad = 1344;
    if ((int32_t) tokens.size() < n_bad) {
        fprintf(stderr, "FAIL: c3-bisect needs >= %d prompt tokens (got %zu)\n",
                n_bad, tokens.size());
        return false;
    }

    capture_state cap_proto;
    if (const char * filt = std::getenv("TASK066_BISECT_FILTER"); filt && filt[0]) {
        try {
            cap_proto.filters.emplace_back(filt);
        } catch (const std::regex_error & e) {
            fprintf(stderr, "FAIL: TASK066_BISECT_FILTER invalid regex: %s\n", e.what());
            return false;
        }
        fprintf(stderr, "bisect_filter=%s\n", filt);
    }
    if (const char * ho = std::getenv("TASK066_BISECT_HASH_ONLY"); ho && ho[0] == '1') {
        cap_proto.keep_row0 = false;
        fprintf(stderr, "bisect_hash_only=1\n");
    }

    auto one_pair = [&](const char * label, int32_t n) -> bool {
        // Match C3/H10 pure pair: same outer n_batch=n, ub=8 vs ub=n (internal
        // ubatch). cb_eval stays armed for the single llama_decode; capture_cb
        // resets on each micro-batch embd so only the last ubatch remains.
        printf("C3_BISECT n=%d multi b=%d ub=8 vs oneshot b=%d ub=%d "
               "(last-ubatch named last-token)\n",
               n, n, n, n);
        capture_state cap_a = cap_proto;
        capture_state cap_b = cap_proto;
        const bisect_prefill_result a =
                run_prefill_bisect(model, params, tokens, n, /*n_batch=*/n, /*n_ubatch=*/8, &cap_a);
        const bisect_prefill_result b =
                run_prefill_bisect(model, params, tokens, n, /*n_batch=*/n, /*n_ubatch=*/n, &cap_b);
        if (!a.ok || !b.ok) {
            printf("DIAG %s: FAIL_RUN a.ok=%d b.ok=%d\n", label, a.ok ? 1 : 0, b.ok ? 1 : 0);
            return false;
        }
        const logits_cmp c = compare_logits(a.logits.data(), b.logits.data(), n_vocab);
        print_cmp(label, c);
        report_c3_bisect(label, a.snaps, b.snaps);
        return true;
    };

    printf("UBATCH-S8-S1024-C3-BISECT (evidence only; C1/C2/C3 gates unchanged)\n");
    printf("  model=%s n_prompt_tokens=%zu\n", params.model.path.c_str(), tokens.size());
    if (!one_pair("C3_BISECT_n1280_control", n_ok)) {
        return false;
    }
    if (!one_pair("C3_BISECT_n1344", n_bad)) {
        return false;
    }
    printf("C3_BISECT_DONE: use FIRST_DIFF_TENSOR_SUMMARY on n1344; n1280 should be <none>\n");
    return true;
}

bool is_turbo_kv_type(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

// Minimal name→type for diag overrides (product names only).
bool parse_kv_type_name(const char * name, ggml_type & out) {
    if (!name || !name[0]) {
        return false;
    }
    const ggml_type candidates[] = {
            GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_TURBO4_0,
            GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0,
    };
    for (ggml_type t : candidates) {
        if (std::strcmp(ggml_type_name(t), name) == 0) {
            out = t;
            return true;
        }
    }
    return false;
}

bool run_localize(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & tokens) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_chunk = std::min(512, (int32_t) tokens.size());
    const int32_t n_full = (int32_t) tokens.size();

    printf("LOCALIZE: n_chunk=%d n_full=%d (bit-exact logits; not a PASS gate)\n", n_chunk, n_full);

    auto one = [&](const char * label, int32_t n_pref, int32_t b0, int32_t ub0, int32_t b1, int32_t ub1,
                   bool fa_on, ggml_type ctv) -> bool {
        common_params p = params;
        p.cache_type_v = ctv;
        p.flash_attn_type = fa_on ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (!fa_on && is_turbo_kv_type(ctv)) {
            printf("DIAG %s: SKIP — FA-off + turbo ctv is force-re-enabled by llama; "
                   "use ctv=f16 for true FA-off\n", label);
            return true;
        }
        const logits_result a = run_prefill_logits(model, p, tokens, n_pref, b0, ub0);
        const logits_result b = run_prefill_logits(model, p, tokens, n_pref, b1, ub1);
        if (!a.ok || !b.ok) {
            printf("DIAG %s: FAIL_RUN a.ok=%d b.ok=%d\n", label, a.ok ? 1 : 0, b.ok ? 1 : 0);
            return false;
        }
        const logits_cmp c = compare_logits(a.logits.data(), b.logits.data(), n_vocab);
        print_cmp(label, c);
        return true;
    };

    const ggml_type ctv_prod = params.cache_type_v;

    // H1: pure ubatch split inside one outer batch of 512.
    if (!one("H1_chunk512_ub8_vs_ub512_same_b512_faON", n_chunk, 512, 8, 512, 512, true, ctv_prod)) {
        return false;
    }
    // H2: product-shaped first chunk (different -b and -ub).
    if (!one("H2_chunk512_b512ub8_vs_b1024ub1024_faON", n_chunk, 512, 8, 1024, 1024, true, ctv_prod)) {
        return false;
    }
    // H3: full prefill product pair (explains first_token_diff=0).
    if (!one("H3_full_prefill_b512ub8_vs_b1024ub1024_faON", n_full, 512, 8, 1024, 1024, true, ctv_prod)) {
        return false;
    }
    // H4: FA-off is invalid with turbo — skipped unless ctv already non-turbo.
    if (!one("H4_chunk512_ub8_vs_ub512_faOFF_turbo_SKIP", n_chunk, 512, 8, 512, 512, false, ctv_prod)) {
        return false;
    }

    // H5–H7: scale the *number* of internal ubatch steps (same outer n).
    if (!one("H5_n8_b8_ub8_vs_ub8_sanity", 8, 8, 8, 8, 8, true, ctv_prod)) {
        return false;
    }
    if (!one("H6_n16_b16_ub8_vs_ub16", 16, 16, 8, 16, 16, true, ctv_prod)) {
        return false;
    }
    if (!one("H7_n32_b32_ub8_vs_ub32", 32, 32, 8, 32, 32, true, ctv_prod)) {
        return false;
    }

    // H8+: post-tile threshold — n<=1280 GREEN, n=1536 RED (stamp 233840Z).
    // Bisect + pure-ubatch twin to separate product (-b/-ub) from multi-step alone.
    for (int32_t n_try : {256, 768, 1024, 1280, 1344, 1408, 1472, 1536}) {
        if (n_try > n_full) {
            continue;
        }
        char label[128];
        snprintf(label, sizeof(label), "H8plus_n%d_b512ub8_vs_b1024ub1024_faON", n_try);
        if (!one(label, n_try, 512, 8, 1024, 1024, true, ctv_prod)) {
            return false;
        }
    }
    // Pure ubatch at first RED width (same outer n=batch): ub=8 vs ub=n.
    if (n_full >= 1536) {
        if (!one("H9_n1536_b1536_ub8_vs_ub1536_faON", 1536, 1536, 8, 1536, 1536, true, ctv_prod)) {
            return false;
        }
        if (!one("H9f16_n1536_b1536_ub8_vs_ub1536_q8_f16_faON", 1536, 1536, 8, 1536, 1536, true,
                 GGML_TYPE_F16)) {
            return false;
        }
        // True FA-off isolate: if GREEN → residual still in FA; if RED → non-FA class (066.5 fuse?).
        if (!one("H9faOFF_n1536_b1536_ub8_vs_ub1536_q8_f16", 1536, 1536, 8, 1536, 1536, false,
                 GGML_TYPE_F16)) {
            return false;
        }
        // Boundary pair: last GREEN vs first RED (1280 vs 1344) for 066.9 Phase A TDD.
        if (!one("H10_n1280_b1280_ub8_vs_ub1280_faON", 1280, 1280, 8, 1280, 1280, true, ctv_prod)) {
            return false;
        }
        if (!one("H10_n1344_b1344_ub8_vs_ub1344_faON", 1344, 1344, 8, 1344, 1344, true, ctv_prod)) {
            return false;
        }
    }
    // H3 sibling: f16 V — is residual still inside FA when turbo is out?
    if (!one("H3f16_full_prefill_b512ub8_vs_b1024ub1024_q8_f16_faON", n_full, 512, 8, 1024, 1024, true,
             GGML_TYPE_F16)) {
        return false;
    }

    printf("LOCALIZE_DONE: see TASK066_DIAG=fa-ablation for true FA-off (ctv=f16)\n");
    return true;
}

// C2-shaped FA/KV ablation. Localization only — not a product PASS gate.
bool run_fa_ablation(
        llama_model * model,
        common_params params,
        const std::vector<llama_token> & tokens) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if ((int32_t) tokens.size() < 32) {
        fprintf(stderr, "FAIL: fa-ablation needs >=32 prompt tokens\n");
        return false;
    }

    printf("FA_ABLATION: C2 n=32 ub=8 vs ub=32 — LOCALIZATION ONLY "
           "(product remains FA-on + turbo4)\n");

    auto row = [&](const char * label, ggml_type ctk, ggml_type ctv, bool fa_on) -> bool {
        if (!fa_on && is_turbo_kv_type(ctv)) {
            printf("DIAG %s: SKIP — refuse FA-off with turbo ctv "
                   "(llama would force FA back on)\n", label);
            return true;
        }
        common_params p = params;
        p.cache_type_k = ctk;
        p.cache_type_v = ctv;
        p.flash_attn_type = fa_on ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        fprintf(stderr, "fa_ablation_row: %s ctk=%s ctv=%s fa=%s\n",
                label, ggml_type_name(ctk), ggml_type_name(ctv), fa_on ? "on" : "off");
        const logits_result a = run_prefill_logits(model, p, tokens, 32, 32, 8);
        const logits_result b = run_prefill_logits(model, p, tokens, 32, 32, 32);
        if (!a.ok || !b.ok) {
            printf("DIAG %s: FAIL_RUN a.ok=%d b.ok=%d\n", label, a.ok ? 1 : 0, b.ok ? 1 : 0);
            return false;
        }
        const logits_cmp c = compare_logits(a.logits.data(), b.logits.data(), n_vocab);
        print_cmp(label, c);
        return true;
    };

    // A0: product lock (known RED).
    if (!row("A0_C2_product_q8_turbo4_faON", GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0, true)) {
        return false;
    }
    // A1: same FA-on, V cache f16 — isolates turbo V vs FA.
    if (!row("A1_C2_q8_f16_faON", GGML_TYPE_Q8_0, GGML_TYPE_F16, true)) {
        return false;
    }
    // A2: true FA-off (requires non-turbo V).
    if (!row("A2_C2_q8_f16_faOFF", GGML_TYPE_Q8_0, GGML_TYPE_F16, false)) {
        return false;
    }
    // A3: both KV f16, FA-off.
    if (!row("A3_C2_f16_f16_faOFF", GGML_TYPE_F16, GGML_TYPE_F16, false)) {
        return false;
    }
    // A4: both KV f16, FA-on (control vs A3).
    if (!row("A4_C2_f16_f16_faON", GGML_TYPE_F16, GGML_TYPE_F16, true)) {
        return false;
    }

    printf("FA_ABLATION_DONE: compare A0 vs A1 (turbo V?) vs A2/A3 (FA-off) vs A4 (FA-on f16)\n");
    printf("NOTE: any GREEN under FA-off does NOT waive product FA-on+turbo4 debugging\n");
    return true;
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
    if (params.prompt.empty()) {
        fprintf(stderr,
                "FAIL: missing required -f / -p prompt (use ladder fixture); "
                "refusing silent prompt substitution\n");
        return 2;
    }
    if (params.n_ctx < 4) {
        fprintf(stderr, "FAIL: -c / n_ctx must be set (>=4); refusing silent ctx\n");
        return 2;
    }

    params.n_gpu_layers = 999;
    params.cache_type_k = GGML_TYPE_Q8_0;
    params.cache_type_v = GGML_TYPE_TURBO4_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_ram_mib = 0;
    params.cpuparams.n_threads = 4;
    params.cpuparams_batch.n_threads = 4;
    params.n_parallel = 1;

    // Optional KV overrides for localization (not product PASS).
    if (const char * ctk = std::getenv("TASK066_CTK"); ctk && ctk[0]) {
        ggml_type t = GGML_TYPE_COUNT;
        if (!parse_kv_type_name(ctk, t)) {
            fprintf(stderr, "FAIL: TASK066_CTK unsupported type name: %s\n", ctk);
            return 2;
        }
        params.cache_type_k = t;
        fprintf(stderr, "DIAG_ONLY: TASK066_CTK=%s\n", ggml_type_name(t));
    }
    if (const char * ctv = std::getenv("TASK066_CTV"); ctv && ctv[0]) {
        ggml_type t = GGML_TYPE_COUNT;
        if (!parse_kv_type_name(ctv, t)) {
            fprintf(stderr, "FAIL: TASK066_CTV unsupported type name: %s\n", ctv);
            return 2;
        }
        params.cache_type_v = t;
        fprintf(stderr, "DIAG_ONLY: TASK066_CTV=%s\n", ggml_type_name(t));
    }

    if (const char * fa = std::getenv("TASK066_FA"); fa && std::strcmp(fa, "off") == 0) {
        if (is_turbo_kv_type(params.cache_type_v) || is_turbo_kv_type(params.cache_type_k)) {
            fprintf(stderr,
                    "FAIL: TASK066_FA=off with turbo KV is invalid "
                    "(llama force-enables FA). Set TASK066_CTV=f16 (and ideally non-turbo CTK).\n"
                    "NOTE: FA-off is localization-only; product debug stays FA-on+turbo4.\n");
            return 2;
        }
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        fprintf(stderr,
                "DIAG_ONLY: TASK066_FA=off (ctk=%s ctv=%s) — not a product waiver; "
                "later debugging remains FA-on\n",
                ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v));
    }

    int32_t n_predict = 64;
    if (const char * e = std::getenv("TASK066_N_PREDICT"); e && e[0]) {
        n_predict = (int32_t) std::strtol(e, nullptr, 10);
        if (n_predict < 1) {
            fprintf(stderr, "FAIL: TASK066_N_PREDICT must be >= 1\n");
            return 2;
        }
    }

    const char * diag = std::getenv("TASK066_DIAG");
    const bool do_localize = diag && std::strcmp(diag, "localize") == 0;
    const bool do_fa_ablation = diag && std::strcmp(diag, "fa-ablation") == 0;
    const bool do_c3_bisect = diag && std::strcmp(diag, "c3-bisect") == 0;
    const bool do_c3_thresh = diag && std::strcmp(diag, "c3-thresh") == 0;
    const bool do_c1j = diag && std::strcmp(diag, "c1j") == 0;

    const int32_t ub_a = 8;
    const int32_t ub_b = 1024;
    const int32_t b_a = 512;
    const int32_t b_b = 1024;

    fprintf(stderr,
            "product_lock: ngl=%d ctk=%s ctv=%s fa=%s cram=%d t=%d tb=%d np=%d ctx=%d\n",
            params.n_gpu_layers, ggml_type_name(params.cache_type_k),
            ggml_type_name(params.cache_type_v),
            params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED ? "off" : "on",
            params.cache_ram_mib,
            params.cpuparams.n_threads, params.cpuparams_batch.n_threads,
            params.n_parallel, (int) params.n_ctx);
    fprintf(stderr, "ubatch_pair: A b=%d ub=%d  B b=%d ub=%d  n_predict=%d diag=%s\n",
            b_a, ub_a, b_b, ub_b, n_predict, diag && diag[0] ? diag : "<none>");
    if (const char * mm = std::getenv("GGML_METAL_MUL_MM_MIN"); mm && mm[0]) {
        fprintf(stderr, "GGML_METAL_MUL_MM_MIN=%s\n", mm);
    } else {
        fprintf(stderr, "GGML_METAL_MUL_MM_MIN=<unset:default 0>\n");
    }

    const bool require_gpu = []() {
        const char * e = std::getenv("TASK066_REQUIRE_GPU");
        return e && e[0] == '1';
    }();

    ggml_backend_load_all();
    llama_backend_init();

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

    params.n_batch = b_b;
    params.n_ubatch = ub_b;
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) {
        fprintf(stderr, "FAIL: model load\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    tokens.resize(params.prompt.size() + 16);
    const int32_t n_tok = llama_tokenize(
            vocab, params.prompt.c_str(), (int32_t) params.prompt.size(),
            tokens.data(), (int32_t) tokens.size(),
            /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) {
        tokens.resize(size_t(-n_tok));
        const int32_t n2 = llama_tokenize(
                vocab, params.prompt.c_str(), (int32_t) params.prompt.size(),
                tokens.data(), (int32_t) tokens.size(), true, true);
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

    const int64_t need_ctx = (int64_t) tokens.size() + (int64_t) n_predict + 8;
    if ((int64_t) params.n_ctx < need_ctx) {
        fprintf(stderr,
                "FAIL: n_ctx=%d too small for prompt=%zu + n_predict=%d (need >= %lld)\n",
                (int) params.n_ctx, tokens.size(), n_predict, (long long) need_ctx);
        return 2;
    }

    if (do_c1j) {
        // Fast A2 loop: skip C2/C3/C1 raw; exit code is the C1j gate (RED/GREEN).
        printf("UBATCH-S8-S1024-DIAG C1j only (066.9 Phase A2)\n");
        return run_c1j_gate(model, vocab, params, params.prompt, n_predict, b_a, ub_a, b_b, ub_b);
    }

    if (do_localize) {
        printf("UBATCH-S8-S1024-LOCALIZE (evidence only; C1/C2 gates unchanged)\n");
        printf("  model=%s n_prompt_tokens=%zu\n", params.model.path.c_str(), tokens.size());
        if (!run_localize(model, params, tokens)) {
            return 1;
        }
        printf("RESULT LOCALIZE_OK (re-run without TASK066_DIAG for C1/C2 RED/GREEN)\n");
        return 0;
    }
    if (do_fa_ablation) {
        printf("UBATCH-S8-S1024-FA-ABLATION (evidence only; product FA-on+turbo4 untouched)\n");
        printf("  model=%s n_prompt_tokens=%zu\n", params.model.path.c_str(), tokens.size());
        if (!run_fa_ablation(model, params, tokens)) {
            return 1;
        }
        printf("RESULT FA_ABLATION_OK\n");
        return 0;
    }
    if (do_c3_bisect) {
        if (!run_c3_bisect(model, params, tokens)) {
            return 1;
        }
        printf("RESULT C3_BISECT_OK (re-run without TASK066_DIAG for C1/C2/C3 RED/GREEN)\n");
        return 0;
    }
    if (do_c3_thresh) {
        if (!run_c3_thresh(model, params, tokens)) {
            return 1;
        }
        printf("RESULT C3_THRESH_OK\n");
        return 0;
    }

    // C2 — fast TDD corner from localize H7 (stamp 20261106T214747Z):
    // n=32, same -b=32, ub=8 (4 steps) vs ub=32 (1 step). H6 n=16 was exact.
    {
        const int32_t n32 = 32;
        if ((int32_t) tokens.size() < n32) {
            fprintf(stderr, "FAIL: prompt needs >= 32 tokens for C2 (got %zu)\n", tokens.size());
            return 2;
        }
        printf("UBATCH-S8-S1024-C2 prefill logits n=32 ub=8 vs ub=32 (fast localize H7)\n");
        printf("  model=%s\n", params.model.path.c_str());
        const logits_result c2a = run_prefill_logits(model, params, tokens, n32, 32, 8);
        const logits_result c2b = run_prefill_logits(model, params, tokens, n32, 32, 32);
        if (!c2a.ok || !c2b.ok) {
            return 1;
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const logits_cmp c2 = compare_logits(c2a.logits.data(), c2b.logits.data(), n_vocab);
        print_cmp("C2", c2);
        if (!c2.exact) {
            printf("RESULT FAIL (RED) — C2 n=32 ubatch multi-step vs one-shot; "
                   "066.5/066.6 TDD gate open (skip C1 until C2 GREEN)\n");
            return 1;
        }
        printf("C2 PASS — n=32 ub=8 vs ub=32 bit-exact; continuing to C3 long-prefill gate\n");
    }

    // C3 — 066.9 Phase A: post-tile threshold (localize 234258Z / 235017Z).
    // n=1280 pure ubatch bit-exact; n=1344 first RED. FA-off still RED at n=1536
    // → residual class is not FA-only (fuse / other ops). C3-1280 = REGRESS lock;
    // C3-1344 = primary TDD RED until fixed.
    {
        constexpr int32_t n_ok = 1280;
        constexpr int32_t n_bad = 1344;
        if ((int32_t) tokens.size() < n_bad) {
            fprintf(stderr, "FAIL: prompt needs >= %d tokens for C3 (got %zu)\n", n_bad,
                    tokens.size());
            return 2;
        }
        printf("UBATCH-S8-S1024-C3 long-prefill ubatch threshold n=%d GREEN / n=%d RED gate\n",
               n_ok, n_bad);
        const logits_result ok_a = run_prefill_logits(model, params, tokens, n_ok, n_ok, 8);
        const logits_result ok_b = run_prefill_logits(model, params, tokens, n_ok, n_ok, n_ok);
        if (!ok_a.ok || !ok_b.ok) {
            return 1;
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const logits_cmp c_ok = compare_logits(ok_a.logits.data(), ok_b.logits.data(), n_vocab);
        print_cmp("C3_n1280", c_ok);
        if (!c_ok.exact) {
            printf("RESULT FAIL (REGRESS) — C3 n=1280 was GREEN after always-tile; bit-exact broken\n");
            return 1;
        }
        const logits_result bad_a = run_prefill_logits(model, params, tokens, n_bad, n_bad, 8);
        const logits_result bad_b = run_prefill_logits(model, params, tokens, n_bad, n_bad, n_bad);
        if (!bad_a.ok || !bad_b.ok) {
            return 1;
        }
        const logits_cmp c_bad = compare_logits(bad_a.logits.data(), bad_b.logits.data(), n_vocab);
        print_cmp("C3_n1344", c_bad);
        if (c_bad.exact) {
            printf("C3 n=1344 unexpectedly GREEN — continuing to C1\n");
        } else {
            printf("RESULT FAIL (RED) — C3 n=1344 multi-step ubatch (066.9 Phase A; "
                   "FA-off also RED at n=1536 → not FA-only). Skip C1 until C3 GREEN\n");
            return 1;
        }
    }

    printf("UBATCH-S8-S1024-C1 greedy token-stream bit-exact\n");
    printf("  model=%s\n", params.model.path.c_str());
    printf("  n_prompt_tokens=%zu n_predict=%d n_vocab=%d\n",
           tokens.size(), n_predict, llama_vocab_n_tokens(vocab));
    printf("HYPOTHESIS: greedy ids diverge under ubatch 8 vs 1024 (ladder DIFF@text class)\n");

    const gen_result a = run_greedy_ubatch(model, params, tokens, n_predict, b_a, ub_a);
    if (!a.ok) {
        return 1;
    }
    const gen_result b = run_greedy_ubatch(model, params, tokens, n_predict, b_b, ub_b);
    if (!b.ok) {
        return 1;
    }

    const int32_t diff_i = first_token_diff(a.tokens, b.tokens);
    printf("  gen_len_a=%zu gen_len_b=%zu first_token_diff=%d\n",
           a.tokens.size(), b.tokens.size(), diff_i);
    if (diff_i >= 0 && diff_i < (int32_t) a.tokens.size() && diff_i < (int32_t) b.tokens.size()) {
        printf("  tok_a=%d tok_b=%d\n", a.tokens[size_t(diff_i)], b.tokens[size_t(diff_i)]);
    }

    if (diff_i >= 0) {
        printf("RESULT FAIL (RED) — greedy token stream not bit-exact S8 vs S1024; "
               "066.5/066.6 TDD gate open\n");
        return 1;
    }

    printf("RESULT PASS (GREEN) — greedy S8 vs S1024 identical for n_predict=%d on this prompt\n",
           n_predict);
    printf("note: raw C1 GREEN ≠ A2 Done; continuing to C1j (jinja chat)\n");

    // Phase A2 — product ladder class (jinja-wrapped prompt).
    return run_c1j_gate(model, vocab, params, params.prompt, n_predict, b_a, ub_a, b_b, ub_b);
}
