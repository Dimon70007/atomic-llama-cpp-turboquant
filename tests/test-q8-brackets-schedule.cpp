// TASK-068 ST-10 - CPU unit for the q8 head/tail bracket layer planner.
//
// The planner is the single source of truth for which layers get q8_0 V. This
// test calls the SAME function the model loader calls, with synthetic model
// facts, so a divergence between "what the test believes" and "what the loader
// does" cannot happen.
//
// Regression this file exists for: on a shared-KV model (Gemma-4-E2B declares
// shared_kv_layers=20, so only layers 0..14 own a cache) a positional tail
// named layers 27..34, which have no cache at all. 8 of 10 q8 slots were dead
// and the bracket silently did nothing. The old test missed it because it
// checked a model-free helper that could not know about shared KV.
//
// Env: TURBO_LAYER_Q8_BRACKETS / HEAD / TAIL / ADAPTIVE / PROTECT_DENSE
//      (cleared between cases).

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Declared in src/llama-kv-cache.h; keep this unit free of the heavy KV header.
bool llama_kv_cache_q8_brackets_active(void);
bool llama_kv_cache_q8_brackets_dense_mode(void);
std::vector<uint8_t> llama_kv_cache_q8_brackets_plan_layers(
        uint32_t n_layer,
        const std::vector<bool> & has_kv,
        const std::vector<bool> & is_swa,
        bool dense_mode,
        uint32_t head_req,
        uint32_t tail_req);

namespace {

int g_fails = 0;

void expect_true(const char * name, bool cond) {
    if (cond) {
        printf("ok   %s\n", name);
        return;
    }
    printf("FAIL %s\n", name);
    ++g_fails;
}

// Render a plan as "0,1,4" for readable failures.
std::string plan_str(const std::vector<uint8_t> & want) {
    std::string s;
    for (size_t i = 0; i < want.size(); ++i) {
        if (!want[i]) {
            continue;
        }
        s += (s.empty() ? "" : ",") + std::to_string(i);
    }
    return s.empty() ? "<none>" : s;
}

void expect_plan(const char * name, const std::vector<uint8_t> & got, const std::string & want) {
    const std::string g = plan_str(got);
    if (g == want) {
        printf("ok   %s (%s)\n", name, g.c_str());
        return;
    }
    printf("FAIL %s (got=%s want=%s)\n", name, g.c_str(), want.c_str());
    ++g_fails;
}

void expect_throws(const char * name, uint32_t n_layer,
                   const std::vector<bool> & has_kv, const std::vector<bool> & is_swa,
                   bool dense, uint32_t head, uint32_t tail) {
    bool threw = false;
    try {
        (void) llama_kv_cache_q8_brackets_plan_layers(n_layer, has_kv, is_swa, dense, head, tail);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect_true(name, threw);
}

std::vector<bool> all_true(size_t n, bool v = true) {
    return std::vector<bool>(n, v);
}

// Gemma-4 shapes differ per model, which is the whole point: the schedule must
// be read from the model, never assumed.
struct ModelShape {
    const char * name;
    uint32_t n_layer;
    uint32_t swa_period;    // dense layer every swa_period-th layer
    uint32_t dense_offset;  // index of the first dense layer
    uint32_t shared_kv;     // trailing layers with no cache of their own
};

// Real shapes, read from the GGUFs (see ST-10.11).
const ModelShape kModels[] = {
    { "gemma-4-E2B",    35, 5, 4, 20 },  // dense 4,9,14,19,24,29,34; kv 0..14
    { "gemma-4-26B-A4B", 30, 6, 5,  0 },  // dense 5,11,17,23,29; kv 0..29
};

std::vector<bool> make_swa(const ModelShape & m) {
    std::vector<bool> s(m.n_layer, true);
    for (uint32_t i = m.dense_offset; i < m.n_layer; i += m.swa_period) {
        s[i] = false;
    }
    return s;
}

std::vector<bool> make_has_kv(const ModelShape & m) {
    std::vector<bool> kv(m.n_layer, true);
    for (uint32_t i = m.n_layer - m.shared_kv; i < m.n_layer; ++i) {
        kv[i] = false;
    }
    return kv;
}

// Independent oracle: declarative filtering, written differently from the
// planner's greedy loops on purpose, so a shared bug is unlikely.
std::string expected_plan(const ModelShape & m, bool dense, uint32_t head, uint32_t tail) {
    const auto swa = make_swa(m);
    const auto kv  = make_has_kv(m);

    std::vector<bool> sel(m.n_layer, false);
    if (dense) {
        for (uint32_t i = 0; i < m.n_layer; ++i) {
            sel[i] = kv[i] && (i < head || !swa[i]);
        }
    } else {
        // Head: first `head` layers that own KV.
        uint32_t left = head;
        for (uint32_t i = 0; i < m.n_layer && left > 0; ++i) {
            if (kv[i]) { sel[i] = true; --left; }
        }
        // Tail: last `tail` layers that own KV (skipping already selected).
        left = tail;
        for (uint32_t i = m.n_layer; i-- > 0 && left > 0; ) {
            if (kv[i] && !sel[i]) { sel[i] = true; --left; }
        }
    }

    std::string s;
    for (uint32_t i = 0; i < m.n_layer; ++i) {
        if (sel[i]) {
            s += (s.empty() ? "" : ",") + std::to_string(i);
        }
    }
    return s.empty() ? "<none>" : s;
}

void check_model(const ModelShape & m, bool dense, uint32_t head, uint32_t tail) {
    const auto swa = make_swa(m);
    const auto kv  = make_has_kv(m);
    const std::string want = expected_plan(m, dense, head, tail);

    char label[128];
    snprintf(label, sizeof(label), "%s %s head%u_tail%u", m.name, dense ? "dense" : "positional", head, tail);

    std::vector<uint8_t> got;
    try {
        got = llama_kv_cache_q8_brackets_plan_layers(m.n_layer, kv, swa, dense, head, tail);
    } catch (const std::runtime_error & e) {
        // The oracle decides legality via the half cap; report what happened.
        printf("FAIL %s threw (%s), oracle expected %s\n", label, e.what(), want.c_str());
        ++g_fails;
        return;
    }
    expect_plan(label, got, want);
}

// Number of layers with a cache; the useful denominator for "how many live slots".
uint32_t live_kv(const ModelShape & m) { return m.n_layer - m.shared_kv; }

}  // namespace

int main() {
    printf("TASK-068 q8 brackets planner unit (CPU; no model)\n");

    // ---- env gates -------------------------------------------------------
    unsetenv("TURBO_LAYER_Q8_BRACKETS");
    unsetenv("TURBO_LAYER_PROTECT_DENSE");
    unsetenv("TURBO_LAYER_HEAD");
    unsetenv("TURBO_LAYER_TAIL");
    unsetenv("TURBO_LAYER_ADAPTIVE");
    expect_true("inactive_when_env_unset", !llama_kv_cache_q8_brackets_active());
    expect_true("dense_off_by_default", !llama_kv_cache_q8_brackets_dense_mode());

    setenv("TURBO_LAYER_Q8_BRACKETS", "0", 1);
    expect_true("inactive_when_env_zero", !llama_kv_cache_q8_brackets_active());
    setenv("TURBO_LAYER_Q8_BRACKETS", "1", 1);
    expect_true("active_when_env_set", llama_kv_cache_q8_brackets_active());

    setenv("TURBO_LAYER_PROTECT_DENSE", "1", 1);
    expect_true("dense_on_when_env_set", llama_kv_cache_q8_brackets_dense_mode());
    unsetenv("TURBO_LAYER_PROTECT_DENSE");

    // ---- every real model shape, expectations computed from its parameters ----
    // No literal index list here: expected_plan() derives the set from the same
    // ModelShape data, so adding a model only means adding one row to kModels.
    for (const auto & m : kModels) {
        printf("-- %s: n_layer=%u dense every %u from %u, shared_kv=%u (live %u)\n",
               m.name, m.n_layer, m.swa_period, m.dense_offset, m.shared_kv, live_kv(m));

        check_model(m, false, 2, 8);
        check_model(m, false, 0, 8);
        check_model(m, true,  2, 8);

        // The regression: on a shared-KV model every planned layer must own a
        // cache, otherwise the slot is a silent no-op.
        const auto kv  = make_has_kv(m);
        const auto swa = make_swa(m);
        const auto got = llama_kv_cache_q8_brackets_plan_layers(m.n_layer, kv, swa, false, 2, 8);
        uint32_t dead = 0;
        for (uint32_t i = 0; i < m.n_layer; ++i) {
            if (got[i] && !kv[i]) {
                ++dead;
            }
        }
        char label[128];
        snprintf(label, sizeof(label), "%s positional_has_no_dead_slots", m.name);
        expect_true(label, dead == 0);

        // Dense mode must also stay inside the KV-bearing range.
        const auto got_d = llama_kv_cache_q8_brackets_plan_layers(m.n_layer, kv, swa, true, 2, 8);
        uint32_t dead_d = 0;
        for (uint32_t i = 0; i < m.n_layer; ++i) {
            if (got_d[i] && !kv[i]) {
                ++dead_d;
            }
        }
        snprintf(label, sizeof(label), "%s dense_has_no_dead_slots", m.name);
        expect_true(label, dead_d == 0);
    }

    // ---- half-layer cap ---------------------------------------------------
    {
        const uint32_t n = 30;
        const auto kv = all_true(n);
        const auto swa = all_true(n, false);
        // 15 of 30 is exactly half -> allowed.
        expect_true("half_exactly_ok",
                    llama_kv_cache_q8_brackets_plan_layers(n, kv, swa, false, 7, 8).size() == n);
        // 16 of 30 -> abort.
        expect_throws("over_half_throws", n, kv, swa, false, 8, 8);
        // A 3-layer model cannot hold 2 q8 layers (2 of 3 > half).
        expect_throws("small_model_over_half_throws", 3, all_true(3), all_true(3, false), false, 1, 1);
        // 4 layers with 1+1 is exactly half -> allowed, plan = 0,3.
        expect_plan("small_model_half_ok",
                    llama_kv_cache_q8_brackets_plan_layers(4, all_true(4), all_true(4, false), false, 1, 1),
                    "0,3");
    }

    // ---- fail-closed corners ----------------------------------------------
    {
        const uint32_t n = 30;
        const auto kv = all_true(n);
        const auto swa = all_true(n, false);
        expect_throws("head_ge_n_layer_throws", n, kv, swa, false, 30, 0);
        expect_throws("tail_ge_n_layer_throws", n, kv, swa, false, 0, 30);
        // No layer owns KV -> nothing to protect -> abort, never a silent no-op.
        expect_throws("no_kv_layers_throws", n, all_true(n, false), swa, false, 2, 8);
        // Dense mode with every layer SWA and HEAD=0 -> empty plan -> abort.
        expect_throws("dense_mode_empty_throws", n, kv, all_true(n), true, 0, 8);
    }

    if (g_fails == 0) {
        printf("all ok\n");
        return 0;
    }
    printf("%d FAILED\n", g_fails);
    return 1;
}
