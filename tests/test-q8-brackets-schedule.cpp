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
uint32_t llama_kv_cache_q8_brackets_percent_to_layers(uint32_t n_layer, uint32_t percent);
struct llama_kv_cache_q8_widths {
    uint32_t head = 0;
    uint32_t tail = 0;
    uint32_t head_pct = 0;
    uint32_t tail_pct = 0;
};
llama_kv_cache_q8_widths llama_kv_cache_q8_brackets_resolve_widths(uint32_t n_layer, uint32_t n_kv_layer = 0);
std::vector<uint8_t> llama_kv_cache_q8_brackets_plan_layers(
        uint32_t n_layer,
        const std::vector<bool> & has_kv,
        const std::vector<bool> & is_swa,
        bool dense_mode,
        uint32_t head_req,
        uint32_t tail_req);
ggml_type llama_kv_cache_q8_brackets_layer_type_v(
        ggml_type mid_v,
        const std::vector<uint8_t> & plan,
        uint32_t il);

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
        // Dense mode: head first, then the layers AFTER each dense layer,
        // counted outwards from the last dense layer.
        uint32_t left = head;
        for (uint32_t i = 0; i < m.n_layer && left > 0; ++i) {
            if (kv[i]) { sel[i] = true; --left; }
        }
        left = tail;
        for (uint32_t off = 1; off < m.n_layer && left > 0; ++off) {
            for (uint32_t i = m.n_layer; i-- > 0 && left > 0; ) {
                if (swa[i]) { continue; }
                const uint32_t nb = i + off;
                if (nb >= m.n_layer || !kv[nb] || sel[nb] || !swa[nb]) { continue; }
                sel[nb] = true; --left;
            }
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

// Dense-mode throw check: dense=true, custom tail.
void expect_throws_on(const char * name, const ModelShape & m,
                      const std::vector<bool> & has_kv, const std::vector<bool> & is_swa,
                      uint32_t head, uint32_t tail) {
    bool threw = false;
    try {
        (void) llama_kv_cache_q8_brackets_plan_layers(
                m.n_layer, has_kv, is_swa, true, head, tail);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect_true(name, threw);
}

// The bracket schedule is computed once and applied per layer. The planner is
// mid-agnostic, so every test above proves the INDICES and nothing about the
// actual V type of a layer: turbo4.8 could silently resolve to uniform turbo4
// and still pass. Exercise the application step for both mids.
//
// This is the level that would catch "turbo4.8 behaves like turbo4".
// Dense mode (TASK-068 10.13 / 10.31): q8 goes to the layers AFTER the dense
// ones, and tail_req decides how many such slots exist. Walks dense layers
// backwards taking offsets +1, +2, ... so the tail hugs the top of the stack.
void check_dense_neighbour(const ModelShape & m) {
    const auto kv  = make_has_kv(m);
    const auto swa = make_swa(m);
    const uint32_t head = 2;
    const uint32_t tail = 8;

    // A short tail is now clamped (with a warning) instead of refused, because a
    // shared-KV model simply has fewer usable slots than the width asks for.
    uint32_t slots_available = 0;
    for (uint32_t offset = 1; offset < m.n_layer; ++offset) {
        for (uint32_t il = 0; il < m.n_layer; ++il) {
            if (swa[il]) {
                continue;
            }
            const uint32_t nb = il + offset;
            if (nb < m.n_layer && kv[nb] && swa[nb]) {
                ++slots_available;
            }
        }
    }
    {
        const auto plan = llama_kv_cache_q8_brackets_plan_layers(
                m.n_layer, kv, swa, true, head, tail);
        uint32_t selected = 0;
        for (uint32_t il = 0; il < m.n_layer; ++il) {
            selected += plan[il] ? 1 : 0;
        }
        char label[192];
        // Spend min(tail, available) slots, never more and never zero-by-refusal.
        const uint32_t want_tail = slots_available < tail ? slots_available : tail;
        snprintf(label, sizeof(label), "%s dense_tail_clamped_to_available", m.name);
        expect_true(label, selected == head + want_tail);

        // Nothing selected may be a layer without a cache (the 10.15 bug class).
        uint32_t dead = 0;
        for (uint32_t il = 0; il < m.n_layer; ++il) {
            if (plan[il] && !kv[il]) {
                ++dead;
            }
        }
        snprintf(label, sizeof(label), "%s dense_never_selects_dead_layer", m.name);
        expect_true(label, dead == 0);
    }
    if (slots_available < tail) {
        return;  // further invariants assume the full tail fitted
    }

    const auto plan = llama_kv_cache_q8_brackets_plan_layers(
            m.n_layer, kv, swa, true, head, tail);

    // Exact set for the 26B shape, which is the operator spec for 10.31.
    // Dense layers 5,11,17,23,29 -> offsets +1 and +2, capped at tail=8.
    if (m.n_layer == 30 && m.shared_kv == 0 && m.dense_offset == 5 && m.swa_period == 6) {
        expect_plan("gemma-4-26B-A4B dense_10_31_exact", plan,
                    "0,1,6,7,12,13,18,19,24,25");
    }

    // Every selected layer past the head must directly follow a dense layer.
    uint32_t misplaced = 0;
    uint32_t selected  = 0;
    for (uint32_t il = 0; il < m.n_layer; ++il) {
        if (!plan[il]) {
            continue;
        }
        ++selected;
        if (il < head) {
            continue;  // head layers keep their q8 for instruction fidelity
        }
        // Any selected layer must sit at some offset after a dense layer.
        bool hangs_off_dense = false;
        for (uint32_t d = 1; d <= il; ++d) {
            if (!swa[il - d]) {
                hangs_off_dense = true;
                break;
            }
        }
        if (!kv[il] || !hangs_off_dense) {
            ++misplaced;
        }
    }
    char label[192];
    snprintf(label, sizeof(label), "%s dense_q8_follows_dense", m.name);
    expect_true(label, selected > head && misplaced == 0);

    // The dense layers themselves must stay on the mid type - that is the whole
    // point of the mode (protecting them instead costs the full-length cache).
    uint32_t dense_q8 = 0;
    for (uint32_t il = head; il < m.n_layer; ++il) {
        if (!swa[il] && plan[il]) {
            ++dense_q8;
        }
    }
    snprintf(label, sizeof(label), "%s dense_never_quantizes_dense", m.name);
    expect_true(label, dense_q8 == 0);

    // tail_req must drive the count, not the number of dense layers: with the
    // same model, asking for 4 must select exactly 4 slots past the head.
    const auto plan4 = llama_kv_cache_q8_brackets_plan_layers(
            m.n_layer, kv, swa, true, head, 4);
    uint32_t selected4 = 0;
    for (uint32_t il = 0; il < m.n_layer; ++il) {
        selected4 += plan4[il] ? 1 : 0;
    }
    snprintf(label, sizeof(label), "%s dense_tail_drives_count", m.name);
    expect_true(label, selected4 == head + 4);

    // tail=0 means head-only, which the old one-per-dense implementation could
    // not express (it always added one slot per dense layer).
    const auto plan0 = llama_kv_cache_q8_brackets_plan_layers(
            m.n_layer, kv, swa, true, head, 0);
    uint32_t selected0 = 0;
    for (uint32_t il = 0; il < m.n_layer; ++il) {
        selected0 += plan0[il] ? 1 : 0;
    }
    snprintf(label, sizeof(label), "%s dense_tail_zero_is_head_only", m.name);
    expect_true(label, selected0 == head);

    // A tail wider than available slots clamps to what exists; a tail so wide it
    // would break the half-layer cap still aborts (that cap is a hard invariant).
    {
        const auto plan = llama_kv_cache_q8_brackets_plan_layers(
                m.n_layer, kv, swa, true, head, tail + 2);
        uint32_t selected = 0;
        for (uint32_t il = 0; il < m.n_layer; ++il) {
            selected += plan[il] ? 1 : 0;
        }
        snprintf(label, sizeof(label), "%s dense_wide_tail_clamps", m.name);
        expect_true(label, selected > head && selected * 2 <= m.n_layer);
    }
    // Whatever the requested width, the plan must never exceed the half cap:
    // clamping happens before the cap check, so an absurd tail degrades to the
    // available slots instead of aborting.
    {
        const uint32_t absurd = m.n_layer - 1;
        const auto plan = llama_kv_cache_q8_brackets_plan_layers(
                m.n_layer, kv, swa, true, head, absurd);
        uint32_t selected = 0;
        for (uint32_t il = 0; il < m.n_layer; ++il) {
            selected += plan[il] ? 1 : 0;
        }
        snprintf(label, sizeof(label), "%s dense_absurd_tail_stays_under_cap", m.name);
        expect_true(label, selected > 0 && selected * 2 <= m.n_layer);
    }
}

// Purely dense model (n_swa = 0): every KV layer is full-context, so neither
// dense mode has anything to hang off. The budget must still be spent, spread
// evenly, instead of refusing - the operator asked for N q8 layers, not N or none.
void check_dense_only_model(const ModelShape & m) {
    // Real Qwen3.5-9B shape: n_layer 32, all dense, all with KV.
    const uint32_t n_layer = 32;
    const uint32_t head = 2;
    const uint32_t tail = 8;
    const auto kv  = all_true(n_layer);
    const auto swa = all_true(n_layer, false);  // every layer is dense

    // No SWA split, so dense mode falls back to the even spread.
    {
        const auto plan = llama_kv_cache_q8_brackets_plan_layers(
                n_layer, kv, swa, true, head, tail);
        const char * mode = "dense";

        uint32_t selected = 0;
        for (uint32_t il = 0; il < n_layer; ++il) {
            selected += plan[il] ? 1 : 0;
        }
        char label[192];
        snprintf(label, sizeof(label), "%s dense_only_%s_spends_budget",
                 m.name, mode);
        expect_true(label, selected == head + tail);

        // Head keeps its contiguous slots; the rest must be spread, not clustered
        // at one end, so no three consecutive layers are selected past the head.
        uint32_t longest_run = 0;
        uint32_t run = 0;
        for (uint32_t il = head; il < n_layer; ++il) {
            run = plan[il] ? run + 1 : 0;
            longest_run = run > longest_run ? run : longest_run;
        }
        snprintf(label, sizeof(label), "%s dense_only_%s_spread_not_clustered",
                 m.name, mode);
        expect_true(label, longest_run <= 2);

        // The last stretch of the stack must be covered: an even spread cannot
        // leave the final quarter untouched.
        uint32_t in_tail_quarter = 0;
        for (uint32_t il = n_layer * 3 / 4; il < n_layer; ++il) {
            in_tail_quarter += plan[il] ? 1 : 0;
        }
        snprintf(label, sizeof(label), "%s dense_only_%s_covers_top_quarter",
                 m.name, mode);
        expect_true(label, in_tail_quarter > 0);

        // Half-layer cap still applies.
        snprintf(label, sizeof(label), "%s dense_only_%s_within_half", m.name, mode);
        expect_true(label, selected * 2 <= n_layer);
    }
}

// Percentages are shares of the layers that own a KV cache, not of n_layer.
// On a shared-KV model the old denominator overshot badly: 6/25 of E4B's 42
// layers gave 14 slots, i.e. 58% of its 24 real cache layers instead of ~31%.
void check_percent_denominator() {
    unsetenv("TURBO_LAYER_HEAD");
    unsetenv("TURBO_LAYER_TAIL");
    setenv("TURBO_LAYER_HEAD_PERCENT", "6", 1);
    setenv("TURBO_LAYER_TAIL_PERCENT", "25", 1);

    struct Case { uint32_t n_layer, n_kv; const char * name; };
    const Case cases[] = {
        { 30, 30, "denominator_26b_no_shared" },
        { 42, 24, "denominator_e4b_shared_kv" },
        { 35, 15, "denominator_e2b_shared_kv" },
        { 32, 32, "denominator_qwen_no_shared" },
    };
    for (const auto & c : cases) {
        const auto w = llama_kv_cache_q8_brackets_resolve_widths(c.n_layer, c.n_kv);
        // 6% and 25% of the KV layers, rounded to nearest, at least 1 each.
        const uint32_t want_head = llama_kv_cache_q8_brackets_percent_to_layers(c.n_kv, 6);
        const uint32_t want_tail = llama_kv_cache_q8_brackets_percent_to_layers(c.n_kv, 25);
        char label[160];
        snprintf(label, sizeof(label), "%s head_is_pct_of_kv", c.name);
        expect_true(label, w.head == want_head);
        snprintf(label, sizeof(label), "%s tail_is_pct_of_kv", c.name);
        expect_true(label, w.tail == want_tail);
        // The share must land near 31% of the real cache, never near the 58-73%
        // the old n_layer denominator produced on shared-KV models.
        snprintf(label, sizeof(label), "%s share_of_real_cache_near_31pc", c.name);
        const double share = 100.0 * (w.head + w.tail) / c.n_kv;
        expect_true(label, share > 24.0 && share < 38.0);
    }

    unsetenv("TURBO_LAYER_HEAD_PERCENT");
    unsetenv("TURBO_LAYER_TAIL_PERCENT");
}

void check_mid_application(const ModelShape & m, ggml_type mid) {
    char label[192];
    const auto kv  = make_has_kv(m);
    const auto swa = make_swa(m);
    const auto plan = llama_kv_cache_q8_brackets_plan_layers(m.n_layer, kv, swa, false, 2, 8);

    uint32_t q8 = 0;
    uint32_t wrong = 0;
    for (uint32_t il = 0; il < m.n_layer; ++il) {
        if (!kv[il]) {
            continue;  // no cache: no per-layer type to resolve
        }
        const ggml_type want = plan[il] ? GGML_TYPE_Q8_0 : mid;
        const ggml_type got  = llama_kv_cache_q8_brackets_layer_type_v(mid, plan, il);
        if (got != want) {
            ++wrong;
        }
        if (got == GGML_TYPE_Q8_0) {
            ++q8;
        }
    }
    snprintf(label, sizeof(label), "%s %s_layer_types_match_plan", m.name, ggml_type_name(mid));
    expect_true(label, wrong == 0);

    // The mid layers really are the mid type: an all-q8 plan would pass the
    // check above only if the plan were all-q8, which the half cap forbids.
    snprintf(label, sizeof(label), "%s %s_mid_layers_keep_mid", m.name, ggml_type_name(mid));
    expect_true(label, q8 * 2 <= m.n_layer && q8 > 0);

    // Out-of-range index: never a q8 layer, never a crash.
    snprintf(label, sizeof(label), "%s %s_out_of_range_is_mid", m.name, ggml_type_name(mid));
    expect_true(label, llama_kv_cache_q8_brackets_layer_type_v(mid, plan, m.n_layer) == mid);

    // turbo3.8 and turbo4.8 must differ on the mid layers and agree on q8 ones.
    // If both arms resolved to the same type everywhere, the aliases would be
    // indistinguishable - the exact failure this test exists for.
    uint32_t both_q8   = 0;
    uint32_t same_mid  = 0;
    uint32_t mid_layers = 0;
    for (uint32_t il = 0; il < m.n_layer; ++il) {
        if (!kv[il]) {
            continue;
        }
        const ggml_type a = llama_kv_cache_q8_brackets_layer_type_v(GGML_TYPE_TURBO3_0, plan, il);
        const ggml_type b = llama_kv_cache_q8_brackets_layer_type_v(GGML_TYPE_TURBO4_0, plan, il);
        if (plan[il]) {
            if (a == GGML_TYPE_Q8_0 && b == GGML_TYPE_Q8_0) {
                ++both_q8;
            }
        } else {
            ++mid_layers;
            if (a == b) {
                ++same_mid;
            }
        }
    }
    snprintf(label, sizeof(label), "%s arms_agree_on_q8_layers", m.name);
    expect_true(label, both_q8 > 0);
    snprintf(label, sizeof(label), "%s arms_differ_on_all_mid_layers", m.name);
    expect_true(label, mid_layers > 0 && same_mid == 0);
}

// Chain the two halves the loader actually runs: env percent -> resolve_widths
// -> plan_layers. Asserts the resulting LAYER INDICES, not the counts.
//
// This is the gap 10.24 covers: resolve_widths and plan_layers are each tested
// on their own, so a regression in the chain (wrong rounding feeding a correct
// planner, or a wrong slide with the right count) would go unnoticed - which is
// exactly how the original shared-KV bug looked (`[0,1]` is a legal count).
void check_env_chain(const ModelShape & m, uint32_t head_pct, uint32_t tail_pct) {
    char label[160];
    snprintf(label, sizeof(label), "%s env head%u%%_tail%u%%", m.name, head_pct, tail_pct);

    setenv("TURBO_LAYER_HEAD_PERCENT", std::to_string(head_pct).c_str(), 1);
    setenv("TURBO_LAYER_TAIL_PERCENT", std::to_string(tail_pct).c_str(), 1);
    unsetenv("TURBO_LAYER_HEAD");
    unsetenv("TURBO_LAYER_TAIL");

    llama_kv_cache_q8_widths w;
    try {
        w = llama_kv_cache_q8_brackets_resolve_widths(m.n_layer);
    } catch (const std::runtime_error & e) {
        printf("FAIL %s resolve_widths threw (%s)\n", label, e.what());
        ++g_fails;
        return;
    }

    // The resolved counts must match the percent->layers rule for this depth.
    const uint32_t want_head = llama_kv_cache_q8_brackets_percent_to_layers(m.n_layer, head_pct);
    const uint32_t want_tail = llama_kv_cache_q8_brackets_percent_to_layers(m.n_layer, tail_pct);
    if (w.head != want_head || w.tail != want_tail) {
        printf("FAIL %s resolved head=%u tail=%u, want %u/%u\n", label, w.head, w.tail,
               want_head, want_tail);
        ++g_fails;
        return;
    }

    const auto kv  = make_has_kv(m);
    const auto swa = make_swa(m);
    const std::string want = expected_plan(m, false, w.head, w.tail);

    std::vector<uint8_t> got;
    try {
        got = llama_kv_cache_q8_brackets_plan_layers(m.n_layer, kv, swa, false, w.head, w.tail);
    } catch (const std::runtime_error & e) {
        printf("FAIL %s plan_layers threw (%s), oracle expected %s\n", label, e.what(),
               want.c_str());
        ++g_fails;
        return;
    }
    expect_plan(label, got, want);

    // The chain must never select a layer without a cache.
    uint32_t dead = 0;
    for (uint32_t i = 0; i < m.n_layer; ++i) {
        if (got[i] && !kv[i]) {
            ++dead;
        }
    }
    char dead_label[176];
    snprintf(dead_label, sizeof(dead_label), "%s_no_dead_slots", label);
    expect_true(dead_label, dead == 0);
}

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

        // Chain env percent -> resolved widths -> planned indices (10.24).
        // The default pair must be a no-op relative to the code default, and a
        // different pair must move the indices, not just the count.
        check_env_chain(m, 6, 25);
        check_env_chain(m, 12, 10);
        check_env_chain(m, 0, 25);
        check_env_chain(m, 6, 0);

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

        // E2: the schedule must also be APPLIED for both turbo mids, not only
        // computed. turbo3.8 and turbo4.8 share one planner, so only this step
        // can tell them apart.
        check_mid_application(m, GGML_TYPE_TURBO3_0);
        check_mid_application(m, GGML_TYPE_TURBO4_0);

        // 10.13: q8 after dense layers, not on them.
        check_dense_neighbour(m);
    }

    // Purely dense models (Qwen3.5 family) fall back to an even spread.
    check_dense_only_model(kModels[0]);

    // Percentages must be shares of the real cache, not of n_layer.
    check_percent_denominator();

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

    // ---- DEGRADATION: the tail must land inside the KV range, every time ----
    // These are the cases that broke before 10.15. Each one asserts the invariant
    // "no selected layer lacks a cache" rather than a literal index string, so a
    // future refactor cannot pass by matching a stale expectation.
    {
        struct Degenerate {
            const char * name;
            uint32_t n_layer;
            uint32_t kv_prefix;   // layers [0, kv_prefix) own KV
            uint32_t head;
            uint32_t tail;
        };
        const Degenerate cases[] = {
            // Shared KV: only the first 15 of 35 own a cache (Gemma-4-E2B).
            { "deg_shared_kv_tail_slides",      35, 15, 2, 8 },
            // Tail wider than the KV prefix: must clamp to what exists, not abort
            // and not name dead layers. 10 KV layers, tail=20.
            { "deg_tail_wider_than_kv_range",   30, 10, 2, 20 },
            // Exactly one KV layer, head=1 tail=8: the single layer must be picked
            // once (head), and the tail cannot double-count it.
            { "deg_single_kv_layer",            20,  1, 1, 8 },
            // Minimum useful stack: 4 layers, all with KV.
            { "deg_min_stack",                   4,  4, 1, 1 },
        };
        for (const auto & c : cases) {
            std::vector<bool> kv(c.n_layer, false);
            for (uint32_t i = 0; i < c.kv_prefix && i < c.n_layer; ++i) {
                kv[i] = true;
            }
            const auto swa = all_true(c.n_layer, false);

            std::vector<uint8_t> got;
            bool threw = false;
            try {
                got = llama_kv_cache_q8_brackets_plan_layers(
                        c.n_layer, kv, swa, false, c.head, c.tail);
            } catch (const std::runtime_error &) {
                threw = true;
            }

            char label[128];
            if (threw) {
                // A refusal is acceptable only when nothing could be planned at all.
                snprintf(label, sizeof(label), "%s_refuses_or_plans", c.name);
                expect_true(label, c.kv_prefix == 0);
                continue;
            }

            // Invariant 1: every selected layer owns a cache.
            uint32_t dead = 0;
            uint32_t selected = 0;
            for (uint32_t i = 0; i < c.n_layer; ++i) {
                if (got[i]) {
                    ++selected;
                    if (!kv[i]) {
                        ++dead;
                    }
                }
            }
            snprintf(label, sizeof(label), "%s_no_dead_slots", c.name);
            expect_true(label, dead == 0);

            // Invariant 2: the plan is never empty and never over half.
            snprintf(label, sizeof(label), "%s_selected_nonzero", c.name);
            expect_true(label, selected > 0);
            snprintf(label, sizeof(label), "%s_within_half", c.name);
            expect_true(label, selected * 2 <= c.n_layer);

            // Invariant 3: the plan covers the head first. When tail and head
            // overlap on a short KV range they merge, so compare the count of
            // selected layers within the first `head` KV-bearing layers.
            uint32_t head_expect = 0;
            for (uint32_t i = 0; i < c.n_layer && head_expect < c.head; ++i) {
                if (kv[i]) {
                    ++head_expect;
                }
            }
            uint32_t head_got = 0;
            for (uint32_t i = 0; i < c.n_layer && head_got < head_expect; ++i) {
                if (kv[i]) {
                    if (!got[i]) {
                        break;
                    }
                    ++head_got;
                }
            }
            snprintf(label, sizeof(label), "%s_head_covers_first_kv_layers", c.name);
            expect_true(label, head_got == head_expect);
        }

        // The specific regression: with 15 of 35 KV layers the tail must end at 14,
        // never at 34. Asserted explicitly because this is the exact bug.
        {
            std::vector<bool> kv(35, false);
            for (uint32_t i = 0; i < 15; ++i) {
                kv[i] = true;
            }
            const auto swa = all_true(35, false);
            const auto got = llama_kv_cache_q8_brackets_plan_layers(35, kv, swa, false, 2, 8);
            expect_true("regression_tail_never_reaches_34", !got[34]);
            expect_true("regression_tail_ends_at_14", got[14]);
            expect_true("regression_keeps_head_0_1", got[0] && got[1]);
        }
    }

    // ---- percentage -> layer count (ST-10.20) -----------------------------
    // The default 6% head / 25% tail must reproduce the historical HEAD=2/TAIL=8
    // on a 30-layer model and scale with depth on everything else.
    {
        struct PctCase { uint32_t n, pct, want; const char * name; };
        const PctCase cases[] = {
            { 30,  6, 2, "pct_6_of_30_is_2" },
            { 30, 25, 8, "pct_25_of_30_is_8" },
            { 32,  6, 2, "pct_6_of_32_is_2" },
            { 35, 25, 9, "pct_25_of_35_is_9" },
            { 24, 25, 6, "pct_25_of_24_is_6" },
            { 60, 25, 15, "pct_25_of_60_is_15" },
            { 42,  6, 3, "pct_6_of_42_is_3" },
            { 30,  0, 0, "pct_0_is_zero" },
            // A non-zero share of a deep stack still selects at least one layer.
            { 100, 1, 1, "pct_min_one_layer" },
            // Rounding is nearest, not floor: 4% of 35 = 1.4 -> 1, 10% of 35 = 3.5 -> 4.
            { 35,  4, 1, "pct_rounds_down_below_half" },
            { 35, 10, 4, "pct_rounds_up_at_half" },
        };
        for (const auto & c : cases) {
            const uint32_t got = llama_kv_cache_q8_brackets_percent_to_layers(c.n, c.pct);
            if (got == c.want) {
                printf("ok   %s (%u%% of %u = %u)\n", c.name, c.pct, c.n, got);
            } else {
                printf("FAIL %s (%u%% of %u = %u, want %u)\n", c.name, c.pct, c.n, got, c.want);
                ++g_fails;
            }
        }

        // The share must stay roughly constant across depth, which is the point of
        // the knob: a fixed 2/8 would be 33% at 30 layers but 13% at 60.
        for (uint32_t n : { 24u, 30u, 32u, 35u, 42u, 60u }) {
            const uint32_t h = llama_kv_cache_q8_brackets_percent_to_layers(n, 6);
            const uint32_t t = llama_kv_cache_q8_brackets_percent_to_layers(n, 25);
            const double share = 100.0 * (h + t) / n;
            char label[96];
            snprintf(label, sizeof(label), "pct_share_near_31_percent_n%u", n);
            // 6+25 = 31%; allow a couple of points for rounding at small n.
            expect_true(label, share > 27.0 && share < 37.0);
        }
    }

    // ---- percent vs absolute knobs are mutually exclusive (ST-10.21) ------
    {
        auto clear_all = [] {
            unsetenv("TURBO_LAYER_HEAD");
            unsetenv("TURBO_LAYER_TAIL");
            unsetenv("TURBO_LAYER_HEAD_PERCENT");
            unsetenv("TURBO_LAYER_TAIL_PERCENT");
        };
        auto throws_now = [](const char * name) {
            bool threw = false;
            try {
                (void) llama_kv_cache_q8_brackets_resolve_widths(30);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            expect_true(name, threw);
        };

        clear_all();
        setenv("TURBO_LAYER_HEAD", "2", 1);
        setenv("TURBO_LAYER_HEAD_PERCENT", "6", 1);
        throws_now("conflict_head_abs_and_pct");

        clear_all();
        setenv("TURBO_LAYER_TAIL", "8", 1);
        setenv("TURBO_LAYER_TAIL_PERCENT", "25", 1);
        throws_now("conflict_tail_abs_and_pct");

        // Percent alone resolves to the historical 2/8 on 30 layers.
        clear_all();
        {
            const auto w = llama_kv_cache_q8_brackets_resolve_widths(30);
            expect_true("defaults_head_is_2_of_30", w.head == 2);
            expect_true("defaults_tail_is_8_of_30", w.tail == 8);
            expect_true("defaults_pct_is_6_25", w.head_pct == 6 && w.tail_pct == 25);
        }

        // Absolute alone still works (deprecated path) and reports a percentage.
        clear_all();
        setenv("TURBO_LAYER_HEAD", "2", 1);
        setenv("TURBO_LAYER_TAIL", "8", 1);
        {
            const auto w = llama_kv_cache_q8_brackets_resolve_widths(30);
            expect_true("abs_head_still_works", w.head == 2 && w.tail == 8);
            expect_true("abs_reports_pct", w.head_pct == 6 && w.tail_pct == 26);
        }

        // Out-of-range percentages abort.
        clear_all();
        setenv("TURBO_LAYER_HEAD_PERCENT", "101", 1);
        throws_now("percent_above_100_throws");

        clear_all();
        setenv("TURBO_LAYER_TAIL_PERCENT", "abc", 1);
        throws_now("percent_not_a_number_throws");

        clear_all();
    }

    if (g_fails == 0) {
        printf("all ok\n");
        return 0;
    }    printf("%d FAILED\n", g_fails);
    return 1;
}
