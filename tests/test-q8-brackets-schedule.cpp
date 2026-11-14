// TASK-068 ST-6.1 - CPU unit for q8 head/tail V brackets (turbo3.8 / turbo4.8).
//
// Locks schedule wiring without GGUF / GPU:
//   defaults HEAD=2 TAIL=8 on n_layer=30 (+ HEAD=1 / HEAD=0 / min 1+1)
//   ADAPTIVE + brackets -> throw
//   n_layer==0 / width overflow / bad env ints -> throw
//   Q8_BRACKETS=0 / unset -> inactive passthrough
//
// Env: TURBO_LAYER_Q8_BRACKETS / HEAD / TAIL / ADAPTIVE (cleared between cases).

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// Declared in src/llama-kv-cache.h; keep this unit free of the heavy KV header.
bool llama_kv_cache_q8_brackets_active(void);
ggml_type llama_kv_cache_q8_brackets_type_v(ggml_type type_v, uint32_t il, uint32_t n_layer);

#if defined(_WIN32)
#    include <stdlib.h>
static void env_set(const char * name, const char * value) {
    _putenv_s(name, value ? value : "");
}
static void env_unset(const char * name) {
    _putenv_s(name, "");
}
#else
static void env_set(const char * name, const char * value) {
    if (value == nullptr || value[0] == '\0') {
        unsetenv(name);
        return;
    }
    setenv(name, value, 1);
}
static void env_unset(const char * name) {
    unsetenv(name);
}
#endif

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

void expect_type(const char * name, ggml_type got, ggml_type want) {
    if (got == want) {
        printf("ok   %s (got=%s)\n", name, ggml_type_name(got));
        return;
    }
    printf("FAIL %s (got=%s want=%s)\n", name, ggml_type_name(got), ggml_type_name(want));
    ++g_fails;
}

void expect_throws(const char * name, ggml_type mid, uint32_t il, uint32_t n_layer) {
    bool threw = false;
    try {
        (void) llama_kv_cache_q8_brackets_type_v(mid, il, n_layer);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect_true(name, threw);
}

void clear_bracket_env() {
    env_unset("TURBO_LAYER_Q8_BRACKETS");
    env_unset("TURBO_LAYER_HEAD");
    env_unset("TURBO_LAYER_TAIL");
    env_unset("TURBO_LAYER_ADAPTIVE");
}

void arm_brackets_defaults() {
    env_set("TURBO_LAYER_Q8_BRACKETS", "1");
    env_unset("TURBO_LAYER_HEAD");
    env_unset("TURBO_LAYER_TAIL");
    env_unset("TURBO_LAYER_ADAPTIVE");
}

}  // namespace

int main() {
    printf("TASK-068 q8 brackets schedule unit (CPU; no model)\n");

    clear_bracket_env();
    expect_true("inactive_when_env_unset", !llama_kv_cache_q8_brackets_active());
    expect_type(
        "inactive_passthrough_turbo3",
        llama_kv_cache_q8_brackets_type_v(GGML_TYPE_TURBO3_0, 0, 30),
        GGML_TYPE_TURBO3_0);

    // Explicit off
    env_set("TURBO_LAYER_Q8_BRACKETS", "0");
    expect_true("inactive_when_env_zero", !llama_kv_cache_q8_brackets_active());
    expect_type(
        "inactive_zero_passthrough",
        llama_kv_cache_q8_brackets_type_v(GGML_TYPE_TURBO4_0, 5, 30),
        GGML_TYPE_TURBO4_0);

    arm_brackets_defaults();
    expect_true("active_when_env_set", llama_kv_cache_q8_brackets_active());

    // n_layer=30, HEAD=2 TAIL=8 -> front 0..1 q8, mid turbo3, tail 22..29 q8
    const uint32_t n_layer = 30;
    const ggml_type mid3 = GGML_TYPE_TURBO3_0;
    const ggml_type mid4 = GGML_TYPE_TURBO4_0;
    expect_type("head0_q8", llama_kv_cache_q8_brackets_type_v(mid3, 0, n_layer), GGML_TYPE_Q8_0);
    expect_type("head1_q8", llama_kv_cache_q8_brackets_type_v(mid3, 1, n_layer), GGML_TYPE_Q8_0);
    expect_type("mid2_turbo3", llama_kv_cache_q8_brackets_type_v(mid3, 2, n_layer), mid3);
    expect_type("mid21_turbo3", llama_kv_cache_q8_brackets_type_v(mid3, 21, n_layer), mid3);
    expect_type("tail22_q8", llama_kv_cache_q8_brackets_type_v(mid3, 22, n_layer), GGML_TYPE_Q8_0);
    expect_type("tail29_q8", llama_kv_cache_q8_brackets_type_v(mid3, 29, n_layer), GGML_TYPE_Q8_0);

    // Same schedule with mid turbo4 (turbo4.8 arm)
    expect_type("mid2_turbo4", llama_kv_cache_q8_brackets_type_v(mid4, 2, n_layer), mid4);
    expect_type("head0_q8_with_turbo4_mid", llama_kv_cache_q8_brackets_type_v(mid4, 0, n_layer), GGML_TYPE_Q8_0);

    // HEAD=1 TAIL=8 (R&D A/B): front only layer 0, mid 1..21, tail 22..29
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "1");
        env_set("TURBO_LAYER_TAIL", "8");
        expect_type("head1_only_il0_q8", llama_kv_cache_q8_brackets_type_v(mid3, 0, n_layer), GGML_TYPE_Q8_0);
        expect_type("head1_il1_mid", llama_kv_cache_q8_brackets_type_v(mid3, 1, n_layer), mid3);
        expect_type("head1_tail22_q8", llama_kv_cache_q8_brackets_type_v(mid3, 22, n_layer), GGML_TYPE_Q8_0);
    }

    // HEAD=0 TAIL=8: no front q8; first mid at 0; tail still 22..29
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "0");
        env_set("TURBO_LAYER_TAIL", "8");
        expect_type("head0_il0_mid", llama_kv_cache_q8_brackets_type_v(mid3, 0, n_layer), mid3);
        expect_type("head0_il21_mid", llama_kv_cache_q8_brackets_type_v(mid3, 21, n_layer), mid3);
        expect_type("head0_tail22_q8", llama_kv_cache_q8_brackets_type_v(mid3, 22, n_layer), GGML_TYPE_Q8_0);
    }

    // Minimal fit under the half-layer policy: n_layer=4, HEAD=1, TAIL=1 ->
    // q8 | mid | mid | q8 = 2 of 4 layers (exactly half, allowed).
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "1");
        env_set("TURBO_LAYER_TAIL", "1");
        expect_type("min_il0_q8", llama_kv_cache_q8_brackets_type_v(mid3, 0, 4), GGML_TYPE_Q8_0);
        expect_type("min_il1_mid", llama_kv_cache_q8_brackets_type_v(mid3, 1, 4), mid3);
        expect_type("min_il2_mid", llama_kv_cache_q8_brackets_type_v(mid3, 2, 4), mid3);
        expect_type("min_il3_q8", llama_kv_cache_q8_brackets_type_v(mid3, 3, 4), GGML_TYPE_Q8_0);
    }
    {
        // Same widths on a 3-layer model are 2 of 3 (over half) and must abort.
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "1");
        env_set("TURBO_LAYER_TAIL", "1");
        expect_throws("min_over_half_throws", mid3, 0, 3);
    }

    // Fail-closed corners
    {
        arm_brackets_defaults();
        expect_throws("n_layer0_throws", mid3, 0, 0);
    }
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_ADAPTIVE", "7");
        expect_throws("adaptive_plus_brackets_throws", mid3, 0, n_layer);
        env_unset("TURBO_LAYER_ADAPTIVE");
    }
    {
        // 0 and empty are the documented "disabled" values, not a conflict.
        arm_brackets_defaults();
        env_set("TURBO_LAYER_ADAPTIVE", "0");
        expect_type("adaptive0_ok_head2", llama_kv_cache_q8_brackets_type_v(mid3, 2, n_layer), mid3);
        env_set("TURBO_LAYER_ADAPTIVE", "");
        expect_type("adaptive_empty_ok_head2", llama_kv_cache_q8_brackets_type_v(mid3, 2, n_layer), mid3);
    }
    {
        // Hard policy: q8_0 covers at most half the layers (n_layer=30 here,
        // so 15 max). 15 is allowed, 16 must abort.
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "7");
        env_set("TURBO_LAYER_TAIL", "8");
        expect_type("q8_exactly_half_ok", llama_kv_cache_q8_brackets_type_v(mid3, 0, n_layer), GGML_TYPE_Q8_0);
        env_set("TURBO_LAYER_HEAD", "8");
        env_set("TURBO_LAYER_TAIL", "8");
        expect_throws("q8_over_half_throws", mid3, 0, n_layer);
    }
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "30");
        env_set("TURBO_LAYER_TAIL", "8");
        expect_throws("head_ge_n_layer_throws", mid3, 0, n_layer);
    }
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "2");
        env_set("TURBO_LAYER_TAIL", "30");
        expect_throws("tail_ge_n_layer_throws", mid3, 0, n_layer);
    }
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "15");
        env_set("TURBO_LAYER_TAIL", "15");
        expect_throws("head_plus_tail_no_mid_throws", mid3, 0, n_layer);
    }
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "1");
        env_set("TURBO_LAYER_TAIL", "1");
        expect_throws("n_layer2_too_small_throws", mid3, 0, 2);
    }
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "abc");
        expect_throws("head_non_integer_throws", mid3, 0, n_layer);
    }
    {
        arm_brackets_defaults();
        env_set("TURBO_LAYER_HEAD", "2");
        env_set("TURBO_LAYER_TAIL", "-1");
        expect_throws("tail_negative_throws", mid3, 0, n_layer);
    }

    clear_bracket_env();

    if (g_fails != 0) {
        printf("%d FAIL(s)\n", g_fails);
        return 1;
    }
    printf("all ok\n");
    return 0;
}
