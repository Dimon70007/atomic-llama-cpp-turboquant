// TURBO_LAYER_ADAPTIVE schedule unit (CPU; no model).
//
// Locks the mode legend in llama-kv-cache.cpp without GGUF / GPU:
//   mode select: env atoi / auto-7 for turbo2-V / opt-out=0
//   mode 1: q8 K+V first+last 4 (V gated on type_k being turbo*)
//   mode 2: q8 K+V last 8
//   mode 5: Boundary V turbo4 edges / turbo2 mid (K unchanged)
//   mode 6: last8 V=turbo4, rest V=turbo2
//   mode 7: first2+last2 V=q8, rest V=turbo2
//   modes 3/4: no-op (passthrough) — documented gap
//   traps: n_layer<8; non-turbo V under 5/6/7; mode1 with non-turbo K
//
// Env: TURBO_LAYER_ADAPTIVE (cleared between cases).

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int llama_kv_cache_turbo_layer_adaptive_mode(ggml_type type_v, uint32_t n_layer);
ggml_type llama_kv_cache_turbo_layer_adaptive_type_k(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);
ggml_type llama_kv_cache_turbo_layer_adaptive_type_v(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);

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

void expect_int(const char * name, int got, int want) {
    if (got == want) {
        printf("ok   %s (got=%d)\n", name, got);
        return;
    }
    printf("FAIL %s (got=%d want=%d)\n", name, got, want);
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

void clear_adaptive_env() {
    env_unset("TURBO_LAYER_ADAPTIVE");
}

}  // namespace

int main() {
    printf("TURBO_LAYER_ADAPTIVE schedule unit (CPU; no model)\n");

    const uint32_t n30 = 30;
    const ggml_type t2 = GGML_TYPE_TURBO2_0;
    const ggml_type t3 = GGML_TYPE_TURBO3_0;
    const ggml_type t4 = GGML_TYPE_TURBO4_0;
    const ggml_type q8 = GGML_TYPE_Q8_0;
    const ggml_type f16 = GGML_TYPE_F16;

    // --- mode selection ---
    clear_adaptive_env();
    expect_int("mode_default_turbo3_is_0",
               llama_kv_cache_turbo_layer_adaptive_mode(t3, n30), 0);
    expect_int("mode_auto_turbo2_is_7",
               llama_kv_cache_turbo_layer_adaptive_mode(t2, n30), 7);
    expect_int("mode_auto_turbo2_short_layers_is_0",
               llama_kv_cache_turbo_layer_adaptive_mode(t2, 7), 0);

    env_set("TURBO_LAYER_ADAPTIVE", "0");
    expect_int("mode_env_0_opts_out_of_auto7",
               llama_kv_cache_turbo_layer_adaptive_mode(t2, n30), 0);

    env_set("TURBO_LAYER_ADAPTIVE", "5");
    expect_int("mode_env_5", llama_kv_cache_turbo_layer_adaptive_mode(t3, n30), 5);

    env_set("TURBO_LAYER_ADAPTIVE", "1");
    expect_int("mode_env_1", llama_kv_cache_turbo_layer_adaptive_mode(t4, n30), 1);
    clear_adaptive_env();

    // --- mode 0: uniform passthrough ---
    expect_type("m0_k_passthrough",
                llama_kv_cache_turbo_layer_adaptive_type_k(0, t3, t3, 0, n30), t3);
    expect_type("m0_v_passthrough",
                llama_kv_cache_turbo_layer_adaptive_type_v(0, t3, t3, 0, n30), t3);

    // --- mode 1: q8 K+V first+last 4 ---
    // layers 0..3 and 26..29 q8; mid 4..25 stay mid
    expect_type("m1_k_head0_q8",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, t3, t3, 0, n30), q8);
    expect_type("m1_k_head3_q8",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, t3, t3, 3, n30), q8);
    expect_type("m1_k_mid4_turbo3",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, t3, t3, 4, n30), t3);
    expect_type("m1_k_mid25_turbo3",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, t3, t3, 25, n30), t3);
    expect_type("m1_k_tail26_q8",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, t3, t3, 26, n30), q8);
    expect_type("m1_k_tail29_q8",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, t3, t3, 29, n30), q8);

    expect_type("m1_v_head0_q8",
                llama_kv_cache_turbo_layer_adaptive_type_v(1, t3, t3, 0, n30), q8);
    expect_type("m1_v_mid4_turbo3",
                llama_kv_cache_turbo_layer_adaptive_type_v(1, t3, t3, 4, n30), t3);
    expect_type("m1_v_tail29_q8",
                llama_kv_cache_turbo_layer_adaptive_type_v(1, t3, t3, 29, n30), q8);

    // TRAP: mode 1 V override is gated on type_k being turbo*, not type_v.
    // Product-like K=q8_0 V=turbo3 under ADAPTIVE=1 must NOT rewrite V.
    expect_type("m1_trap_v_gated_on_type_k",
                llama_kv_cache_turbo_layer_adaptive_type_v(1, q8, t3, 0, n30), t3);
    expect_type("m1_trap_k_non_turbo_passthrough",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, q8, t3, 0, n30), q8);

    // --- mode 2: q8 K+V last 8 ---
    expect_type("m2_k_head0_mid",
                llama_kv_cache_turbo_layer_adaptive_type_k(2, t3, t3, 0, n30), t3);
    expect_type("m2_k_il21_mid",
                llama_kv_cache_turbo_layer_adaptive_type_k(2, t3, t3, 21, n30), t3);
    expect_type("m2_k_tail22_q8",
                llama_kv_cache_turbo_layer_adaptive_type_k(2, t3, t3, 22, n30), q8);
    expect_type("m2_k_tail29_q8",
                llama_kv_cache_turbo_layer_adaptive_type_k(2, t3, t3, 29, n30), q8);
    expect_type("m2_v_tail22_q8",
                llama_kv_cache_turbo_layer_adaptive_type_v(2, t3, t3, 22, n30), q8);
    expect_type("m2_v_il21_mid",
                llama_kv_cache_turbo_layer_adaptive_type_v(2, t3, t3, 21, n30), t3);

    // --- modes 3/4: no schedule wired (passthrough) ---
    expect_type("m3_v_noop",
                llama_kv_cache_turbo_layer_adaptive_type_v(3, t3, t3, 0, n30), t3);
    expect_type("m4_v_noop",
                llama_kv_cache_turbo_layer_adaptive_type_v(4, t3, t3, 29, n30), t3);
    expect_type("m3_k_noop",
                llama_kv_cache_turbo_layer_adaptive_type_k(3, t3, t3, 0, n30), t3);

    // --- mode 5: Boundary V turbo4 edges / turbo2 mid; K unchanged ---
    expect_type("m5_k_unchanged",
                llama_kv_cache_turbo_layer_adaptive_type_k(5, t3, t3, 0, n30), t3);
    expect_type("m5_v_head0_turbo4",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, t3, t3, 0, n30), t4);
    expect_type("m5_v_head1_turbo4",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, t3, t3, 1, n30), t4);
    expect_type("m5_v_mid2_turbo2",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, t3, t3, 2, n30), t2);
    expect_type("m5_v_mid27_turbo2",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, t3, t3, 27, n30), t2);
    expect_type("m5_v_tail28_turbo4",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, t3, t3, 28, n30), t4);
    expect_type("m5_v_tail29_turbo4",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, t3, t3, 29, n30), t4);
    // TRAP: mid is forced to turbo2 even when operator asked for turbo3/turbo4 V.
    expect_type("m5_trap_mid_forces_turbo2_from_turbo4_v",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, q8, t4, 10, n30), t2);

    // --- mode 6: last8 V=turbo4, rest V=turbo2 ---
    expect_type("m6_v_il21_turbo2",
                llama_kv_cache_turbo_layer_adaptive_type_v(6, t3, t3, 21, n30), t2);
    expect_type("m6_v_tail22_turbo4",
                llama_kv_cache_turbo_layer_adaptive_type_v(6, t3, t3, 22, n30), t4);
    expect_type("m6_v_tail29_turbo4",
                llama_kv_cache_turbo_layer_adaptive_type_v(6, t3, t3, 29, n30), t4);
    expect_type("m6_k_unchanged",
                llama_kv_cache_turbo_layer_adaptive_type_k(6, t3, t3, 29, n30), t3);

    // --- mode 7: first2+last2 V=q8, rest V=turbo2 ---
    expect_type("m7_v_head0_q8",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, t2, 0, n30), q8);
    expect_type("m7_v_head1_q8",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, t2, 1, n30), q8);
    expect_type("m7_v_mid2_turbo2",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, t2, 2, n30), t2);
    expect_type("m7_v_mid27_turbo2",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, t2, 27, n30), t2);
    expect_type("m7_v_tail28_q8",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, t2, 28, n30), q8);
    expect_type("m7_v_tail29_q8",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, t2, 29, n30), q8);
    // TRAP: ADAPTIVE=7 with -ctv turbo3 still forces mid to turbo2 (not turbo3).
    expect_type("m7_trap_mid_forces_turbo2_from_turbo3_v",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, q8, t3, 15, n30), t2);

    // --- short models / non-turbo V ---
    expect_type("short_n_layer_passthrough_v",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, t3, 0, 7), t3);
    expect_type("short_n_layer_passthrough_k",
                llama_kv_cache_turbo_layer_adaptive_type_k(1, t3, t3, 0, 7), t3);
    expect_type("m7_non_turbo_v_passthrough",
                llama_kv_cache_turbo_layer_adaptive_type_v(7, t3, f16, 0, n30), f16);
    expect_type("m5_non_turbo_v_passthrough",
                llama_kv_cache_turbo_layer_adaptive_type_v(5, t3, q8, 0, n30), q8);

    clear_adaptive_env();

    if (g_fails != 0) {
        printf("%d FAIL(s)\n", g_fails);
        return 1;
    }
    printf("all ok\n");
    return 0;
}
