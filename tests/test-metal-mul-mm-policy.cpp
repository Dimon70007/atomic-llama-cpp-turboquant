// TASK-066.9 Phase A2 — CPU unit for Metal mul_mm vs mul_mv dispatch policy.
//
// Locks the rem cross-kernel class without GPU / model:
//   jinja ladder prompt n=1708, ub=8 → rem=4
//   under MUL_MM_MIN=7: full chunks on mm, rem on mv → hazard
//   under product default (1): both on mm → no hazard
//
// Bad merge that bumps GGML_METAL_MUL_MM_MIN_DEFAULT back to 7 fails here immediately.
// Runner: SUITE=mul-mat-residual (policy gate first) or build target alone.
// No Metal required.

#include "../ggml/src/ggml-metal/ggml-metal-mul-mm-policy.h"

#include <cstdio>

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

void expect_eq_i32(const char * name, int32_t got, int32_t want) {
    if (got == want) {
        printf("ok   %s (got=%d)\n", name, got);
        return;
    }
    printf("FAIL %s (got=%d want=%d)\n", name, got, want);
    ++g_fails;
}

void expect_eq_int(const char * name, int got, int want) {
    if (got == want) {
        printf("ok   %s (got=%d)\n", name, got);
        return;
    }
    printf("FAIL %s (got=%d want=%d)\n", name, got, want);
    ++g_fails;
}

}  // namespace

int main() {
    printf("TASK-066 mul_mm policy unit (CPU; no Metal)\n");
    printf("  GGML_METAL_MUL_MM_MIN_DEFAULT=%d\n", GGML_METAL_MUL_MM_MIN_DEFAULT);

    // Product pin after A2 — do not silently restore 7.
    expect_eq_int("product_default_min_is_1", GGML_METAL_MUL_MM_MIN_DEFAULT, 1);

    // Dispatch rule: ne11 > mm_min → mul_mm (matches ggml-metal-ops.cpp).
    expect_true("n1_min1_on_mv", !ggml_metal_mul_mm_ne11_uses_mm(1, 1));
    expect_true("n2_min1_on_mm", ggml_metal_mul_mm_ne11_uses_mm(2, 1));
    expect_true("n8_min7_on_mm", ggml_metal_mul_mm_ne11_uses_mm(8, 7));
    expect_true("n4_min7_on_mv", !ggml_metal_mul_mm_ne11_uses_mm(4, 7));
    expect_true("n4_min1_on_mm", ggml_metal_mul_mm_ne11_uses_mm(4, 1));
    expect_true("n8_min8_on_mv", !ggml_metal_mul_mm_ne11_uses_mm(8, 8));

    // Last-chunk sizes (prefill split).
    expect_eq_i32("last_1708_ub8_rem4", ggml_metal_ubatch_last_chunk_ne11(1708, 8), 4);
    expect_eq_i32("last_1696_ub8_full", ggml_metal_ubatch_last_chunk_ne11(1696, 8), 8);
    expect_eq_i32("last_1708_ub16_rem12", ggml_metal_ubatch_last_chunk_ne11(1708, 16), 12);
    expect_eq_i32("last_100_ub8_rem4", ggml_metal_ubatch_last_chunk_ne11(100, 8), 4);
    expect_eq_i32("last_single_chunk", ggml_metal_ubatch_last_chunk_ne11(5, 8), 5);

    // A2 class: jinja 1708 + ub8 + legacy min7 → cross-kernel hazard.
    expect_true("hazard_jinja_ub8_min7",
                ggml_metal_ubatch_rem_cross_kernel_hazard(1708, 8, 7));
    expect_true("no_hazard_jinja_ub8_min1",
                !ggml_metal_ubatch_rem_cross_kernel_hazard(1708, 8, 1));
    expect_true("no_hazard_raw_c1_ub8_min7",
                !ggml_metal_ubatch_rem_cross_kernel_hazard(1696, 8, 7));
    expect_true("no_hazard_jinja_ub16_min7",
                !ggml_metal_ubatch_rem_cross_kernel_hazard(1708, 16, 7));
    expect_true("no_hazard_exact_multiple",
                !ggml_metal_ubatch_rem_cross_kernel_hazard(1024, 8, 7));

    // Lock product default against reintroducing the A2 hole on the ladder prompt.
    expect_true("product_default_safe_for_jinja_1708_ub8",
                !ggml_metal_ubatch_rem_cross_kernel_hazard(
                        1708, 8, GGML_METAL_MUL_MM_MIN_DEFAULT));

    if (g_fails != 0) {
        printf("RESULT FAIL (%d assertion(s)) — mul_mm policy regress\n", g_fails);
        return 1;
    }
    printf("RESULT PASS — mul_mm policy unit GREEN\n");
    return 0;
}
