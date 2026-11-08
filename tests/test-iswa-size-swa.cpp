// ISWA size_swa unit (CPU; no model).
//
// Locks TASK-066.5 KEEP:
//   - n_ubatch must NOT affect size_swa (batch-invariance / C3-n1344)
//   - headroom = size_base => size_swa == size_base when ctx >= n_swa
//     (oneshot ubatch up to n_ctx fits; pure windowed PAD(n_swa) RED on C3)
//   - swa_full still forces size_base

#include <cstdio>
#include <cstdint>

// Declared in src/llama-kv-cache-iswa.h; keep this unit free of the heavy KV header.
uint32_t llama_kv_cache_iswa_calc_size_swa(
        uint32_t size_base,
        uint32_t n_swa,
        uint32_t n_seq_max,
        bool     unified,
        bool     swa_full,
        uint32_t n_ubatch);

namespace {

int g_fails = 0;

void expect_u32(const char * name, uint32_t got, uint32_t want) {
    if (got == want) {
        printf("ok   %s (%u)\n", name, got);
        return;
    }
    printf("FAIL %s: got %u want %u\n", name, got, want);
    ++g_fails;
}

} // namespace

int main() {
    printf("ISWA size_swa unit (CPU; no model)\n");

    // Gemma4-shaped: n_swa=1024, validation-class ctx, unified single seq.
    const uint32_t n_swa = 1024;
    const uint32_t ctx   = 3072; // ladder / C3 class (~2k prompt + headroom)

    // size_base headroom => full cells; identical for every -ub.
    expect_u32(
            "gemma4_ub8_equals_size_base",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 1, true, false, /*n_ubatch*/ 8),
            ctx);
    expect_u32(
            "gemma4_ub1024_same",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1024),
            ctx);
    expect_u32(
            "gemma4_ub2048_same",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 2048),
            ctx);

    // Long product ctx: same KEEP, same side effect (size_swa == n_ctx).
    {
        const uint32_t long_ctx = 160000;
        expect_u32(
                "long_ctx_equals_size_base",
                llama_kv_cache_iswa_calc_size_swa(long_ctx, n_swa, 1, true, false, 512),
                long_ctx);
    }

    expect_u32(
            "swa_full_uses_size_base",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, /*swa_full*/ true, 8),
            ctx);

    // size_base smaller than window -> clamp to size_base (then pad).
    expect_u32(
            "clamp_to_size_base",
            llama_kv_cache_iswa_calc_size_swa(/*size_base*/ 512, n_swa, 1, true, false, 8),
            512);

    // unified multi-seq: swa_window grows but + size_base still clamps to size_base.
    expect_u32(
            "unified_n_seq_max_2_still_size_base",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 2, true, false, 8),
            ctx);

    // non-unified: ignore n_seq_max for window width; still size_base headroom.
    expect_u32(
            "non_unified_still_size_base",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 4, false, false, 8),
            ctx);

    if (g_fails != 0) {
        printf("RESULT FAIL fails=%d\n", g_fails);
        return 1;
    }
    printf("RESULT PASS\n");
    return 0;
}
