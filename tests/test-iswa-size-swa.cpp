// ISWA size_swa unit (CPU; no model).
//
// Locks TASK-066.5 ideas + TASK-066.13 C2:
//   n_swa_eff = ceil(n_swa / 128) * 128
//   size_swa  = PAD(min(size_base, n_swa_eff * (unified ? n_seq_max : 1) + n_swa_headroom), 256)
//   n_swa_headroom = n_batch (NOT live -ub)
//   swa_full still forces size_base

#include <cstdio>
#include <cstdint>

// Declared in src/llama-kv-cache-iswa.h; keep this unit free of the heavy KV header.
uint32_t llama_kv_cache_iswa_calc_size_swa(
        uint32_t size_base,
        uint32_t n_swa,
        uint32_t n_seq_max,
        bool     unified,
        bool     swa_full,
        uint32_t n_swa_headroom);

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
    printf("ISWA size_swa unit (CPU; no model) — ideas formula\n");

    // Gemma4-shaped: n_swa=1024 (already multiple of 128).
    const uint32_t n_swa = 1024;
    const uint32_t ctx   = 3072;

    // headroom=8  -> PAD(1024+8, 256) = 1280
    expect_u32(
            "gemma4_headroom8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 1, true, false, /*headroom*/ 8),
            1280);
    // headroom=512 -> PAD(1536, 256) = 1536
    expect_u32(
            "gemma4_headroom512",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 512),
            1536);
    // headroom=1344 -> PAD(2368, 256) = 2560
    expect_u32(
            "gemma4_headroom1344",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1344),
            2560);
    // product-ish -b=2048 headroom (066.13 C2)
    expect_u32(
            "gemma4_headroom2048",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 2048),
            3072);

    // Long product ctx: windowed, not full size_base.
    {
        const uint32_t long_ctx = 160000;
        expect_u32(
                "long_ctx_windowed_ub512",
                llama_kv_cache_iswa_calc_size_swa(long_ctx, n_swa, 1, true, false, 512),
                1536);
    }

    expect_u32(
            "swa_full_uses_size_base",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, /*swa_full*/ true, 8),
            ctx);

    // size_base smaller than window+ub -> clamp to size_base.
    expect_u32(
            "clamp_to_size_base",
            llama_kv_cache_iswa_calc_size_swa(/*size_base*/ 512, n_swa, 1, true, false, 8),
            512);

    // pad n_swa=1000 -> 1024; ub=8 -> PAD(1032)=1280
    expect_u32(
            "pad128_n_swa_1000_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, /*n_swa*/ 1000, 1, true, false, 8),
            1280);

    // unified multi-seq: window = n_swa_eff * n_seq_max
    // 1024*2+8=2056 -> PAD=2304
    expect_u32(
            "unified_n_seq_max_2_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 2, true, false, 8),
            2304);

    // non-unified: ignore n_seq_max for window width
    expect_u32(
            "non_unified_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 4, false, false, 8),
            1280);

    // TASK-066.12 P1: uint32 overflow in n_swa_eff * n_seq_max must not wrap.
    // 2^31 * 2 overflows uint32; fixed path clamps to size_base.
    {
        const uint32_t n_swa_huge = 1u << 31; // already % 128 == 0
        const uint32_t size_base  = 100000;
        expect_u32(
                "overflow_unified_n_seq_max_2_clamps_to_size_base",
                llama_kv_cache_iswa_calc_size_swa(size_base, n_swa_huge, /*n_seq_max*/ 2, true, false, 8),
                size_base);
    }

    if (g_fails != 0) {
        printf("RESULT FAIL fails=%d\n", g_fails);
        return 1;
    }
    printf("RESULT PASS\n");
    return 0;
}
