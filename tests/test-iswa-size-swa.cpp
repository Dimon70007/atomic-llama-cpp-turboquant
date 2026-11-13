// ISWA size_swa unit (CPU; no model).
//
// Locks TASK-066.5 / 071:
//   n_swa_eff = ceil(n_swa / n_swa_align) * n_swa_align
//   headroom  = n_ubatch
//   size_swa  = min(PAD512(min(size_base, n_swa_eff * (unified ? n_seq_max : 1) + headroom)), size_base)
//   swa_full still forces size_base
//   overflow on the window sum clamps to size_base (upstream parity)
//
// calc_size_swa is a pure function: no env, no globals. If this test ever starts
// depending on the caller's shell, the purity has been lost again.

#include <cstdint>
#include <cstdio>
#include <exception>

#include "llama-kv-cache-iswa.h"

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

// KV block sizes are 128 for the turbo types and 32 for q8_0.
constexpr uint32_t k_turbo_align = 128;
constexpr uint32_t k_q8_0_align  = 32;

} // namespace

int main() {
    printf("ISWA size_swa unit (CPU; no model) - headroom n_ubatch + pad512 + clamp\n");

    // --- headroom helper ---
    // headroom is exactly one ubatch: the room a single split needs to write.
    expect_u32("headroom_ub512", llama_kv_cache_iswa_calc_swa_headroom(512), 512);
    expect_u32("headroom_ub256", llama_kv_cache_iswa_calc_swa_headroom(256), 256);
    expect_u32("headroom_ub8",   llama_kv_cache_iswa_calc_swa_headroom(8),   8);

    // --- alignment selection ---
    // turbo/vs turbo -> 128; q8_0 on either side cannot lower it; f16 -> 1... but
    // ggml_blck_size(f16) == 1, so f16/f16 needs no rotation-driven padding.
    expect_u32("align_turbo_turbo",
            llama_kv_cache_iswa_kv_block_size(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0), 128);
    expect_u32("align_turbo_k_q8_0_v",
            llama_kv_cache_iswa_kv_block_size(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0), 128);
    expect_u32("align_q8_0_q8_0",
            llama_kv_cache_iswa_kv_block_size(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0), 32);
    expect_u32("align_f16_f16",
            llama_kv_cache_iswa_kv_block_size(GGML_TYPE_F16, GGML_TYPE_F16), 1);

    // Gemma4-shaped: n_swa=1024 (already a multiple of 128), ctx=3072.
    const uint32_t n_swa = 1024;
    const uint32_t ctx   = 3072;
    const uint32_t a     = k_turbo_align;

    // headroom=8 -> 1024 + 8 = 1032 -> PAD512 = 1536
    expect_u32("gemma4_headroom8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 8, a), 1536);
    // headroom=384 -> 1408 -> PAD512 = 1536 (same band as 8..511)
    expect_u32("gemma4_headroom384",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 384, a), 1536);
    expect_u32("gemma4_headroom512",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 512, a), 1536);
    // headroom=1024 -> 2048 -> next band
    expect_u32("gemma4_headroom1024",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1024, a), 2048);
    expect_u32("gemma4_headroom1344",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1344, a), 2560);
    expect_u32("gemma4_headroom2048",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 2048, a), 3072);

    // Banded invariance: two ubatches in the same band give the same ring, two
    // across a band boundary do not. This is what -ub parity actually buys.
    {
        const uint32_t band_lo = llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 256, a);
        const uint32_t band_hi = llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 512, a);
        const uint32_t band_nx = llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1024, a);
        expect_u32("band_256_eq_512", band_lo, band_hi);
        if (band_hi == band_nx) {
            printf("FAIL band_512_ne_1024: both %u, expected a band change\n", band_hi);
            ++g_fails;
        } else {
            printf("ok   band_512_ne_1024 (%u vs %u)\n", band_hi, band_nx);
        }
    }

    {
        const uint32_t long_ctx = 160000;
        expect_u32("long_ctx_windowed_ub512",
                llama_kv_cache_iswa_calc_size_swa(long_ctx, n_swa, 1, true, false, 512, a), 1536);
    }

    expect_u32("swa_full_uses_size_base",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, /*swa_full*/ true, 8, a), ctx);

    expect_u32("clamp_to_size_base",
            llama_kv_cache_iswa_calc_size_swa(/*size_base*/ 512, n_swa, 1, true, false, 8, a), 512);

    // n_swa_eff pad: 1000 -> 1024; +8 = 1032 -> PAD512 = 1536
    expect_u32("pad_n_swa_1000_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, /*n_swa*/ 1000, 1, true, false, 8, a), 1536);

    // unified n_seq_max=2: 1024*2 + 8 = 2056 -> PAD512 = 2560
    expect_u32("unified_n_seq_max_2_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 2, true, false, 8, a), 2560);

    // non-unified ignores n_seq_max: 1024 + 8 -> PAD512 = 1536
    expect_u32("non_unified_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 4, false, false, 8, a), 1536);

    // q8_0 KV: block 32, no rotation, so n_swa=1000 stays 1000 before the pad.
    // 1000 + 8 = 1008 -> PAD512 = 1536 either way, but n_swa=1024+8 must not
    // depend on the alignment to land in the right band.
    expect_u32("q8_0_align_window",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 8, k_q8_0_align), 1536);

    // PAD512 must not overshoot size_base. size_base is PAD256(n_ctx_seq), so it
    // can be an odd multiple of 256: 1280 + 512 headroom reaches the base, and
    // the old pad pushed the ring to 1536.
    expect_u32("no_overshoot_pad512",
            llama_kv_cache_iswa_calc_size_swa(/*size_base*/ 1280, n_swa, 1, true, false, 512, a), 1280);
    // Small base, window fits entirely: 1 -> 128 (align) + 8 = 136 -> PAD512 = 512,
    // below the 768 base, so the clamp does not kick in.
    expect_u32("small_base_windowed",
            llama_kv_cache_iswa_calc_size_swa(/*size_base*/ 768, /*n_swa*/ 1, 1, true, false, 8, a), 512);

    // Overflow clamps to size_base, matching upstream. Documented behaviour: only a
    // nonsensical n_swa / n_seq_max reaches this, and the cost is a full-context SWA cache.
    {
        const uint32_t n_swa_huge = 1u << 31;
        const uint32_t size_base  = 100000;
        expect_u32("overflow_clamps_to_size_base",
                llama_kv_cache_iswa_calc_size_swa(size_base, n_swa_huge, /*n_seq_max*/ 2, true, false, 8, a),
                size_base);
    }

    // Zero alignment is a programming error, not a silent "no padding".
    {
        bool threw = false;
        try {
            (void) llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 8, 0);
        } catch (const std::exception &) {
            threw = true;
        }
        if (threw) {
            printf("ok   zero_align_throws\n");
        } else {
            printf("FAIL zero_align_throws: no exception raised\n");
            ++g_fails;
        }
    }

    if (g_fails != 0) {
        printf("RESULT FAIL fails=%d\n", g_fails);
        return 1;
    }
    printf("RESULT PASS\n");
    return 0;
}
