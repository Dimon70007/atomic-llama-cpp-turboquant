// ISWA size_swa unit (CPU; no model).
//
// Locks TASK-066.5 / 066.17:
//   n_swa_eff = ceil(n_swa / 128) * 128
//   headroom  = PAD128(n_batch)   // ⊥ub under equal -b (066.13 KEEP restored)
//   size_swa  = PAD(outer_pad, min(size_base, n_swa_eff * (unified ? n_seq_max : 1) + headroom))
//   product outer_pad = 256; H-SWA-1 block_pad = PAD256(n_ubatch)
//   swa_full still forces size_base


#include <cstdio>
#include <cstdint>

uint32_t llama_kv_cache_iswa_calc_swa_headroom(uint32_t n_batch, uint32_t n_ubatch);

uint32_t llama_kv_cache_iswa_calc_block_pad(uint32_t n_ubatch);

uint32_t llama_kv_cache_iswa_calc_size_swa(
        uint32_t size_base,
        uint32_t n_swa,
        uint32_t n_seq_max,
        bool     unified,
        bool     swa_full,
        uint32_t n_swa_headroom,
        uint32_t outer_pad);

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
    printf("ISWA size_swa unit (CPU; no model) — PAD128(n_batch) + outer_pad\n");

    // --- headroom helper (066.17: from -b only) ---
    // Product YAML batch=2048 ub=512 -> PAD128(2048)=2048.
    expect_u32(
            "headroom_product_2048_512",
            llama_kv_cache_iswa_calc_swa_headroom(2048, 512),
            2048);
    // Same -b, different -ub -> identical headroom (C2p class).
    expect_u32(
            "headroom_c2p_b512_ub256",
            llama_kv_cache_iswa_calc_swa_headroom(512, 256),
            512);
    expect_u32(
            "headroom_c2p_b512_ub512",
            llama_kv_cache_iswa_calc_swa_headroom(512, 512),
            512);
    // Idiot / C3: headroom tracks -b (not ub).
    expect_u32(
            "headroom_idiot_1536_8",
            llama_kv_cache_iswa_calc_swa_headroom(1536, 8),
            1536);
    expect_u32(
            "headroom_idiot_1536_1536",
            llama_kv_cache_iswa_calc_swa_headroom(1536, 1536),
            1536);
    expect_u32(
            "headroom_c3_1280_8",
            llama_kv_cache_iswa_calc_swa_headroom(1280, 8),
            1280);
    // ub=8 b=8 -> PAD128(8)=128.
    expect_u32(
            "headroom_tiny_8_8",
            llama_kv_cache_iswa_calc_swa_headroom(8, 8),
            128);

    // --- H-SWA-1 block_pad = PAD256(n_ubatch) ---
    expect_u32("block_pad_ub512", llama_kv_cache_iswa_calc_block_pad(512), 512);
    expect_u32("block_pad_ub2048", llama_kv_cache_iswa_calc_block_pad(2048), 2048);
    expect_u32("block_pad_ub1000", llama_kv_cache_iswa_calc_block_pad(1000), 1024);
    expect_u32("block_pad_ub8", llama_kv_cache_iswa_calc_block_pad(8), 256);

    // Gemma4-shaped: n_swa=1024 (already multiple of 128).
    const uint32_t n_swa = 1024;
    const uint32_t ctx   = 3072;
    const uint32_t pad256 = 256;

    // headroom=8  -> PAD256(1024+8) = 1280  (direct headroom arg)
    expect_u32(
            "gemma4_headroom8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 1, true, false, /*headroom*/ 8, pad256),
            1280);
    expect_u32(
            "gemma4_headroom384_idiot_c3",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 384, pad256),
            1536);
    expect_u32(
            "gemma4_headroom512",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 512, pad256),
            1536);
    expect_u32(
            "gemma4_headroom1024_metal_max",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1024, pad256),
            2048);
    expect_u32(
            "gemma4_headroom1344",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1344, pad256),
            2560);
    expect_u32(
            "gemma4_headroom2048",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 2048, pad256),
            3072);

    {
        const uint32_t long_ctx = 160000;
        expect_u32(
                "long_ctx_windowed_ub512",
                llama_kv_cache_iswa_calc_size_swa(long_ctx, n_swa, 1, true, false, 512, pad256),
                1536);
    }

    expect_u32(
            "swa_full_uses_size_base",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, /*swa_full*/ true, 8, pad256),
            ctx);

    expect_u32(
            "clamp_to_size_base",
            llama_kv_cache_iswa_calc_size_swa(/*size_base*/ 512, n_swa, 1, true, false, 8, pad256),
            512);

    expect_u32(
            "pad256_n_swa_1000_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, /*n_swa*/ 1000, 1, true, false, 8, pad256),
            1280);

    expect_u32(
            "unified_n_seq_max_2_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 2, true, false, 8, pad256),
            2304);

    expect_u32(
            "non_unified_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 4, false, false, 8, pad256),
            1280);

    // H-SWA-1: outer_pad=block_pad(ub).
    // ub=512 -> PAD512(1024+8)=1536 (vs pad256=1280).
    // ub=2048 + headroom 2048 -> PAD2048(1024+2048)=PAD2048(3072)=4096.
    expect_u32(
            "h_swa1_outer_ub512_headroom8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 8, 512),
            1536);
    expect_u32(
            "h_swa1_outer_ub2048_headroom2048",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 2048, 2048),
            4096);

    {
        const uint32_t n_swa_huge = 1u << 31;
        const uint32_t size_base  = 100000;
        expect_u32(
                "overflow_unified_n_seq_max_2_clamps_to_size_base",
                llama_kv_cache_iswa_calc_size_swa(size_base, n_swa_huge, /*n_seq_max*/ 2, true, false, 8, pad256),
                size_base);
    }

    if (g_fails != 0) {
        printf("RESULT FAIL fails=%d\n", g_fails);
        return 1;
    }
    printf("RESULT PASS\n");
    return 0;
}
