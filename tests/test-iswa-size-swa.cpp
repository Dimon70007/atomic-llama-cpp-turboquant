// ISWA size_swa unit (CPU; no model).
//
// Locks TASK-066.5 ideas:
//   n_swa_eff = ceil(n_swa / 128) * 128
//   headroom  = PAD128(max(n_ubatch, n_batch / 4))  // product anti-footgun + TQ block
//   size_swa  = PAD256(min(size_base, n_swa_eff * (unified ? n_seq_max : 1) + headroom))
//   swa_full still forces size_base
// Metal product envelope: n_ubatch <= 1024 (larger ub -> oversized size_swa).


#include <cstdio>
#include <cstdint>

uint32_t llama_kv_cache_iswa_calc_swa_headroom(uint32_t n_batch, uint32_t n_ubatch);

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
    printf("ISWA size_swa unit (CPU; no model) — PAD128(max(ub,b/4)) + pad256\n");

    // --- headroom helper ---
    // Product YAML batch=2048 ub=512 -> PAD128(512)=512.
    expect_u32(
            "headroom_product_2048_512",
            llama_kv_cache_iswa_calc_swa_headroom(2048, 512),
            512);
    // Idiot autotest: ub=8 b=1536 -> PAD128(max(8,384))=384 (already %128).
    expect_u32(
            "headroom_idiot_1536_8",
            llama_kv_cache_iswa_calc_swa_headroom(1536, 8),
            384);
    // Oneshot ub=n: PAD128(1536)=1536.
    expect_u32(
            "headroom_idiot_1536_1536",
            llama_kv_cache_iswa_calc_swa_headroom(1536, 1536),
            1536);
    // b=1280 ub=8 -> max(8,320)=320 -> PAD128(320)=384.
    expect_u32(
            "headroom_c3_1280_8",
            llama_kv_cache_iswa_calc_swa_headroom(1280, 8),
            384);
    // ub=8 b=8 -> max(8,2)=8 -> PAD128(8)=128.
    expect_u32(
            "headroom_tiny_8_8",
            llama_kv_cache_iswa_calc_swa_headroom(8, 8),
            128);

    // Gemma4-shaped: n_swa=1024 (already multiple of 128).
    const uint32_t n_swa = 1024;
    const uint32_t ctx   = 3072;

    // headroom=8  -> PAD256(1024+8) = 1280  (direct headroom arg; product would floor first)
    expect_u32(
            "gemma4_headroom8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 1, true, false, /*headroom*/ 8),
            1280);
    // idiot C3 multi effective headroom 384 -> PAD256(1024+384)=PAD256(1408)=1536
    expect_u32(
            "gemma4_headroom384_idiot_c3",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 384),
            1536);
    expect_u32(
            "gemma4_headroom512",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 512),
            1536);
    expect_u32(
            "gemma4_headroom1024_metal_max",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1024),
            2048);
    expect_u32(
            "gemma4_headroom1344",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 1344),
            2560);
    expect_u32(
            "gemma4_headroom2048",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, 1, true, false, 2048),
            3072);

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

    expect_u32(
            "clamp_to_size_base",
            llama_kv_cache_iswa_calc_size_swa(/*size_base*/ 512, n_swa, 1, true, false, 8),
            512);

    expect_u32(
            "pad256_n_swa_1000_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, /*n_swa*/ 1000, 1, true, false, 8),
            1280);

    expect_u32(
            "unified_n_seq_max_2_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 2, true, false, 8),
            2304);

    expect_u32(
            "non_unified_ub8",
            llama_kv_cache_iswa_calc_size_swa(ctx, n_swa, /*n_seq_max*/ 4, false, false, 8),
            1280);

    {
        const uint32_t n_swa_huge = 1u << 31;
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
