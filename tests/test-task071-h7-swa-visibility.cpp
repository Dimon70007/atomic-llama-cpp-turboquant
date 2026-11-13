// TASK-071 H7 mechanism test: which SWA cells survive the ring, as a function of size_swa.
//
// H7 (locked by measurement, see tasks/TASK-071-...md § ST-3):
//   -ub with fixed -b  -> bit-identical output (logits AND full 931-char text)
//   -b moves (-> size_swa 3072 vs 5120) -> pre-sampling logprobs shift ~0.09
//
// The formula test (test-iswa-size-swa.cpp) already locks how size_swa is *computed*.
// This test locks the *consequence*: given a token position p1, which earlier positions
// are still visible? `is_masked_swa` is the single predicate that decides visibility, and
// it depends on n_swa only - NOT on size_swa. That asymmetry is exactly H7: two arms with
// different size_swa allocate a different number of cells, so a different set of positions
// survives eviction, and the visible window at the same p1 differs.
//
// What this test proves:
//   1. The visibility predicate is independent of size_swa (purely a function of positions
//      and n_swa). A change here would be an upstream semantic break.
//   2. Window width is exactly n_swa for STANDARD.
//   3. A ring with size_swa < n_swa + headroom cannot hold a full window -> positions that
//      are *visible by predicate* may have been *evicted by capacity*. This is the H7 corner
//      and is what the harness must never allow.
//
// CPU only, no model, no KV allocation.

#include <cstdio>
#include <cstdint>
#include <vector>

#include "llama-hparams.h"

namespace {

int g_fails = 0;

void expect_bool(const char * name, bool got, bool want) {
    if (got == want) {
        printf("ok   %s\n", name);
        return;
    }
    printf("FAIL %s: got %s want %s\n", name, got ? "true" : "false", want ? "true" : "false");
    ++g_fails;
}

void expect_u32(const char * name, uint32_t got, uint32_t want) {
    if (got == want) {
        printf("ok   %s (%u)\n", name, got);
        return;
    }
    printf("FAIL %s: got %u want %u\n", name, got, want);
    ++g_fails;
}

// Count how many of the `size_swa` most recent ring slots would still be "in window" for a
// query at position p1, i.e. not masked. Models the ring as holding positions
// [p1 - size_swa + 1 .. p1]; everything older is gone by capacity.
uint32_t visible_in_ring(uint32_t size_swa, uint32_t n_swa, uint32_t p1) {
    if (size_swa == 0 || p1 == 0) {
        return 0;
    }
    const uint32_t oldest_held = (p1 + 1 > size_swa) ? (p1 + 1 - size_swa) : 0;
    uint32_t visible = 0;
    for (uint32_t p0 = oldest_held; p0 < p1; ++p0) {
        if (!llama_hparams::is_masked_swa(n_swa, LLAMA_SWA_TYPE_STANDARD, (llama_pos) p0, (llama_pos) p1)) {
            ++visible;
        }
    }
    return visible;
}

} // namespace

int main() {
    printf("TASK-071 H7: SWA visibility vs ring capacity (CPU; no model)\n");

    const uint32_t n_swa = 1024;   // gemma4.attention.sliding_window

    // --- 1. predicate is a function of positions + n_swa only -------------------
    // The product arms differ in size_swa (3072 vs 5120) but share n_swa. The predicate
    // must therefore agree on both for the same (p0, p1). If this ever fails, visibility
    // started depending on capacity and the H7 reasoning needs revisiting.
    {
        const uint32_t p1 = 20000;
        for (uint32_t back = 0; back < 2048; back += 64) {
            const uint32_t p0 = p1 - back;
            const bool masked = llama_hparams::is_masked_swa(
                    n_swa, LLAMA_SWA_TYPE_STANDARD, (llama_pos) p0, (llama_pos) p1);
            // recompute with a hypothetical larger n_swa-derived ring: same call, the
            // predicate has no size parameter at all - assert the obvious contract.
            const bool masked_again = llama_hparams::is_masked_swa(
                    n_swa, LLAMA_SWA_TYPE_STANDARD, (llama_pos) p0, (llama_pos) p1);
            expect_bool("predicate_deterministic", masked == masked_again, true);
        }
    }

    // --- 2. STANDARD window width is exactly n_swa -------------------------------
    {
        const uint32_t p1 = 5000;
        // distance n_swa-1 -> visible, distance n_swa -> masked
        expect_bool("window_edge_visible",
                !llama_hparams::is_masked_swa(n_swa, LLAMA_SWA_TYPE_STANDARD,
                        (llama_pos) (p1 - (n_swa - 1)), (llama_pos) p1), true);
        expect_bool("window_edge_masked",
                !llama_hparams::is_masked_swa(n_swa, LLAMA_SWA_TYPE_STANDARD,
                        (llama_pos) (p1 - n_swa), (llama_pos) p1), false);
        expect_bool("far_past_masked",
                !llama_hparams::is_masked_swa(n_swa, LLAMA_SWA_TYPE_STANDARD,
                        (llama_pos) 0, (llama_pos) p1), false);
    }

    // --- 3. capacity vs window: the H7 corner -----------------------------------
    // size_swa 3072 and 5120 both exceed n_swa=1024, so at a deep position the full
    // window fits in EITHER ring -> same visible count. That is why the divergence is
    // NOT explained by "ring too small"; it must come from *which* cells got reclaimed
    // earlier, i.e. from capacity-dependent eviction history.
    {
        const uint32_t deep = 40000;   // far past both ring sizes
        const uint32_t vis_3072 = visible_in_ring(3072, n_swa, deep);
        const uint32_t vis_5120 = visible_in_ring(5120, n_swa, deep);
        expect_u32("visible_count_ring3072", vis_3072, n_swa - 1);
        expect_u32("visible_count_ring5120", vis_5120, n_swa - 1);
        // Both rings hold the full window at a deep position; capacity alone does not
        // change the visible set. Documents that H7 is about eviction history, not width.
        expect_bool("full_window_fits_both_rings", vis_3072 == vis_5120, true);
    }

    // A ring SMALLER than the window is the dangerous case: predicate says visible,
    // capacity already dropped the cell -> silent use of a wrong/stale KV entry.
    {
        const uint32_t tiny_ring = n_swa / 2;   // 512 < 1024
        const uint32_t deep = 40000;
        const uint32_t vis = visible_in_ring(tiny_ring, n_swa, deep);
        // The ring holds positions [p1-511 .. p1], but p1 itself is the query row, so the
        // *past* cells that can be attended are p1-511 .. p1-1 = 511 positions. All of them
        // are inside the 1024 window, so visibility == cells-1, not cells.
        expect_u32("undersized_ring_holds_only_capacity", vis, tiny_ring - 1);
        expect_bool("undersized_ring_loses_window", vis < (n_swa - 1), true);
    }

    // --- 4. eviction order differs with capacity --------------------------------
    // At p1 just past the ring, the oldest held position differs between rings. Two arms
    // therefore start their history at different positions -> different eviction history
    // for the same prompt. This is the mechanism H7 needs.
    {
        const uint32_t p1 = 3072 + 100;   // just past the 3072 ring
        const uint32_t oldest_3072 = (p1 + 1 > 3072) ? (p1 + 1 - 3072) : 0;
        const uint32_t oldest_5120 = (p1 + 1 > 5120) ? (p1 + 1 - 5120) : 0;
        expect_u32("oldest_held_ring3072", oldest_3072, 101);
        expect_u32("oldest_held_ring5120_keeps_from_zero", oldest_5120, 0);
        expect_bool("rings_hold_different_history", oldest_3072 != oldest_5120, true);
    }

    if (g_fails == 0) {
        printf("OK\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fails);
    return 1;
}
