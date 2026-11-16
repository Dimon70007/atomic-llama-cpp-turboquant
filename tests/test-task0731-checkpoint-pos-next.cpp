// TASK-073.1 ST-3 - CPU unit for the checkpoint resume position.
//
// A checkpoint blob stamped [pos_min, pos_max] covers pos_max + 1 tokens, so a
// restore must resume at pos_max + 1. The server used to compute this as
// max(pos_min + 1, pos_max), which equals pos_max + 1 only when the range is a
// single position. Groot's checkpoints are exactly that ([47, 47]), so the
// production model masked the bug; Gemma-4's [0, 55] dropped token 55, the
// token was re-evaluated in a 1-token batch where the original request used a
// 48-token batch, and the text moved because the model is not batch-invariant.
//
// No model is loaded: this is the position arithmetic only.

#include "server-checkpoint-pos.h"

#include <algorithm>
#include <cstdio>

namespace {

int g_fails = 0;

void expect_pos(const char * name, llama_pos got, llama_pos want) {
    if (got == want) {
        printf("ok   %s (resume = %d)\n", name, (int) got);
        return;
    }
    printf("FAIL %s: resume = %d, want %d\n", name, (int) got, (int) want);
    ++g_fails;
}

void test_blob_covering_many_positions() {
    // Gemma-4-26B: checkpoint [0, 55] holds 56 tokens, so the next position is
    // 56. The old max(pos_min + 1, pos_max) returned 55 and replayed token 55.
    expect_pos("checkpoint ending at 55 resumes at 56", server_prompt_checkpoint_pos_next(55), 56);
}

void test_blob_covering_one_position() {
    // Groot-19B-A3B: checkpoint [47, 47] holds 48 tokens. This is the case the
    // old formula got right by coincidence, so it must stay right.
    expect_pos("checkpoint ending at 47 resumes at 48", server_prompt_checkpoint_pos_next(47), 48);
}

void test_wide_range() {
    expect_pos("checkpoint ending at 5049 resumes at 5050", server_prompt_checkpoint_pos_next(5049), 5050);
    expect_pos("checkpoint ending at 5021 resumes at 5022", server_prompt_checkpoint_pos_next(5021), 5022);
}

void test_degenerate_input() {
    expect_pos("negative pos_max yields 0", server_prompt_checkpoint_pos_next(-1), 0);
    expect_pos("pos_max of -1 does not wrap", server_prompt_checkpoint_pos_next(-1), 0);
}

// A corrupt stamp can have pos_min > pos_max. The replaced formula returned
// pos_min + 1 there; this helper returns pos_max + 1, which is never larger than
// the prompt and never negative. Records the behaviour instead of leaving it to
// whoever refactors next.
void test_inverted_range() {
    expect_pos("inverted stamp [55, 10] resumes at 11",
               server_prompt_checkpoint_pos_next(10), 11);
    if (server_prompt_checkpoint_pos_next(10) >= 55) {
        printf("FAIL inverted stamp must not use pos_min as a floor\n");
        ++g_fails;
    } else {
        printf("ok   inverted stamp does not use pos_min as a floor\n");
    }
}

// llama_pos is int32_t, so pos_max + 1 can overflow. A stamped position that
// large cannot come from a real prompt, but the helper must not return a
// negative resume point if it ever does.
void test_no_overflow() {
    const llama_pos huge = 2147483647; // INT32_MAX
    expect_pos("INT32_MAX stamp does not wrap negative",
               server_prompt_checkpoint_pos_next(huge), huge);
}

// The caller clamps with min(pos_next, it->n_tokens) and, further down, with
// size_up_to_pos(). A resume point past the end of the blob's own token count
// must therefore be clamped by the caller, not by this helper - this records
// that contract explicitly so a future refactor does not "simplify" the min away.
void test_caller_clamp_contract() {
    // Groot [47, 47] with n_tokens 48: resume 48 == n_tokens, nothing to clamp.
    expect_pos("resume equals n_tokens for a tight blob",
               std::min(server_prompt_checkpoint_pos_next(47), (llama_pos) 48), 48);

    // Gemma [0, 55] with n_tokens 56: resume 56 == n_tokens, nothing to clamp.
    expect_pos("resume equals n_tokens for [0, 55]",
               std::min(server_prompt_checkpoint_pos_next(55), (llama_pos) 56), 56);

    // An over-claiming stamp must lose to the blob's own token count.
    expect_pos("caller clamp wins over an over-claiming stamp",
               std::min(server_prompt_checkpoint_pos_next(200), (llama_pos) 56), 56);
}

// Proves the test has teeth: the expression this fix replaced must disagree with
// the correct answer on the ranges that matter, and agree only on the
// single-position range that let the bug hide on the production model.
void test_old_formula_would_be_caught() {
    auto old_formula = [](llama_pos pos_min, llama_pos pos_max) {
        return pos_min + 1 > pos_max ? pos_min + 1 : pos_max;
    };

    const llama_pos gemma_old = old_formula(0, 55);
    const llama_pos gemma_new = server_prompt_checkpoint_pos_next(55);
    expect_pos("old formula drops a token on [0, 55]", gemma_old, 55);
    if (gemma_old == gemma_new) {
        printf("FAIL old formula was expected to differ from the fix on [0, 55]\n");
        ++g_fails;
    } else {
        printf("ok   old formula differs from the fix on [0, 55] (%d vs %d)\n",
               (int) gemma_old, (int) gemma_new);
    }

    expect_pos("old formula agrees on [47, 47]", old_formula(47, 47), 48);
    expect_pos("new formula agrees on [47, 47]", server_prompt_checkpoint_pos_next(47), 48);
}

} // namespace

int main() {
    test_blob_covering_many_positions();
    test_blob_covering_one_position();
    test_wide_range();
    test_degenerate_input();
    test_inverted_range();
    test_no_overflow();
    test_caller_clamp_contract();
    test_old_formula_would_be_caught();

    if (g_fails != 0) {
        printf("\n%d check(s) failed\n", g_fails);
        return 1;
    }

    printf("\nall checks passed\n");
    return 0;
}
