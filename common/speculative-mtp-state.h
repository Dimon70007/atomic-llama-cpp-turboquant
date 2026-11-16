#pragma once

#include <cstdint>
#include <vector>

// Per-sequence state of the MTP draft driver, as carried across calls to
// process()/draft()/accept(). Context checkpoints stash it next to the KV blobs:
// restoring a checkpoint reloads the KV of both contexts, so the driver must be
// put back into the same state or the next draft pairs a token with a hidden row
// from a different request.
struct common_speculative_mtp_seq_state {
    // row width the carried hidden rows are expressed in; a blob is only valid
    // for the width it was written with
    int32_t n_embd = 0;

    // last position the driver committed a hidden row for; -1 = nothing carried
    int32_t pos = -1;

    int32_t i_last = -1; // batch index of the last draft row
    int32_t n_last = 0;  // drafts attempted in the most recent draft() call
    int32_t n_cap  = 0;  // effective draft cap of the most recent draft() call

    // adaptive depth controller (draft-mtp-adaptive); unused when adaptive is off
    int32_t adaptive_n_cur   = 0;
    int32_t adaptive_n_climb = 0;
    int32_t adaptive_n_drop  = 0;

    // hidden row paired with the next token: [n_embd]
    std::vector<float> pending_h;

    // hidden rows of the most recent verify batch, [verify_h_rows][n_embd];
    // accept() copies one of them into pending_h
    int32_t            verify_h_rows = 0;
    std::vector<float> verify_h;

    // catch-up rows deferred from process() and prepended to the next draft
    std::vector<int32_t> defer_tok;
    std::vector<int32_t> defer_pos;
    std::vector<int32_t> defer_seq;
    std::vector<float>   defer_embd; // [defer_tok.size()][n_embd]
};

// Serialize `state` into `blob`. Refuses (returns false, leaves `blob` empty)
// when the state is internally inconsistent, so a caller that forgot to resize a
// buffer cannot write a blob that restores as garbage.
bool common_speculative_mtp_state_encode(
        const common_speculative_mtp_seq_state & state,
        std::vector<uint8_t> & blob);

// Restore `state` from `blob`. `n_embd` is the row width of the context that
// will consume the state; a blob written for another width, another version, or
// a different length is rejected (returns false, leaves `state` untouched)
// rather than partially applied.
bool common_speculative_mtp_state_decode(
        const std::vector<uint8_t> & blob,
        int32_t n_embd,
        common_speculative_mtp_seq_state & state);

// True when a carry stamped at `carry_pos` belongs to a checkpoint whose last
// covered position is `restored_pos_max`. The carry is the hidden row of the
// last processed token, so the two are the same position; a mismatch means the
// restore landed on another prefix and the row must not be used.
//
// The comparison is against the checkpoint's own pos_max, not against the live
// memory: at restore time the memory still holds cells from the request being
// replaced, above the checkpoint, and they are trimmed only afterwards.
bool common_speculative_mtp_state_position_ok(int32_t carry_pos, int32_t restored_pos_max);
