#pragma once

#include "llama.h"

#include <cstdint>

// TASK-073.1: where a restored context checkpoint resumes.
//
// A checkpoint blob is stamped [pos_min, pos_max] and covers pos_max + 1
// tokens, so the first position the server still has to evaluate is pos_max + 1.
// Computing it as max(pos_min + 1, pos_max) is correct only when the range is a
// single position (Groot-19B-A3B stamps [47, 47]); for a wider range it drops
// the last covered token, which then gets re-evaluated in a batch of its own.
// The models here are not batch-invariant, so that re-evaluation changes the
// text - see tasks/TASK-073.1-swa-checkpoint-unsound.md (F8-F11).
//
// Header-only and free of server types so it can be unit tested without a
// model, a KV cache or mtmd.
inline llama_pos server_prompt_checkpoint_pos_next(llama_pos pos_max) {
    // a stamp that covers no position cannot resume anywhere
    if (pos_max < 0) {
        return 0;
    }

    // llama_pos is int32_t, so the increment can overflow on a nonsense stamp.
    // Such a position cannot come from a real prompt, but returning a negative
    // resume point would be worse than returning the stamp itself.
    if (pos_max == INT32_MAX) {
        return pos_max;
    }

    return pos_max + 1;
}
