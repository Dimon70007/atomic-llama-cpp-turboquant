#include "llama-kv-cache-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// TurboQuant KV block size (turbo2/3/4). SWA window must be a multiple so
// quantized blocks align with cache rotation (ideas/llamacpp_swa_size_calculate.md).
#ifndef TURBO_KV_BLOCK_SIZE
#define TURBO_KV_BLOCK_SIZE 128
#endif

// TASK-066.14 R&D override: requires BOTH env vars (avoids poisoning unit tests /
// leftover shell exports). Unset = product headroom from caller (MAX_UBATCH KEEP).
static uint32_t task066_resolve_swa_headroom(uint32_t n_swa_headroom) {
    const char * allow = std::getenv("TASK066_ALLOW_SIZE_SWA_HEADROOM");
    const char * e = std::getenv("TASK066_SIZE_SWA_HEADROOM");
    if (!allow || allow[0] != '1') {
        return n_swa_headroom;
    }
    if (!e || !e[0]) {
        LLAMA_LOG_ERROR("%s: FAIL TASK066_ALLOW_SIZE_SWA_HEADROOM=1 but "
                        "TASK066_SIZE_SWA_HEADROOM unset\n",
                        __func__);
        GGML_ABORT("TASK066_SIZE_SWA_HEADROOM required when ALLOW=1");
    }
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || v < 1 || v > (long) UINT32_MAX) {
        LLAMA_LOG_ERROR("%s: FAIL TASK066_SIZE_SWA_HEADROOM=%s (need positive int); "
                        "refusing silent headroom fallback (caller had %u)\n",
                        __func__, e, n_swa_headroom);
        GGML_ABORT("invalid TASK066_SIZE_SWA_HEADROOM");
    }
    const uint32_t locked = (uint32_t) v;
    LLAMA_LOG_WARN("%s: DIAG_ONLY TASK066_SIZE_SWA_HEADROOM=%u (caller had %u) — "
                   "R&D override; unset ALLOW for product KEEP\n",
                   __func__, locked, n_swa_headroom);
    return locked;
}

uint32_t llama_kv_cache_iswa_calc_block_pad(uint32_t n_ubatch) {
    if (n_ubatch == 0) {
        throw std::runtime_error("n_ubatch == 0; SWA block_pad requires n_ubatch > 0");
    }
    // H-SWA-1/1b: one module (>= ubatch and multiple of 256).
    return GGML_PAD(n_ubatch, 256u);
}

uint32_t llama_kv_cache_iswa_calc_swa_headroom(uint32_t n_batch, uint32_t n_ubatch) {
    if (n_ubatch == 0) {
        throw std::runtime_error("n_ubatch == 0; SWA size_swa headroom requires n_ubatch > 0");
    }
    if (n_batch == 0) {
        throw std::runtime_error("n_batch == 0; SWA size_swa headroom requires n_batch > 0");
    }
#ifdef TASK066_DISABLE_SIZE_SWA_UB
    // Ablation master-like: headroom = live n_ubatch only (∝ub; 066.18 lever).
    (void) n_batch;
    return n_ubatch;
#elif defined(TASK066_ENABLE_H_SWA_1)
    // TASK-069 H-SWA-1b: headroom = block_pad = PAD256(n_ubatch).
    // Outer-only H-SWA-1 was NO_EFFECT (…032102Z); Cell C MATCH needs headroom∝ub.
    (void) n_batch;
    return llama_kv_cache_iswa_calc_block_pad(n_ubatch);
#else
    // TASK-066.17 (restore 066.13 C2 KEEP): headroom from -b only.
    // Equal -b + differing -ub ⇒ identical size_swa (ubatch bit-exact).
    // Safe: llama guarantees n_ubatch <= n_batch, so one-write room holds.
    // PAD128 = TurboQuant KV block. n_ubatch only fail-closed above (⊥ub sizing).
    (void) n_ubatch;
    return GGML_PAD(n_batch, (uint32_t) TURBO_KV_BLOCK_SIZE);
#endif
}

uint32_t llama_kv_cache_iswa_calc_size_swa(
        uint32_t size_base,
        uint32_t n_swa,
        uint32_t n_seq_max,
        bool     unified,
        bool     swa_full,
        uint32_t n_swa_headroom,
        uint32_t outer_pad) {
    if (swa_full) {
        return size_base;
    }
    if (outer_pad == 0) {
        throw std::runtime_error("outer_pad == 0; SWA size_swa outer align requires outer_pad > 0");
    }
    // TASK-066.5 (ideas): windowed SWA; n_swa_eff aligned to TurboQuant block (128).
    // Product headroom = PAD128(n_batch) from calc_swa_headroom (066.17 / 066.13 KEEP).
    // Optional TASK066_SIZE_SWA_HEADROOM overrides for R&D only.
    // Outer pad: product 256; H-SWA-1 uses PAD256(n_ubatch) from caller.
    // TASK-066.12 P1: compute window in uint64 to avoid uint32 wrap.
    n_swa_headroom = task066_resolve_swa_headroom(n_swa_headroom);
    uint32_t n_swa_eff = n_swa;
#ifndef TASK066_DISABLE_ISWA_PADS
    if (n_swa % TURBO_KV_BLOCK_SIZE != 0) {
        n_swa_eff = ((n_swa + TURBO_KV_BLOCK_SIZE - 1) / TURBO_KV_BLOCK_SIZE) * TURBO_KV_BLOCK_SIZE;
        LLAMA_LOG_WARN("%s: n_swa (%u) is not a multiple of TurboQuant block size (%d), "
                       "padding effective SWA size to %u to prevent context drift\n",
                       __func__, n_swa, TURBO_KV_BLOCK_SIZE, n_swa_eff);
    }
#endif

    const uint64_t swa_window_64 = (uint64_t) n_swa_eff * (unified ? n_seq_max : 1);
    if (swa_window_64 + (uint64_t) n_swa_headroom > (uint64_t) UINT32_MAX) {
        LLAMA_LOG_WARN("%s: SWA window calculation overflow, clamping to base size\n", __func__);
        return size_base;
    }
    const uint32_t swa_window = (uint32_t) swa_window_64;
    const uint32_t raw = std::min(size_base, swa_window + n_swa_headroom);
#ifdef TASK066_ENABLE_H_BUF_1
    // H-BUF-1: one outer_pad of ring slack past PAD(raw). ub512@1536 was slack=0.
    // Fail-closed if raw+outer_pad would wrap uint32 (should not under product ctx).
    if (raw > UINT32_MAX - outer_pad) {
        throw std::runtime_error("size_swa H-BUF-1: raw + outer_pad overflows uint32");
    }
    return GGML_PAD(raw + outer_pad, outer_pad);
#else
    return GGML_PAD(raw, outer_pad);
#endif
}

//
// llama_kv_cache_iswa
//

llama_kv_cache_iswa::llama_kv_cache_iswa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_swa_headroom,
                 uint32_t   size_swa_outer_pad,
                 uint32_t   n_pad,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
                     size_t kv_stream_stage_bytes,
                     void * kv_stream_phase_arena,
                     size_t kv_stream_maximum_pool_bytes) :
    llama_kv_cache_iswa(model, model.hparams, type_k, type_v, v_trans, offload, swa_full, unified,
            kv_size, n_seq_max, n_swa_headroom, size_swa_outer_pad, n_pad, mem_other, filter, reuse, share,
            kv_stream_stage_bytes, kv_stream_phase_arena, kv_stream_maximum_pool_bytes) {
}

llama_kv_cache_iswa::llama_kv_cache_iswa(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_swa_headroom,
                 uint32_t   size_swa_outer_pad,
                 uint32_t   n_pad,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
                     size_t kv_stream_stage_bytes,
                     void * kv_stream_phase_arena,
                     size_t kv_stream_maximum_pool_bytes) : unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
    }

    const uint32_t size_swa = llama_kv_cache_iswa_calc_size_swa(
            size_base, hparams.n_swa, n_seq_max, unified, swa_full, n_swa_headroom,
            size_swa_outer_pad);

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);
    // WARN: product-measure captures verbosity=3; INFO size_swa lines were dropped from stamps.
    LLAMA_LOG_WARN("%s: size_swa calc: n_swa=%u headroom=%u "
                   "unified=%d swa_full=%d -> size_swa=%u (n_swa_eff pad %d; outer_pad=%u)\n",
            __func__, hparams.n_swa, n_swa_headroom, (int) unified, (int) swa_full,
            size_swa, TURBO_KV_BLOCK_SIZE, size_swa_outer_pad);

    llama_memory_t mem_other_base = nullptr;
    if (mem_other) {
        mem_other_base = static_cast<llama_kv_cache_iswa *>(mem_other)->get_base();
    }

    llama_memory_t mem_other_swa = nullptr;
    if (mem_other) {
        mem_other_swa = static_cast<llama_kv_cache_iswa *>(mem_other)->get_swa();
    }

    // Block KV streaming: the base cache scales with context length and is
    // the one worth streaming; the SWA cache is window-bounded (small by
    // construction) and stays always-resident, UNLESS swa_full makes it
    // full-context-length too (size_swa == size_base), in which case it
    // needs streaming just as much as the base cache does. Concurrently
    // streaming both requires the phase arena to support more than one
    // lease at once, which it doesn't yet (see get_kv_stream_targets() and
    // the multi-cache streaming plan) - forwarding the same
    // kv_stream_phase_arena to both here is safe today only because no
    // caller reaches the swa_full branch with streaming enabled yet.
    kv_base = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share,
            "", kv_stream_stage_bytes, kv_stream_phase_arena, kv_stream_maximum_pool_bytes);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    const bool swa_is_full_length = size_swa == size_base;
    kv_swa = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, mem_other_swa, filter_swa, reuse, share,
            "", swa_is_full_length ? kv_stream_stage_bytes : 0,
            swa_is_full_length ? kv_stream_phase_arena : nullptr,
            swa_is_full_length ? kv_stream_maximum_pool_bytes : 0);
}

void llama_kv_cache_iswa::clear(bool data) {
    kv_base->clear(data);
    kv_swa ->clear(data);
}

bool llama_kv_cache_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TASK-066.12 P0: if the trim starts before the SWA window, SWA already
    // rotated those rows away and cannot mirror the base trim. Fail closed
    // before mutating either cache (upstream iSWA semantics).
    if (seq_id >= 0 && p0 >= 0) {
        const llama_pos swa_min = kv_swa->seq_pos_min(seq_id);
        if (swa_min >= 0 && swa_min > p0) {
            LLAMA_LOG_WARN("%s: cannot remove from pos %d for seq %d: "
                           "SWA cache starts at %d (window already rotated)\n",
                           __func__, (int) p0, (int) seq_id, (int) swa_min);
            return false;
        }
    }

    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_iswa::seq_keep(llama_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llama_kv_cache_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // TASK-066.12 P0: base keeps the full sequence; SWA is only a window.
    return kv_base->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_base->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

bool llama_kv_cache_iswa::has_kv_stream_targets() const {
    return kv_base->has_kv_stream_targets() || kv_swa->has_kv_stream_targets();
}

std::vector<llama_kv_stream_target> llama_kv_cache_iswa::get_kv_stream_targets() const {
    std::vector<llama_kv_stream_target> targets = kv_base->get_kv_stream_targets();
    for (auto & target : kv_swa->get_kv_stream_targets()) {
        targets.push_back(target);
    }
    return targets;
}

llama_memory_context_ptr llama_kv_cache_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        // TASK-066.12 P2: assert is stripped under NDEBUG; mismatch must not
        // build a context with unequal slot infos.
        if (sinfos_base.size() != sinfos_swa.size()) {
            LLAMA_LOG_WARN("%s: base/swa sinfos size mismatch (%zu vs %zu), "
                           "retrying with equal split\n",
                           __func__, sinfos_base.size(), sinfos_swa.size());
            break;
        }

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        if (sinfos_base.size() != sinfos_swa.size()) {
            LLAMA_LOG_WARN("%s: base/swa sinfos size mismatch (%zu vs %zu) on equal split\n",
                           __func__, sinfos_base.size(), sinfos_swa.size());
            break;
        }

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_full() {
    return std::make_unique<llama_kv_cache_iswa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_iswa_context>(this, lctx, optimize);
}

bool llama_kv_cache_iswa::get_can_shift() const {
    // TASK-066.12 P2 KEEP: partial SWA (swa_full=false) sizes base != swa, so
    // context shift stays disabled. Equal-size check is intentional, not a bug.
    return kv_base->get_can_shift() &&
           kv_swa->get_can_shift() &&
           kv_base->get_size() == kv_swa->get_size();
}

bool llama_kv_cache_iswa::get_has_shared_cells() const {
    return (kv_base && kv_base->get_has_shared_cells()) || (kv_swa && kv_swa->get_has_shared_cells());
}

void llama_kv_cache_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_kv_cache_iswa::get_base() const {
    return kv_base.get();
}

llama_kv_cache * llama_kv_cache_iswa::get_swa() const {
    return kv_swa.get();
}

//
// llama_kv_cache_iswa_context
//

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv) :
    ctx_base(kv->get_base()->init_full()),
    ctx_swa (kv->get_swa ()->init_full()),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_swa,
        std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_base(new llama_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa (new llama_kv_cache_context(kv->get_swa (), std::move(sinfos_swa),  this->ubatches)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context:: ~llama_kv_cache_iswa_context() = default;

bool llama_kv_cache_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();

    return res;
}

llama_memory_status llama_kv_cache_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

bool llama_kv_cache_iswa_context::has_kv_stream_targets() const {
    return ctx_base->has_kv_stream_targets() || ctx_swa->has_kv_stream_targets();
}

std::vector<llama_kv_stream_active_target> llama_kv_cache_iswa_context::get_kv_stream_active_targets() const {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS) {
        return {};
    }
    std::vector<llama_kv_stream_active_target> targets = ctx_base->get_kv_stream_active_targets();
    for (auto & target : ctx_swa->get_kv_stream_active_targets()) {
        targets.push_back(target);
    }
    return targets;
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_base.get());
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_swa()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}
