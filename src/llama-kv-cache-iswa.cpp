#include "llama-kv-cache-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

uint32_t llama_kv_cache_iswa_kv_block_size(ggml_type type_k, ggml_type type_v) {
    // A rotated block must never straddle the ring eviction boundary, so the ring length
    // has to be a multiple of the KV block size. Take the coarser of the two types: the
    // auto-asymmetric rewrite in llama_kv_cache only ever downgrades turbo K to q8_0
    // (block 32), which is still covered by the turbo block.
    const uint32_t block_k = (uint32_t) ggml_blck_size(type_k);
    const uint32_t block_v = (uint32_t) ggml_blck_size(type_v);

    return std::max(block_k, block_v);
}

uint32_t llama_kv_cache_iswa_calc_swa_headroom(uint32_t n_ubatch) {
    if (n_ubatch == 0) {
        throw std::runtime_error("n_ubatch == 0; SWA size_swa headroom requires n_ubatch > 0");
    }

    // Upstream (llama.cpp master / fork dev): headroom is the live n_ubatch, i.e. exactly
    // the room one ubatch needs to write.
    //
    // TASK-071: the TASK-066.17 variant (headroom = PAD128(n_batch)) was removed. It made
    // size_swa independent of -ub by dropping -ub from the formula, which greened the
    // ubatch-invariance tests without addressing why different ubatches differed, and in
    // exchange turned -b into a quality lever (measured: raising -b shifted pre-sampling
    // logprobs by ~0.09 and flipped greedy argmax). The idea was rejected, not defaulted.
    return n_ubatch;
}

uint32_t llama_kv_cache_iswa_calc_size_swa(
        uint32_t size_base,
        uint32_t n_swa,
        uint32_t n_seq_max,
        bool     unified,
        bool     swa_full,
        uint32_t n_swa_headroom,
        uint32_t n_swa_align) {
    if (n_swa_align == 0) {
        throw std::runtime_error("n_swa_align == 0; SWA ring alignment requires a positive block size");
    }
    if (swa_full) {
        return size_base;
    }

    uint32_t n_swa_eff = n_swa;
#ifndef TASK066_DISABLE_ISWA_PADS
    if (n_swa % n_swa_align != 0) {
        n_swa_eff = ((n_swa + n_swa_align - 1) / n_swa_align) * n_swa_align;
        LLAMA_LOG_WARN("%s: n_swa (%u) is not a multiple of the KV block size (%u), "
                       "padding effective SWA size to %u to prevent context drift\n",
                       __func__, n_swa, n_swa_align, n_swa_eff);
    }
#else
    GGML_UNUSED(n_swa_align);
#endif

    // TASK-066.12 P1: compute the window in uint64 to avoid uint32 wrap.
    const uint64_t swa_window_64 = (uint64_t) n_swa_eff * (unified ? n_seq_max : 1);
    const uint64_t needed_64     = swa_window_64 + (uint64_t) n_swa_headroom;
    if (needed_64 > (uint64_t) UINT32_MAX) {
        // Upstream parity: upstream clamps here too. Only reachable with a nonsensical
        // n_swa / n_seq_max; the result is a full-context SWA cache.
        LLAMA_LOG_WARN("%s: SWA window calculation overflow, clamping to base size\n", __func__);
        return size_base;
    }

    const uint32_t raw = (uint32_t) std::min<uint64_t>(size_base, needed_64);

    // TASK-071: outer pad 256 -> 512.
    //
    // Upstream pads to 256 for FA graph stability / reuse (ggml#17037), not because a kernel
    // requires it. With headroom = n_ubatch the raw values are
    //   n_ubatch 256 -> 1280, 512 -> 1536, 1024 -> 2048, 2048 -> 3072.
    // Padding to 512 makes 256 and 512 collapse to the same ring (1536), so changing -ub
    // between those values keeps size_swa identical. TASK-066.14 Exp A/H6 established that
    // an equal size_swa gives GREEN even with unequal decode chunks; unequal size_swa is the
    // causal factor. Cost: 256 cells more at -ub 256, none at 512+.
    //
    // Invariance is per band, not global: 512 vs 1024 still give different rings.
    constexpr uint32_t k_size_swa_outer_pad = 512;
    static_assert((k_size_swa_outer_pad & (k_size_swa_outer_pad - 1)) == 0,
            "outer pad must be a power of two: GGML_PAD is a bit mask");

    // Padded in uint64: raw can sit near UINT32_MAX and the +pad-1 would wrap.
    const uint64_t padded_64 = GGML_PAD((uint64_t) raw, (uint64_t) k_size_swa_outer_pad);

    // size_base is only padded to 256, so PAD512 can overshoot it. The ring cannot hold
    // more positions than the context provides - clamp instead of allocating dead cells.
    return (uint32_t) std::min<uint64_t>(padded_64, size_base);
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
                 uint32_t   n_pad,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
                     size_t kv_stream_stage_bytes,
                     void * kv_stream_phase_arena,
                     size_t kv_stream_maximum_pool_bytes) :
    llama_kv_cache_iswa(model, model.hparams, type_k, type_v, v_trans, offload, swa_full, unified,
            kv_size, n_seq_max, n_swa_headroom, n_pad, mem_other, filter, reuse, share,
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

    const uint32_t n_swa_align = llama_kv_cache_iswa_kv_block_size(type_k, type_v);

    const uint32_t size_swa = llama_kv_cache_iswa_calc_size_swa(
            size_base, hparams.n_swa, n_seq_max, unified, swa_full, n_swa_headroom, n_swa_align);

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);
    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells (n_swa = %u, headroom = %u, align = %u)\n",
            __func__, size_swa, hparams.n_swa, n_swa_headroom, n_swa_align);

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
    // construction) and stays always-resident. When the ring ends up as long
    // as the base cache (swa_full, or a context fully covered by the window)
    // it is full-context-length too and needs streaming just as much.
    // Concurrently streaming both requires the phase arena to support more
    // than one lease at once, which it doesn't yet - see get_kv_stream_targets()
    // and the multi-cache streaming plan. The swa_full case is rejected up
    // front by llama_kv_stream_config_validate() (swa_full_conflict); the
    // covered-context case is not reachable in practice because streaming
    // requires a context far larger than any SWA window.
    kv_base = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share,
            "", kv_stream_stage_bytes, kv_stream_phase_arena, kv_stream_maximum_pool_bytes);

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
    //
    // This makes seq_rm a two-phase operation for every caller that also owns
    // another cache: check can_seq_rm() first, mutate second. A caller that
    // mutates a second cache before calling this one would leave the two
    // caches disagreeing.
    if (!can_seq_rm(seq_id, p0, p1)) {
        return false;
    }

    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

bool llama_kv_cache_iswa::can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    GGML_UNUSED(p1);

    if (seq_id < 0 || p0 < 0) {
        // "any sequence" or "from the start": removes whole sequences, always removable
        return true;
    }

    const llama_pos swa_min = kv_swa->seq_pos_min(seq_id);
    if (swa_min >= 0 && swa_min > p0) {
        LLAMA_LOG_WARN("%s: cannot remove from pos %d for seq %d: "
                       "SWA cache starts at %d (window already rotated)\n",
                       __func__, (int) p0, (int) seq_id, (int) swa_min);
        return false;
    }

    return true;
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

    // Both sub-caches must agree on the split: the context holds one sinfo list
    // per cache and drives them in lockstep, so unequal sizes are unusable.
    auto try_split = [&](bool equal, const char * what) -> llama_memory_context_ptr {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = equal ? balloc.split_equal(n_ubatch, !unified, 0)
                                : balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            return nullptr;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            return nullptr;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            return nullptr;
        }

        // TASK-066.12 P2: assert is stripped under NDEBUG; mismatch must not
        // build a context with unequal slot infos.
        if (sinfos_base.size() != sinfos_swa.size()) {
            LLAMA_LOG_WARN("%s: base/swa sinfos size mismatch (%zu vs %zu) on %s\n",
                           __func__, sinfos_base.size(), sinfos_swa.size(), what);
            return nullptr;
        }

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    };

    // simple split first; it requires equal splits, so non-unified starts at equal
    if (unified) {
        if (auto res = try_split(/*equal*/ false, "simple split")) {
            return res;
        }
    }
    if (auto res = try_split(/*equal*/ true, "equal split")) {
        return res;
    }

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
    return kv_base->get_has_shared_cells() || kv_swa->get_has_shared_cells();
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
