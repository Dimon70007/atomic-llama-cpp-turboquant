#pragma once

#include "llama-kv-cache.h"

#include <vector>

//
// llama_kv_cache_iswa
//

// utilizes two instances of llama_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class llama_kv_cache_iswa : public llama_memory_i {
public:
    llama_kv_cache_iswa(
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
                          size_t kv_stream_stage_bytes = 0,
                          void * kv_stream_phase_arena = nullptr,
                          size_t kv_stream_maximum_pool_bytes = 0);

    llama_kv_cache_iswa(
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
                          size_t kv_stream_stage_bytes = 0,
                          void * kv_stream_phase_arena = nullptr,
                          size_t kv_stream_maximum_pool_bytes = 0);

    ~llama_kv_cache_iswa() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    bool get_has_shared_cells() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_target> get_kv_stream_targets() const override;

    //
    // llama_kv_cache_iswa specific API
    //

    llama_kv_cache * get_base() const;
    llama_kv_cache * get_swa () const;

private:

    const bool unified;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_swa;
};

class llama_kv_cache_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_iswa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv);

    // used to create an update context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_iswa_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_active_target> get_kv_stream_active_targets() const override;

    //
    // llama_kv_cache_iswa_context specific API
    //

    const llama_kv_cache_context * get_base() const;
    const llama_kv_cache_context * get_swa()  const;

private:
    //llama_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_swa;

    const llama_memory_status status;
};

// Product SWA headroom (TASK-066.17 — restore 066.13 C2 KEEP):
//   headroom = PAD128(n_batch)   // ⊥ live -ub; equal -b ⇒ equal size_swa
// With TASK066_ENABLE_H_SWA_1 (H-SWA-1b): headroom = PAD256(n_ubatch).
// With TASK066_DISABLE_SIZE_SWA_UB: headroom = n_ubatch.
// n_ubatch must be > 0 (fail-closed).
uint32_t llama_kv_cache_iswa_calc_swa_headroom(uint32_t n_batch, uint32_t n_ubatch);

// TASK-069 H-SWA-1/1b: block_pad = PAD256(n_ubatch) (>= ubatch, multiple of 256).
// Example: ub=512 -> 512; ub=2048 -> 2048; ub=1000 -> 1024. Fail-closed on 0.
uint32_t llama_kv_cache_iswa_calc_block_pad(uint32_t n_ubatch);

// Pure size_swa helper (TASK-066.5 ideas).
// Pads n_swa up to TurboQuant KV block (128), then:
//   size_swa = PAD(min(size_base, n_swa_eff * (unified ? n_seq_max : 1) + n_swa_headroom), outer_pad)
// With TASK066_ENABLE_H_BUF_1: PAD(raw + outer_pad, outer_pad) — ring slack vs ring-full@1536.
// Product default outer_pad = 256 (FA get_n_kv pad). H-SWA-1 passes block_pad instead.
// Used by ctor + CPU unit test-iswa-size-swa.
uint32_t llama_kv_cache_iswa_calc_size_swa(
        uint32_t size_base,
        uint32_t n_swa,
        uint32_t n_seq_max,
        bool     unified,
        bool     swa_full,
        uint32_t n_swa_headroom,
        uint32_t outer_pad);
