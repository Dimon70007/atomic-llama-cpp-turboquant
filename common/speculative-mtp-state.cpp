#include "speculative-mtp-state.h"

#include <cstring>

namespace {

constexpr uint32_t MTP_STATE_MAGIC   = 0x5350544Du; // "MTPS"
constexpr uint32_t MTP_STATE_VERSION = 1u;

// Blob layout: a fixed header, then the variable-length rows. Every count is in
// the header, so the decoder can compute the exact expected size and reject
// anything else instead of trusting the buffer it was handed.
constexpr size_t HEADER_BYTES = 4 * 4 /*magic, version, n_embd, pos*/ +
                                5 * 4 /*i_last, n_last, n_cap, verify_h_rows, defer_count*/ +
                                3 * 4 /*adaptive n_cur, n_climb, n_drop*/;

void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    uint8_t buf[4];
    std::memcpy(buf, &v, sizeof(v));
    out.insert(out.end(), buf, buf + sizeof(buf));
}

void put_floats(std::vector<uint8_t> & out, const std::vector<float> & v) {
    if (v.empty()) {
        return;
    }
    const size_t bytes = v.size() * sizeof(float);
    const size_t off   = out.size();
    out.resize(off + bytes);
    std::memcpy(out.data() + off, v.data(), bytes);
}

void put_i32s(std::vector<uint8_t> & out, const std::vector<int32_t> & v) {
    if (v.empty()) {
        return;
    }
    const size_t bytes = v.size() * sizeof(int32_t);
    const size_t off   = out.size();
    out.resize(off + bytes);
    std::memcpy(out.data() + off, v.data(), bytes);
}

// Bounds-checked sequential reader: every get() either advances or fails, and a
// failure leaves the caller with a partially read blob it will discard.
struct reader {
    const uint8_t * data = nullptr;
    size_t          size = 0;
    size_t          off  = 0;
    bool            ok   = true;

    bool take(void * dst, size_t bytes) {
        if (!ok || bytes > size - off) {
            ok = false;
            return false;
        }
        std::memcpy(dst, data + off, bytes);
        off += bytes;
        return true;
    }

    uint32_t u32() {
        uint32_t v = 0;
        take(&v, sizeof(v));
        return v;
    }

    int32_t i32() {
        int32_t v = 0;
        take(&v, sizeof(v));
        return v;
    }

    bool floats(std::vector<float> & dst, size_t n) {
        dst.assign(n, 0.0f);
        return n == 0 || take(dst.data(), n * sizeof(float));
    }

    bool i32s(std::vector<int32_t> & dst, size_t n) {
        dst.assign(n, 0);
        return n == 0 || take(dst.data(), n * sizeof(int32_t));
    }
};

bool rows_consistent(int32_t n_embd, int32_t rows, const std::vector<float> & buf) {
    return rows >= 0 && buf.size() == (size_t) rows * (size_t) n_embd;
}

} // namespace

bool common_speculative_mtp_state_encode(
        const common_speculative_mtp_seq_state & state,
        std::vector<uint8_t> & blob) {
    blob.clear();

    const int32_t n_embd = state.n_embd;
    if (n_embd <= 0) {
        return false;
    }
    if (state.pending_h.size() != (size_t) n_embd) {
        return false;
    }
    if (!rows_consistent(n_embd, state.verify_h_rows, state.verify_h)) {
        return false;
    }

    const size_t n_defer = state.defer_tok.size();
    if (state.defer_pos.size() != n_defer || state.defer_seq.size() != n_defer) {
        return false;
    }
    if (state.defer_embd.size() != n_defer * (size_t) n_embd) {
        return false;
    }
    if (n_defer > INT32_MAX) {
        return false;
    }

    blob.reserve(HEADER_BYTES + (size_t) n_embd * (1 + (size_t) state.verify_h_rows + n_defer) * sizeof(float) +
                 n_defer * 3 * sizeof(int32_t));

    put_u32(blob, MTP_STATE_MAGIC);
    put_u32(blob, MTP_STATE_VERSION);
    put_u32(blob, (uint32_t) n_embd);
    put_u32(blob, (uint32_t) state.pos);
    put_u32(blob, (uint32_t) state.i_last);
    put_u32(blob, (uint32_t) state.n_last);
    put_u32(blob, (uint32_t) state.n_cap);
    put_u32(blob, (uint32_t) state.verify_h_rows);
    put_u32(blob, (uint32_t) n_defer);
    put_u32(blob, (uint32_t) state.adaptive_n_cur);
    put_u32(blob, (uint32_t) state.adaptive_n_climb);
    put_u32(blob, (uint32_t) state.adaptive_n_drop);

    put_floats(blob, state.pending_h);
    put_floats(blob, state.verify_h);
    put_i32s(blob, state.defer_tok);
    put_i32s(blob, state.defer_pos);
    put_i32s(blob, state.defer_seq);
    put_floats(blob, state.defer_embd);

    return true;
}

bool common_speculative_mtp_state_decode(
        const std::vector<uint8_t> & blob,
        int32_t n_embd,
        common_speculative_mtp_seq_state & state) {
    if (n_embd <= 0 || blob.size() < HEADER_BYTES) {
        return false;
    }

    reader r{blob.data(), blob.size(), 0, true};

    const uint32_t magic   = r.u32();
    const uint32_t version = r.u32();
    const int32_t  w       = r.i32();
    const int32_t  pos     = r.i32();
    const int32_t  i_last  = r.i32();
    const int32_t  n_last  = r.i32();
    const int32_t  n_cap   = r.i32();
    const int32_t  rows    = r.i32();
    const int32_t  n_defer = r.i32();
    const int32_t  n_cur   = r.i32();
    const int32_t  n_climb = r.i32();
    const int32_t  n_drop  = r.i32();
    if (!r.ok) {
        return false;
    }

    if (magic != MTP_STATE_MAGIC || version != MTP_STATE_VERSION) {
        return false;
    }
    // a foreign row width cannot be reinterpreted: the carried rows would be
    // spliced into a buffer of a different shape
    if (w != n_embd || rows < 0 || n_defer < 0) {
        return false;
    }

    common_speculative_mtp_seq_state s;
    s.n_embd           = w;
    s.pos              = pos;
    s.i_last           = i_last;
    s.n_last           = n_last;
    s.n_cap            = n_cap;
    s.adaptive_n_cur   = n_cur;
    s.adaptive_n_climb = n_climb;
    s.adaptive_n_drop  = n_drop;
    s.verify_h_rows    = rows;

    if (!r.floats(s.pending_h, (size_t) w) ||
        !r.floats(s.verify_h, (size_t) rows * (size_t) w) ||
        !r.i32s(s.defer_tok, (size_t) n_defer) ||
        !r.i32s(s.defer_pos, (size_t) n_defer) ||
        !r.i32s(s.defer_seq, (size_t) n_defer) ||
        !r.floats(s.defer_embd, (size_t) n_defer * (size_t) w)) {
        return false;
    }

    // exact length: trailing bytes mean the blob was written by a different
    // layout, and a prefix of it would restore a half-understood state
    if (r.off != blob.size()) {
        return false;
    }

    state = std::move(s);
    return true;
}

bool common_speculative_mtp_state_position_ok(int32_t carry_pos, int32_t restored_pos_max) {
    // -1 means the restored checkpoint covers no position at all, so nothing can
    // be paired with the carry
    if (restored_pos_max < 0) {
        return false;
    }

    return carry_pos == restored_pos_max;
}
