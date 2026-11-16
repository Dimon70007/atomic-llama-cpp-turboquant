// TASK-073 ST-2 - CPU unit for the MTP draft state codec.
//
// A context checkpoint stores the draft driver's per-sequence state next to the
// KV blobs and restores it when the server reuses the checkpoint. The codec is
// the only thing between "the driver looked like this when the checkpoint was
// taken" and "the driver looks like this after the restore", so a silent
// length, version or shape bug there reintroduces exactly the defect this file
// guards: repeated requests emit different text.
//
// No model is loaded: the codec is pure bytes in, bytes out.

#include "speculative-mtp-state.h"

#include <cstdio>
#include <vector>

namespace {

int g_fails = 0;

void expect_true(const char * name, bool cond) {
    if (cond) {
        printf("ok   %s\n", name);
        return;
    }
    printf("FAIL %s\n", name);
    ++g_fails;
}

common_speculative_mtp_seq_state make_state(int32_t n_embd, int32_t verify_rows, int32_t defer_count) {
    common_speculative_mtp_seq_state s;
    s.n_embd      = n_embd;
    s.pos         = 47;
    s.i_last      = 1234;
    s.n_last      = 2;
    s.n_cap       = 3;
    s.adaptive_n_cur   = 2;
    s.adaptive_n_climb = 1;
    s.adaptive_n_drop  = 7;

    s.pending_h.resize((size_t) n_embd);
    for (int32_t i = 0; i < n_embd; ++i) {
        s.pending_h[i] = (float) i * 0.5f;
    }

    s.verify_h_rows = verify_rows;
    s.verify_h.resize((size_t) verify_rows * n_embd);
    for (size_t i = 0; i < s.verify_h.size(); ++i) {
        s.verify_h[i] = (float) i * -0.25f;
    }

    s.defer_tok.resize((size_t) defer_count);
    s.defer_pos.resize((size_t) defer_count);
    s.defer_seq.resize((size_t) defer_count);
    s.defer_embd.resize((size_t) defer_count * n_embd);
    for (int32_t i = 0; i < defer_count; ++i) {
        s.defer_tok[i] = 100 + i;
        s.defer_pos[i] = 50 + i;
        s.defer_seq[i] = 0;
    }
    for (size_t i = 0; i < s.defer_embd.size(); ++i) {
        s.defer_embd[i] = (float) i * 0.125f;
    }
    return s;
}

bool same_state(const common_speculative_mtp_seq_state & a, const common_speculative_mtp_seq_state & b) {
    return a.n_embd == b.n_embd && a.pos == b.pos && a.i_last == b.i_last &&
           a.n_last == b.n_last && a.n_cap == b.n_cap &&
           a.adaptive_n_cur == b.adaptive_n_cur &&
           a.adaptive_n_climb == b.adaptive_n_climb &&
           a.adaptive_n_drop == b.adaptive_n_drop &&
           a.verify_h_rows == b.verify_h_rows &&
           a.pending_h == b.pending_h && a.verify_h == b.verify_h &&
           a.defer_tok == b.defer_tok && a.defer_pos == b.defer_pos &&
           a.defer_seq == b.defer_seq && a.defer_embd == b.defer_embd;
}

void test_round_trip() {
    const common_speculative_mtp_seq_state in  = make_state(8, 3, 2);
    std::vector<uint8_t> blob;

    expect_true("encode succeeds", common_speculative_mtp_state_encode(in, blob));
    expect_true("blob is not empty", !blob.empty());

    common_speculative_mtp_seq_state out;
    expect_true("decode succeeds", common_speculative_mtp_state_decode(blob, 8, out));
    expect_true("round trip preserves every field", same_state(in, out));
}

void test_round_trip_empty_carry() {
    // a sequence with no carried row and no deferred rows is a legal state
    common_speculative_mtp_seq_state in;
    in.n_embd = 4;
    in.pos    = -1;
    in.pending_h.assign(4, 0.0f);

    std::vector<uint8_t> blob;
    expect_true("encode empty carry", common_speculative_mtp_state_encode(in, blob));

    common_speculative_mtp_seq_state out;
    expect_true("decode empty carry", common_speculative_mtp_state_decode(blob, 4, out));
    expect_true("empty carry round trips", same_state(in, out));
    expect_true("empty carry keeps pos -1", out.pos == -1);
}

void test_rejects_foreign_width() {
    // a blob from a different model must be rejected, not memcpy'd into a
    // buffer of the wrong width
    const common_speculative_mtp_seq_state in = make_state(8, 2, 0);
    std::vector<uint8_t> blob;
    expect_true("encode for width 8", common_speculative_mtp_state_encode(in, blob));

    common_speculative_mtp_seq_state out;
    expect_true("reject blob of another n_embd",
                !common_speculative_mtp_state_decode(blob, 16, out));
}

void test_rejects_truncation() {
    const common_speculative_mtp_seq_state in = make_state(8, 2, 1);
    std::vector<uint8_t> blob;
    expect_true("encode for truncation test", common_speculative_mtp_state_encode(in, blob));

    bool all_rejected = true;
    for (size_t len = 0; len < blob.size(); ++len) {
        std::vector<uint8_t> cut(blob.begin(), blob.begin() + (long) len);
        common_speculative_mtp_seq_state out;
        if (common_speculative_mtp_state_decode(cut, 8, out)) {
            printf("  truncated blob of %zu/%zu bytes was accepted\n", len, blob.size());
            all_rejected = false;
            break;
        }
    }
    expect_true("reject every truncated prefix", all_rejected);

    std::vector<uint8_t> trailing = blob;
    trailing.push_back(0);
    common_speculative_mtp_seq_state out;
    expect_true("reject trailing garbage", !common_speculative_mtp_state_decode(trailing, 8, out));
}

void test_rejects_wrong_version() {
    const common_speculative_mtp_seq_state in = make_state(8, 2, 0);
    std::vector<uint8_t> blob;
    expect_true("encode for version test", common_speculative_mtp_state_encode(in, blob));
    if (blob.size() < 2) {
        printf("FAIL version test needs a non-empty blob (got %zu bytes)\n", blob.size());
        ++g_fails;
        return;
    }

    // version is the first field of the header
    blob[0] = (uint8_t) (blob[0] + 1u);

    common_speculative_mtp_seq_state out;
    expect_true("reject a blob from another version",
                !common_speculative_mtp_state_decode(blob, 8, out));
}

void test_rejects_inconsistent_shape() {
    // verify_h_rows and the buffer length must agree in the encoder as well:
    // a caller that forgets to resize gets a loud refusal, not a short blob
    common_speculative_mtp_seq_state bad = make_state(8, 2, 0);
    bad.verify_h_rows = 5; // buffer still holds 2 rows

    std::vector<uint8_t> blob;
    expect_true("encoder rejects rows/buffer mismatch",
                !common_speculative_mtp_state_encode(bad, blob));

    common_speculative_mtp_seq_state bad_pending = make_state(8, 2, 0);
    bad_pending.pending_h.resize(3);

    std::vector<uint8_t> blob2;
    expect_true("encoder rejects wrong pending_h width",
                !common_speculative_mtp_state_encode(bad_pending, blob2));
}

void test_position_guard() {
    // The carry is the hidden row of the last processed token, so a checkpoint
    // covering that position must accept it. Regression this guards: the check
    // was made against the live memory's pos_max instead of the checkpoint's,
    // and at restore time the memory still held the previous request's cells
    // (858) above the checkpoint (55). Every Gemma restore was rejected, the
    // carry stayed stale, and repeats drifted.
    expect_true("accept carry at the checkpoint's last position",
                common_speculative_mtp_state_position_ok(55, 55));
    expect_true("reject carry from another prefix",
                !common_speculative_mtp_state_position_ok(55, 47));
    expect_true("reject carry from a later prefix",
                !common_speculative_mtp_state_position_ok(55, 1063));
    expect_true("reject a checkpoint that covers nothing",
                !common_speculative_mtp_state_position_ok(0, -1));
}

} // namespace

int main() {
    test_round_trip();
    test_round_trip_empty_carry();
    test_rejects_foreign_width();
    test_rejects_truncation();
    test_rejects_wrong_version();
    test_rejects_inconsistent_shape();
    test_position_guard();

    if (g_fails != 0) {
        printf("\n%d check(s) failed\n", g_fails);
        return 1;
    }

    printf("\nall checks passed\n");
    return 0;
}
