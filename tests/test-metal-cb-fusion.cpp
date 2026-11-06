// TASK-066.2 / ggml-org/llama.cpp#31688 + #28164 — Metal CB split vs fuse table.
//
// Invariants (see tests/TASK-066.1-audit-notes.md + TASK-066.2-compose-autotests.md):
//   I1  fusion_max spans encoder-relevant fuse groups (ADD / RMS_NORM+MUL+ADD / SNAKE).
//   I2  CB align walk never lands inside such a group (same loop as ggml-metal-context.m).
//   I3  Empty-ne tensors: fusion_max matches upstream pr28164 (ggml_is_empty => pack alone).
//   I4  MUL_MAT+ADD: STRUCTURAL may pack 2 for CB; FULL/encoder use_mma is false in this fork
//       (do not claim encoder fuse).
//
// Built only with GGML_METAL and static backends (needs ggml_metal_fusion_max).
// No GPU required.

#include "ggml.h"
#include "ggml-cpp.h"

#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-metal/ggml-metal-common.h"
#include "../ggml/src/ggml-metal/ggml-metal-fusion.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool is_encode_node(const ggml_tensor * t) {
    return t != nullptr && !ggml_op_is_empty(t->op) && !ggml_is_empty(t);
}

// Structural ADD-fuse end (exclusive graph index), matching ggml_metal_op_bin
// without the Metal src[1] buffer-id check (over-count OK; under-count is not).
int ref_add_fuse_end(const ggml_cgraph * gf, int idx) {
    if (!gf || idx < 0 || idx >= gf->n_nodes || !gf->nodes[idx]) {
        return idx + 1;
    }
    if (!is_encode_node(gf->nodes[idx]) || gf->nodes[idx]->op != GGML_OP_ADD) {
        return idx + 1;
    }

    std::vector<int> idxs;
    idxs.reserve(size_t(gf->n_nodes - idx));
    for (int i = idx; i < gf->n_nodes; ++i) {
        if (is_encode_node(gf->nodes[i])) {
            idxs.push_back(i);
        }
    }
    if (idxs.empty() || idxs[0] != idx) {
        return idx + 1;
    }

    enum ggml_op fops[8];
    for (int k = 0; k < 8; ++k) {
        fops[k] = GGML_OP_ADD;
    }

    int n_fuse = 1;
    for (int n = 0; n <= 6; ++n) {
        if (n + 1 >= (int) idxs.size()) {
            break;
        }
        const int pair[2] = { idxs[n], idxs[n + 1] };
        if (!ggml_can_fuse_ext(gf, pair, fops + n, 2)) {
            break;
        }
        struct ggml_tensor * f0 = gf->nodes[idxs[n]];
        struct ggml_tensor * f1 = gf->nodes[idxs[n + 1]];
        if (!f0 || !f1 || f0 != f1->src[0]) {
            break;
        }
        if (!f0->src[1] || !f1->src[1] || !ggml_are_same_layout(f0->src[1], f1->src[1])) {
            break;
        }
        n_fuse = n + 2;
    }

    return idxs[n_fuse - 1] + 1;
}

int ref_fusion_max(const ggml_cgraph * gf, int idx) {
    return ref_add_fuse_end(gf, idx) - idx;
}

// Same walk as ggml_metal_graph_compute CB boundary align.
int align_past_target(const ggml_cgraph * gf, int target) {
    int i = 0;
    while (i < target) {
        const int step = ggml_metal_fusion_max(gf, i);
        if (step < 1) {
            return -1;
        }
        i += step;
    }
    return i;
}

struct graph_holder {
    ggml_context_ptr ctx;
    ggml_cgraph * gf = nullptr;
};

graph_holder make_ctx(int n_nodes_cap = 128) {
    graph_holder h;
    const size_t mem = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(n_nodes_cap, false) + 64 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    h.ctx.reset(ggml_init(params));
    h.gf = ggml_new_graph_custom(h.ctx.get(), n_nodes_cap, false);
    return h;
}

int find_node(const ggml_cgraph * gf, const ggml_tensor * t) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }
    return -1;
}

bool check(bool ok, int & fails, const char * name, const std::string & detail) {
    if (ok) {
        printf("ok   %s\n", name);
        return true;
    }
    printf("FAIL %s: %s\n", name, detail.c_str());
    ++fails;
    return false;
}

// ADD, RESHAPE (empty op), ADD with a1->src[0] == a0 — encoder fuses across reshape.
bool case_add_view_add(int & fails) {
    constexpr int64_t n = 64;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b0    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b1    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * other = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    ggml_tensor * a0 = ggml_add(ctx, x, b0);
    ggml_tensor * v  = ggml_reshape_1d(ctx, other, n);
    ggml_tensor * a1 = ggml_add(ctx, a0, b1);

    ggml_build_forward_expand(gf, a0);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, a1);

    const int i0 = find_node(gf, a0);
    const int iv = find_node(gf, v);
    const int i1 = find_node(gf, a1);
    if (i0 < 0 || iv < 0 || i1 < 0 || !(i0 < iv && iv < i1)) {
        return check(false, fails, "add_view_add.topology",
                     "expected node order ADD, RESHAPE, ADD (got i0=" + std::to_string(i0) +
                     " iv=" + std::to_string(iv) + " i1=" + std::to_string(i1) + ")");
    }

    const int want = ref_fusion_max(gf, i0);
    const int got  = ggml_metal_fusion_max(gf, i0);
    check(want > 1 && want == (i1 - i0 + 1), fails, "add_view_add.ref_span",
          "ref span want=" + std::to_string(i1 - i0 + 1) + " got_ref=" + std::to_string(want));
    check(got == want, fails, "add_view_add.fusion_max",
          "fusion_max=" + std::to_string(got) + " want=" + std::to_string(want) +
          " (must skip RESHAPE between fused ADDs)");

    // Any target inside the group must align to the exclusive end.
    const int end = i1 + 1;
    for (int target = i0 + 1; target <= i1; ++target) {
        const int aligned = align_past_target(gf, target);
        check(aligned >= end, fails,
              ("add_view_add.align_t" + std::to_string(target)).c_str(),
              "aligned=" + std::to_string(aligned) + " want>=" + std::to_string(end));
    }
    return fails == 0;
}

// Pure consecutive ADD chain. Upstream table ADD_CHAIN covers 2..7 ADDs (#28164);
// longer chains pack as 7 then leave a tail (over-count within the first group is OK).
bool case_consecutive_adds(int & fails) {
    for (int n_add = 1; n_add <= 8; ++n_add) {
        constexpr int64_t n = 32;
        auto h = make_ctx(64);
        ggml_context * ctx = h.ctx.get();
        ggml_cgraph * gf = h.gf;

        ggml_tensor * acc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor * first_add = nullptr;
        for (int i = 0; i < n_add; ++i) {
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            acc = ggml_add(ctx, acc, b);
            if (i == 0) {
                first_add = acc;
            }
        }
        ggml_build_forward_expand(gf, acc);

        const int i0 = find_node(gf, first_add);
        if (i0 < 0) {
            check(false, fails, ("consec_n" + std::to_string(n_add) + ".find").c_str(), "first ADD missing");
            continue;
        }

        // Table max ADD fuse is 7; fusion_max may also chain further patterns.
        const int want_min = n_add == 1 ? 1 : (n_add <= 7 ? n_add : 7);
        const int got      = ggml_metal_fusion_max(gf, i0);
        check(got >= want_min, fails, ("consec_n" + std::to_string(n_add) + ".fusion_max").c_str(),
              "fusion_max=" + std::to_string(got) + " want>=" + std::to_string(want_min));

        if (n_add > 1) {
            const int end = i0 + want_min;
            for (int target = i0 + 1; target < end; ++target) {
                const int aligned = align_past_target(gf, target);
                check(aligned >= end, fails,
                      ("consec_n" + std::to_string(n_add) + ".align_t" + std::to_string(target)).c_str(),
                      "aligned=" + std::to_string(aligned) + " want>=" + std::to_string(end));
            }
        }
    }
    return true;
}

// Longer chain interrupted by a non-fusable op: fuse only the prefix, then resume.
bool case_fuse_then_mul(int & fails) {
    constexpr int64_t n = 32;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * s  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    ggml_tensor * a0  = ggml_add(ctx, x, b0);
    ggml_tensor * a1  = ggml_add(ctx, a0, b1);
    ggml_tensor * mul = ggml_mul(ctx, a1, s);
    ggml_tensor * a2  = ggml_add(ctx, mul, b2);

    ggml_build_forward_expand(gf, a2);

    const int i0 = find_node(gf, a0);
    const int i2 = find_node(gf, a2);
    if (i0 < 0 || i2 < 0) {
        return check(false, fails, "fuse_then_mul.topology", "missing ADD nodes");
    }

    const int want0 = ref_fusion_max(gf, i0);
    const int got0  = ggml_metal_fusion_max(gf, i0);
    check(want0 == 2, fails, "fuse_then_mul.ref_prefix", "ref=" + std::to_string(want0));
    check(got0 == want0, fails, "fuse_then_mul.fusion_max_prefix",
          "fusion_max=" + std::to_string(got0) + " want=" + std::to_string(want0));

    const int want2 = ref_fusion_max(gf, i2);
    const int got2  = ggml_metal_fusion_max(gf, i2);
    check(want2 == 1 && got2 == 1, fails, "fuse_then_mul.tail_add",
          "tail fusion_max=" + std::to_string(got2));
    return true;
}

// Metal also fuses RMS_NORM -> MUL -> ADD (ggml_metal_op_norm). fusion_max must
// cover that group so a CB split cannot bisect it (follow-up Low).
bool case_norm_mul_add(int & fails) {
    constexpr int64_t n = 64;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    ggml_tensor * rn  = ggml_rms_norm(ctx, x, 1e-5f);
    ggml_tensor * mul = ggml_mul(ctx, rn, w);
    ggml_tensor * add = ggml_add(ctx, mul, b);

    ggml_build_forward_expand(gf, add);

    const int i0 = find_node(gf, rn);
    const int i1 = find_node(gf, mul);
    const int i2 = find_node(gf, add);
    if (i0 < 0 || i1 < 0 || i2 < 0 || !(i0 + 1 == i1 && i1 + 1 == i2)) {
        return check(false, fails, "norm_mul_add.topology",
                     "expected consecutive RMS_NORM, MUL, ADD");
    }

    // Encoder fuses up to 3 ops; CB helper must return the same graph span.
    const int want = 3;
    const int got  = ggml_metal_fusion_max(gf, i0);
    check(got == want, fails, "norm_mul_add.fusion_max",
          "fusion_max=" + std::to_string(got) + " want=" + std::to_string(want));

    const int end = i0 + want;
    for (int target = i0 + 1; target < end; ++target) {
        const int aligned = align_past_target(gf, target);
        check(aligned >= end, fails,
              ("norm_mul_add.align_t" + std::to_string(target)).c_str(),
              "aligned=" + std::to_string(aligned) + " want>=" + std::to_string(end));
    }
    return true;
}

// Same NORM fuse group with an empty op between RMS_NORM and MUL (filtered idxs).
bool case_norm_view_mul_add(int & fails) {
    constexpr int64_t n = 64;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * other = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    ggml_tensor * rn  = ggml_rms_norm(ctx, x, 1e-5f);
    ggml_tensor * v   = ggml_reshape_1d(ctx, other, n);
    ggml_tensor * mul = ggml_mul(ctx, rn, w);
    ggml_tensor * add = ggml_add(ctx, mul, b);

    ggml_build_forward_expand(gf, rn);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, add);

    const int i0 = find_node(gf, rn);
    const int iv = find_node(gf, v);
    const int i2 = find_node(gf, add);
    if (i0 < 0 || iv < 0 || i2 < 0 || !(i0 < iv && iv < i2)) {
        return check(false, fails, "norm_view_mul_add.topology",
                     "expected RMS_NORM, RESHAPE, ... ADD order");
    }

    const int want = i2 - i0 + 1; // span through final ADD, including empty gap
    const int got  = ggml_metal_fusion_max(gf, i0);
    check(got == want, fails, "norm_view_mul_add.fusion_max",
          "fusion_max=" + std::to_string(got) + " want=" + std::to_string(want));

    const int end = i2 + 1;
    for (int target = i0 + 1; target <= i2; ++target) {
        const int aligned = align_past_target(gf, target);
        check(aligned >= end, fails,
              ("norm_view_mul_add.align_t" + std::to_string(target)).c_str(),
              "aligned=" + std::to_string(aligned) + " want>=" + std::to_string(end));
    }
    return true;
}

// A4 / encoder: SNAKE autofuse (MUL,SIN,SQR,MUL,ADD) via ggml_metal_op_bin.
// fusion_max must span all 5 so a CB split cannot bisect the group.
bool case_snake(int & fails) {
    constexpr int64_t D = 64;
    constexpr int64_t C = 32;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, C);
    ggml_tensor * a     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
    ggml_tensor * inv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);

    ggml_tensor * mul0 = ggml_mul(ctx, x, a);
    ggml_tensor * sn   = ggml_sin(ctx, mul0);
    ggml_tensor * sq   = ggml_sqr(ctx, sn);
    ggml_tensor * mul1 = ggml_mul(ctx, sq, inv_b);
    ggml_tensor * add  = ggml_add(ctx, mul1, x);

    ggml_build_forward_expand(gf, add);

    const int i0 = find_node(gf, mul0);
    const int i4 = find_node(gf, add);
    if (i0 < 0 || i4 < 0 || i4 != i0 + 4) {
        return check(false, fails, "snake.topology",
                     "expected consecutive MUL,SIN,SQR,MUL,ADD (i0=" + std::to_string(i0) +
                     " i4=" + std::to_string(i4) + ")");
    }

    const int want = 5;
    const int got  = ggml_metal_fusion_max(gf, i0);
    check(got == want, fails, "snake.fusion_max",
          "fusion_max=" + std::to_string(got) + " want=" + std::to_string(want));

    const int end = i0 + want;
    for (int target = i0 + 1; target < end; ++target) {
        const int aligned = align_past_target(gf, target);
        check(aligned >= end, fails,
              ("snake.align_t" + std::to_string(target)).c_str(),
              "aligned=" + std::to_string(aligned) + " want>=" + std::to_string(end));
    }
    return true;
}

// I4: STRUCTURAL CB packing for MUL_MAT+ADD (may_use_mma) vs FULL encoder (use_mma==false).
bool case_mul_mat_add(int & fails) {
    constexpr int64_t K = 64;
    constexpr int64_t M = 32;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, M);

    ggml_tensor * mm  = ggml_mul_mat(ctx, w, x);
    ggml_tensor * out = ggml_add(ctx, mm, bias);

    ggml_build_forward_expand(gf, out);

    const int i0 = find_node(gf, mm);
    const int i1 = find_node(gf, out);
    if (i0 < 0 || i1 < 0 || i1 != i0 + 1) {
        return check(false, fails, "mul_mat_add.topology",
                     "expected consecutive MUL_MAT, ADD");
    }

    check(ggml_metal_op_mul_mat_may_use_mma(mm), fails, "mul_mat_add.may_use_mma",
          "STRUCTURAL packing expects may_use_mma for F32 src1");
    check(!ggml_metal_op_mul_mat_use_mma(mm), fails, "mul_mat_add.use_mma_false",
          "fork stub: FULL/encoder must NOT claim MMA+ADD fuse (066.1 PARTIAL)");

    const int want = 2; // STRUCTURAL over-pack for CB align
    const int got  = ggml_metal_fusion_max(gf, i0);
    check(got == want, fails, "mul_mat_add.fusion_max_structural",
          "fusion_max=" + std::to_string(got) + " want=" + std::to_string(want) +
          " (CB STRUCTURAL pack; not encoder FULL fuse)");

    const int aligned = align_past_target(gf, i1);
    check(aligned >= i1 + 1, fails, "mul_mat_add.align",
          "aligned=" + std::to_string(aligned) + " want>=" + std::to_string(i1 + 1));
    return true;
}

// I3 / Gap A (066.1): upstream pr28164 fusion_max returns 1 when the start node is
// ggml_is_empty. A lone empty ADD is ambiguous (both paths return 1). Two chained
// empty ADDs expose the fork filter gap: without ggml_is_empty, STRUCTURAL ADD_CHAIN
// can pack 2; upstream packs the empty start alone.
bool case_empty_ne_start(int & fails) {
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
    ggml_tensor * b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
    ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
    ggml_tensor * a0 = ggml_add(ctx, x, b0);
    ggml_tensor * a1 = ggml_add(ctx, a0, b1);
    ggml_build_forward_expand(gf, a1);

    const int i0 = find_node(gf, a0);
    const int i1 = find_node(gf, a1);
    if (i0 < 0 || i1 < 0) {
        return check(false, fails, "empty_ne_start.find", "ADD nodes missing");
    }
    if (!ggml_is_empty(a0) || !ggml_is_empty(a1)) {
        return check(false, fails, "empty_ne_start.is_empty",
                     "fixture must be ggml_is_empty for Gap A");
    }
    if (ggml_op_is_empty(a0->op)) {
        return check(false, fails, "empty_ne_start.op",
                     "fixture needs non-empty op (ADD) with empty shape");
    }

    const int got = ggml_metal_fusion_max(gf, i0);
    check(got == 1, fails, "empty_ne_start.fusion_max",
          "fusion_max=" + std::to_string(got) +
          " want=1 (upstream pr28164: ggml_is_empty start => pack alone; "
          "got>1 means fork omitted empty filter / ADD_CHAIN over-pack)");
    GGML_UNUSED(i1);
    return true;
}

// I1: table STRUCTURAL match for RMS_NORM+MUL+ADD via ggml_metal_fusion_next.
bool case_fusion_next_norm_mul_add(int & fails) {
    constexpr int64_t n = 64;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    ggml_tensor * rn  = ggml_rms_norm(ctx, x, 1e-5f);
    ggml_tensor * mul = ggml_mul(ctx, rn, w);
    ggml_tensor * add = ggml_add(ctx, mul, b);
    ggml_build_forward_expand(gf, add);

    const int i0 = find_node(gf, rn);
    if (i0 < 0) {
        return check(false, fails, "fusion_next_norm.find", "RMS_NORM missing");
    }

    int idxs[8];
    int n_idxs = 0;
    for (int i = i0; i < gf->n_nodes && n_idxs < 8; ++i) {
        if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
            idxs[n_idxs++] = i;
        }
    }
    int len = 0;
    const ggml_metal_fusion * fus =
        ggml_metal_fusion_next(gf, idxs, n_idxs, 0, GGML_METAL_FUSION_STRUCTURAL, &len);
    check(fus != nullptr && len == 3, fails, "fusion_next_norm.len",
          "len=" + std::to_string(len) + " fus=" + std::to_string(fus != nullptr));
    if (fus) {
        check(ggml_metal_fusion_get_id(fus) == GGML_METAL_FUSION_NORM_MUL_ADD, fails,
              "fusion_next_norm.id", "unexpected fusion id");
    }

    const int got = ggml_metal_fusion_max(gf, i0);
    check(got == 3, fails, "fusion_next_norm.fusion_max",
          "fusion_max=" + std::to_string(got) + " want=3");
    return true;
}

// Two ADD groups separated by empty op: each group aligned independently.
bool case_two_groups_with_gap(int & fails) {
    constexpr int64_t n = 32;
    auto h = make_ctx();
    ggml_context * ctx = h.ctx.get();
    ggml_cgraph * gf = h.gf;

    ggml_tensor * x0    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b0    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b1    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * other = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * x1    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b2    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b3    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    ggml_tensor * g0a = ggml_add(ctx, x0, b0);
    ggml_tensor * g0b = ggml_add(ctx, g0a, b1);
    ggml_tensor * gap = ggml_reshape_1d(ctx, other, n);
    ggml_tensor * g1a = ggml_add(ctx, x1, b2);
    ggml_tensor * g1b = ggml_add(ctx, g1a, b3);

    ggml_build_forward_expand(gf, g0b);
    ggml_build_forward_expand(gf, gap);
    ggml_build_forward_expand(gf, g1b);

    const int i0 = find_node(gf, g0a);
    const int i1 = find_node(gf, g1a);
    if (i0 < 0 || i1 < 0) {
        return check(false, fails, "two_groups.topology", "missing group starts");
    }

    // Table-backed fusion_max may chain adjacent ADD groups across empty ops (over-count).
    // That is OK for CB align; only require each group's exclusive end is cleared.
    const int got0 = ggml_metal_fusion_max(gf, i0);
    const int got1 = ggml_metal_fusion_max(gf, i1);
    check(got0 >= 2, fails, "two_groups.g0",
          "got=" + std::to_string(got0) + " want>=2");
    check(got1 >= 2, fails, "two_groups.g1",
          "got=" + std::to_string(got1) + " want>=2");

    const int end0 = find_node(gf, g0b) + 1;
    const int end1 = find_node(gf, g1b) + 1;
    for (int target = i0 + 1; target < end0; ++target) {
        const int aligned = align_past_target(gf, target);
        check(aligned >= end0, fails,
              ("two_groups.align_g0_t" + std::to_string(target)).c_str(),
              "aligned=" + std::to_string(aligned) + " end0=" + std::to_string(end0));
    }
    for (int target = i1 + 1; target < end1; ++target) {
        const int aligned = align_past_target(gf, target);
        check(aligned >= end1, fails,
              ("two_groups.align_g1_t" + std::to_string(target)).c_str(),
              "aligned=" + std::to_string(aligned) + " end1=" + std::to_string(end1));
    }
    GGML_UNUSED(gap);
    return true;
}

} // namespace

int main() {
    printf("CONTRACT TASK-066.2: I1 encoder-class spans; I2 CB align; "
           "I3 empty-ne==upstream pack-alone; I4 MUL_MAT+ADD STRUCTURAL pack / use_mma false\n");

    int fails = 0;
    case_add_view_add(fails);
    case_consecutive_adds(fails);
    case_fuse_then_mul(fails);
    case_two_groups_with_gap(fails);
    case_norm_mul_add(fails);
    case_norm_view_mul_add(fails);
    case_snake(fails);
    case_mul_mat_add(fails);
    case_empty_ne_start(fails);
    case_fusion_next_norm_mul_add(fails);

    if (fails != 0) {
        printf("FAILED: %d check(s) — RED until 066.3 fixes (see TASK-066.1-audit-notes)\n", fails);
        return 1;
    }
    printf("OK: metal CB fusion align (TASK-066.2)\n");
    return 0;
}
