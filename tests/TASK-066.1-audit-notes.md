# TASK-066.1 audit notes — #31688 CB align + #28164 fusion_max

**Date:** 2026-11-06  
**Branch:** `task066-metal-batch-invariance`  
**Artifacts:** `logs/upstream-prs/pr31688.diff`, `pr28164.diff`, `upstream-ggml-metal-context.m`, local `ggml-metal-fusion.{h,cpp}`  
**Verdict:** **PARTIAL**

| Surface | Verdict | Notes |
|---------|---------|--------|
| #31688 `cb_start` walk in `ggml-metal-context.m` | **MATCH** | Same algorithm as PR diff |
| #31688 encode ranges via `cb_start[]` | **MATCH** | `idx_end = cb_start[0]`; extra CB `[cb_start[i], cb_start[i+1])` |
| Old `n_nodes_0/1/n_nodes_per_cb` | **GONE** | Replaced; comments cite TASK-066 / #31688 |
| `ggml-metal-fusion.{h,cpp}` vs `logs/upstream-prs/ggml-metal-fusion.*` | **MATCH** | `diff` empty (surgical copy in tree == saved copy) |
| `ggml_metal_fusion_max` vs **pr28164.diff** body | **PARTIAL → fixed in 066.3** | Empty-tensor filter differed; HELP in 066.3 |
| ADD-only heuristic (`58d1940`) | **STALE** | Removed; table-backed `fusion_max` at HEAD |
| MMA stubs / encoder FULL `MUL_MAT+ADD` | **PARTIAL** | Intentional fork gap |
| Encoder uses shared `ggml_metal_fusion_next` | **PARTIAL** | Ops still hand-roll many fuse paths |

No model ladder in this subtask. Next: [TASK-066.2-compose-autotests.md](TASK-066.2-compose-autotests.md) — harden tests for the PARTIAL gaps below.

---

## 1. #31688 CB walk — MATCH

Fork `ggml/src/ggml-metal/ggml-metal-context.m`:

| Item | Evidence |
|------|----------|
| Struct field | L59–61: `cb_start[GGML_METAL_MAX_COMMAND_BUFFERS + 1]` |
| Boundary walk | L468–481: `n_rest`, `n_per_cb`, `while (i < target) i += ggml_metal_fusion_max(gf, i)`, `cb_start[cb_idx]`, `cb_start[n_cb] = gf->n_nodes` |
| Encode block | L696–701: `idx_end = cb_start[0]`; if `cb_idx < n_cb_l` use `cb_start[cb_idx]` / `cb_start[cb_idx+1]` |
| `n_cb == 0` | Loop body skipped; `cb_start[0] = gf->n_nodes` via `cb_start[n_cb]` — full graph on main CB |

Matches `logs/upstream-prs/pr31688.diff` hunks (struct + walk + encode). Minor comment wording only.

**Not a #31688 miss:** the CB *integration* is present. Residual S8 vs S1024 is not explained by “CB walk never landed”.

---

## 2. #28164 fusion table — MATCH copy, PARTIAL vs PR diff filter

- Local `ggml-metal-fusion.cpp` (1337 lines) is **byte-identical** to `logs/upstream-prs/ggml-metal-fusion.cpp` (the surgical port artifact).
- Table includes: NORM/RMS chains, ADD_CHAIN 2..7, SNAKE, MoE/TOPK, `MUL_MAT+ADD`, etc.
- `ggml_metal_fusion_max` calls `ggml_metal_fusion_next(..., GGML_METAL_FUSION_STRUCTURAL, ...)`.

### Gap A — empty-tensor filtering vs `pr28164.diff`

Upstream PR body (`pr28164.diff`):

```c
if (ggml_op_is_empty(gf->nodes[idx]->op) || ggml_is_empty(gf->nodes[idx])) {
    return 1;
}
// idxs: !ggml_op_is_empty(op) && !ggml_is_empty(tensor)
```

Fork / saved copy (`ggml-metal-fusion.cpp` ~1298+):

```c
if (ggml_op_is_empty(gf->nodes[idx]->op)) {
    return 1;
}
// idxs: !ggml_op_is_empty(op) only — 0-element tensors included
```

**Risk:** CB packing / fusion_max span can differ from upstream on graphs with empty tensors. Autotests in 066.2 should cover empty-tensor nodes if product graphs hit them.

**Note:** `logs/upstream-prs/ggml-metal-fusion.cpp` is **not** a pristine extract of the PR hunk filter — it matches the fork adaptation. Always compare to `pr28164.diff` for upstream truth.

---

## 3. MMA stubs — PARTIAL (documented)

`ggml-metal-common.cpp` L459–468:

- `ggml_metal_op_mul_mat_may_use_mma` — true when `src[1]` is F32 (STRUCTURAL packing can include `MUL_MAT+ADD`).
- `ggml_metal_op_mul_mat_use_mma` — **always false** (FULL / encoder never fuses few-row MMA+ADD).

`ggml_metal_fusion_check_mul_mat_add`: STRUCTURAL can return true; FULL requires `use_mma` → false.

**Implication for #31688:** CB may *keep* MUL_MAT+ADD in one buffer (over-pack), while encoder still runs them unfused. That avoids mid-group CB splits for a fuse the encoder does not do; it does **not** create fused vs unfused rounding for that pattern. Real #31688 risk is patterns the **encoder does fuse** (ADD chain, NORM, SNAKE, …) if `fusion_max` under-counts them.

`#30100` encode path still N/A (hist recheck). Tracked under 066.5 / 066.7 if needed.

---

## 4. Encoder vs table — PARTIAL

Header claims encoders use `ggml_metal_fusion_next` as shared SoT. Fork `ggml-metal-ops.cpp` still uses local helpers (`can_fuse`, `ggml_metal_op_can_fuse_snake`, ADD/NORM hand loops) rather than routing all fuse decisions through `ggml_metal_fusion_next` / fusion ids.

**Risk for 066.2–066.5:** `fusion_max` STRUCTURAL span ≠ what the encoder actually fuses → CB align can over- or under-pack relative to encode. Autotests must assert CB walk vs **encoder-equivalent** fuse ends, not only table STRUCTURAL over-count.

---

## 5. Draft ADD-only criticism — STALE

Commit `58d1940` ADD-chain heuristic is gone. HEAD uses table + `fusion_max` (`16044ea6a` era). Do not re-litigate “port the table” as if missing; focus on Gap A + encoder parity + MMA stubs.

---

## 6. Existing `SUITE=cb-fusion` (input to 066.2)

`tests/test-metal-cb-fusion.cpp` already covers more than ADD:

- ADD + view + ADD, consecutive ADDs, fuse-then-MUL, two groups, RMS_NORM+MUL+ADD (+ view), SNAKE, MUL_MAT+ADD

**Weaknesses for 066.2:**

- Primary `ref_fusion_max` helper is still ADD-oriented for several cases; MUL_MAT+ADD can pass via STRUCTURAL over-count even though encoder `use_mma==false`.
- No case for Gap A (`ggml_is_empty` tensor filter vs upstream).
- No RED test that fails if encoder fuse length ≠ `fusion_max` for a pattern ops actually fuses.
- Comments still point at deleted `TASK-066-cb-fusion-followup.md` (now under `task066-hist/`).

---

## Recommendations → numbered track

| Follow-up | Action |
|-----------|--------|
| **066.2** | Autotests: (1) empty-tensor filter parity or explicit fork WAIVE with reason; (2) encoder-equivalent fuse end vs `fusion_max` for ADD/NORM/SNAKE; (3) MUL_MAT+ADD: document STRUCTURAL over-pack vs FULL non-fuse so suite does not false-green “encoder fuse”. |
| **066.3** | ✅ Align `fusion_max` empty filter with `pr28164.diff` (HELP; suite PASS). |
| **066.5** | Unify encoder fuse paths onto table / `fusion_next` where still hand-rolled. |
| **066.7** | MMA/`#30100` only if residual remains after CB/fuse gates. |

---

## Checklist (066.1 DoD)

- [x] Diff CB block vs pr31688 / upstream context
- [x] Verify cb_start math / fusion_max advance / encode ranges / old path gone
- [x] Diff fusion_max + table vs #28164 artifacts (copy + PR filter)
- [x] Call out STRUCTURAL/FULL stubs / encoder gaps
- [x] Note ADD-only draft criticism is STALE
- [x] This notes file written; verdict **PARTIAL**
