# TASK-066 — Upstream / fork PR audit registry

**Updated:** 2026-11-07  
**Index:** [README-TASK066.md](README-TASK066.md)  
**Artifacts:** `logs/upstream-prs/{pr28164,pr31688,pr30100,pr27781}.diff`  
**Product findings:** ladder confirm + C2 + FA ablation (below).

## Frozen product findings (2026-11-06/07)

| Stamp / gate | Result |
|--------------|--------|
| Ladder `212703Z` (`MUL_MM_MIN=7`) | HELP first-DIFF 80→182; still DIFF@text |
| `SUITE=ubatch-s8-s1024` C1 | RED `first_token_diff=0` |
| Localize H6 n=16 / H7 n=32 | GREEN / **RED** (multi-step ubatch vs one-shot) |
| C2 (=H7) | RED ~4s; primary TDD gate |
| `MUL_MM_MIN` ∈ {1,7,32} on C2 | NO_EFFECT (same argmax) |
| FA ablation `215517Z` | A4 f16+FA-on soft RED; A0 turbo4 decision RED; A1 FA-on+f16 GREEN |
| Agreed read | Base error **inside FA** under multi-step ubatch; turbo4 **amplifies** |
| Web research | [task066-hist/TASK-066-fa-batch-invariance-web-research.md](task066-hist/TASK-066-fa-batch-invariance-web-research.md) |

## PR / patch matrix (fidelity vs original intent)

| ID | Role | Fork status | Fidelity check (2026-11-07) | Autotest gate |
|----|------|-------------|------------------------------|---------------|
| **#28164** | Fuse table + `fusion_max` | ✅ Ported (`ggml-metal-fusion.{h,cpp}`) | MATCH table; empty-ne filter aligned in **066.3** (`ggml_is_empty` same as `pr28164.diff`) | `SUITE=cb-fusion` |
| **#31688** | CB `cb_start[]` + `i += fusion_max` | ✅ Present | MATCH vs `pr31688.diff` walk / encode ranges; old `n_nodes_0/1` gone | `SUITE=cb-fusion` I2 |
| **#30100** | MUL_MAT+ADD when residual is MUL_MAT | ⏸ N/A | No few-row MMA+ADD encode (`use_mma` stub false); recheck [p22-30100](task066-hist/TASK-066-p22-30100-recheck.md) | — |
| **#27781** | Shared KV cells API | ✅ Ported | `SUITE=b1-shared-kv` GREEN (status matrix) | `SUITE=b1-shared-kv` |
| **#25618 FA-vec nsg** | Pin `nsg` so KV-pad boundaries don’t change reduction | ✅ Applied on **vec** path | `ggml-metal-ops.cpp` FA-vec: `nsg = 1` (skoulik recipe). **Tile FA** still `nsg = ne00>=512 ? 8 : 4` — **not** in #25618 comment; C2 likely hits tile → still RED | `SUITE=fa` + **C2** |
| **#25618 mul_mv_ext** | Disable ext for ne11∈[2,8] | ✅ Applied | `if (false && … mul_mv_ext)` — forced off | `SUITE=mul-mat` |
| **#27441** skinny mm | Small-batch mm tiles | ❌ CLOSED | Port REGRESS `175708Z`, reverted; do not blind-reapply | residual / ladder |
| **066.7** float-tile + `MUL_MM_MIN=7` | Unify S8+ on mm | 🟢 KEEP | Orthogonal to C2 (levers NO_EFFECT) | `SUITE=mul-mat-residual` |
| **#26358** (Vulkan) | No GQA pack for multi-Q | ⬜ Not Metal-ported | Research only; check Metal packing gates if C2 stays RED after tile invariance | — |
| **TM / vLLM** fixed split-KV | Batch-invariant FA | ⬜ Not ported | Prefill ubatch class = chunked vs one-shot | C2 / A4 |

## Gaps (not “failed apply”)

1. **FA-vec nsg pin ≠ FA-tile invariance** — #25618 recipe applied; product C2 residual remains → next work is tile / turbo FA path, not re-litigate vec pin.
2. **Encoder still hand-rolls fuse** vs `fusion_next` — 066.1 PARTIAL; 066.5 active.
3. **#30100 / #27441** — correctly deferred / closed; do not reopen without new RED hypothesis.

## Re-audit protocol

After **any** Metal/ops/fusion edit:

```bash
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=fa ./tests/run-task066-metal-corners.sh
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  SUITE=ubatch-s8-s1024 ./tests/run-task066-metal-corners.sh   # expect C2 RED until FA fix
```

Log HELP / NO_EFFECT / REGRESS; never ε-PASS residual or C2.

## Baseline after fidelity audit (no code edits)

**Stamp:** `logs/task066-pr-audit-baseline-20261106T230631Z`

| Suite | rc | Verdict |
|-------|----|---------|
| `cb-fusion` | 0 | GREEN (#28164/#31688 still hold) |
| `mul-mat-residual` | 0 | GREEN (066.7 KEEP) |
| `fa` | 0 | GREEN (vec nsg pin corners; does **not** cover C2 prefill) |
| `ubatch-s8-s1024` C2 | 1 | **RED** expected — FA multi-step class still open |

**Conclusion:** Prior ports match saved upstream diffs / #25618 vec recipe. No silent drift from “applied patches.” Next code change = FA-tile / multi-step invariance with C2 RED→GREEN; re-run this protocol after each edit.
