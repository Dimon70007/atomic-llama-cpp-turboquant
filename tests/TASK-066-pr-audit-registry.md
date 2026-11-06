# TASK-066 — Upstream / fork PR audit registry

**Updated:** 2026-11-07 (FA always-tile)  
**Index:** [README-TASK066.md](README-TASK066.md)  
**Artifacts:** `logs/upstream-prs/{pr28164,pr31688,pr30100,pr27781}.diff`  
**Product findings:** ladder confirm + C2 + FA ablation (below).

## Frozen product findings (2026-11-06/07)

| Stamp / gate | Result |
|--------------|--------|
| Ladder `212703Z` (`MUL_MM_MIN=7`) | HELP first-DIFF 80→182; still DIFF@text |
| Localize H6 n=16 / H7 n=32 | GREEN / was RED (multi-step ubatch vs one-shot) |
| `MUL_MM_MIN` ∈ {1,7,32} on C2 | NO_EFFECT (pre-fix); orthogonal to FA dispatch |
| FA ablation `215517Z` (pre-fix) | A4 f16+FA-on soft RED; A0 turbo4 decision RED; A1 FA-on+f16 GREEN |
| Agreed read (pre-fix) | Base error **inside FA** under multi-step ubatch; turbo4 **amplifies** |
| Web research | [task066-hist/TASK-066-fa-batch-invariance-web-research.md](task066-hist/TASK-066-fa-batch-invariance-web-research.md) |

### FA always-tile fix (2026-11-07)

**Root cause:** `ggml_metal_op_flash_attn_ext_use_vec` gated on `ne01 < 20` → product `ub=8` (ne01=8) took **FA-vec**, `ub=32/1024` took **FA-tile**. H6 n=16 both-vec GREEN; H7/C2 n=32 mixed RED.

**Lever:** default **always tile**; legacy shape-vec only with `GGML_METAL_FA_VEC=1`. Dispatch log: `GGML_METAL_FA_DISPATCH=1`.

| Stamp / gate | Result |
|--------------|--------|
| C7 autotest (FA-op multi-step) legacy `FA_VEC=1` | **RED** n=32 (fails=3); n=16 GREEN — `logs/task066-c7-legacy-vec-*` |
| C7 + `SUITE=fa` default (always-tile) | **GREEN** — `logs/task066-c7-always-tile-*` / regression `231654Z` |
| `SUITE=ubatch-s8-s1024` **C2** | **GREEN** bit-exact (`exact=1`) — `logs/task066-c2-after-tile-*` |
| C1 greedy S8 vs S1024 | still **RED**; HELP `first_token_diff` 0→**8** |
| Ladder `231804Z` (post-tile) | still DIFF@text; S1024 vs S8 first-DIFF **182→329** (HELP vs `212703Z`) |
| FA ablation after fix | **A0–A4 all exact=1** (incl. product turbo4) — `logs/task066-regression-after-tile-20261106T231654Z` |
| Regression pack | `cb-fusion` / `mul-mat-residual` / `fa` **GREEN** |

## PR / patch matrix (fidelity vs original intent)

| ID | Role | Fork status | Fidelity check (2026-11-07) | Autotest gate |
|----|------|-------------|------------------------------|---------------|
| **#28164** | Fuse table + `fusion_max` | ✅ Ported (`ggml-metal-fusion.{h,cpp}`) | MATCH table; empty-ne filter aligned in **066.3** | `SUITE=cb-fusion` |
| **#31688** | CB `cb_start[]` + `i += fusion_max` | ✅ Present | MATCH vs `pr31688.diff` | `SUITE=cb-fusion` I2 |
| **#30100** | MUL_MAT+ADD when residual is MUL_MAT | ⏸ N/A | No few-row MMA+ADD encode | — |
| **#27781** | Shared KV cells API | ✅ Ported | `SUITE=b1-shared-kv` GREEN | `SUITE=b1-shared-kv` |
| **#25618 FA-vec nsg** | Pin `nsg` so KV-pad boundaries don’t change reduction | ✅ On **vec** path; product default is **tile** | Vec pin kept for `GGML_METAL_FA_VEC=1`; C2 fixed by not shape-switching kernels | `SUITE=fa` + **C2 GREEN** |
| **#25618 mul_mv_ext** | Disable ext for ne11∈[2,8] | ✅ Applied | `if (false && … mul_mv_ext)` | `SUITE=mul-mat` |
| **#27441** skinny mm | Small-batch mm tiles | ❌ CLOSED | Do not blind-reapply | residual / ladder |
| **066.7** float-tile + `MUL_MM_MIN=7` | Unify S8+ on mm | 🟢 KEEP | Orthogonal to C2 | `SUITE=mul-mat-residual` |
| **TASK-066 FA always-tile** | No `ne01<20` vec/tile flip | ✅ Default tile | C7 RED→GREEN + C2 GREEN | `SUITE=fa` C7 + C2 |
| **#26358** / TM fixed split-KV | Further prefill invariance | ⬜ Open if C1 stays RED | C1 still DIFF@tok8 after tile unify | C1 / ladder |

## Gaps (not “failed apply”)

1. **C2 closed** via always-tile; **C1 / ladder text** still open (HELP first_token_diff 0→8) — next: larger-prefill / fuse / other op class, not re-open vec pin.
2. **Encoder still hand-rolls fuse** vs `fusion_next` — 066.1 PARTIAL; 066.5 active.
3. **#30100 / #27441** — deferred / closed.

## Re-audit protocol

After **any** Metal/ops/fusion edit:

```bash
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=fa ./tests/run-task066-metal-corners.sh
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  SUITE=ubatch-s8-s1024 ./tests/run-task066-metal-corners.sh   # C2 must stay GREEN; C1 may still RED
```

Log HELP / NO_EFFECT / REGRESS; never ε-PASS residual or C2.

## Baseline after always-tile (2026-11-07)

**Stamps:** `logs/task066-c2-after-tile-*`, `logs/task066-regression-after-tile-20261106T231654Z`

| Suite | rc | Verdict |
|-------|----|---------|
| `cb-fusion` | 0 | GREEN |
| `mul-mat-residual` | 0 | GREEN (066.7 KEEP) |
| `fa` (+ C7) | 0 | GREEN (always-tile; legacy vec RED under `FA_VEC=1`) |
| `ubatch-s8-s1024` **C2** | 0 | **GREEN** |
| `ubatch-s8-s1024` **C1** | 1 | RED HELP first_token_diff=8 |

**Conclusion:** C2 FA multi-step class closed by removing shape-dependent vec/tile selection. Product stays FA-on+turbo4. Next residual = C1/ladder beyond n=32.
