# TASK-066 — Upstream / fork PR audit registry

**Updated:** 2026-11-07 (A1 Done; **A2 jinja-ladder next** > Phase B MTP)  
**Index:** [README-TASK066.md](README-TASK066.md)  
**Artifacts:** `logs/upstream-prs/{pr28164,pr31688,pr30100,pr27781}.diff`  
**Full freeze:** [TASK-066.9 § Frozen results](TASK-066.9-fa-batch-invariance.md)  
**Next code:** [066.5](TASK-066.5-fusion-table-parity.md) on C3-n1344 (not FA pins).

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

### 066.9 Phase A localize (2026-11-07) — frozen

| Stamp / gate | Result |
|--------------|--------|
| Threshold / bisect / faoff | `logs/task066-localize-{threshold,bisect,faoff}-*` |
| n≤1280 product + pure ubatch | **GREEN** |
| n=1344 (C3) | **RED** — primary TDD (`logs/task066-c3-gate-20261106T235908Z`) |
| n=1536 pure = product | **RED** (multi-step, not `-b`) |
| H9faOFF (FA-off q8/f16) | **RED** ⇒ residual **not FA-only** |
| FA C7 ≤256 | GREEN (op-level ≠ C3) |
| Agreed read | Next lever **066.5 fuse/CB**; FA-only pins forbidden |

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
| **066.9 Phase A1** | Nomtp C2/C3/C1 + ISWA | 🟢 Done | C3+C1 GREEN; ISWA `size_swa` KEEP | C2/C3/C1 |
| **066.9 Phase A2** | Jinja-ladder S8 residual | 🔓 **Next** (> MTP) | `012625Z` S8 vs S1024 DIFF@22; C1 raw GREEN ≠ Done | Ladder + autotest |
| **066.9 Phase B** | MTP verify widths 1..4 (#28108 spirit) | ⏸ Gated on **A2 DoD** | Draft: [task066-hist/draft_mtp_research_task-066_fa.md](task066-hist/draft_mtp_research_task-066_fa.md) | MTP FA autotest (TBD) |
| **#26358** / TM fixed split-KV | Further prefill / chat-path | ⬜ Own under **066.9 Phase A2** if ladder stays DIFF | Not Phase B | Ladder + A2 autotest |
| **#28108** (CUDA hist) | Unify FA tile for verify Q widths | ⬜ Metal port in Phase B | **After A2 DoD** | MTP FA corner |

## Gaps (not “failed apply”)

1. **A1 closed** (C2 + ISWA C3/C1). **Next: A2** jinja-ladder ([066.9](TASK-066.9-fa-batch-invariance.md)). MTP Phase B gated on **A2 DoD**.
2. **Encoder still hand-rolls fuse** vs `fusion_next` — 066.1 PARTIAL; **066.5 active**.
3. **#30100 / #27441** — deferred / closed.

## Re-audit protocol

After **any** Metal/ops/fusion edit:

```bash
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=fa ./tests/run-task066-metal-corners.sh
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  SUITE=ubatch-s8-s1024 ./tests/run-task066-metal-corners.sh
# C2 + C3-n1280 GREEN; C3-n1344 RED until 066.5
```

Log HELP / NO_EFFECT / REGRESS; never ε-PASS residual / C2 / C3.

## Baseline after always-tile + localize (2026-11-07)

**Stamps:** `logs/task066-c2-after-tile-*`, `logs/task066-regression-after-tile-20261106T231654Z`, `logs/task066-c3-gate-20261106T235908Z`

| Suite | rc | Verdict |
|-------|----|---------|
| `cb-fusion` | 0 | GREEN |
| `mul-mat-residual` | 0 | GREEN (066.7 KEEP) |
| `fa` (+ C7) | 0 | GREEN (always-tile) |
| `ubatch-s8-s1024` **C2** | 0 | **GREEN** |
| `ubatch-s8-s1024` **C3** | 1 | n=1280 GREEN / **n=1344 RED** |
| `ubatch-s8-s1024` **C1** | — | deferred until C3 GREEN |

**Conclusion:** Short FA multi-step closed (always-tile). Long residual is **non-FA** multi-step (H9faOFF). Next = 066.5 on C3. Product stays FA-on+turbo4.
