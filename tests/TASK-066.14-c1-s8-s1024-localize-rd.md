# TASK-066.14 — R&D: localize C1 S8↔S1024 greedy RED (post-066.13 C2)

**Status:** 🔄 **IN PROGRESS** — **H1 KEEP** (unequal `-b` / `size_swa` is the C1 RED root); pick fix lever next  
**Depends on:** 066.13 C2 KEEP (`size_swa` headroom=`n_batch`); C3-n1536 GREEN under C2  
**Frozen RED evidence:**  
- `logs/task066-13-c2-nbatch-headroom-20261108T220220Z/c3/` — C3 GREEN → C1 RED  
- `logs/task066-13-ctx64k-smoke-20261108T225702Z/corners/` — same @ CTX=65536  
**066.14 stamps (CTX=3072, DIAG=c1, IQ4_XS, MUL_MM_MIN unset):**  
| Stamp | Pair | Verdict |
|-------|------|---------|
| `logs/task066-14-c1-baseline-c1only-20261108T231714Z/` | A `b=512 ub=8` / B `b=1024 ub=1024` | **RED** `first_token_diff=2` `tok_a=237323` `tok_b=1587` |
| `logs/task066-14-c1-equal-b-20261108T231534Z/` | A `b=1024 ub=8` / B `b=1024 ub=1024` | **GREEN** `first_token_diff=-1` |

**Root (named):** C1 RED is the **066.13 residual**: product harness changes **both** `-b` and `-ub` → under C2, `size_swa` = PAD(`n_swa`+`n_batch`) → **1536 vs 2048**. Equal `-b` (equal `size_swa`) + vary `-ub` alone → bit-exact greedy. **H2 REFUTED** by the same stamp. Metal/mul_mm not implicated for this RED.  
**Gate binary:** `tests/test-ubatch-s8-s1024-row0-parity` via `tests/run-task066-ubatch-s8-s1024-corners.sh`  
**Knobs:** `TASK066_C1_BATCH=<n>` (equal `-b`); `TASK066_DIAG=c1` (skip C2/C3)  
**FA / KV:** product `q8_0` / `turbo4`, FA on  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)

## Why

After **066.13 C2** (`size_swa ∝ n_batch`, not live `-ub`):

| Probe | Result |
|-------|--------|
| C2 n=32 | GREEN |
| C3-n1280 / C3-n1536 (same `n_batch=n` both sides) | **GREEN** |
| KLD-UB (fixed `-b=2048`, vary `-ub`) | **PASS** |
| **C1 greedy** `A: b=512 ub=8` vs `B: b=1024 ub=1024` | **RED** (`first_token_diff=2`) |
| **H1** equal `-b=1024`, ub 8 vs 1024 | **GREEN** (`…231534Z`) |

Remaining suite RED was **not** ub-only drift; it was unequal outer `-b` → unequal `size_swa`.

## Goal

1. **Localize** why C1 greedy token streams diverge S8 vs S1024 under the current harness pair. ✅ H1  
2. Stamp decisive controls that prove/refute each hypothesis. ✅ H1/H2  
3. Propose a minimal fix lever (do not silently ship YAML / product KEEP inside this R&D). ⬜ choose below

## Known C1 harness contract (fail-closed facts)

```text
ubatch_pair (product default): A b=512 ub=8  B b=1024 ub=1024  n_predict=64
H1 control:                    A b=1024 ub=8 B b=1024 ub=1024  (TASK066_C1_BATCH=1024)
```

Under C2:

| Side | `-b` | `-ub` | `size_swa` (Gemma4 n_swa=1024) |
|------|------|-------|--------------------------------|
| A product | 512 | 8 | PAD(1024+512)=**1536** |
| B product | 1024 | 1024 | PAD(1024+1024)=**2048** |
| A/B H1 | 1024 | 8 / 1024 | PAD(1024+1024)=**2048** both |

## Hypotheses (localize in order)

| ID | Hypothesis | Result |
|----|------------|--------|
| **H1** | C1 RED because **`-b` differs** → `size_swa` 1536 vs 2048 | **KEEP** — equal-b GREEN `…231534Z` |
| **H2** | C1 RED from **`-ub` alone** even with equal `-b` | **REFUTE** — same stamp GREEN |
| **H3** | Divergence is decode-path / greedy (not prefill) | optional; not needed to name root |
| **H4** | Prompt / `n_predict` / CTX sensitive only | optional; 3072 matches 64k class (`diff=2`) |
| **H5** | Metal / mul_mm residual | **skip** — H1 explains RED |

## Proposed fix levers (pick one → next TASK / KEEP)

1. **Harness lock `-b` (recommended for gate):** change C1/C1j product pair to same `n_batch` (e.g. both `1024`; or both `max(512,1024)`), keep ub 8 vs 1024. Matches C3/KLD-UB invariant. Fastest GREEN for 066.5 gate; ladder cells that still vary `-b` need the same lock.  
2. **Fixed `size_swa` headroom cap** independent of live `-b` (e.g. always headroom=`max_batch` / profile knobs). Product-wide; bigger surface than harness.  
3. **Document C1 as unequal-b intentional RED** and move S8↔S1024 gate to equal-b-only — weak; prefer (1).

**Do not** reopen Metal mul_mm for this RED.

## Scope locks

1. **Localize before fix.** No product YAML / README ahead of named root (§5.0).  
2. Prefer extending existing diag (`TASK066_DIAG`, or temporary equal-`-b` env in the test binary) over new scripts; if a new knob is needed, fail-closed (no silent default pair).  
3. Companion: short ubatch ladder cell with **locked `-b`** only after H1 stamp.  
4. **No ε-PASS.** Document RED/GREEN honestly.  
5. **Not Done** while no explicit next fix TASK / KEEP chosen.

## Operator entry (fork root)

```bash
# Baseline product pair (RED) — stamped …231714Z
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws TASK066_DIAG=c1 \
  ./tests/run-task066-ubatch-s8-s1024-corners.sh logs/task066-14-c1-baseline-c1only-<UTC>

# H1 equal -b (GREEN) — stamped …231534Z
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
TASK066_DIAG=c1 TASK066_C1_BATCH=1024 \
  ./tests/run-task066-ubatch-s8-s1024-corners.sh logs/task066-14-c1-equal-b-<UTC>
```

## Definition of Done — partial

- [x] Baseline C1 RED reaffirmed (`…231714Z` + frozen 64k).  
- [x] **H1 equal-`-b` control** stamped **GREEN** (`…231534Z`).  
- [x] H1 KEEP: equal `size_swa` by C2 formula (both PAD(1024+1024)=2048); H2 REFUTE.  
- [x] H1 REFUTE path N/A.  
- [ ] Frozen root note + **next fix TASK** (lever 1 / 2 / 3 above) — **awaiting choice**.  
- [ ] Pointer + CHANGELOG; 066.14 stays R&D until fix KEEP lands or explicit closeout.

## Non-goals

- Claiming 066.5 Done from C3 GREEN alone.  
- Shipping fixed-cap / harness KEEP without localization stamp.  
- Replacing C1 with ε / “close enough” token match.  
- Full 100k KLD matrix.
