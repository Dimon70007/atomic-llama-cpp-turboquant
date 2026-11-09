# TASK-066.15 — R&D: Metal perf knobs (prefill + decode) since `d60d01875`

**Status:** 🟡 **IN PROGRESS** (env + min=1 quality + full suite; RED triage → **[066.16](TASK-066.16-min1-red-triage.md)**; A+rem still open)  
**Depends on:** 066.9 A1–C Done (always-tile / mul_mm policy shipped); 066.14 pivot evidence `…022827Z`  
**Branch:** `task066-metal-batch-invariance`  
**Baseline commit (perf regression window start):** `d60d01875` (`fix(metal): pin flash_attn_ext_vec nsg…`) — compare vs parent `d60d01875^`  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md) (parent repo)  
**Ideas / prior audit:** chat audit 2026-11-09; size_swa evidence [`logs/task066-14-pivot-evidence-20261109T022827Z/`](../../logs/task066-14-pivot-evidence-20261109T022827Z/) (product tree)  
**Env stamp:** [`logs/task066-15-perf-knobs-20261109T025448Z/`](../../logs/task066-15-perf-knobs-20261109T025448Z/) (product tree; `SUMMARY.md`)

## Why

After the Metal batch-invariance track (`d60d01875` → tip), **both prefill and decode** are slower vs the pre-track Metal defaults on Mini (Gemma4 IQ4_XS, turbo3 V). 066.14 showed `size_swa` 1536 vs 3072 is only a **small** filled-ring decode tax (~+6.5%); it does **not** explain a large dual prefill+decode drop vs `d60d01875^`.

This subtask **localizes which KEEP knobs cost speed**, then picks **optimal defaults per workload** (nomtp decode, nomtp prefill, MTP verify) without silently undoing invariance gates.

## Goal

1. Freeze a **suspect list** (below) mapped to commits + env levers.  
2. Run a **one-lever-at-a-time** matrix (pp + tg) against `d60d01875^` behavior via env where possible.  
3. Recommend product defaults (or documented env profiles) that restore speed **or** document the invariance tax honestly.  
4. Do **not** ship YAML/README product flips until a stamp + gate decision.

## Baseline vs tip (hot path)

| Knob | at `d60d01875^` | tip (product overlay) |
|------|-----------------|------------------------|
| FA path | `use_vec` if `ne01 < 20` | **always tile**; `GGML_METAL_FA_VEC=1` → legacy shape-vec |
| FA vec `nsg` | grows with `ne11` | pinned **`nsg=1`** (only if vec path) |
| `mul_mv_ext` (ne11 2..8) | enabled | **`if (false)`** → force `mul_mv` |
| `MUL_MM_MIN` | hard **8** | default **0** (`GGML_METAL_MUL_MM_MIN_DEFAULT`) |
| mul_mm smem / tile | half-tile offsets; smem ~8k | float-tile q8_0/iq4_xs; **smem 12288** |
| Metal fusion / CB splits | uniform `n_nodes_per_cb` | `ggml_metal_fusion_max` + `cb_start[]` + fusion table |
| `size_swa` | `PAD256(n_swa+n_ubatch)` | product **pad128** + headroom=`n_ubatch` (066.14 pivot) |

## Suspects (priority)

### P0 — likely both prefill and decode

| ID | Commit(s) | Lever | Why both |
|----|-----------|-------|----------|
| **K1** | `1e43ed04d` | `GGML_METAL_FA_VEC=0/1` | Always-tile: decode N=1 leaves vec; prefill rem / small ne01 leave vec |
| **K2** | `a0191ac91`…`77f3167e4` | `GGML_METAL_MUL_MM_MIN` ∈ {0,1,7,8,32} | `min=0` → N=1 on `mul_mm` (decode); mm occupancy / path for large pp |
| **K3** | `a0191ac91` | float-tile + smem 12k (no env) | Fewer concurrent TG on mm → pp heavy; tg if N=1 on mm |

### P1 — both, graph / secondary

| ID | Commit(s) | Lever | Notes |
|----|-----------|-------|-------|
| **K4** | `58d194068`, `16044ea6a`, `d2cc1c2f9` | tip vs `58d194068^` binary | CB fuse boundaries + fusion table; no single env |
| **K5** | `05985ebe4` | restore `mul_mv_ext` (code) or tip-as-is | rem∈[2,8]; weaker than K1/K2 |
| **K6** | `dd5f5d968`…`d1911b404` | headroom / pad (066.14) | Filled-ring `n_kv∝size_swa`; **not** primary vs `d60d01875^` |

### P2 — decode / MTP mostly

| ID | Commit(s) | Lever | Notes |
|----|-----------|-------|-------|
| **K7** | `d60d01875` | only with `FA_VEC=1` | vec `nsg` pin; dormant under always-tile |
| **K8** | `16044ea6a` speculative shared-KV | MTP only | skip for nomtp A/B |

## Hypotheses (one lever at a time)

| ID | Hypothesis | Pass criterion | Verdict (`…025448Z`) |
|----|------------|----------------|----------------------|
| H1 | Restoring shape-vec (`FA_VEC=1`) recovers most **decode** vs tip | tg↑ ≥50% of gap to baseline tip-vs-parent, or clear plateau | **REFUTE** — tg 10.15→10.15; pp noise |
| H2 | `MUL_MM_MIN=8` recovers most **decode** (N=1 back on mv) | same | **KEEP** — tg 10.15→**24.84** (~2.45×); `mm-min1` same (~24.86) |
| H3 | `FA_VEC=1` + `MUL_MM_MIN=8` recovers **both** pp and tg near `d60d01875^` class | pp+tg within ~5% of parent rebuild **or** stamped residual gap named | **PARTIAL** — tg via mm-min only; FA noop; **pp flat** across all env cells on tip binary (~349–352 / ~329) |
| H4 | Fusion/CB (K4) is a material dual tax independent of K1/K2 | tip@K1+K2 still slow vs `58d194068^`@same knobs | **DEFER** — needs binary bisect; env cannot isolate K4 |
| H5 | Float-tile/smem (K3) alone costs ≥5% pp | need rebuild without 12k smem / half-tile — only if H3 insufficient | **DEFER** — only if pp still slow vs `d60d01875^` rebuild |
| H6 | `size_swa` pivot vs fixed-2048 is not the dual regression | already supported by `…022827Z`; reconfirm if product ctx A/B needed | **KEEP** (prior stamp) |

## Operator matrix (DoD harness)

**Model lock:** Gemma4 `UD-IQ4_XS`, `-ctk q8_0 -ctv turbo3`, `-fa on`, `-b 2048 -ub 512`, **nomtp** (no draft).  
**Fail-closed:** explicit `MODEL=` / GGUF path; no silent alias.  
**Entry (fork or product Mini):** stop Metal server first if sharing GPU; use install-prefix `llama-bench` / `llama-cli`.

```bash
# Suggested stamp root (product tree):
#   logs/task066-15-perf-knobs-<UTC>/
BIN="$HOME/.local/opt/linux_to_mac/minillm/forks/turboquant/bin"
MODEL=/Users/otvazhniy/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf  # fail-closed: set explicitly

run_bench () {
  local tag="$1"; shift
  env "$@" "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on \
    -ctk q8_0 -ctv turbo3 -b 2048 -ub 512 -t 4 \
    -p 512 -n 128 -p 2048 -n 128 -r 3 -o md \
    >"bench-${tag}.md" 2>"bench-${tag}.stderr.txt"
}

# Cells (env only — no rebuild):
run_bench tip-default          # unset FA_VEC / MUL_MM_MIN
run_bench fa-vec1              GGML_METAL_FA_VEC=1
run_bench mm-min8              GGML_METAL_MUL_MM_MIN=8
run_bench fa-vec1-mm-min8      GGML_METAL_FA_VEC=1 GGML_METAL_MUL_MM_MIN=8
run_bench mm-min1              GGML_METAL_MUL_MM_MIN=1
run_bench mm-min0              GGML_METAL_MUL_MM_MIN=0
```

Optional filled-ring decode (066.14 class): `llama-cli -c 32768 -lv 4 -n 64` with long prompt; record tg only as secondary.

**Binary bisect (if env insufficient):** rebuild/overlay at `d60d01875^`, `1e43ed04d^`, `a0191ac91^`, `58d194068^`, tip — same bench argv.

### Planned automation (§5.8)

| Piece | Plan |
|-------|------|
| Orchestrator | `tests/run-task066-perf-knobs-matrix.sh` → thin wrap of env cells + stamp dir |
| Make (product) | `make minillm-task066-perf-knobs` → Darwin-only, `MODEL=` required |
| Config | optional `configs/minillm/task066_perf_knobs_matrix.json` (cells with explicit env) |
| Resume | `--from-cell` / `--dry-run` |

Do **not** invent ad-hoc one-off scripts as the primary operator path once the orchestrator exists.

## Invariance constraints (do not ignore)

Speed wins that **re-open** RED gates are not product KEEP unless explicitly waived:

- 066.9 C2 / FA tile — always-tile closed ub8 vs ub32 FA path split.  
- 066.9 A2/B — `MUL_MM_MIN=0` closed N=1 vs verify {2,3,4} and last-ubatch rem cross-kernel.  
- 066.7 float-tile — numeric drift vs half-tile.

Any recommended default must cite which suite stays GREEN (or name the accepted tax).

## Recommendation (from `…025448Z`; draft — no YAML KEEP)

| Audience | Knob | Value | Why |
|----------|------|-------|-----|
| **Product default (066.15 probe)** | `MUL_MM_MIN` | **1** | decode speed; A2 rem OK; MTP matmul hazard accepted — quality stamp `…041653Z` no duel regression vs min=0 |
| **Invariance override** | `GGML_METAL_MUL_MM_MIN` | **0** | Phase B all-mm; ~2.45× tg tax |
| **R&D / A+rem target** | `MUL_MM_MIN` | **8** + rem-rebalance | MTP on mv + rem closed without min=0 (design ack pending implement) |
| **Product default** | FA path | keep **always-tile** | `FA_VEC=1` no tg/pp win on flat nomtp bench; C2 stays |
| **Prefill** | — | no env fix | pp invariant under K1/K2 env; next lever = binary bisect K3/K4 if still slow vs `d60d01875^` |

**Invariance tax named:** keeping product `MUL_MM_MIN=0` costs ~**2.45×** decode t/s on Gemma4 IQ4_XS / turbo3 / flat `llama-bench` tg128 vs speed profile. FA always-tile is **not** the primary decode tax here.

## Definition of Done

- [x] Stamp dir with full env matrix (pp512/pp2048 + tg128) + summary table — `…025448Z`  
- [x] H1–H3 decided (KEEP / REFUTE / partial) with numbers  
- [x] H4 gated (env-optimal tip vs pre-fusion tip) or deferred with reason — **DEFER** (binary bisect)  
- [x] Written **recommendation**: product default vs R&D env profile(s); invariance impact named  
- [ ] Orchestrator + make target (`run-task066-perf-knobs-matrix.sh` / `make minillm-task066-perf-knobs`) **or** deferred follow-up TASK  
- [x] Pointer + `README-TASK066.md` + CHANGELOG updated; **no** product YAML / README runbook flip without separate implementation KEEP  
- [ ] Optional: MTP short-bench cell after nomtp KEEP decision  
- [ ] Optional next: binary bisect for residual **pp** vs `d60d01875^` (H4/H5)

## Non-goals

- Replacing 066.5 long-ctx KLD / 066.10 closeout  
- Restoring fixed-2048 `size_swa` headroom as product  
- Claiming ÷2 decode from `size_swa` alone (contradicted by `…022827Z`)  
- Upstream PR text (TASK-067) until recommendation frozen  

## Stamps

| Stamp | What | Verdict |
|-------|------|---------|
| `logs/task066-14-pivot-evidence-20261109T022827Z/` | size_swa 1536 vs 3072 | size_swa **not** primary dual tax |
| `logs/task066-15-perf-knobs-20261109T025448Z/` | tip env matrix (6 cells); build `bc49022c4` / 11459 | **H1 REFUTE / H2 KEEP / H3 PARTIAL**; decode tax = `MUL_MM_MIN=0`; pp flat under env |
| `logs/task066-15-min1-mtp-quality-20261109T041653Z/` | paired mtp-vs-nomtp min=0 vs min=1 | **no quality drop** on duel fixture; acc 0.884→0.893; duel_hits 1/2→2/2; DEFAULT pin **1** |
| `logs/task066-15-full-suite-20261109T042939Z/` | full TASK-066 suites @ DEFAULT=1 | residual/FA/longctx/cb **PASS**; mul-mat/moe* **RED expected**; ubatch C3_n1280 **REGRESS**; mtp-vs-nomtp duel 2/2 |
