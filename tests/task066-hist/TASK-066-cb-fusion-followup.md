# TASK-066 follow-up — status pointer

**Authoritative execution plan (no contradictions):** [TASK-066-execution-plan.md](TASK-066-execution-plan.md)  
**Decisions:** [draft_mtp_research_task-066_answers.md](draft_mtp_research_task-066_answers.md)  
**Research / testing notes:** [draft_mtp_research_task-066_testing.md](draft_mtp_research_task-066_testing.md)  
**Unit gates:** [README-TASK066.md](README-TASK066.md) · `./tests/run-task066-followup-all.sh`  
**MUL_MAT inventory:** [TASK-066-mul-mat-batch-inventory.md](TASK-066-mul-mat-batch-inventory.md)  
**Upstream port history:** [TASK-066-upstream-patches-plan.md](TASK-066-upstream-patches-plan.md) (Phases 1-3 done)

## Unit status (post-#27441-revert, 2026-11-06)

Must stay green after Metal ports (separate from product ladder).

| Gate | Status | Stamp / note |
|------|--------|--------------|
| `SUITE=all` (fa, mul-mat, longctx, moe, moe-router) | **PASS** | `logs/task066-post-revert-suite-all` |
| Track A CB/fuse `#28164`/`#31688` | **PASS** | followup `logs/task066-post-revert-followup` |
| B1 `#27781` | **PASS** | same |
| B5 residual pin `ne11_mm_min=32` | **PASS** on M4 | same; env-gated default 32 |
| `#30100` encode | **N/A** | [TASK-066-p22-30100-recheck.md](TASK-066-p22-30100-recheck.md) |
| B2 `mtp-acc-trace` | FAIL until log attached | expected without `TASK066_ACC_TRACE_LOG` |
| B4 `mtp-e2e` | SKIP without MODEL/CTX | Stage 2 / P3 |

## Product status

- **Stage 0 done:** stamps `170944Z` (min=8) / `172821Z` (min=32)
- **#27441 CLOSED:** surgical port tried; stamp `175708Z` **regressed** vs `170944Z` (earlier first-disagree); **reverted**. Audit: [TASK-066-p22-27441-audit.md](TASK-066-p22-27441-audit.md)
- **Product default:** keep `GGML_METAL_MUL_MM_MIN=32` (S16==S8 on `172821Z`)
- **Product ubatch LOCK:** floor **256**, target **512**; owned large also **`{1024, 2048}`**. S16 = diagnostic near-baseline (pin 32), not prod gate.
- **Stage 1 CLOSED:** `{256,512,1024,2048}` vs S8 **WAIVEd** — [TASK-066-stage1-waive.md](TASK-066-stage1-waive.md). Evidence: `170944Z` S1024==S2048, same-ubatch OK, all four vs S8 DIFF@80; `184055Z` confirms S256/S512 under pin 32. Ready mm PRs exhausted; no new knobs.
- **P3 MTP:** **UNLOCKED** at ubatch **512** (floor 256); prior stamp `...173712Z` INVALID — need `-lv 3` engagement.

Locked highlights:

- Done = noise low enough + WAIVE (not bit-exact to ubatch=1) — **met**
- Ladder `{1,8,16,33,256,512,1024,2048}`; 2x same-ubatch on `{1024,2048}`; debug with `GGML_METAL_MUL_MM_MIN=8`
- Fixture: Gemma4 UD-IQ4_XS, ~2k, FA on, KV **q8_0/turbo4**, temp=0; server yaml canon / cli fast
