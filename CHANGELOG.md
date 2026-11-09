# Changelog

One section per release, keyed by the exact tag. `verify-version` refuses to cut
a release whose tag has no section here, so this gets written *before* the tag is
pushed — the release notes are generated from it verbatim.

Write it for the person who downloads the build: what they get, what changed for
them, what to watch out for. Not a commit dump — the notes already carry the full
commit list underneath.

Releases before `b10269-1.5.0` predate this file; see the git history.

## Unreleased

### Changed

- **C1j ubatch pair:** A-side `ub=8` removed (`ub < 9` banned); C1j now `b=512 ub=16` vs
  `b=1024 ub=1024` (both multiples of 8). Fail-closed in `run_c1j_gate`. C1/C2/C3 keep
  historical `ub=8` vs `1024`. Binary: `test-ubatch-s8-s1024-row0-parity`.
- **`GGML_METAL_MUL_MM_MIN_DEFAULT` → 8 (upstream):** revert 066.15/16 probe `=1`.
  jinja `ub=8` rem+full both on `mul_mv` (A2 hole was `min=7`, not 8); MTP verify
  `N≤4` also all on `mul_mv`. Policy unit + residual suite gate updated; last-chunk
  / C1j must be re-stamped GREEN (not waived).
- **TASK-066.16 DONE — KEEP `MUL_MM_MIN_DEFAULT=1`:** R1–R9 all final **P**; no **Q**.
  *(superseded for product DEFAULT by upstream-8 revert above; stamp history kept.)*
  Control stamp `logs/task066-16-r1r5-control-20261109T062918Z` — min=0 GREEN /
  min=1 RED for mul-mat/moe*/mtp-verify/fullgraph/mtp-e2e; residual vs N=8 GREEN
  both; moe-router **ids match**. R8/R9 size_swa∝ub orthogonal. A+rem optional.
- **TASK-066.16 R9 final (P):** equal-`-b` C1 (`TASK066_C1_BATCH=1024`) still RED @
  min=0 and min=1 — same `first_token_diff` as product pair (3 / 8). Equal-b does
  not equalize `size_swa` under tip `max(ub,b/4)` (1280 vs 2048). Residual =
  `size_swa∝ub` policy, not chunking, not min=1 alone. Stamp
  `logs/task066-16-r9-equalb-c1-20261109T061835Z`.
- **TASK-066.16 headroom anti-footgun:** `PAD128(max(n_ubatch, n_batch/4))` + outer
  pad256; stamp `logs/task066-16-headroom-pad128max-ub-b4-20261109T060048Z` —
  unit PASS; C2 + C3_n1280 + **C3_n1536 GREEN** (min0/min1); C1 S8 vs S1024 still RED.
- **TASK-066.16 pad256 probe:** outer `size_swa` pad 128→256; unit retargeted;
  stamp `logs/task066-16-pad256-ubatch-c3-20261109T052740Z` — C3_n1280 GREEN (min0/min1);
  C3_n1536 RED mad≈7.63 (oneshot ub=1536 → size_swa=2560; above Metal ub≤1024).
- **TASK-066.16 R8 final (P):** ubatch C3 n=1280 REGRESS is **not** `MUL_MM_MIN=1` —
  control stamp `logs/task066-16-r8-ubatch-c3-20261109T044702Z` (same binary min=0 vs 1):
  C2 PASS both; C3 mad≈14.35 both; `SWA ring full size=1152` on ub=8. Root =
  066.14 `size_swa∝n_ubatch` (ub8=1152 vs ub1280=2304). Track under 066.5/066.14;
  does not argue revert DEFAULT→0.
- **TASK-066.16 (spec):** RED triage after `MUL_MM_MIN_DEFAULT=1` — rubric T/P/Q/S/F;
  checklist R1–R9 from stamp `logs/task066-15-full-suite-20261109T042939Z`.
  SSOT `tests/TASK-066.16-min1-red-triage.md`.
- **TASK-066.15 MUL_MM_MIN DEFAULT 0→1 (speed probe):** N=1 decode back on `mul_mv`;
  A2 rem lock retained; MTP verify cross-kernel accepted. Quality stamp
  `logs/task066-15-min1-mtp-quality-20261109T041653Z` — mtp-vs-nomtp min=0 vs min=1:
  no duel regression (hits 1/2→2/2); acceptance 0.884→0.893. Full suite stamp
  `logs/task066-15-full-suite-20261109T042939Z` — residual/FA/longctx/cb PASS;
  mul-mat/moe RED expected (N=1 vs N≥2); ubatch C3_n1280 REGRESS (triage);
  mtp-vs-nomtp duel 2/2. Override `GGML_METAL_MUL_MM_MIN=0` restores Phase B all-mm.
- **TASK-066.15 env matrix:** stamp `logs/task066-15-perf-knobs-20261109T025448Z`
  (Mini MTL, Gemma4 IQ4_XS, turbo3, build `bc49022c4`/11459). Decode tax =
  `MUL_MM_MIN=0` → tg128 ~10.15 vs ~24.8–24.9 at `MUL_MM_MIN∈{1,8}` (~2.45×);
  `FA_VEC=1` noop on flat bench; pp512/pp2048 flat across env (~±1%). H1 REFUTE /
  H2 KEEP / H3 PARTIAL; H4/H5 defer to binary bisect. Draft: keep product
  `MUL_MM_MIN=0` (A2/B tax); R&D speed `GGML_METAL_MUL_MM_MIN=1`.
- **TASK-066.15 (spec):** Metal perf-knobs R&D since `d60d01875` — suspects
  FA always-tile / `MUL_MM_MIN` / float-tile smem / fusion CB; env matrix + invariance
  constraints. SSOT `tests/TASK-066.15-metal-perf-knobs-rd.md`.
- **TASK-066.14 pivot evidence:** stamp `logs/task066-14-pivot-evidence-20261109T022827Z`
  — live `size_swa calc` / occupancy hwm need `-lv 4` (INFO→TRACE map); pivot
  `size_swa=1536` / fixed-2048 `3072`; filled-ring nomtp decode 8.61 vs 8.08 tok/s
  (~+6.5%; ÷2 not reproduced); short `llama-bench` flat. Product stays `n_ubatch`+pad128.
- **TASK-066.14 docs:** `tests/TASK-066.14-…md` § «What `calc_size_swa` is» —
  allocates `kv_swa` only; window vs `--swa-full`; FA does not set ring length;
  pad128 = `QK_TURBO3/4`, 256 = FA `get_n_kv` pad.
- **TASK-066.14 product pivot (speed):** `size_swa` headroom back to live `n_ubatch`;
  outer pad **128** (TurboQuant block), not 256. Fixed-2048 KEEP doubled product
  `size_swa` (1536→3072 @ `-ub 512`). Telemetry: create `size_swa calc` + apply_ubatch
  `SWA occupancy hwm` (used vs allocated). Estimator default headroom = `--n-ubatch`.
  R&D override unchanged (`TASK066_ALLOW_SIZE_SWA_HEADROOM` + `TASK066_SIZE_SWA_HEADROOM`).
- **TASK-066.14 H6 KEEP (superseded for product headroom):** was fixed
  `LLAMA_SWA_HEADROOM_MAX_UBATCH` (2048). Full corners GREEN
  `logs/task066-14-c1-keep-full-20261109T004421Z` (C2/C3/C1/C1j). Kept as R&D
  evidence that fixed headroom restores C1; product chose minimal `n_ubatch` instead.
- **TASK-066.13 C2 KEEP (superseded for headroom source by 066.14):** was `n_batch`
  headroom (not live `-ub`). Stamp `logs/task066-13-c2-nbatch-headroom-20261108T220220Z`
  — C3-n1536 GREEN; KLD-UB PASS @4k/16k; C1 residual when `-b` differed (066.14 root).
  CTX=64k smoke `logs/task066-13-ctx64k-smoke-20261108T225702Z` — corners C3 GREEN/C1 RED.

- **TASK-066.11 root named:** cross-ub divergence = `size_swa ∝ n_ubatch`. Stamp
  `logs/task066-11-localize-20261108T212103Z` — ub8↔ub256 (same size_swa=1280)
  KLD cross-ub 0; ub 256/512/1024 turbo3 FAIL; C3-bisect FIRST_DIFF `kqv_out-0`
  (turbo3/4) / `cache_v_l0` (f16) @ n1536. Fix lever: size_swa independent of live `-ub`.
- **TASK-066.12 ISWA KV logic fixes (Done; 066.5 gates still RED):** `seq_pos_min`/
  `seq_pos_max` read from `kv_base`; `seq_rm` fails closed before mutate when SWA
  window already rotated past `p0`; `calc_size_swa` uint64 + overflow clamp (unit);
  `init_batch` release-safe sinfos mismatch; `get_can_shift` equal-size KEEP documented.
  Re-stamp `logs/task066-triple-ubatch-256-512-1024-20261108T210925Z-post06612` —
  C3-n1536 RED / KLD-UB FAIL / ladder DIFF **unchanged** vs pre-fix (orthogonal to
  `size_swa∝ub`).
- **TASK-066.5 ISWA `size_swa` ideas formula:** replace size_base headroom KEEP with
  windowed `PAD(min(size_base, n_swa_eff*(unified?n_seq_max:1)+n_ubatch),256)` where
  `n_swa_eff` is ceil-aligned to TurboQuant KV block 128. Helper
  `llama_kv_cache_iswa_calc_size_swa` + CPU unit `test-iswa-size-swa`. Long-ctx SWA RAM
  drops; C3 cross-ub must be re-stamped (capacity depends on `-ub` again).
- **TASK-066.10 estimator:** mirrors ideas formula; fail-closed `N_UBATCH` /
  `--n-ubatch`; unit stamps @ N_CTX=100000 / N_UBATCH=512 (`size_swa=1536`).

### Added

- **TASK-066.5 host stamps (2026-11-08):** CPU `test-iswa-size-swa` PASS; **T-066.5-C3 RED**
  (gate **n=1536** ub8 vs ub1536; historical n1344 also RED); **T-066.5-KLD** CTX curve
  `…194452Z-merged` **FAIL honest** (turbo4 4K->16K up); stream logits in `perplexity.cpp`
  cleared 64K HOST_OOM. **T-066.5-KLD-UB** runner `tests/run-task066-iswa-kld-ubatch.sh`
  (same CTX, ub `%128`, tables vs-f16 + cross-ub); stamp
  `logs/task066-triple-ubatch-256-512-1024-20261108T203714Z` — cross-ub KLD != 0 + ladder
  S512/S1024 DIFF. **066.5 remains TODO (not Done)** while gates RED/FAIL.
- **TASK-066.11 (spec only):** R&D subtask
  `tests/TASK-066.11-small-ctx-q8-turbo3-cross-ub-rd.md` — investigate small-CTX logit
  divergence under `-ctk q8_0 -ctv turbo3` across ub 256/512/1024. Status TODO / not started;
  not a product fix KEEP.
- **TASK-066.9 Phase C MTP knobs matrix:** `tests/task066_mtp_knob_matrix.json` +
  `tests/task066_mtp_knob_matrix.py` + `tests/run-task066-mtp-knob-matrix.sh` compose the
  MTP-vs-nomtp ladder across `n_max∈{1,2,3,4}` × `p_min∈{0.85,0.9,0.95}` (ub=512 locks).
  Shared nomtp baseline once (`TASK066_CELL_MODES`); stamp+Pareto recommend vs product KEEP
  `(2,0.95)`; no YAML auto-promote. Ladder gains `TASK066_CELL_MODES=both|nomtp|mtp`.
  Stamp `logs/task066-mtp-knob-matrix-20261107T043032Z` — all cells HARD+duel 2/2;
  speed-Pareto candidate `n4-p085` (tg≈13.05); **operator KEEP** product `(2,0.95)` for
  acceptance≈0.88 (YAML untouched). Acceptance-sorted result table frozen in
  `TASK-066.9` Phase C + stamp `summary.md`.
- **TASK-066.9 Phase B e2e (docs+runner+stamp):** `run-task066-mtp-vs-nomtp-ladder.sh` supports
  `TASK066_SAMPLE_MODE=product|greedy`, product `p_min=0.95` / Unsloth sampling; HARD gate =
  engagement + non-empty gen; MTP↔nomtp DIFF is telemetry (not FAIL) unless
  `TASK066_REQUIRE_IDENTICAL=1`. Timing-strip in generation capture. Stamp
  `logs/task066-mtp-vs-nomtp-20261107T041336Z` (ub=512, acceptance≈0.88, duel 2/2).
- **TASK-066.9 Phase B MTP verify gate:** `test-mul-mat-batch-invariance --mtp-verify` locks
  row0 N=1 vs `{2,3,4}` (q8_0 / iq4_xs / MoE). Wired into `SUITE=mul-mat-residual` after residual.
  Policy helper `ggml_metal_mtp_verify_cross_kernel_hazard` locks min=1 RED / min=0 GREEN.
- **TASK-066.9 A2 mul_mm policy unit:** header `ggml-metal-mul-mm-policy.h` + CPU test
  `test-metal-mul-mm-policy` (no Metal). Locks rem cross-kernel hazard (`1708%8=4` under min=7)
  and (Phase B) product `GGML_METAL_MUL_MM_MIN_DEFAULT=0`. Runs first in `SUITE=mul-mat-residual`.
- **TASK-066.9 Phase A2 C1j gate:** `test-ubatch-s8-s1024-row0-parity` applies jinja chat template
  (`enable_thinking=false`) then greedy S8 vs S1024 after raw C1. `TASK066_DIAG=c1j` for fast loop.
  Covers ladder class that raw C1 misses (1708 tok prompt).
- **TASK-066.5/066.6 TDD gate:** `test-ubatch-s8-s1024-row0-parity` + `SUITE=ubatch-s8-s1024` —
  C2 + C3 (n=1280/1344) + C1 product greedy S8 vs S1024. Diag:
  `TASK066_DIAG=localize|fa-ablation|c3-bisect|c3-thresh`; true FA-off needs `TASK066_CTV=f16`
  (fail-closed with turbo). Requires `TASK066_MODEL` + `TASK066_CTX`. Not in `SUITE=all`.
- **TASK-066.8 R&D (docs + telemetry):** mathematical FP comparison metrics for ubatch S1/S8/S256/S1024
  and batch-invariance literature notes; `test-mul-mat-batch-invariance` dual-reports `max_ulp`
  beside `mad` while **bit-exact** remains the PASS gate.

### Fixed

- **TASK-066.5 KLD@64K host OOM:** `tools/perplexity/perplexity.cpp` streams logit rows per
  ubatch (and chunks single-batch windows) instead of `reserve(n_ctx*n_vocab)` f32 /
  full-window u16 `log_probs`. Unlocks `llama-perplexity --save-all-logits` /
  `--kl-divergence` at CTX=65536 with `-b 2048` on ~25GiB Mac (base file ~16GiB on disk).
- **TASK-066.9 Phase B (MTP verify widths):** after A2 (`MUL_MM_MIN=1`), N=1 decode stayed on
  `mul_mv` while verify `{2,3,4}` used float-tile `mul_mm` → row0 ULP drift (autotest
  `--mtp-verify` RED@min1). Product default **1 → 0** (`ne11≥1` → mm). FA op-level C1/C2 already
  GREEN (always-tile covers #28108 FA class on Metal). Residual suite gate: unset or `0`.
  A2 rem lock retained (`!hazard(1708,8,0)`).
- **TASK-066.9 Phase A2 (jinja-ladder S8 vs S1024):** last prefill ubatch remainder with
  `ne11∈{2..7}` stayed on `mul_mv` under `MUL_MM_MIN=7` while full chunks used float-tile
  `mul_mm` (jinja prompt 1708 toks → `1708%8=4`). Product default `GGML_METAL_MUL_MM_MIN`
  **7 → 1** (`ne11≥2` → mm; N=1 decode still mv). C1j RED→GREEN; ladder
  `logs/task066-ubatch-ladder-20261107T023433Z` S8≡S256≡S512≡S1024. Residual suite gate:
  unset or `1` (superseded by Phase B → `0`).
- **TASK-066.5 ISWA `size_swa` batch-invariance (C3-n1344):** stop sizing the SWA KV cache from
  this context’s `n_ubatch` (`n_swa+n_ubatch` made ub=8 → 1280 cells vs large-ub ≫1280). Prefill
  past 1280 then slid only on the small cache → multi-ubatch vs one-shot logits diverged
  (`c3-thresh` n=1288 first RED; `FIRST_DIFF=kqv_out`). Headroom is now `size_base` so every `-ub`
  shares capacity. `SUITE=ubatch-s8-s1024` C3+C1 GREEN; ladder HELP (`012625Z` S512≡S1024).
- **TASK-066 fullgraph runner:** classify prefix/KV-slot failures as `FIXTURE_FAIL` (not logit RED);
  prior status-matrix “fullgraph RED” with `CTX=2048`/`-b 2048` was fixture misuse.

### Changed

- **TASK-066.9 Phase B started / matmul KEEP:** FA geometry for MTP widths already GREEN under
  always-tile; Phase B closes post-A2 mul_mm/mv hole (`MUL_MM_MIN` default **0**). Spec:
  `tests/TASK-066.9-fa-batch-invariance.md`. Turbo4 × speculative e2e still follow-up.
- **TASK-066.9 Phase A2 DONE:** ordered **A1 → A2 → B (MTP)**; MTP gated on A2 DoD ✅ + explicit
  start. Spec: `tests/TASK-066.9-fa-batch-invariance.md`.
- **TASK-066 docs:** PR audit registry + frozen findings (ladder/C2/FA ablation); baseline suites
  `cb-fusion`/`mul-mat-residual`/`fa` GREEN, C2 RED (`logs/task066-pr-audit-baseline-20261106T230631Z`).

### Fixed

- **Metal MUL_MAT residual vs ubatch S8 (TASK-066.7):** float-tile `mul_mm` / `mul_mm_id` for
  `q8_0` and `iq4_xs` (cuts half-MMA row0 drift ~1e-4 → ~1 ULP vs `mul_mv`), and product default
  `GGML_METAL_MUL_MM_MIN` **32 → 7** so `ne11`/`ne21` ≥ 8 share that mm path with golden N=8.
  `SUITE=mul-mat-residual` GREEN; `SUITE=mul-mat` stays GREEN on widths `{2,3,4,5,7}`.

## b10298-2.1.0

### Added

- **Unlimited-OCR-2** (arch `unlimited-ocr2`): Baidu's 3.3B OCR model that turns page images into text
  with layout. Page-aware 2D RoPE and routed sparse attention (DASH), ported from the community patch by
  RX5950XT; GGUFs with a matching chat template at `AtomicChat/Unlimited-OCR-2-GGUF`. One request at a
  time (`--parallel 1`, the Atomic Chat default); context shift is off for this model. The same change
  fixes a CUDA graph crash on graphs with source-less ops such as `arange`.

## b10298-2.0.0

### Added

- **New TurboQuant+ base: `tqp-v0.4.0`** (TheTom's fork as of 2026-09-28; our
  work was rebased onto it). What it brings:
  - MoE expert cache (`--moe-cache`, see `docs/backend/MOE-CACHE.md`): faster
    decode for MoE models whose experts stay in host memory. CUDA/HIP, Metal and
    Vulkan; anything it cannot handle falls back to the CPU path.
  - KV streaming (`--kv-stream-arena-mib`, `--kv-stream-stage-mib`, see
    `docs/kv-stream.md`).
  - ConvRot weight types `Q8_CR`, `Q5_CR`, `Q6_CR` in `llama-quantize`.
  - TurboQuant K/V (`turbo2`/`turbo3`/`turbo4`) flash attention on Vulkan.
  - Faster TQ4_1S/TQ3_1S on AMD (RDNA and CDNA MMQ), HIP graph capture, and many
    CUDA/HIP/Vulkan/SYCL fixes.
  - New architectures: Qwen4-Exp, Muse Glimmer; DFlash2 speculative decoding.
- **Community Laya fine-tunes load with `-m <dir>`.** Both converters (C++ and
  `convert_hf_to_gguf.py`) now ignore a root `config.json` next to
  `rl_agent_config.json` + `encoder/config.json` and a
  `tokenizer/special_tokens_map.json` that only restates `tokenizer_config.json`;
  until now both files made the conversion fail (Python mapped the checkpoint as a
  plain ModernBERT). This opens fine-tunes such as `cklxx/laya-browser`,
  `Wouze/laya-ara-rag`, `impacte/mimir-laya-router` or `gtm-k/*`. The output is
  the same as for the directory without those files. Their accuracy is the
  fine-tune author's; only the official checkpoints are measured in `DECISION.md`.
- **Hints for Laya repackagings this engine cannot load**: `ggmlc` GGUFs
  (`mys/laya-*-GGUF`), encoder-only `modern-bert` GGUFs with a separate head
  (`Weidows/*`, `fr0stbit3/*`, with `--decision`) and `laya-head` files now fail
  with a pointer to the official checkpoint and `--decision -m <dir>`.
- **Cloudflare Clef decision models** (`Cloudflare/clef`, `Cloudflare/clef-flash`)
  in `llama-server --decision` (layout `clef`, see "Clef" in `DECISION.md`). The
  GGUF is the upstream one: `ggml-org/Clef-Flash-GGUF` / `ggml-org/Clef-GGUF` load
  as they are, and `convert_hf_to_gguf.py` converts the Hugging Face repo. All
  questions of a request go in one prompt and one forward pass (the joint schema
  head), on the CPU or one GPU (`--decision-device gpu`); `-c` sets the longest
  prompt (default 16384 tokens). Requests follow the reference: `instructions`
  are optional, the confidence is the probability of the chosen option. Text
  only for now (no images).

### Notes

- **Context shift / `--cache-reuse` with a quantized K cache:** `q8_0`, `q4_0`
  and the other ggml types no longer crash and now shift correctly. With
  `turbo2`/`turbo3`/`turbo4` K the shift is still skipped, as in earlier
  releases: no crash, but output quality drops after the context shifts. Size
  `-c` so it does not shift, or use a ggml type for `-ctk` when you need it.
- On Vulkan devices without int64/int8/fp16 shader arithmetic or 8-bit storage
  (older iGPUs, some legacy AMD) the MoE cache turns itself off and the normal
  expert path runs.
- **Windows signing scope**: release archives Authenticode-sign what Atomic Chat
  runs, `llama-server.exe` and the DLLs it loads (`llama*.dll`, `mtmd.dll`,
  `ggml*.dll`, the bundled CUDA runtime). The other tools in the archive
  (`llama-cli`, `llama-bench`, `llama-quantize`, ...) are no longer signed, so
  SmartScreen may warn when you start one by hand. `dev-latest` builds are
  unsigned unless a run asks for signing.

### Changed

- **Supported platforms are unchanged:** same CUDA 12.4 / 13.3, ROCm 7.2.1 /
  TheRock 10.0.0, Vulkan SDK and GPU architecture lists as b10269-1.7.0.
  Released GGUFs (TQ3_1S/TQ4_1S, NVFP4, Kimi K3, BailingMoeV3, Inkling, laya)
  load as before.
- **`Q2_0` type id is now 42**, the same as upstream llama.cpp, so upstream Q2_0
  GGUFs load. The fork used 47 before; a Q2_0 GGUF made by an older fork build
  does not load.
- **Breaking:** `--slot-save-path` / state files saved with a turbo KV cache by an
  older build do not load (the turbo type ids moved). RPC client and server must
  run the same build.
- **Breaking (upstream):** `llama-tts` now runs Qwen3-TTS through mtmd; the
  OuteTTS vocoder flags (`--model-vocoder`/`-mv`, `--hf-repo-v`, `--hf-file-v`,
  `--tts-oute-default`, `--tts-use-guide-tokens`) are gone.
- **Gemma 4 with `--cache-reuse`:** cache reuse is off again unless
  `--swa-full` is set (upstream keeps the SWA cache size check for
  correctness). Add `--swa-full` to get the fast time-to-first-token back.
- Model suppress tokens are applied by the sampler instead of the Gemma 4 graph;
  sampled output is the same, raw logits from the C API no longer carry `-inf`
  for them.
- `--spec-type mtp` / `nextn` and `--mtp-head` keep working (aliases of
  `draft-mtp` and `--spec-draft-model`).

## b10269-1.7.0

### Added

- **Decision models in `llama-server`** (`--decision`, see `DECISION.md`). A
  separate server mode for Jev-class decision models: no chat routes, one
  worker (on the CPU unless `--decision-device` says otherwise), calibrated
  probabilities instead of generated text.
  - `POST /v1/systemone`: TypeSafe-compatible typed questions (noul, choice,
    score) over a state.
  - `POST /v1/router/score`: executor cards in, an independent calibrated
    `p_success` per candidate out (Platt calibration from the model's spec; 501
    unless calibrated or `--decision-allow-uncalibrated`).
  - `/health`, `/props` (`decision` block with limits, plan and calibration),
    `/v1/models` capabilities, an error envelope with stable reason codes, and a
    bounded queue (429 + `Retry-After`).
  - New flags: `--decision`, `--decision-spec`, `--decision-plan`,
    `--decision-queue`, `--decision-max-items`, `--decision-allow-uncalibrated`,
    `--decision-debug`, `--decision-kernels`, `--decision-precision`,
    `--decision-device`, `--decision-gpu`, `--decision-strict-placement`.
  - `systemone` answers carry `action.act_probability` (the reference agent's
    action head, `round(softmax(act logits)[0], 4)`); with `--decision-debug`
    the raw act logits are in `debug.act_logits`. Router answers do not have it.
  - An internal error answers 500 with a fixed message; the exception text goes
    to the server log only.
  - `--decision-precision default|strict`: the matmul precision request of the
    laya graph. The CPU computes the same bits in both modes; on CUDA `strict`
    is the parity-audit mode (PEDANTIC on every matmul, slower), not a serving
    mode.
  - Experimental: `--decision-device gpu|auto` (and `--decision-gpu`,
    `--decision-strict-placement`) runs the laya graph on a GPU through the
    stock ggml backends; the default stays `cpu`. `/props.decision` reports the
    device and where the graph nodes run. Metal passes its parity tier
    (`f16-class`) against the CPU on the full parity corpus; CUDA (RTX 4090)
    passes its tiers, `--decision-precision strict` included, and Vulkan
    (NVIDIA) passes `f16-class` on the multilingual model. On CUDA the graph
    dequantizes Q8_0 weights to F32 (the quantized-activation kernels lose this
    model's outliers) and, in `strict`, F16 weights too, and `strict` computes
    the RoPE tables on the host.
  - `--decision-device auto` uses the GPU when there is one and falls back to
    the CPU, with a warning, when the device fails to load, initialize or warm
    up, or when CPU-only kernels are requested. A device whose weight buffer
    cannot be allocated now fails the load (`gpu`) or falls back (`auto`)
    instead of computing from host memory; `/props.decision.placement` counts
    weights the device cannot run (`host_weights`), and
    `--decision-strict-placement` refuses them. ROCm / MUSA builds, other
    backends and quantized types other than Q8_0 load with a warning: they have
    no parity run.
  - `llama-laya-cli` gets `--plan packed|sequential` (default `packed`;
    `sequential` runs one graph per question, as the server does),
    `--jsonl FILE` (one load, one output line per input line), `--device`,
    `--gpu`, `--strict-placement` and `--precision`.
  - `LAYA_TRACE_DIR=<dir>` dumps the per-layer outputs of every forward pass
    (`tools/laya/trace_diff.py` compares two dumps) for `llama-laya-cli` and
    `llama-decision-bench`; the server honours it only with `--decision-debug`,
    since the dump holds the activations of every request.
  - Measured on servers: an RTX 4090 answers 13-24x faster at p50 than 32
    threads of a 64-core EPYC on `laya-multilingual` F16 (systemone with one
    question: 7.1 vs 93.8 ms), an RTX 5060 Ti 9-13x faster than 28 threads of
    an EPYC 9334, and the decision process then uses almost no CPU. Use an F16
    GGUF on a GPU: Q8_0 is not faster there. `DECISION.md` ("GPU backends") has
    the tables and the recommendation. Macs are not measured yet, so the app
    default stays `cpu`.
  - `scripts/bench-decision-device.py`: paired CPU vs GPU bench (alternating
    blocks, load and GPU sampling, time to ready, optionally a chat model
    generating on the same GPU). `scripts/bench-decision-report.py --parity`
    prints a speed number only when a passing parity gate of the same build,
    GGUF and device covers it.
- **Laya decision model support** (`tools/laya`, `llama-laya-cli`), based on
  upstream PR #29363: converter for `laya-multilingual`, GGUF arch `laya`, a
  self-contained CPU graph, and a tokenizer that matches the HF tokenizer exactly.
  Inputs follow the laya 0.3.21 PyTorch reference. On a 2605-question parity set
  the F32 GGUF matches the reference to 6.6e-4 in scorer logits with identical
  answers; F16 changes 2 answers, both near-ties.
- **English Laya checkpoints** `convaiinnovations/laya` (512 ctx) and
  `laya-typed-decisions` (1024 ctx), ModernBERT-large encoder: the converter
  handles them, `tools/laya` gets a second tokenizer (ModernBERT / OLMo
  byte-level BPE with NFC, 0 mismatches against the HF tokenizer on 13.6k strings,
  10.7k of them English-heavy), the special ids come from the tokenizer (`[CLS]`
  50281, `[MASK]` 50284 with lstrip; a GGUF whose mask id is not
  `laya.marker_token_id` no longer loads), and the checkpoint's
  `temperature_by_options` buckets are stored in the GGUF and applied like the
  reference. On 1491 English questions the F32 GGUFs match the PyTorch reference
  with identical answers (max |dlogit| 4.2e-4 and 6.8e-5). F16 through the server
  keeps every answer with the `blas` kernels, the macOS default (1.3e-2 and
  4.6e-3); with the `default` ggml kernels, the Linux and Windows default, it
  reaches 0.24 and 3.8e-2 with one near-tie answer change on `laya`
  (`llama-laya-cli`, which packs the questions of an item: 0.44, two changes). About 2.2x the compute of `laya-multilingual` per token on an
  M4 Max (router N=1 221 vs 111 ms at `-t 12`) and 944 vs 519 MiB footprint in
  F16 (see `DECISION.md`, "English checkpoints").
  `laya-multilingual` results are unchanged bit for bit with the `default`
  kernels (`llama-laya-cli`, and the server with `--decision-kernels default`).
- **Model metadata `decision.spec`**, stamped into a GGUF without reconversion
  with `gguf-py/gguf/scripts/gguf_decision_spec.py`. A calibration marked
  required must cover every question type and option count, or the model does not
  load.
- **`llama-decision-bench`** and `scripts/bench-decision.sh`: in-process latency
  (p50/p95/p99 per phase), CPU seconds, memory and logits-hash determinism of
  decision requests, plus server time-to-ready and first request after idle, with
  the machine's power source and load recorded next to the numbers.
- **Router training and calibration tools** (`DECISION.md`, "Training and
  calibration tools"): `tools/decision/reference.py`, a stdlib Python reference of
  the router input (strict JSON, executor cards, `card-v1`, router state,
  `escape-control` from the GGUF vocabulary), checked against the engine byte for
  byte on 540 golden requests (`test-decision-reference`) and through the server's
  render route; the card as JSON Schema; `scripts/fit-router-calibration.py`
  (collects the server's raw router logits, fits the Platt `router.calibration`
  block with bootstrap-interval ECE / Brier before and after, and prints the
  `gguf_decision_spec.py` command instead of touching the GGUF); and
  `scripts/router-baselines.py` (pass-rate heuristic and logistic regression on
  card metrics, same data format and metrics).
- **Decision CPU levers** (laya engine): `token_embd` stays in a read-only file
  mapping and the GGUF metadata is freed after the load (Q8_0 with the `default`
  kernels: RSS 637 -> 262 MiB and footprint 764 -> 389 MiB after load; the macOS
  default `auto` adds the BLAS backend, 408 MiB; results bitwise unchanged); one persistent ggml
  threadpool; a warm-up pass at load (`--no-warmup` skips it); `--load-mode mlock`
  / `mmap+mlock` lock the model; the default thread count is the performance cores
  (Windows: highest `EfficiencyClass`); `KMP_BLOCKTIME=0` / `OMP_WAIT_POLICY=passive`
  unless set. New `--decision-kernels auto|default|repack|blas|repack+blas` (also
  spec `plan.kernels`, shown in `/props.decision.plan.kernels`): `auto`, the new
  default, uses the Accelerate BLAS backend on macOS, which is 2-2.3x faster, uses
  3-4x fewer CPU seconds and is closer to the PyTorch reference than the ggml kernels
  for F16 and Q8_0 (F16: 2604/2604 argmax agreement); `repack` enables
  the CPU repack buffers for quantized weights. The BLAS backend runs with
  min(`-t`, 8) threads (it starts new threads for every weight conversion; at
  `-t 12`, 12 threads were up to 10% slower than 4 or 8 on `laya-multilingual`, and
  4 was 3-5% slower than 8 on the larger English encoder). `llama-laya-cli` gets `--kernels`,
  `--no-mmap` and `--mlock` and keeps the `default` kernels. `/props.decision`
  also shows the BLAS threads and the weight bytes loaded / mapped / repacked.

- **Laya checkpoints without Python.** `llama-server --decision -m <dir>` and
  `llama-laya-cli -m <dir>` take a laya Hugging Face checkpoint directory (an HF
  snapshot works as it is). The first start converts it into a GGUF cache,
  later starts reuse it ("converted" / "cache hit" in the log), and the cached
  GGUF is loaded exactly like `-m FILE`. The cache lives in
  `$LLAMA_CACHE/laya/gguf-cache` or the user cache (macOS
  `~/Library/Caches/llama.cpp/laya/gguf-cache`, Linux `~/.cache/...`, Windows
  `%LOCALAPPDATA%\...`), never in the checkpoint; new flags
  `--decision-convert-cache DIR` and `--decision-convert-type f16|f32`
  (default f16; the converter's q8_0 is not the precision-protected Q8_0 recipe,
  use `tests/laya/quantize.sh` on the f16 GGUF). The cache key covers the
  converter version, the outtype and the files the converter can read (root,
  `encoder/`, `tokenizer/`; files up to 8 MiB by content, larger ones such as the
  weights by size + mtime + symlink target). Writes are atomic and synced to disk.
  `/props.decision` shows `source` (`gguf` / `checkpoint-dir`) and `cache_path`.
  The conversion is the new `llama-laya-convert` (C++, no Python; also
  usable on its own), byte-identical to `convert_hf_to_gguf.py` for all three
  published checkpoints in f32, f16 and q8_0, with and without `--model-name`
  (18 of 18 files), and 6-19x faster (0.2-1.8 s per f16 file). A first start of
  `laya-multilingual` takes about 1.2-1.3 s instead of 0.3 s and peaks at about
  515 MiB instead of 372 MiB RSS.

### Notes

- Decision mode runs on the CPU by default (`--decision-device cpu`); next to a
  GPU chat server keep it there and start it with `--device none`.
- **Stricter laya GGUF checks**: a GGUF without `laya.attention.sliding_window`,
  `laya.attention.sliding_window_pattern`, `laya.rope.freq_base` or
  `laya.rope.freq_base_swa`, or with `laya.act_classes` below 1, no longer
  loads. Every GGUF made by this converter has them; hand-made or older
  third-party files may need reconversion.
- A laya GGUF started without `--decision` fails fast with a hint.
- **Decision server numerics on macOS change**: the default kernels there are now
  `cpu+blas` (Accelerate), so macOS server logits differ slightly from Phase 1 and
  from Linux and Windows, where `auto` stays on the ggml kernels and nothing
  changes. `--decision-kernels default` restores the Phase 1 numerics bit for bit on
  every platform. An explicit `--decision-kernels blas` fails at load on a
  build without a BLAS backend.
- Recommended laya precision is F16: Q8_0 is about 1.5x faster but changes about
  2% of answers on the parity set.
- Windows: decision model, spec and input paths may contain non-ASCII characters
  (for example a Cyrillic user folder), and `llama-laya-cli` prints LF line ends on
  every platform, so its output compares byte for byte with the golden files.
- A router calibration fitted with `scripts/fit-router-calibration.py` pins the
  kernels its logits came from (`plan.kernels`): one fitted on a Mac pins `blas`
  and needs a BLAS build elsewhere. Collect on the target platform, or start the
  collecting server with `--decision-kernels default`.

### Changed

- **Qwen3.5 / Qwen3-Next / Kimi-Linear / Kimi-K3 / BailingMoeV3 numerics:** the
  Gated DeltaNet q/k normalization now uses upstream's `rsqrt(sum(x^2) + eps)`
  form (upstream 5fdfa6282). Logits of these models shift slightly; results
  measured on earlier builds are not bit-identical.

## b10269-1.6.0

### Added

- **Windows AMD ROCm archive** `llama-turboquant-windows-x64-rocm.zip`. HIP
  backend for RDNA2 through RDNA4 and Ryzen AI 300 / Ryzen AI Max (gfx1030,
  gfx1100/1101/1102/1103, gfx1150/1151, gfx1200/1201), self-contained: the HIP
  runtime and the BLAS DLLs it links are bundled (about 100 MB), only a current
  Adrenalin driver is needed, no ROCm SDK install. Until now Windows + Radeon
  meant the Vulkan build; HIP is the faster prompt-processing path on these
  GPUs. Everything in the archive is Authenticode-signed, the AMD DLLs with
  AMD's signature where they ship one.
- **Linux ROCm archive** now also targets gfx950 (Instinct MI350/MI355X),
  gfx1150 (Ryzen AI 300) and gfx1103 (Radeon 780M/760M).
- `docs/amd/`: what ships for AMD, the ROCm vs Vulkan benchmark plan for the
  TurboQuant KV cache, and the open items. `scripts/bench-amd.sh` runs that
  matrix per backend build, `scripts/bench-amd-report.py` renders one table.

### Changed

- ROCm builds no longer pass `GGML_HIP_ROCWMMA_FATTN`; upstream removed the
  rocWMMA flash-attention path and the flag was a no-op.

### Notes

- **The Windows ROCm archive is beta.** It is built, signed and checked for
  completeness in CI, but has not been run on a Radeon yet. `llama-server
  --list-devices` must show a HIP device; if it does not, or if loading fails,
  report the GPU, driver version and the error, and use the Vulkan archive in
  the meantime.
- Atomic Chat does not select the Windows ROCm backend yet; the client change
  follows separately. Until then it is a manual download.

## b10269-1.5.1

### Fixed

- **Ling-3.0-flash (BailingMoeV3) no longer emits garbage token bursts.** The
  model is trained with clamped SwiGLU activations in its late layers, and the
  per-layer limits live in `config.json` under `expert_swiglu_limit_list` and
  `share_expert_swiglu_limit_list`. The public HF modeling code ignores those
  keys and so did this port, which caused deterministic transient logit
  collapse - output like `count += 1eville` dropped into otherwise fine
  generations. Measured at roughly -20 pass@1 on HumanEval (72.6% -> 93%+ with
  the fix); the garbage-token repro is eliminated.

### Notes

- **Re-convert your Ling-3.0-flash GGUF to get the fix.** The clamp limits are
  written by the converter into two new KVs (`{arch}.swiglu_clamp_exp` and
  `{arch}.swiglu_clamp_shexp`); a GGUF produced before this release does not
  carry them, and the runtime then defaults to no clamping. Re-download the
  quant or re-run `conversion/bailingmoe.py`.
- Both KVs are optional and default to zero, so existing GGUFs and every other
  architecture are unaffected. The graph needed no change - the SwiGLU clamp
  branches in `build_ffn` / `build_moe_ffn` already trigger on a nonzero
  per-layer limit, matching the vLLM `SwigluStepAndMul` semantics.

## b10269-1.5.0

### Added

- **NVIDIA DGX Spark (GB10) support.** New archive
  `llama-turboquant-linux-arm64-cuda-13.3`, built natively for aarch64 with
  CUDA 13.3 and sm_121 SASS. Other arm64 NVIDIA machines (GH200, GB200, Jetson
  Thor) run it too, JITing the kernels from PTX on first launch. This is the
  first Linux arm64 build the fork ships — until now arm64 meant macOS only.
- **BailingMoeV3 (Ling 3.0) architecture support**, including the KDA gate
  handling.

### Changed

- **Linux CUDA archives are roughly half the size** — 1657 → 956 MB (12.4) and
  1879 → 1028 MB (13.3) measured across both the `.zip` and `.tar.gz`. The zips
  were storing `libcublas.so` → `.so.13` → `.so.13.5.1.27` as three full copies
  because `zip` followed the symlinks.
- **CUDA 13.3 builds ship Ampere PTX (`80-virtual`).** A100/H100/B200 were
  falling back to the Turing PTX floor, which silently disabled `cp.async` and
  the Ampere MMA path — both gated on `__CUDA_ARCH__ >= 800`. Those cards get
  Ampere-class kernels now. No architecture lost support in this release.
- Windows CUDA builds got their architecture lists pinned, all runner cores, a
  ccache that can actually hold a CUDA build, and 7-Zip instead of
  `Compress-Archive`. Release turnaround drops accordingly.

### Notes

- The DGX Spark archive has **not yet been validated on real GB10 hardware** —
  it is built and arch-checked in CI (`cuobjdump` asserts sm_121 SASS is
  present), but nobody has run it on a Spark yet. Treat this one as beta and
  report back.
- The CUDA 13.3 archives now use `-compress-mode=size`. Kernel SASS is
  unchanged and inference speed is unaffected; the fatbin is decompressed once
  at module load. It needs a driver from the CUDA 12.4 era or newer.
