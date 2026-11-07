# PR draft — metal : batch-invariance for FA / ISWA / mul_mm (Apple Metal)

Fill into `.github/pull_request_template.md` when opening. Do not open until the
operator asks.

## Overview

Metal decode/prefill and speculative verify (MTP) used to change results when
`n_ubatch` or verify width changed (class ggml-org/llama.cpp#25618).

This PR keeps three fixes together:

1. Flash-attn **always-tile** on Metal (legacy shape-vec only with `GGML_METAL_FA_VEC=1`)
2. ISWA `size_swa` independent of `n_ubatch` (long-prefill multi-step parity)
3. Product `GGML_METAL_MUL_MM_MIN` default **0** + CPU policy unit so decode N=1
   and MTP verify widths share float-tile `mul_mm`

Public before/after + ctest commands:
[docs/backend/metal-batch-invariance.md](metal-batch-invariance.md).

Product MTP vs nomtp / knobs matrix:
[examples/metal-batch-invariance/](../../examples/metal-batch-invariance/)
(not ctest); fixtures:
[benchmarks/metal-batch-invariance/](../../benchmarks/metal-batch-invariance/).

## Additional information

- Autotests: `test-flash-attn-batch-invariance`, `test-mul-mat-batch-invariance`,
  `test-metal-mul-mm-policy`, `test-metal-cb-fusion`, MoE/longctx row0 corners,
  `test-ubatch-s8-s1024-row0-parity` (SKIP 77 without model/ctx/Metal)
- Measured MTP e2e (HARD+duel) and knobs matrix stamps are summarized in the
  public doc; product knobs KEEP `(n_max=2, p_min=0.95)` — YAML not changed here
- Suggested title: `metal : batch-invariance for FA, ISWA, and mul_mm`

## Requirements

- I have read and agree with the contributing guidelines
- AI usage disclosure: YES - agents used for research, implementation, test
  wiring, documentation, and PR draft text; human operator owns review and merge
