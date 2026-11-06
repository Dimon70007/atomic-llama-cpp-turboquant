# TASK-066 P2.2 — Re-check ggml-org/llama.cpp#30100 after #27441 close

Date: 2026-11-06  
Context: #27441 tried + reverted (stamp `175708Z` regressed). Remaining Stage 1 noise is S33+ / large `mul_mm`.

## What #30100 fixes

One-line change in upstream `ggml_metal_op_mul_mat_mma`: when fusing `MUL_MAT+ADD`, pick the residual as the ADD operand that is **not** the fused `mul_mat` (`dst->src[0] == op ? src[1] : src[0]`), instead of `src[0]->op == MUL_MAT ? ...`. Needed when both ADD operands are `MUL_MAT` outputs.

Diff: `logs/upstream-prs/pr30100.diff`.

## Fork re-check (2026-11-06)

| item | status |
|------|--------|
| `ggml_metal_op_mul_mat_mma` | **absent** |
| `ggml_metal_op_mul_mat_use_mma` | stub, always `false` (`ggml-metal-common.cpp`) |
| `ggml_metal_op_mul_mat_may_use_mma` | STRUCTURAL only (fusion table packing / CB align) |
| Encoder fused few-row MMA+ADD store | **not wired** |
| Fusion residual helper | identity-based `ggml_metal_mul_mat_add_residual` already (table path); encode never reads the buggy line |

**Verdict:** #30100 has **nothing to apply**. Encode port remains DEFERRED until/unless the few-row MMA+ADD path lands. Not a lever for current S33+ ladder DIFF under stock `mul_mm` / `mul_mv`.

## Implication for Stage 1

After #27441 CLOSED and #30100 N/A: no remaining ready upstream Metal mm patch in the TASK-066 list for S16/S33. Product path stays B5 `GGML_METAL_MUL_MM_MIN=32` (stamp `172821Z`: S16==S8). Residual `{256,512,1024,2048}` vs S8 **WAIVEd** — [TASK-066-stage1-waive.md](TASK-066-stage1-waive.md); not another blind PR port.
