# RTX 4090 Parity — Critical Path

## Baseline: 74.2s/step (M2 Pro) → ~5s/step (RTX 4090) = **~15× gap**

### Bottleneck: mul_mat_f32 casts f16 weights → f32 → slow f32×f32 Metal kernel
### Root: `kernel_mul_mm_f32_f32` (scalar) vs `kernel_mul_mm_f16_f32` (native simdgroup HALF8x8)

---

## Critical path (must do to close gap)

| Step | Item | Effort | Expected gain |
|------|------|--------|---------------|
| **A** | Per-tensor f16 gate in `mul_mat_f32`: check `max_abs` of weights before f16→f32 cast. If ≤ 65504, keep f16 → Metal selects native `kernel_mul_mm_f16_f32` | ~20 lines | **~5-8×** on conv/attn linears |
| **B** | Weight range sweep: iterate all model tensors, record max_abs per layer. Identify which exceed f16 range | ~1 script | Informs (A) coverage |
| **C** | Conditional dispatch: f16-safe tensors → `kernel_mul_mm_f16_f32`, overflow-risk tensors → keep `kernel_mul_mm_f32_f32` | ~10 lines | Same as (A) |
| **G** | Verify: run Lean witness gate, check NaN, IoU vs f32 baseline | ~1 test run | Confidence |
| **H** | Profile step time. Target <10s/step @ 768px. Iterate if needed | ~1 run | Closure |

### Already done (green)

| Item | Gain |
|------|------|
| `linear_fast` body pass: f16 native (no cast) | ~20% on body UNet |
| Flash-attn f32 K/V: Metal selects f32 kernel variant | Passes, no NaN |
| Flash-attn enabled on Metal (removed `!pipe_is_metal` guard) | — |
| `direct_conv` + `conv_row_chunk` enabled on Metal | — |
| Latent scaling (clamp ±100) prevents activation overflow | Enables f16 linears |

### Optional

| Item | Gain | Note |
|------|------|------|
| Flash-attn f16 Q (keep Q in f16 shared memory) | ~10% | Needs new Metal kernel variant |
| Q4_0 quantized models | 2-3× throughput | Fallback if f16 matmul still too slow |

## Expected timeline

1. **A + B + C**: 2-3 hours (code + sweep + dispatch) → **target ~5-10s/step**
2. **G + H**: 1 hour (verify + profile) → **confirm parity**
3. **Optional**: indefinite (nice-to-have, not blocking)

PERT chart at `docs/parity-pert-chart.svg`