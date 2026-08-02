# PERT — verified Fast Metal/SPIR-V Execution Path

Objective: wire the emitted Slang compute shaders through the in-process
compiler and onto the GPU, verified on-par with the Accelerate/CPU oracle
(cos>0.999 vs ggml) at every step, shipping ONLY the shader path in release.

## Task table (hours; O/M/P = optimistic/most-likely/pessimistic)

| ID | Task | Pred | O | M | P | E=(O+4M+P)/6 |
|----|------|------|---|---|---|--------------|
| A | Emit Compute shaders (GEMM/Conv2d/Attn/Norm) | — | 0 | 0 | 0 | 0.0 (DONE) |
| B | Slang→SPIR-V in-process FFI on macOS | — | 0 | 0 | 0 | 0.0 (DONE) |
| C | Compile REAL production shaders → SPIR-V (not fixtures) | A,B | 1 | 2 | 4 | 2.17 |
| D | GPU execution harness (dispatch kernel, read results) | C | 6 | 10 | 18 | 10.67 |
| E | Accelerate-CPU oracle, full op set (GEMM/MHA/LN/GN/GEGLU/Conv/resnet) | — | 4 | 8 | 16 | 8.67 |
| F | Per-op parity gate cos>0.999 both sides + f32 parity vs scalar | C,D,E | 3 | 6 | 12 | 6.5 |
| G | M2 chain integration: mid_block transformer3d +5 temporal | F | 10 | 20 | 40 | 21.67 |
| H | UNet-loop perf on Metal/SPIR-V (1-2s/loop target) | G | 3 | 8 | 16 | 8.5 |

## Forward pass (ES/EF, hours)

| ID | ES | E | EF |
|----|----|----|----|
| A | 0 | 0.0 | 0 |
| B | 0 | 0.0 | 0 |
| C | 0 | 2.17 | 2.17 |
| D | 2.17 | 10.67 | 12.83 |
| E | 0 | 8.67 | 8.67 |
| F | max(D=12.83, E=8.67) | 6.5 | 19.33 |
| G | 19.33 | 21.67 | 41.0 |
| H | 41.0 | 8.5 | 49.5 |

## Critical path

A → C → D → F → G → H   =   **49.5 hours**

E (Accelerate oracle, 8.67h) is NOT on the critical path — it finishes
(8.67) before D (12.83) and never delays F.

Critical-path ranking by slack: **G (21.67h) and D (10.67h) are the long
poles.** G is the mid_block temporal chain-integration blocker (cos 0.9958
at chain vs 1.0 isolated). D is the missing GPU execution harness.

## Follow-through (in this session, starting now)

1. C — prove the real Gemm/Conv2d/Attn/Norm shaders compile to SPIR-V
   through the FFI (not just fixtures). Small, concrete, unblocks D.
2. D — GPU execution harness.
...
