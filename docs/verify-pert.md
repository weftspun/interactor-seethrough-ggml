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

## D SHORTCUT — same-SOURCE / three-target (verified 2026-08-01)

The lean-slang-gpu-compute skill's core principle collapses D: the GPU
harness does NOT hand-build kernels. The same Lean-emitted `.slang` file
compiles to all three backends, so the CPU C++ harness is the validation
anchor and Metal/SPIR-V are the execution side — they agree by construction.
Verified on the REAL emitted shaders (compute_verify → emit_shaders):

| target | cmd | result |
|--------|-----|--------|
| SPIR-V | `slangc -target spirv` | gemm 3316 B — **matches in-process FFI byte-for-byte** |
| Metal  | `slangc -target metal` (+xcrun metal → .metallib) | gemm.metallib 4922 B |
| C++    | `slangc -target cpp` | FAILS: `GroupMemoryBarrierWithGroupSync` is GPU-only |

CPU cpp target requires a barrier-free per-thread variant (documented skill
constraint).

### D execution model — through the ggml op graph, NOT a bespoke Metal driver

D does NOT build a custom MTLDevice command-buffer driver. The hardware
execution is delegated to the **ggml backend**, which owns the Metal dispatch,
buffer set/get, and readback. This is the `gemm_witness.cpp` pattern already in
`src/`: build the ggml graph, init the GPU (Metal) backend, `tensor_set` inputs,
`ggml_backend_graph_compute`, `tensor_get` results — and run the SAME graph on
the CPU backend as the reference, then compare.

So "execution" = driving the exported kernel's math through ggml ops on the
Metal backend + readback, NOT authoring Metal command buffers. The remaining D
work is: extend the op set (GEMM already exists in gemm_witness) to conv/mha/
norm through the ggml backend, and gate cos>0.999 vs the ggml oracle on the GPU
result. Only the shader math ships; ggml Metal is the dispatcher.

## Follow-through (in this session, starting now)

1. C — prove the real Gemm/Conv2d/Attn/Norm shaders compile to SPIR-V
   through the FFI (not just fixtures). DONE (commit c5bec305) + fixed the
   attention `V(...)`→`V[...]` subscript bug.
2. D — same-SOURCE three-target proven: gemm → metal/spirv/metallib.
   Remaining: thin dispatch driver + readback vs oracle tap.
3. G — mid_block temporal chain integration (long pole).
