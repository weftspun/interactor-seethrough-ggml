import PlausibleWitnessDag

/-! # Optimization ladder

The same discipline as `docs/quantization-ladder.md`, applied to speed instead
of weights: optimizations are **stepped up to, not assumed**. Each rung records
the baseline it was measured against, the noise floor of that measurement, and
what happened to the output. A rung is kept only if it clears all three gates.

Three gates, and a rung must pass every one:

1. **Venue.** A rung must be measured in the same venue as its own baseline.
   A WSL number against a Windows baseline is not a comparison; it is two
   numbers. This gate exists because that mistake produced a 594.8s figure that
   looked like a verdict on the CUDA backend and was partly a verdict on WSL.

2. **Noise floor.** The delta must exceed the worst warm spread of the
   measurement. A 2.6s delta against a 1.95s floor is not a result, however
   much one wants it to be.

3. **Output.** Faster-but-different is a failed rung, not a tradeoff --
   `quantization-ladder.md`'s rule, restated. Precision rungs are expected to
   change bits, so `geometryHeld` (all layers present, bounding boxes within
   10%) is the honest gate for those; bit-identity is the gate for rungs that
   claim to be semantic no-ops.

A *witness* here is a rung that fails a gate. Unlike the kernel gate, finding
one is the normal outcome: every rung measured so far has failed one.

All figures below are measured, not estimated. Sessions differ -- the same
binary and config drifted ~9s between two sessions on one afternoon -- which is
why every rung carries its own baseline rather than sharing a global one.
-/

open PlausibleWitnessDag

/-- What happened to the output. -/
inductive Output where
  /-- byte-identical PSD; the rung is a proven semantic no-op -/
  | identical
  /-- bits differ, but all layers present and bounding boxes within 10%.
      The honest gate for a precision change, which cannot be bit-identical. -/
  | geometryHeld
  /-- output moved in a way nobody validated -/
  | changed
  | unmeasured
  deriving Repr, BEq, Inhabited

/-- Where a measurement was taken. Rungs are only comparable within a venue. -/
inductive Venue where
  | windowsVulkan
  | windowsCuda
  | wslCuda
  deriving Repr, BEq, Inhabited

structure Rung where
  name : String
  flags : String
  venue : Venue
  /-- baseline measured in the SAME session and venue -/
  baselineS : Float
  baselineVenue : Venue
  medianS : Float
  /-- worst warm spread across the compared configs -/
  noiseFloorS : Float
  output : Output
  deriving Repr, Inhabited

def Rung.deltaS (r : Rung) : Float := r.medianS - r.baselineS

def Rung.pctS (r : Rung) : Float := 100.0 * r.deltaS / r.baselineS

inductive Verdict where
  | win
  | regression
  | withinNoise
  | outputFailed
  | notComparable
  deriving Repr, BEq, Inhabited

def Rung.verdict (r : Rung) : Verdict :=
  if r.venue != r.baselineVenue then .notComparable
  else if r.output == .changed || r.output == .unmeasured then .outputFailed
  else if r.deltaS > r.noiseFloorS then .regression
  else if r.deltaS < -r.noiseFloorS then .win
  else .withinNoise

def Rung.kept (r : Rung) : Bool := r.verdict == .win

/-- Every optimization measured against this port, in the order attempted.

    Sources: MADR 0013 for the rowchunk rung; this session's sweeps for the
    rest. `profiling/sweep-lf/sweep.json`, `profiling/measure-nocopy/`,
    `profiling/measure-bf16/`. -/
def optLadder : Array Rung := #[
  { name := "rowchunk-1024"
    flags := "--rowchunk-budget-mb 1024"
    venue := .windowsVulkan, baselineVenue := .windowsVulkan
    baselineS := 358.7, medianS := 349.0, noiseFloorS := 1.4
    output := .identical },

  { name := "linear-fast (all passes)"
    flags := "--linear-fast"
    venue := .windowsVulkan, baselineVenue := .windowsVulkan
    baselineS := 356.0, medianS := 353.3, noiseFloorS := 1.95
    -- drifts the small head-pass facial layers; never validated as acceptable
    output := .changed },

  { name := "copy elimination (cross_frame_block)"
    flags := "source change, reverted"
    venue := .windowsVulkan, baselineVenue := .windowsVulkan
    baselineS := 346.86, medianS := 361.44, noiseFloorS := 3.10
    -- sha256 f14427a0..82299 on both: a proven semantic no-op
    output := .identical },

  { name := "bf16 weights (UNet only, f32 accum)"
    flags := "--bf16"
    venue := .windowsVulkan, baselineVenue := .windowsVulkan
    baselineS := 356.0, medianS := 414.5, noiseFloorS := 1.95
    -- 27/27 layers, no bbox beyond 10%; VAEs excluded, conv kept f16
    output := .geometryHeld },

  { name := "CUDA backend + flash (WSL)"
    flags := "GGML_CUDA=ON --no-tiled-attn, WSL Fedora"
    venue := .wslCuda, baselineVenue := .windowsVulkan
    baselineS := 346.9, medianS := 594.8, noiseFloorS := 3.10
    output := .unmeasured },

  -- The same experiment, re-run natively so it can actually be scored. The
  -- WSL figure was disqualified on venue, not on merit; this one is not.
  -- Venue was worth 83.5s of the 594.8s (14%); the remaining 155.3s is CUDA's
  -- own deficit. Built with MSVC 14.44 + nvcc 12.4, arch 89, GGML_VULKAN=OFF.
  { name := "CUDA backend (native Windows)"
    flags := "GGML_CUDA=ON GGML_VULKAN=OFF, VS2022 + nvcc 12.4"
    venue := .windowsCuda, baselineVenue := .windowsCuda
    baselineS := 356.0, medianS := 511.28, noiseFloorS := 2.40
    -- 27/27 layers, no bbox beyond 10%: a correct build that is simply slower
    output := .geometryHeld }
]

/-! ## Why torch is faster

Measured, not inferred. `seethrough-torch` was read to check whether it simply
does less work. It does not: identical structure -- two passes (13 body tags,
11 head tags), both at resolution 1280, both 30 steps, `guidance_scale = 1.0`
so no classifier-free guidance, one UNet forward per step. ggml matches on
every one of those.

Per-step cost, from torch's own tqdm output against MADR 0013's spans:

| stage        | torch      | ggml       | ratio |
|--------------|------------|------------|-------|
| body loop    | 1.88 s/step| 4.16 s/step| 2.21x |
| head loop    | 1.59 s/step| 3.69 s/step| 2.32x |
| loops total  | ~103s      | 238.5s     | 2.31x |
| non-loop     | ~60s       | ~117s      | 1.95x |

The tax is **uniform** -- UNet steps, VAE decode and postproc are all ~2.2x.
A single factor across unrelated stages is not a hotspot; it is a per-op cost
paid everywhere. That is why every rung above failed: each addressed one slice
(convolution, copies, GEMM precision flag, backend) of a factor applying to
every op.

The systemic difference is **activation dtype**. torch puts the whole graph in
bfloat16 -- unet, vae, trans_vae, and both text encoders are all
`.to(dtype=torch.bfloat16)`. ggml runs f32 activations throughout:
`mul_mat_f32` casts every activation up to f32, and geglu_ff's comment records
"the (2*inner, chunk, B) f32 projection transient ... ~3.4GB". f32 activations
double memory traffic on every op and forgo the native tensor-core dtype.

This also explains two failed rungs precisely. `--linear-fast` drops only the
precision *flag*; activations stay f32. `--bf16` converted only *weights*;
activations stayed f32, so ggml materialised conversions and it got slower.
Neither reproduced torch's configuration, so neither tested this hypothesis. -/

/-- Untried rung: the whole graph in bf16, not just the weights.

    This is the only remaining candidate whose magnitude matches the measured
    2.2x, and it is deliberately recorded as `unmeasured` rather than scored --
    the ladder does not credit predictions.

    **BLOCKED, and not by our code.** ggml-vulkan's entire bf16 surface is
    matmul (`pipeline_matmul_bf16`, `_id_bf16`), f32<->bf16 conversion, and two
    unrelated ops (`col2im_1d`, `snake`). There are NO bf16 kernels for ADD,
    MUL, NORM, GROUP_NORM, SOFTMAX, GLU, FLASH_ATTN or CONT -- which is
    ADD=1752, MUL=376, NORM=330, GLU=360, CONT=1662 in this graph, i.e. most
    of it. Every such op would round-trip bf16->f32->compute->f32->bf16, the
    exact materialised-conversion pattern that made weights-only bf16 17%
    slower; graph-wide it would be worse.

    f16 is no better: NORM, GROUP_NORM, GLU and the activations have zero f16
    pipelines either (790 nodes here).

    So this rung requires implementing a half-precision path across
    ggml-vulkan's elementwise/norm/GLU shaders. That is upstream-scale work,
    not a flag. It is the root of the 2.2x: ggml-vulkan is an f32-activation
    engine and PyTorch is a bf16 engine, and no amount of configuration
    changes which one this port is running on.

    The VAE's near-cancelling reduction (MADR 0009) would additionally have to
    stay f32 regardless. -/
def untried : Rung :=
  { name := "whole-graph bf16 activations (NOT ATTEMPTED)"
    flags := "source change across ops/unet_frame/vae"
    venue := .windowsVulkan, baselineVenue := .windowsVulkan
    baselineS := 356.0, medianS := 356.0, noiseFloorS := 1.95
    output := .unmeasured }

def verdictStr : Verdict → String
  | .win => "WIN"
  | .regression => "REGRESSION"
  | .withinNoise => "within noise"
  | .outputFailed => "OUTPUT FAILED"
  | .notComparable => "NOT COMPARABLE (venue)"

def main : IO UInt32 := do
  IO.println s!"optimization ladder: {optLadder.size} rungs"
  IO.println "gates: same venue | delta > noise floor | output holds"
  IO.println ""

  for r in optLadder do
    let sign := if r.deltaS >= 0.0 then "+" else ""
    IO.println s!"  {r.name}"
    IO.println s!"    {r.flags}"
    IO.println s!"    {r.medianS}s vs {r.baselineS}s baseline  ({sign}{r.deltaS}s, floor {r.noiseFloorS}s)  output={repr r.output}"
    IO.println s!"    -> {verdictStr r.verdict}"

  let kept := optLadder.filter Rung.kept
  IO.println ""
  IO.println s!"rungs kept: {kept.size} of {optLadder.size}"
  for k in kept do
    IO.println s!"  KEPT: {k.name} ({k.flags})"

  -- A witness is a rung that fails a gate. Finding one is the expected
  -- outcome here, not an alarm: the ladder exists to record which rungs
  -- failed and why, so they are not re-attempted.
  let failing : Array Bool := optLadder.map (fun r => !r.kept)
  let firstWitness := failing.findIdx? id
  let (found, lvl, trace) ← resolve (α := Bool) "optimization-ladder-failed-rung"
    (fun _ i => failing.getD i false)
    (fun _steps =>
      { value := firstWitness.isSome
        found := firstWitness.isSome
        witnessIdx := firstWitness.getD 0
        budgetHit := false })
  IO.println s!"resolve: level {lvl}, outcome {repr trace.outcome}"

  if found then
    let idx := firstWitness.getD 0
    IO.println s!"first failing rung: {(optLadder[idx]!).name} -- {verdictStr (optLadder[idx]!).verdict}"
    IO.println ""
    IO.println "Do not re-attempt a failed rung without changing the mechanism."
    IO.println ""
    IO.println "torch does NOT do less work: same 2 passes, same 1280, same 30 steps,"
    IO.println "guidance_scale=1.0 so no CFG, one UNet forward per step -- ggml matches"
    IO.println "on all of them. But ggml is ~2.2x slower per step in BOTH loops and"
    IO.println "~1.95x in everything else. A uniform factor across unrelated stages is"
    IO.println "a per-op cost, not a hotspot -- which is why each rung above, addressing"
    IO.println "one slice, moved nothing."
    IO.println ""
    IO.println s!"UNTRIED: {untried.name}"
    IO.println "  torch runs the whole graph in bfloat16; ggml runs f32 activations"
    IO.println "  throughout. --linear-fast changed only the precision flag and --bf16"
    IO.println "  only the weights, so neither actually tested this."
  pure 0
