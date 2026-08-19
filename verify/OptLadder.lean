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

  { name := "CUDA backend + flash"
    flags := "GGML_CUDA=ON --no-tiled-attn"
    venue := .wslCuda, baselineVenue := .windowsVulkan
    baselineS := 346.9, medianS := 594.8, noiseFloorS := 3.10
    output := .unmeasured }
]

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
    IO.println "The 2.12x gap against seethrough-torch (163.4s) is not explained by"
    IO.println "backend, convolution, copy count, copy traffic, GEMM precision, or"
    IO.println "activation dtype. Each was measured and each was refuted."
  pure 0
