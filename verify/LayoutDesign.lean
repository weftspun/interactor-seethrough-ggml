import PlausibleWitnessDag

/-! # Copy-layout design search

Not a correctness gate. This searches the *layout* design space for the
materialising copies (`CONT`/`CPY`) in the layerdiff UNet graph, ranking them
by predicted memory cost and looking for the ones worth changing.

Why it exists: op counts alone motivated a copy elimination that measured 4.2%
SLOWER with bit-identical output (361.4s vs 346.9s warm median, RTX 4090,
driver 610.88, res 1280). The general refutation is proved in
`2-contract/tensor-copy-cost-model`; this applies it to the graph's real
shapes.

Unlike the kernel gate, a "witness" here is not a defect. It is a copy whose
predicted cost exceeds the budget below — i.e. a copy worth fixing. The search
answers "which copies dominate, and what would fixing their stride buy?", not
"is anything wrong?".

The shapes are measured, not assumed: `copy_cost_report` in `src/copy_cost.cpp`
emitted them from a real res-1280 graph. Totals over all 2244 copies in one
UNet step: 67,481,250,304 elements, 6,437,971,592 cache lines, 1204 copies
page-scattered (31.7% of elements).
-/

open PlausibleWitnessDag

/-- Elements per cache line: a 64-byte line over f32. -/
def lineElems : Nat := 16

/-- Bytes per page, for the scatter classification. -/
def pageBytes : Nat := 4096

/-- One materialising copy, as measured from the graph. -/
structure Copy where
  name : String
  elems : Nat
  /-- innermost contiguous run, in elements -/
  run : Nat
  /-- byte stride between consecutive runs; 0 = fully contiguous -/
  stride : Nat
  deriving Repr, Inhabited

/-- Cache lines touched. A run shorter than a line still pays for a whole
    line, hence `min run lineElems`. Mirrors `cost` in
    `2-contract/tensor-copy-cost-model` and `copy_cost.cpp`. -/
def Copy.lines (c : Copy) : Nat := c.elems / max 1 (min c.run lineElems)

/-- Consecutive runs land in different pages: each run costs a TLB entry. -/
def Copy.pageScattered (c : Copy) : Bool := c.stride ≥ pageBytes

/-- What this copy would cost if its permutation kept the channel axis
    innermost, i.e. `run = channels` instead of `run = 1`. This is the
    quantity a layout change is trying to buy. -/
def Copy.linesIfContiguous (c : Copy) (channels : Nat) : Nat :=
  c.elems / max 1 (min channels lineElems)

/-- The measured top copies from a res-1280 layerdiff step.

    Every one has `run = 1`: the permutation puts a non-unit stride on the
    fastest-varying axis, so each element touches its own cache line. That is
    the worst case the cost model admits, and it was NOT what was assumed
    before measuring -- the earlier reasoning supposed dim 0 stayed contiguous.

    Copy 1 is identifiable from its stride: 1280 bytes = 320 f32 = C, a
    channel-axis transpose. That is `transformer3d`'s
    `ggml_permute(h, 1, 0, 2, 3)` (src/unet_frame.cpp:158 and its mirror at
    :173) -- the single most expensive copy in the graph. -/
def measured : Array Copy := #[
  { name := "transformer3d.proj_in.transpose",   elems := 106496000, run := 1, stride := 1280 },
  { name := "attn.qkv.permute.a",                elems := 53248000,  run := 1, stride := 25600 },
  { name := "attn.qkv.permute.b",                elems := 53248000,  run := 1, stride := 25600 },
  { name := "attn.qkv.permute.c",                elems := 53248000,  run := 1, stride := 25600 },
  { name := "attn.qkv.permute.d",                elems := 53248000,  run := 1, stride := 25600 },
  { name := "attn.qkv.permute.e",                elems := 53248000,  run := 1, stride := 25600 },
  { name := "attn.out.permute.a",                elems := 53248000,  run := 1, stride := 2560 },
  { name := "attn.out.permute.b",                elems := 53248000,  run := 1, stride := 2560 },
  { name := "attn.out.permute.c",                elems := 53248000,  run := 1, stride := 2560 },
  { name := "attn.out.permute.d",                elems := 53248000,  run := 1, stride := 2560 },
  { name := "attn.out.permute.e",                elems := 53248000,  run := 1, stride := 2560 },
  { name := "transformer3d.proj_out.transpose",  elems := 27289600,  run := 1, stride := 2560 }
]

/-- A copy is worth fixing when it touches more lines than this. Set at 1% of
    the measured whole-step line total (6,437,971,592), so a "witness" is a
    copy carrying at least a percent of all copy traffic on its own. -/
def budget : Nat := 64379715

/-- Channel widths the SDXL CrossFrame UNet actually uses. A layout fix has to
    hold at all three, not just the widest. -/
def channelWidths : Array Nat := #[320, 640, 1280]

def main : IO UInt32 := do
  IO.println s!"copy-layout design search over {measured.size} measured copies"
  IO.println s!"  line = {lineElems} f32, page = {pageBytes}B, budget = {budget} lines"
  IO.println ""

  let mut totalNow := 0
  let mut totalFixed := 0
  for c in measured do
    let now := c.lines
    -- The narrowest channel width is the conservative estimate: it is the
    -- least a contiguous layout could buy. Claiming the 1280 figure would
    -- flatter the result at the two narrower levels that also run.
    let fixed := c.linesIfContiguous (channelWidths.foldl min 1280)
    totalNow := totalNow + now
    totalFixed := totalFixed + fixed
    let flag := if c.pageScattered then " PAGE-SCATTERED" else ""
    IO.println s!"  {c.name}: run={c.run} stride={c.stride}B lines={now} -> {fixed} if contiguous{flag}"

  IO.println ""
  IO.println s!"top-{measured.size} lines now      : {totalNow}"
  IO.println s!"top-{measured.size} lines if fixed : {totalFixed}"
  IO.println s!"predicted reduction on these copies: {totalNow - totalFixed} lines"

  -- A witness here = a copy over budget, i.e. one worth the engineering.
  let over : Array Bool := measured.map (fun c => c.lines > budget)
  let firstWitness := (over.findIdx? id)
  let (found, lvl, trace) ← resolve (α := Bool) "copy-layout-over-budget"
    (fun _ i => over.getD i false)
    (fun _steps =>
      { value := firstWitness.isSome
        found := firstWitness.isSome
        witnessIdx := firstWitness.getD 0
        budgetHit := false })
  IO.println s!"resolve: level {lvl}, outcome {repr trace.outcome}"

  if found then
    let idx := firstWitness.getD 0
    IO.println s!"WITNESS (worth fixing): {(measured[idx]!).name}"
    IO.println "Fixing means changing the permutation so the fastest-varying axis"
    IO.println "is contiguous -- NOT deleting the copy. Deleting one copy and"
    IO.println "worsening another's stride is what measured 4.2% slower."
  else
    IO.println "no copy exceeds budget; layout is not the dominant cost here"
  pure 0
