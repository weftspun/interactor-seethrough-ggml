import Compute.Gemm
import Compute.Conv2d
import Compute.Attention
import Compute.Norm

open LeanSlang

/-!
# `ComputeVerify` — standalone GPU-shader emission verifier (lean_exe)

Imports the `Compute.*` shader-library modules and drives their emitters,
checking that each produces non-empty Slang output. Returns 0 on PASS,
non-zero on any FAIL.

Run with:
  lake build compute_verify && ./.lake/build/bin/compute_verify
-/

namespace ComputeVerify

def check (name : String) (emitted : String) : IO Bool := do
  if emitted == "" then
    IO.println s!"FAIL — {name} emitted empty shader"
    pure false
  else
    IO.println s!"PASS — {name} ({emitted.length} chars)"
    pure true

/-- Emit every compute shader and confirm each is non-empty. -/
def runAll : IO UInt32 := do
  let g := Compute.Gemm.emitGemmShaderLit 128 128 64
  let c := Compute.Conv2d.emitConv2dShader 8 8
      (.litUint 4) (.litUint 8) (.litUint 8) (.litUint 3) 3 3 1 1
  let a := LeanSlang.emit (Compute.Attention.attnShader 8 8 (.litUint 128) (.litUint 64))
  let ln := LeanSlang.emit (Compute.Norm.layerNormShader 32 (.litUint 320))
  let sl := LeanSlang.emit (Compute.Norm.siluShader 256)

  let okG ← check "gemm" g
  let okC ← check "conv2d" c
  let okA ← check "attention" a
  let okL ← check "layer_norm" ln
  let okS ← check "silu" sl

  let all : Bool := okG && okC && okA && okL && okS
  if all then
    IO.println "PASS — all compute shaders emitted"
    pure 0
  else
    IO.println "FAIL — one or more compute shaders failed to emit"
    pure 1

end ComputeVerify

def main : IO UInt32 := ComputeVerify.runAll
