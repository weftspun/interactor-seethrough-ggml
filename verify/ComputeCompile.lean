import Compute.Gemm
import Compute.Conv2d
import Compute.Attention
import Compute.Norm
import LeanSlang.Compile

/-!
# `ComputeCompile` — round-trip REAL production shaders to SPIR-V

Bridges tasks A→C of the critical path: not just trivial fixtures, but the
actual `Compute.*` production shaders emitted by the library are passed
through the in-process FFI compiler (`LeanSlang.Compile.spirvSize`). Proves
the shader text we author is accepted by libslang and lowers to a real
SPIR-V blob — the prerequisite for the GPU execution harness (task D).

Run with:
  lake build compute_compile && ./.lake/build/bin/compute_compile
-/

open LeanSlang

namespace ComputeCompile

def check (name : String) (m : SlangShaderModule) : IO Bool := do
  let n := spirvSize m
  if n ≤ 0 then
    IO.println s!"FAIL — {name} → SPIR-V compile error (code {n})"
    pure false
  else
    IO.println s!"PASS — {name} → {n} bytes SPIR-V"
    pure true

/-- Compile every real production shader and confirm each lowers to SPIR-V. -/
def runAll : IO UInt32 := do
  let okG ← check "gemm" (Compute.Gemm.gemmShader (.litUint 128) (.litUint 128) (.litUint 64))
  let okC ← check "conv2d"
    (Compute.Conv2d.conv2dShader 8 8 (.litUint 4) (.litUint 8) (.litUint 8) (.litUint 3) 3 3 1 1)
  let okA ← check "attention" (Compute.Attention.attnShader 8 8 (.litUint 128) (.litUint 64))
  let okL ← check "layer_norm" (Compute.Norm.layerNormShader 32 (.litUint 320))

  let all : Bool := okG && okC && okA && okL
  if all then
    IO.println "PASS — all real production shaders compile to SPIR-V"
    pure 0
  else
    IO.println "FAIL — one or more production shaders failed to compile"
    pure 1

end ComputeCompile

def main : IO UInt32 := ComputeCompile.runAll
