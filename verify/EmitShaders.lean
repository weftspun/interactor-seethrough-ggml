import Compute.Gemm
import Compute.Conv2d
import Compute.Attention
import Compute.Norm

/-!
# `EmitShaders` — write the real Compute.* shaders to `.slang` files

Task D shortcut (same-SOURCE / three-target): one Lean-emitted `.slang` file
compiles to all three backends via `slangc`:

  slangc -target cpp   -o x.cpp    x.slang     # CPU validation anchor
  slangc -target spirv -o x.spv    x.slang     # GPU Vulkan/MoltenVK
  slangc -target metal -o x.metal  x.slang     # GPU native Apple Silicon

Writes to `build-shaders/` under the verify project dir. Run:
  lake build emit_shaders && ./.lake/build/bin/emit_shaders
-/

def write (name : String) (src : String) : IO Unit := do
  IO.FS.createDirAll "build-shaders"
  IO.FS.writeFile ("build-shaders" / (name ++ ".slang")) src
  IO.println s!"wrote build-shaders/{name}.slang ({src.length} chars)"

def main : IO UInt32 := do
  write "gemm" (Compute.Gemm.emitGemmShaderLit 128 128 64)
  write "conv2d" (Compute.Conv2d.emitConv2dShader 8 8
    (.litUint 4) (.litUint 8) (.litUint 8) (.litUint 3) 3 3 1 1)
  write "attention" (LeanSlang.emit (Compute.Attention.attnShader 8 8 (.litUint 128) (.litUint 64)))
  write "layer_norm" (LeanSlang.emit (Compute.Norm.layerNormShader 32 (.litUint 320)))
  IO.println "PASS — all production shaders written"
  pure 0
