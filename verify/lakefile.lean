import Lake
open Lake DSL

package verify where
  -- Witness-search quality gates + verified Lean->Slang GPU compute shaders
  -- over the see-through graph configuration space.

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

require LeanSlang from git
  "https://github.com/V-Sekai-fire/lean-slang.git" @ "main"

-- import library for the seethrough_c DLL; override with
--   lake build -Kseethrough_c_lib=<path>
def seethroughCLib :=
  #[((get_config? seethrough_c_lib).getD
      (__dir__ / ".." / "build-vulkan" / "seethrough_c.lib").toString)]

lean_lib Case

lean_lib Compute where
  globs := #[.submodules `Compute]

@[default_target] lean_exe kernel_gate where
  root := `KernelGate
  moreLinkArgs := seethroughCLib

lean_exe compute_verify where
  root := `ComputeVerify

lean_exe compute_compile where
  root := `ComputeCompile
  moreLinkArgs := #[
    s!"-Wl,-rpath,{(__dir__ / ".lake" / "packages" / "LeanSlang" / "vendor" / "lib").toString}"]

lean_exe emit_shaders where
  root := `EmitShaders

lean_exe quant_design where
  root := `QuantDesign
  moreLinkArgs := seethroughCLib

-- Copy-layout design search. No GPU: pure cost arithmetic over shapes measured
-- by src/copy_cost.cpp, so it needs no seethrough_c link.
lean_exe layout_design where
  root := `LayoutDesign

-- Optimization ladder: which speed rungs were tried, measured, and why each
-- failed its gate. Pure arithmetic over measured figures; no GPU, no link.
lean_exe opt_ladder where
  root := `OptLadder
