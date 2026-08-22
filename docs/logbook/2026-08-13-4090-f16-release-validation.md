# 2026-08-13 — v0.1.1 f16 release binary on a bare RTX 4090: 1280/30 measured

* Operator: Claude (agent session)
* Objective: establish whether the **packaged** release runs at production
  quality on a machine with no build toolchain, and get honest numbers for the
  f16 weight set — every timing on record was Q4 from a local build.
* Outcome: **358.7 s and 27 correct layers** at 1280px/30 steps. One hypothesis
  confirmed, one partial, one **refuted by operator error** (see H2 — the
  refutation is the most useful part of this entry).

## Apparatus

| Component | Value |
|---|---|
| Host | Windows 11 Pro 26200, no build toolchain used |
| GPU | NVIDIA GeForce RTX 4090, 24 GB, driver **591.86** (CUDA 13.1) |
| Backend | ggml **Vulkan**; `fp16: 1 bf16: 0 int dot: 0`, matrix cores **KHR_coopmat** |
| coopmat2 | force-disabled by `disable_broken_coopmat2()` (as designed) |
| Binary | `see-through-windows-latest-vulkan.zip` from release `v0.1.1`, unmodified |
| Commit | `v0.1.1` == `569ea6c`, i.e. `main` at the time |
| Weights | `v0.1.1` **f16** GGUF set, ~11 GB packed; `layerdiff-unet.gguf` 8.1 GB |
| Input | `art/concept/anime_with_caption_cc0_0023.jpg` (CC0, in-repo), 1024x1024 |
| Settings | `--res 1280 --steps 30 --seed 42` |
| Peak VRAM | ~16.6 GB observed; no WDDM host-memory spill (shared-usage counters checked) |

## Hypotheses

### H1 — The packaged binary runs end to end on a host with no toolchain. **CONFIRMED**

*Why predicted:* `CMakeLists.txt` carries three deliberate fixes for exactly
this failure mode — `GGML_OPENMP OFF`, static MinGW runtime,
`BUILD_SHARED_LIBS OFF` — each with a comment describing the
`STATUS_DLL_NOT_FOUND` it prevents. If the packaging were still broken the
binary would fail to *start*, not fail slowly, so this is a cheap, decisive
test. PRs #1, #2 and #5 merged that work but nobody had run the artifact on a
clean machine.

*Result:* confirmed. The exe launched with no DLL errors and completed a full
production run, writing 27 layers plus the `_depth.psd` companion and
`.psd.json` sidecar.

### H2 — 1280px cost is super-linear in step count. **REFUTED — operator error**

*Why predicted:* a 2-step probe measured 4.11 s/step, predicting a ~163 s body
pass at 30 steps, yet the 8- and 30-step runs appeared to sit in
`layerdiff.body` for over an hour with the GPU pegged at 100% and ~430 W. A
mechanism was even proposed: later timesteps driving activations into
denormal range, which can cost 10-50x on some Vulkan implementations while
still showing full utilisation.

*Result:* **refuted, and the hypothesis should never have been raised.** No
blow-up exists. The full 30-step run completes in 358.7 s, and per-step cost is
flat at 4.16 s. The "hours" were an artifact of the *observer*: elapsed time
was inferred by counting the agent's own `sleep` calls, which do not track wall
time in that environment. Two healthy runs were killed on this false premise.

Corroborating facts available at the time that should have prevented the
conclusion, and were not checked:

* The 512px smoke run reported a **program-measured** 67.1 s while the same
  faulty observer method suggested ~30 minutes had passed. That contradiction
  was rationalised as stale file caching instead of being investigated.
* `see-through.exe` had accumulated only ~43 s of **CPU time** across a
  supposed three-hour run — consistent with normal GPU-bound execution, not
  with a pathological stall.

Rules adopted, both cheap:

1. **Timings come from the process**, never an observer's clock — the `[perf]`
   lines, `profiling/spans.jsonl`, or a stopwatch around the process. This repo
   already emits all three (`otel_jsonl.h`, MADR 0011's accumulators) precisely
   so nobody has to guess.
2. **Wait by blocking on the process** (`Wait-Process -Id`), not by sleeping
   and polling. Related trap: a file being appended by a running Windows
   process can read back stale through a Git-Bash `cat`/`tail` even though
   `spans_path` is `fflush`'d per span; re-read via PowerShell `Get-Content`
   when freshness matters.

This entry is left standing rather than rewritten, per this logbook's
append-only rule: the wrong measurement is part of the record.

### H3 — The `rowchunk_budget_mb` default is wrong for f16. **PARTIAL**

*Why predicted:* every knob A/B in MADR 0010/0011 was measured with Q4 weights
on an older ggml, so the defaults are *inherited, not verified* for this
configuration. `rowchunk_budget_mb`'s justification is the most dtype-sensitive
of them — *"2048 OOM'd; 256 and 512 measured the same speed at 1280, so take
the headroom"* was decided when the resident UNet was ~3 GB, not 8.1 GB — and
it governs the conv path that dominates this graph (IM2COL 220, CONCAT 402
nodes at 1280). If the OOM ceiling had moved down, the default might now be
both slower *and* closer to the edge.

*Result:* **partial.** The direction was right but the magnitude is marginal,
and the feared OOM did not appear. 1024 MB is ~9 s faster against a 1.4 s
baseline spread — real (about 6x run-to-run noise) but only 2.4-2.7%. Output is
**bit-identical**: per-layer alpha IoU 1.0000 across all 27 layers at threshold
0.125, matching pixel counts on every tag, satisfying MADR 0011's rule that
"faster but different output does not count". The graph thins exactly as the
mechanism predicts (CONCAT 402 -> 247, CONT 1662 -> 1292, MUL_MAT 1705 -> 1550,
VIEW 650 -> 435), and the gain lands in the decode stages while the UNet loops
barely move.

The default was **not changed**: one image on one host is not grounds for it,
and the honest gain is under 3%. The useful finding is the ceiling — no OOM at
4x the default budget on a 24 GB card *with* f16 weights resident.

## Observations

All *measured* unless labelled otherwise.

### Stage spans, `--res 1280 --steps 30 --seed 42` (measured)

| span | duration |
|---|---|
| `clip.body` | 1.7 s |
| `layerdiff.body` | 168.5 s — loop 126.3 s (first 5.76 s, 4.16 s/step) |
| `clip.head` | 1.4 s |
| `layerdiff.head` | 149.5 s — loop 112.2 s (3.69 s/step) |
| `marigold` | 32.6 s |
| `postproc` | 0.9 s |
| **`run`** | **358.7 s** |

Model loads 12.2 s across 52 graphs; backend init + graph build + alloc ~0.2 s
combined, consistent with MADR 0011 finding 5.

### Rowchunk A/B (measured)

| `--rowchunk-budget-mb` | `run` span |
|---|---|
| 256 (default) | 358.7 s |
| 256 (repeat) | 357.3 s |
| 1024 | 349.0 s |

The repeat exists because a single pair could not be distinguished from noise.

### Output quality (measured)

27 layers. Re-composited over white at each layer's sidecar `xyxy`: the
character reconstructs cleanly and the occluded-content layers are fully
inpainted — `back hair` carries the complete bob silhouette including the
region behind the head, `bottomwear` the whole hakama including the portion the
sash covers, `topwear-front` the complete kimono torso, `face` the bare
inpainted skin with eyes/brows/nose/mouth correctly split out. No pad-boundary
slivers, no seams, no blank layers. This is an independent non-repro of MADR
0008's open layer-quality item, on a fourth character/pose.

### Cross-configuration comparison (third-party / not directly comparable)

MADR 0011 records 5 m 05.9 s for Q4 on an RTX 4090, and 183.2 s for the
upstream HuggingFace Space. Neither was measured in this session or on this
apparatus: the Q4 figure predates the current ggml and used a different weight
set, and the Space figure is different hardware running a newer checkpoint,
reached over the network. The ~17% f16-over-Q4 gap implied by 359 s vs 306 s is
therefore an **estimate**, not a controlled measurement — plausible as weight
bandwidth, since the per-step profile (4.16 s vs the recorded 4.42 s) is
unchanged.

## Open threads

* `NV_coopmat2` remains force-disabled on a witness-confirmed 2026-07-20
  failure. Upstream has since landed im2col/matmul optimisations (llama.cpp
  #10942, reporting 3.68 -> 5.0 it/s with NV_coopmat2 for stable-diffusion) and
  a 2026 workaround for a conv2d coopmat2 compiler bug. Re-running
  `verify/KernelGate.lean` against current ggml is cheap and is the highest
  expected-value single change left that needs no training. **Not attempted
  here** (would require a local build).
* `docs/ggml-upstream-issues.md` is cited four times across records 0008-0011
  but does not exist in the tree; those citations dead-end.
* A torch-vs-ggml comparison on identical hardware has never been run. Every
  such number quoted to date is the hosted Space, which is not a like-for-like
  engine comparison.
