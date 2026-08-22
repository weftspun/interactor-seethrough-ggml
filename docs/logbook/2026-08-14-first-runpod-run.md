# 2026-08-14 — first containerised GPU run: Vulkan on RunPod, 1280/30 measured

* Operator: Claude (agent session)
* Objective: **reduce cost and latency** of a full-quality run. Speed numbers
  below are means to that end, not the end.
* Outcome: end-to-end run works. **653.8 s and $0.117 per 1280px/30-step run**
  on an A40. One hypothesis confirmed, one partial, one refuted, one opened.
* Total spend this session: **$0.586** of a $10 ceiling.

## Apparatus

| Component | Value |
|---|---|
| Provider | RunPod, on-demand pod, `cloudType: ALL` |
| GPU | NVIDIA A40, 48 GB, driver 570.195.03 |
| Rate | **$0.44/hr** as billed (GPU-types query advertised $0.35 — not what was charged) |
| Image | `ghcr.io/weftspun/interactor-seethrough-ggml:add-runpod-image` |
| Base | `nvidia/cuda:12.8.0-runtime-ubuntu24.04` |
| Backend | ggml **Vulkan** (`GGML_VULKAN=ON`); Vulkan instance 1.3.275 |
| Weights | `v0.1.0`, f16, 14 GB resident from ~10.7 GB compressed |
| Input | `art/concept/anime_with_caption_cc0_0023.jpg` (CC0, in-repo) |
| Container disk | 80 GB; host RAM 515 GB, 24 GB requested |

## Hypotheses

### H1 — A Vulkan-only ggml binary can run in a RunPod container. **CONFIRMED**

*Why predicted:* NVIDIA exposes Vulkan through its own driver, and llama.cpp
(also ggml-Vulkan) publishes working Vulkan container images, so the
capability plainly exists in this exact configuration. If it did not work, the
cause would be packaging rather than platform.

*Result:* confirmed, after three packaging fixes (H2, H3). `vulkaninfo`
reports instance 1.3.275 and `VK_LAYER_NV_optimus`; inference executes with
`FLASH_ATTN_EXT=180`, `MUL_MAT=1083`, which is GPU compute, not CPU fallback.

### H2 — `NVIDIA_DRIVER_CAPABILITIES=graphics` is the key setting. **PARTIAL**

*Why predicted:* that capability is what makes the container runtime inject
the Vulkan ICD manifest; the default `compute,utility` does not.

*Result:* necessary but **not sufficient**. With `graphics` set, the ICD JSON
*was* injected (`/etc/vulkan/icd.d/nvidia_icd.json`, pointing at
`libGLX_nvidia.so.0`) yet `vkCreateInstance` still failed. Two further
packages were required, in order:

1. X11 libs (`libxext6`, `libx11-6`, …) — without them the loader cannot
   `dlopen` the ICD at all (`libXext.so.6: cannot open shared object file`).
2. **The GLVND stack** (`libglvnd0`, `libgl1`, `libglx0`, `libegl1`,
   `libgles2`) — `libGLX_nvidia.so.0` is a GLVND *vendor* library and
   registers through `libGLdispatch`. Without it the library loads but cannot
   export `vk_icdGetInstanceProcAddr`.

Capabilities were also widened to `all`. The GLVND stack was the actual fix.

### H3 — The failure is structural (NVIDIA container-toolkit bug). **REFUTED**

*Why predicted:* the observed error matched open issue
`NVIDIA/nvidia-container-toolkit#1952` verbatim — "Could not get
`vkCreateInstance` via `vk_icdGetInstanceProcAddr` for ICD
`libGLX_nvidia.so.0`" — which attributes it to `libGLX` initialising
windowing in a headless container, with no confirmed workaround.

*Result:* **refuted.** It was missing packages. Two corroborating facts that
should have prevented this conclusion: `/dev/dri` *is* present in RunPod pods
(`card3`, `renderD130`), removing the headless premise; and llama.cpp's
Vulkan images demonstrably work in the same environment. Generalising from one
matching upstream issue, without checking a known-working counterexample, was
the error.

### H4 — The workload fits a 24 GB 4090, which is cheaper *and* faster. **UNTESTED**

*Why predicted:* ggml loads and frees per stage rather than holding everything
resident — the trace shows sequential `load layerdiff-vae 0.20s`,
`load marigold-unet 1.01s`, `load marigold-vae 0.10s` — so peak VRAM should be
well under the 14 GB on-disk total. A 4090 is **$0.34/hr against the A40's
$0.44** and ~2.2x its fp32 throughput.

*Prediction, falsifiable:* ~306 s and ~$0.029 per run — a 75% cost reduction
and 54% latency reduction versus today. Refuted if it OOMs or exceeds ~400 s.

*Not tested.* The A40 was chosen only because the first 4090 deploy returned a
capacity error, and 48 GB removed a risk that now looks unnecessary.

## Observations — measured

Two predictions, one input, same pod.

| Run | res | steps | wall | layers |
|---|---|---|---|---|
| smoke | 512 | 4 | 92.97 s | 30 + PSD |
| **full** | **1280** | **30** | **653.79 s** | **27 + PSD** |

Full-run breakdown:

    layerdiff unet (body)  271.9s   30 steps, first 10.91s, rest_avg 9.00s
    layerdiff unet (head)  213.1s   30 steps, first  7.25s, rest_avg 7.10s
    marigold unet           20.8s    4 steps
    model_load              12.6s
    compute (graph)          8.1s   graphs=52

Outputs verified as real: valid PNG signatures, per-layer dimensions varying
(`112x130`, `284x357`, `498x490` — cropped bounding boxes), 8.4 MB PSD,
13.6 MB total, 29/31 entries >1 KB.

## Analysis — cost and latency

Measured cost, at the billed $0.44/hr:

| Config | Latency | $/run |
|---|---|---|
| A40, f16, cold (incl. ~5 min weight fetch) | ~16 min | **$0.117** |
| A40, f16, warm | 10.9 min | $0.080 |
| 4090, f16, warm *(H4, estimated)* | ~5 min | ~$0.029 |

**Cross-check.** The project's own README records 306 s on an RTX 4090. An A40
is ~2.2x slower in fp32, predicting ~673 s; we measured 653.8 s — within 3%.
The containerised Vulkan path therefore carries no measurable overhead versus
native, and the setup is not misconfigured. This is the strongest evidence in
the entry, because it validates the apparatus rather than the result.

**Against the hosted reference — NOT A MEASUREMENT.** The upstream PyTorch
implementation has **never been run by us, on any hardware**. The only figure
available is 183.2 s reported for the HF Space, on half an RTX PRO 6000
Blackwell, under unknown conditions (unknown precision, unknown whether the
timing includes ZeroGPU queue or cold start, unknown layer count).

Comparing it to our A40 number requires normalising for hardware, and the
result is dominated by the assumed ratio:

| assumed HF hardware advantage | our time normalised | implied gap |
|---|---|---|
| 1.66x (fp32 TFLOPS) | ~394 s | 2.15x |
| 3x | ~218 s | 1.19x |
| 4x | ~163 s | **0.89x — we would be faster** |

fp32 TFLOPS is a poor proxy for a diffusion workload, which runs on
fp16/tensor cores and is often bandwidth-bound. Blackwell's advantage on those
axes is considerably larger than its fp32 ratio, so 1.66x is probably an
underestimate and the gap is probably **overstated** — possibly to zero.

> **No claim about relative implementation efficiency is supported by this
> session.** An earlier estimate of ~2x, derived the same way from the same
> third-party figure, is not independent corroboration — it is the same
> unverified number reused.

The only way to settle it is to run the torch implementation on the same A40
and compare directly. That repo has been archived, which makes this
harder, not impossible.

**Against the Mac.** 22,225 s (M2 Pro / Metal) → 653.8 s is a **34x**
improvement. Under a cost-and-latency objective this is the headline result.

## Conclusions

1. Vulkan ggml runs in a container on rented NVIDIA with no measurable
   overhead. The blocker was packaging, not the platform.
2. A full-quality run currently costs **$0.117 and ~16 minutes** cold.
3. The largest untaken lever is the **GPU choice** (H4): potentially −75% cost
   and −54% latency with no code change.
4. **Whether ggml is slower than upstream PyTorch is unknown.** The apparent
   ~2x gap rests entirely on one third-party figure from more powerful
   hardware, normalised by a proxy (fp32 TFLOPS) that is wrong for this
   workload. It could be 2x, 1x, or favourable to us. Any decision made on
   the basis of that gap is currently unsupported.

## Open questions

* **H4 unresolved** — does it fit 24 GB? Cheapest and highest-value next test
  (~$0.03).
* **No parity check exists.** Nothing here shows output *correctness*, only
  that plausible layers are produced. Any speed claim is unvalidated until the
  IoU harness exists (PERT task N, 4.67 h, 68% of remaining work).
* **Single run, no repeats, no variance.** n=1 at each setting.
* **The torch implementation has never been measured by us.** Running it on
  the same A40, same input, same settings is the only way to get a real
  comparison, and would cost roughly $0.10. Everything else is arithmetic on
  someone else's number.
* Is 183.2 s pure compute, or does it include ZeroGPU queue/cold start?
  Unknown, and it moves the comparison in the opposite direction to the
  hardware-ratio uncertainty.
* Layer counts differ between our runs (30 at 512/4, 27 at 1280/30) and are
  unknown for the reference — so workloads are not provably identical.

## Failures worth keeping

* A synchronous prediction cannot return through RunPod's proxy: **HTTP 524 at
  ~100 s**. Fixed by implementing Cog's `Prefer: respond-async` + polling.
* The log endpoint does **not** stream during inference —
  `subprocess.run(capture_output=True)` buffers until exit — despite being
  built for live polling. Unfixed (PERT task Q).
* A first run's logs were lost entirely: the container exited and the poll loop
  overwrote captured records with empty 404 responses. Both fixed.
* The bash log server and the Python prediction server bound the same port;
  `exec` made the latter PID 1, so the collision killed the container.
