# see-through.cpp — ARCHIVED

**This port is archived. `seethrough-torch` is the line of work.** Nothing here is
deleted and nothing is retracted; the measurements below stand. Development stopped
because the question it existed to answer got answered, and the answer was no.

## Why it stopped

It is **2.12x slower than the torch implementation**, and the cause is not
configuration. ggml runs f32 activations where torch runs bf16, uniformly across every
stage, and ggml-vulkan has no half-precision kernels for NORM, GROUP_NORM or GLU. There
is no flag that closes it.

Six optimizations were measured against that gap. One survived:

```
rowchunk-1024        -9.7s  identical       WIN
linear-fast          -2.7s  changed         OUTPUT FAILED
copy elimination    +14.6s  identical       REGRESSION  (predicted a win)
bf16                +58.5s  geometry held   REGRESSION
CUDA + flash (WSL) +247.9s  unmeasured      NOT COMPARABLE (venue)
CUDA (native Win)  +155.3s  geometry held   REGRESSION
```

The CUDA rung is worth reading twice: re-run natively so it could be scored properly, a
correct build came in 155.3s slower than Vulkan on the same venue. Changing backend did
not help either.

**What was actually given up by archiving this: single-binary, no-Python deployment.**
Not portability — torch runs on Mac, Windows and Linux. The C++ port bought a
distributable without an interpreter, and that is the cost being paid.

## What is worth keeping

* `verify/OptLadder.lean` — the ladder above as a checked artifact, including the rungs
  that failed. A reader who knows which roads are dead ends is better off than one who
  only knows the current answer.
* `scripts/repack_q4_to_q8.py` — exact Q4_0 to Q8_0 container change, not a
  requantization. `models-q8run/` holds the repacked unet (1071 tensors, 2.78 -> 4.65 GB)
  and **it was never run**. The MMQ question it isolates is still open.
* `2-contract/tensor-copy-cost-model` — the Lean cost model that predicted the copy
  elimination would win. It was wrong by +14.6s, which is why the theorem
  `copy_count_is_not_a_cost_model` exists.
* MADR 0010/0011/0013 and the optimization ladder, whose timings are all against
  `art/concept/anime_with_caption_cc0_0023.jpg` — a fixed benchmark input, kept for
  comparability even though its source dataset is blocklisted for training.

Turns a single anime character illustration into up to 23 separate,
fully-inpainted layers (hair, face, clothing, accessories, etc.) plus a
depth map, saved as a layered PSD. C++
port of [See-Through](https://github.com/shitagaki-lab/see-through)
(Shitagaki Lab, SIGGRAPH 2026).

Get the weights (Or grab a prebuilt release binary instead of building).

```bash
# Download and unpack into a `models/` folder, then build.
mkdir models && cd models
gh release download v0.0.2-dev --repo weftspun/interactor-seethrough-ggml \
    -p "lama.gguf" -p "layerdiff-te1.gguf" -p "layerdiff-te2.gguf" \
    -p "layerdiff-vae.gguf" -p "marigold-te.gguf" -p "marigold-vae.gguf" \
    -p "trans-vae.gguf"
gh release download v0.0.2-dev --repo weftspun/interactor-seethrough-ggml \
    -p "layerdiff-unet.gguf.zst.part*" -p "marigold-unet.gguf.zst.part*"
cat layerdiff-unet.gguf.zst.part* | zstd -d -o layerdiff-unet.gguf
cat marigold-unet.gguf.zst.part* | zstd -d -o marigold-unet.gguf
rm *.zst.part*
cmake -B build -G Ninja -DGGML_VULKAN=ON && cmake --build build
# PNG, JPEG, BMP, TGA, GIF, and PSD are supported by `see-through.cpp`.
./build/see-through -m models -i art/concept/anime_with_caption_cc0_0023.jpg -o art/concept/anime_with_caption_cc0_0023.psd
# `-o` must end in `.psd`. Produces a flat, layered `out.psd` (plus an
# `out_depth.psd` companion and an `out.psd.json` metadata sidecar), matching
# upstream's `dump_parts_psd`.
```
