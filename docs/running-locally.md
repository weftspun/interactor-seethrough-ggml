# Running locally on an RTX 4090

Two routes. The container is the faster path to a first run because the Vulkan
packaging problems are already solved in it; the native binary is leaner if you
already have a working Vulkan setup.

Expect **~306 s** for 1280px / 30 steps on a 4090 (the figure in this repo's
README). Measured on an A40 at 653.8 s, and an A40 is ~2.2x slower, so the two
numbers agree to within 3%.

## Why local is dramatically cheaper

| | per 1280/30 PSD |
|---|---|
| RunPod A40 (measured) | **$0.080** compute, $0.117 cold |
| RunPod 4090 (est., $0.74/hr billed) | ~$0.063 |
| **Local 4090, electricity only** | **~$0.006** |

At ~450 W for ~306 s that is 0.038 kWh, about $0.006 at $0.15/kWh — roughly
**13x cheaper than renting**, with no cold start, no image pull, and no
weight fetch after the first time. Latency also drops: the ~5 min weight fetch
and ~30 s image pull disappear entirely on repeat runs.

## Route A — container (recommended first run)

Needs Docker or Podman with the NVIDIA container toolkit.

    mkdir -p models out

    docker run --rm --gpus all \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      -v "$PWD/models:/models" \
      -v "$PWD/out:/out" \
      -p 8080:8080 \
      ghcr.io/weftspun/interactor-seethrough-ggml:add-runpod-image serve

First start downloads ~10.7 GB of f16 weights into `models/` and skips that on
every later run (`fetch-weights.sh` is idempotent). The image itself is
0.14 GB.

Then, from another shell:

    curl -X POST http://localhost:8080/predictions \
      -H 'Content-Type: application/json' \
      -H 'Prefer: respond-async' \
      -d '{"input":{"image":"file:///dev/null","res":1280,"steps":30}}'

`input.image` takes a URL or a `data:` URL. For a local file:

    IMG="data:image/png;base64,$(base64 -w0 mypic.png)"
    curl -X POST http://localhost:8080/predictions \
      -H 'Content-Type: application/json' -H 'Prefer: respond-async' \
      -d "{\"input\":{\"image\":\"$IMG\",\"res\":1280,\"steps\":30}}"

That returns a prediction id; poll it:

    curl -s http://localhost:8080/predictions/<id> | jq -r .status

Outputs land in `out/<id>/layers/*.png` plus `out/<id>/out.psd`, and are also
returned as base64 data URLs in the prediction JSON.

`GET http://localhost:8080/` streams OTel NDJSON log records.

### Checking the GPU is actually used

    docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
      ghcr.io/weftspun/interactor-seethrough-ggml:add-runpod-image selftest

Look for `Vulkan Instance Version:` and a listed device. If it reports
"Found no drivers!", the container is missing the GLVND stack or
`NVIDIA_DRIVER_CAPABILITIES` lacks `graphics` — see
`docs/logbook/2026-08-14-first-runpod-run.md`, H2/H3.

## Route B — native binary

    gh release download v0.1.0 --repo weftspun/interactor-seethrough-ggml \
      -p "see-through-ubuntu-latest-vulkan.zip"
    unzip see-through-ubuntu-latest-vulkan.zip

    # populate ./models from the same release (handles .zst.partNN reassembly)
    MODELS_DIR=./models bash scripts/fetch-weights.sh

    ./see-through -m models -i mypic.png -o out.psd \
      --png-dir layers --res 1280 --steps 30

Native needs a working Vulkan loader and the NVIDIA driver
(`vulkaninfo --summary` should list the 4090). No CUDA required — this is a
Vulkan build.

## Useful flags

    --res 1280          inference resolution
    --steps 30          diffusion steps. 4 produces noise; do not benchmark with it
    --png-dir DIR       per-layer PNGs (this is what makes layers viewable)
    --seed N            reproducibility
    --device vulkan     explicit; auto-selects the first GPU otherwise

`--device cpu` is rejected: the build is GPU-only.

## VRAM

Unverified on 24 GB. f16 weights total 14 GB on disk, but the binary loads and
frees per stage (`load layerdiff-vae 0.20s`, `load marigold-unet 1.01s`), so
peak residency should be well under that. If it OOMs, try the q4 set:

    WEIGHTS_TAG=v0.0.2-dev WEIGHTS_VARIANT=q4 bash scripts/fetch-weights.sh

which is ~5.6 GB. Note q4 quality is unmeasured — see
`docs/quantization-ladder.md`.
