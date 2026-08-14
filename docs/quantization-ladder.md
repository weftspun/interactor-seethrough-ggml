# Quantization ladder

Quantization is stepped down to, not started at. f16 establishes the reference
output; each rung below is measured against it and kept only if it holds the
parity gate.

## Gate

Per-layer alpha-mask IoU against the f16 reference PSD, threshold 0.125, with
**no layer below 0.99** — the same gate this project already uses against the
upstream reference. A rung that is faster but drops a layer below 0.99 is a
failed rung, not a tradeoff.

## Rungs

| Rung | Weights | Download | Purpose |
|---|---|---|---|
| 0 | f16, `v0.1.0` | ~10.7 GB | production baseline |
| 1 | f16, `v0.0.2-dev` | ~10.7 GB | reference for the ladder |
| 2 | q4, `v0.0.2-dev` | ~5.6 GB | first quantized rung |

    WEIGHTS_TAG=v0.0.2-dev WEIGHTS_VARIANT=f16 ./scripts/fetch-weights.sh
    WEIGHTS_TAG=v0.0.2-dev WEIGHTS_VARIANT=q4  ./scripts/fetch-weights.sh

## Why the ladder runs inside v0.0.2-dev

`v0.1.0` ships f16 only; q4 exists solely in `v0.0.2-dev`. Comparing
`v0.1.0` f16 against `v0.0.2-dev` q4 would confound quantization with
whatever changed between the two releases (`v0.1.0` is "head-res speedup +
f16 weights", so the weights themselves differ). That comparison cannot
attribute a quality change to quantization.

`v0.0.2-dev` contains both complete sets from the same export, so rung 1 → 2
isolates quantization as the only variable. Rung 0 is measured separately, as
the production baseline, and is not the denominator for the ladder.

## Coverage

Only four models have q4 variants:

    layerdiff-te2  layerdiff-unet  marigold-te  marigold-unet

The remaining five (`lama`, `layerdiff-te1`, `layerdiff-vae`,
`marigold-vae`, `trans-vae`) ship f16 only and total <1.4 GB, so quantizing
them would buy little. `fetch-weights.sh` falls back to f16 per model
automatically, so `WEIGHTS_VARIANT=q4` means "q4 where it exists" — a mixed
set, not an all-q4 set. Bear that in mind when attributing quality changes:
the ladder moves four models, not nine.

## What is not yet known

No rung has been measured. The IoU harness for comparing a candidate PSD
against an f16 reference is not written. Until it exists, the download and
VRAM savings above are the only established facts about q4 — nothing about
its output quality on this pipeline has been tested.
