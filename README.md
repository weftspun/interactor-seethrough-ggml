# see-through.cpp

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
