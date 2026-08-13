# see-through (ggml/Vulkan) — linux/amd64 image for RunPod Serverless.
#
# Vulkan, not CUDA: src/pipeline.cpp hardwires Vulkan device selection and
# rejects every other --device value, so GGML_CUDA is not a flag flip here.
# NVIDIA exposes Vulkan through its driver, so this runs unmodified on RunPod
# provided NVIDIA_DRIVER_CAPABILITIES includes "graphics" (set below) --
# the default "compute,utility" does NOT install the Vulkan ICD.
#
# Weights are NOT baked in. models/ is ~14GB, which overflows the GitHub
# Actions runner disk. Mount a RunPod network volume at /models instead.

# ---- build ----
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates wget gnupg \
    && rm -rf /var/lib/apt/lists/*

# Full LunarG SDK, not distro Vulkan packages. ggml-vulkan's shader codegen
# needs SPIRV-Headers/SPIRV-Tools CMake config, which libvulkan-dev+glslc do
# not provide -- an apt-only build fails at configure with
# "Could not find a package configuration file provided by SPIRV-Headers".
# .github/workflows/build.yml reaches the same conclusion for the bare
# runner and pins the official SDK bundle for the same reason.
RUN wget -qO /etc/apt/trusted.gpg.d/lunarg.asc \
      https://packages.lunarg.com/lunarg-signing-key-pub.asc \
 && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list \
      https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list \
 && apt-get update && apt-get install -y --no-install-recommends vulkan-sdk \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Mirrors .github/workflows/build.yml so CI and image builds cannot drift.
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_VULKAN=ON \
      -DSEETHROUGH_BUILD_H3_SERVER=ON \
 && cmake --build build --config Release --target see-through -j"$(nproc)"

# ---- runtime ----
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 vulkan-tools ca-certificates curl jq zstd netcat-openbsd \
      libxext6 libx11-6 libxcb1 libxau6 libxdmcp6 python3 \
      libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# libGLX_nvidia.so.0 is a GLVND *vendor* library: it registers through
# GLVND's dispatch layer (libGLdispatch.so.0 / libGLX.so.0, from libglvnd0).
# Without the GLVND stack the loader can dlopen it but cannot resolve
# vk_icdGetInstanceProcAddr, so vulkaninfo reports "Found no drivers!".
# Measured on an A40 2026-08-13: X11 libs alone advanced the error from
# "libXext.so.6: cannot open shared object file" to the ICD entry-point
# failure; the GLVND stack is the other half. This package set matches what
# llama.cpp's Vulkan container images install.
#
# The X11 libs are not cosmetic either. NVIDIA's Vulkan ICD links against
# libXext/libX11; without them the loader finds the driver-injected ICD JSON,
# fails to dlopen libGLX_nvidia.so.0, and reports "vkCreateInstance: Found no
# drivers! / ERROR_INCOMPATIBLE_DRIVER" -- which reads like a missing GPU or a
# capabilities problem but is a missing shared object. Measured on an A40,
# 2026-08-13: NVIDIA_DRIVER_CAPABILITIES=graphics was already correct; this
# was the whole failure.

# "graphics" is what installs the NVIDIA Vulkan ICD. Without it the binary
# exits with "no GPU device found (Vulkan)" despite a working GPU.
# Working Vulkan containers use the full capability set; graphics alone
# injects the ICD JSON but not everything the driver needs.
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

COPY --from=build /src/build/see-through /usr/local/bin/see-through
COPY scripts/fetch-weights.sh scripts/entrypoint.sh scripts/cog_server.py /usr/local/bin/
RUN chmod +x /usr/local/bin/fetch-weights.sh /usr/local/bin/entrypoint.sh

# Weights are fetched from GitHub Releases on first run (~10.7GB compressed
# for f16). Mounting a persistent volume here makes that a one-time cost;
# without one it re-runs per cold start, which is still cheaper than a
# RunPod network volume when runs are batched.
ENV MODELS_DIR=/models
ENV WEIGHTS_TAG=v0.1.0
ENV WEIGHTS_VARIANT=f16
VOLUME ["/models"]

# entrypoint.sh fetches weights, then: see-through -m /models "$@"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# `serve` exposes the Cog-compatible wire API on :8080 (POST /predictions,
# GET /health-check) and is the mode this image exists for. `selftest` probes
# the Vulkan device. see-through rejects --help ("unknown arg").
EXPOSE 8080
CMD ["serve"]
