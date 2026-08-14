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
      libvulkan1 vulkan-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# "graphics" is what installs the NVIDIA Vulkan ICD. Without it the binary
# exits with "no GPU device found (Vulkan)" despite a working GPU.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all

COPY --from=build /src/build/see-through /usr/local/bin/see-through

# Mount weights here (RunPod network volume).
ENV SEETHROUGH_MODELS=/models
VOLUME ["/models"]

# ./see-through -m models -i in.png -o out.psd
ENTRYPOINT ["/usr/local/bin/see-through"]
CMD ["--help"]
