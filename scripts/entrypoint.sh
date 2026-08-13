#!/usr/bin/env bash
# Container entrypoint: ensure weights are present, then run see-through.
#
# Weights default to f16 (WEIGHTS_TAG=v0.1.0). Quantization is a deliberate
# ladder measured against an f16 reference, not a starting point --
# see docs/quantization-ladder.md. Do not flip the default to q4.
#
# Set SKIP_WEIGHT_FETCH=1 when MODELS_DIR is a pre-populated volume.

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"

# `selftest` answers the three questions this image cannot otherwise be asked
# on a host: is there a Vulkan ICD, does the loader enumerate the GPU, and
# does the binary run. Kept here because ENTRYPOINT is fixed -- RunPod's
# dockerArgs overrides CMD, not ENTRYPOINT, so there is no other way in.
# Skips the weight fetch: this is a device probe, not an inference run.
if [[ "${1:-}" == "selftest" ]]; then
  rc=0
  echo "=== NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"

  echo "=== ICD manifests (installed by the driver when caps include graphics)"
  ls -1 /usr/share/vulkan/icd.d/ 2>&1 || { echo "NO ICD DIRECTORY"; rc=1; }

  echo "=== vulkaninfo --summary"
  if ! vulkaninfo --summary 2>&1 | sed -n '1,40p'; then
    echo "vulkaninfo FAILED"; rc=1
  fi

  echo "=== see-through --help"
  /usr/local/bin/see-through --help 2>&1 | sed -n '1,15p' || rc=1

  echo "=== selftest exit rc=$rc"
  exit "$rc"
fi

if [[ "${SKIP_WEIGHT_FETCH:-0}" != "1" ]]; then
  /usr/local/bin/fetch-weights.sh
fi

exec /usr/local/bin/see-through -m "$MODELS_DIR" "$@"
