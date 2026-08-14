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

if [[ "${SKIP_WEIGHT_FETCH:-0}" != "1" ]]; then
  /usr/local/bin/fetch-weights.sh
fi

exec /usr/local/bin/see-through -m "$MODELS_DIR" "$@"
