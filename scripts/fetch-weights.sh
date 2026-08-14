#!/usr/bin/env bash
# Populate a models/ directory from this repo's GitHub Releases.
#
# Release assets come in three shapes -- plain .gguf, single .gguf.zst, and
# split .gguf.zst.partNN (GitHub caps release assets at 2GB, and the f16
# layerdiff unet is ~5.9GB). All three are normalised here to the canonical
# filenames src/pipeline.cpp loads, e.g. "<dir>/layerdiff-unet.gguf".
#
# Public repo, so no auth: plain HTTPS, no gh CLI or token needed.
#
#   WEIGHTS_TAG=v0.1.0      f16 (default)
#   WEIGHTS_TAG=v0.0.2-dev  required for WEIGHTS_VARIANT=q4
#   WEIGHTS_VARIANT=q4      prefer <name>-q4 assets, written to canonical names
#
# Idempotent: an already-present model is skipped, so mounting a persistent
# volume at MODELS_DIR turns this into a one-time cost.

set -euo pipefail

REPO="${WEIGHTS_REPO:-weftspun/interactor-seethrough-ggml}"
TAG="${WEIGHTS_TAG:-v0.1.0}"
VARIANT="${WEIGHTS_VARIANT:-f16}"
MODELS_DIR="${MODELS_DIR:-/models}"

MODELS=(lama layerdiff-te1 layerdiff-te2 layerdiff-vae layerdiff-unet
        marigold-te marigold-unet marigold-vae trans-vae)

mkdir -p "$MODELS_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "fetch-weights: repo=$REPO tag=$TAG variant=$VARIANT dir=$MODELS_DIR"

assets="$tmp/assets.json"
curl -fsSL "https://api.github.com/repos/$REPO/releases/tags/$TAG" -o "$assets"

# name -> download URL
declare -A URL
while IFS=$'\t' read -r n u; do URL["$n"]="$u"; done < <(
  jq -r '.assets[] | "\(.name)\t\(.browser_download_url)"' "$assets"
)

have() { [[ -n "${URL[$1]:-}" ]]; }

fetch_to() {  # $1=asset name, $2=dest path
  echo "  get $1"
  curl -fsSL --retry 3 --retry-delay 2 "${URL[$1]}" -o "$2"
}

for m in "${MODELS[@]}"; do
  dest="$MODELS_DIR/$m.gguf"
  if [[ -s "$dest" ]]; then echo "  skip $m.gguf (present)"; continue; fi

  # Candidate basenames, most-preferred first.
  cands=("$m")
  [[ "$VARIANT" == "q4" ]] && cands=("$m-q4" "$m")

  done_one=0
  for c in "${cands[@]}"; do
    if have "$c.gguf"; then
      fetch_to "$c.gguf" "$dest"; done_one=1; break

    elif have "$c.gguf.zst"; then
      fetch_to "$c.gguf.zst" "$tmp/$c.gguf.zst"
      zstd -d -q -o "$dest" "$tmp/$c.gguf.zst"; done_one=1; break

    elif have "$c.gguf.zst.part00"; then
      parts=$(jq -r --arg p "$c.gguf.zst.part" \
        '.assets[].name | select(startswith($p))' "$assets" | sort)
      for p in $parts; do fetch_to "$p" "$tmp/$p"; done
      # shellcheck disable=SC2086
      cat $(printf "$tmp/%s " $parts) | zstd -d -q -o "$dest"
      rm -f "$tmp/$c.gguf.zst.part"*
      done_one=1; break
    fi
  done

  if [[ "$done_one" -eq 0 ]]; then
    echo "fetch-weights: no asset for '$m' in $REPO@$TAG (variant=$VARIANT)" >&2
    exit 1
  fi
done

echo "fetch-weights: complete"
du -sh "$MODELS_DIR" 2>/dev/null || true
