#!/usr/bin/env bash
# Container entrypoint: ensure weights are present, then run see-through.
#
# Weights default to f16 (WEIGHTS_TAG=v0.1.0). Quantization is a deliberate
# ladder measured against an f16 reference, not a starting point --
# see docs/quantization-ladder.md. Do not flip the default to q4.
#
# Set SKIP_WEIGHT_FETCH=1 when MODELS_DIR is a pre-populated volume.
#
# Observability follows OpenTelemetry conventions: log records are emitted as
# NDJSON in OTLP LogRecord shape (Timestamp / SeverityText / SeverityNumber /
# Body / Attributes) with a Resource block built from OTEL_SERVICE_NAME and
# OTEL_RESOURCE_ATTRIBUTES. No collector is required -- records are appended
# to a file and served over a log endpoint (see serve_log). If a collector is
# ever wanted, this file is already valid OTLP/JSON input.

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
LOGF="${LOGF:-/tmp/otel-logs.ndjson}"
LOG_PORT="${LOG_PORT:-8080}"
OUT_DIR="${OUT_DIR:-/out}"

OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-interactor-seethrough-ggml}"
OTEL_SERVICE_VERSION="${OTEL_SERVICE_VERSION:-unknown}"

# --- OTel resource ----------------------------------------------------------
# Semantic conventions: service.*, container.*, host.*. RunPod is not in the
# cloud.provider enum, so it is carried as a non-normative attribute rather
# than pretending to be one of the registered values.
build_resource() {
  local extra='{}'
  if [[ -n "${OTEL_RESOURCE_ATTRIBUTES:-}" ]]; then
    extra=$(printf '%s' "$OTEL_RESOURCE_ATTRIBUTES" | jq -Rc '
      split(",") | map(select(length>0) | split("=") | {(.[0]): (.[1] // "")}) | add // {}')
  fi
  jq -nc --arg svc "$OTEL_SERVICE_NAME" \
         --arg ver "$OTEL_SERVICE_VERSION" \
         --arg host "$(hostname 2>/dev/null || echo unknown)" \
         --arg img "${CONTAINER_IMAGE_NAME:-ghcr.io/weftspun/interactor-seethrough-ggml}" \
         --arg pod "${RUNPOD_POD_ID:-unknown}" \
         --argjson extra "$extra" '
    {"service.name":$svc, "service.version":$ver,
     "host.name":$host, "container.image.name":$img,
     "runpod.pod.id":$pod} + $extra'
}
RESOURCE="$(build_resource)"

# SeverityNumber per the OTel log data model.
sev_num() { case "$1" in TRACE) echo 1;; DEBUG) echo 5;; INFO) echo 9;;
                          WARN) echo 13;; ERROR) echo 17;; FATAL) echo 21;;
                          *) echo 0;; esac; }

emit() { # emit <SEVERITY> <body> [attr-json]
  local sev="$1" body="$2" attrs="${3:-{\}}"
  jq -nc --arg ts "$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)" \
         --arg sev "$sev" --argjson sevn "$(sev_num "$sev")" \
         --arg body "$body" --argjson attrs "$attrs" \
         --argjson res "$RESOURCE" '
    {Timestamp:$ts, SeverityText:$sev, SeverityNumber:$sevn,
     Body:$body, Attributes:$attrs, Resource:$res}' >> "$LOGF"
}

emit_stream() { # emit_stream <SEVERITY> <event.name>  -- one record per stdin line
  local sev="$1" ev="$2"
  while IFS= read -r line; do
    emit "$sev" "$line" "$(jq -nc --arg e "$ev" '{"event.name":$e}')"
    printf '%s\n' "$line"
  done
}

# --- log endpoint -----------------------------------------------------------
# RunPod exposes container logs only in its web console: there is no logs path
# in the REST API (checked against /v1/openapi.json) and GraphQL introspection
# is disabled. Serve the records ourselves instead.
#
# Re-read per request and served from the start, so a long run is pollable
# while it is still computing:
#     curl https://<podId>-8080.proxy.runpod.net
# Deploy the pod with ports "8080/http" so the proxy hostname exists.
# Requests are dispatched by path so artifacts can be pulled off the pod --
# a PSD is the actual deliverable and there is no other way out of a RunPod
# container (no logs endpoint, no file API, no console download).
#
#   GET /           OTel NDJSON records (pollable during a run)
#   GET /ls         listing of OUT_DIR
#   GET /f/<name>   raw artifact, Content-Length set so binaries transfer intact
http_handler() {
  local method path proto h f n body
  read -r method path proto || return
  while IFS= read -r h; do [[ "$h" == $'\r' || -z "$h" ]] && break; done
  path="${path%%\?*}"
  case "$path" in
    /|/logs)
      n=$(wc -c < "$LOGF" 2>/dev/null || echo 0)
      printf 'HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n'
      printf 'Content-Length: %s\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n' "$n"
      cat "$LOGF" 2>/dev/null
      ;;
    /ls)
      body=$( { cd "$OUT_DIR" 2>/dev/null && ls -la; } 2>&1 || echo "no output dir" )
      printf 'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n'
      printf 'Content-Length: %s\r\nConnection: close\r\n\r\n%s\n' "$(( ${#body} + 1 ))" "$body"
      ;;
    /f/*)
      f="$OUT_DIR/$(basename "${path#/f/}")"
      if [[ -f "$f" ]]; then
        printf 'HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n'
        printf 'Content-Length: %s\r\nConnection: close\r\n\r\n' "$(wc -c < "$f")"
        cat "$f"
      else
        printf 'HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'
      fi
      ;;
    *)
      printf 'HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'
      ;;
  esac
}

serve_log() {
  local fifo=/tmp/.httpreq
  rm -f "$fifo"; mkfifo "$fifo"
  while true; do
    http_handler < "$fifo" | nc -l "$LOG_PORT" > "$fifo" 2>/dev/null || sleep 1
  done
}

mkdir -p "$OUT_DIR"
: > "$LOGF"
if [[ "${LOG_SERVE:-1}" == "1" ]] && command -v nc >/dev/null 2>&1; then
  serve_log &
  emit INFO "log endpoint listening" "$(jq -nc --argjson p "$LOG_PORT" \
    '{"event.name":"log.endpoint.start","server.port":$p}')"
fi

# --- serve (Cog wire API) ---------------------------------------------------
# POST /predictions, GET /health-check -- Cog's HTTP contract without the cog
# package. Layers come back as separate outputs (base64 data URLs per the Cog
# spec), so results are viewable layer by layer rather than only as a PSD.
if [[ "${1:-}" == "serve" ]]; then
  if [[ "${SKIP_WEIGHT_FETCH:-0}" != "1" ]]; then
    emit INFO "weight fetch start" "$(jq -nc --arg t "${WEIGHTS_TAG:-}" --arg v "${WEIGHTS_VARIANT:-}" \
      '{"event.name":"weights.fetch.start","weights.tag":$t,"weights.variant":$v}')"
    /usr/local/bin/fetch-weights.sh 2>&1 | emit_stream INFO "weights.fetch"
    emit INFO "weight fetch complete" '{"event.name":"weights.fetch.end"}'
  fi
  exec python3 /usr/local/bin/cog_server.py
fi

# --- selftest ---------------------------------------------------------------
# Answers what this image cannot otherwise be asked on a host: is there a
# Vulkan ICD, does the loader enumerate the GPU, does the binary run.
# ENTRYPOINT is fixed and RunPod's dockerArgs overrides CMD, not ENTRYPOINT,
# so there is no other way in. Skips the weight fetch: a device probe.
if [[ "${1:-}" == "selftest" ]]; then
  rc=0
  emit INFO "selftest start" "$(jq -nc --arg c "${NVIDIA_DRIVER_CAPABILITIES:-unset}" \
    '{"event.name":"selftest.start","nvidia.driver.capabilities":$c}')"

  # The driver may inject its manifest into either location; checking only
  # /usr/share produced a false alarm on 2026-08-13 while the loader was in
  # fact reading /etc/vulkan/icd.d.
  found_icd=0
  for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    if ls -1 "$d" >/dev/null 2>&1; then
      found_icd=1
      ls -1 "$d" 2>&1 | emit_stream INFO "vulkan.icd.manifest"
    fi
  done
  [[ "$found_icd" -eq 1 ]] || { emit ERROR "no ICD manifest directory" '{"event.name":"vulkan.icd.missing"}'; rc=1; }

  # Driver-injection diagnostics. The ICD JSON can be present and its library
  # loadable while still not exporting vk_icdGetInstanceProcAddr, which happens
  # when the runtime injects an incomplete driver set or ldconfig was never
  # re-run after injection. Guessing apt packages against that is a coin flip;
  # these three probes name the cause.
  cat /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null \
    | emit_stream INFO "vulkan.icd.json"
  ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* \
         /usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv.so* \
         /usr/lib/x86_64-linux-gnu/libnvidia-vulkan* 2>&1 \
    | emit_stream INFO "vulkan.driver.libs"
  ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>&1 | grep -i "not found\|=>" \
    | sed -n '1,25p' | emit_stream INFO "vulkan.driver.ldd"
  ldconfig -p 2>/dev/null | grep -ci nvidia | sed 's/^/nvidia libs in ldconfig cache: /' \
    | emit_stream INFO "vulkan.ldconfig"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 \
    | emit_stream INFO "gpu.info"

  # NVIDIA/nvidia-container-toolkit#1952 (open as of 2026-07): libGLX_nvidia
  # tries to initialise windowing in a headless container and fails to export
  # the ICD entry point. Reports there associate it with /dev/dri not being
  # mounted, so record whether it exists here.
  ls -la /dev/dri 2>&1 | sed -n '1,10p' | emit_stream INFO "dev.dri"

  # Re-run ldconfig in case injection left the cache stale, then retry once.
  ldconfig 2>/dev/null || true

  if vulkaninfo --summary 2>&1 | sed -n '1,40p' | emit_stream INFO "vulkan.info"; then :; else
    emit ERROR "vulkaninfo failed" '{"event.name":"vulkan.info.failed"}'; rc=1
  fi

  /usr/local/bin/see-through --help 2>&1 | sed -n '1,15p' | emit_stream INFO "seethrough.help" || rc=1

  emit INFO "selftest complete" "$(jq -nc --argjson rc "$rc" \
    '{"event.name":"selftest.end","process.exit.code":$rc}')"

  # Stay alive so records can be fetched. The pod watchdog terminates; exiting
  # here would drop the log with the container.
  if [[ "${LOG_SERVE:-1}" == "1" ]]; then
    emit INFO "holding open for log retrieval" '{"event.name":"hold.open"}'
    while true; do sleep 30; done
  fi
  exit "$rc"
fi

# --- normal run -------------------------------------------------------------
if [[ "${SKIP_WEIGHT_FETCH:-0}" != "1" ]]; then
  emit INFO "weight fetch start" "$(jq -nc --arg t "${WEIGHTS_TAG:-}" --arg v "${WEIGHTS_VARIANT:-}" \
    '{"event.name":"weights.fetch.start","weights.tag":$t,"weights.variant":$v}')"
  if /usr/local/bin/fetch-weights.sh 2>&1 | emit_stream INFO "weights.fetch"; then
    emit INFO "weight fetch complete" '{"event.name":"weights.fetch.end"}'
  else
    emit ERROR "weight fetch failed" '{"event.name":"weights.fetch.failed"}'
  fi
fi

emit INFO "inference start" '{"event.name":"inference.start"}'
/usr/local/bin/see-through -m "$MODELS_DIR" "$@" 2>&1 | emit_stream INFO "seethrough"
rc="${PIPESTATUS[0]}"
emit INFO "inference complete" "$(jq -nc --argjson rc "$rc" \
  '{"event.name":"inference.end","process.exit.code":$rc}')"

if [[ "${HOLD_OPEN:-0}" == "1" ]]; then
  emit INFO "holding open for log retrieval" '{"event.name":"hold.open"}'
  while true; do sleep 30; done
fi
exit "$rc"
