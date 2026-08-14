#!/usr/bin/env python3
"""Cog-compatible HTTP surface for see-through.

Speaks Cog's *wire* format so any Cog client works, without depending on the
cog package -- this image is a C++ binary, not a Python model, and pulling in
the framework to satisfy an HTTP contract would be the wrong trade.

    GET  /health-check   {"status": "READY"|"STARTING"|"SETUP_FAILED", ...}
    POST /predictions    {"input": {...}} -> prediction object
    GET  /               OTel NDJSON log records (non-Cog, for debugging)

Per the Cog spec, file outputs are returned as base64 data URLs. Layers come
from the binary's own --png-dir, so each layer is a separate output entry and
can be viewed individually rather than unpacked from the PSD.

Stdlib only: the runtime image has no pip and does not need one.
"""

import base64
import json
import threading
import mimetypes
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/out"))
LOGF = Path(os.environ.get("LOGF", "/tmp/otel-logs.ndjson"))
PORT = int(os.environ.get("LOG_PORT", "8080"))
BINARY = os.environ.get("SEETHROUGH_BIN", "/usr/local/bin/see-through")

SETUP_STATUS = {"status": "READY", "started_at": time.time(), "logs": ""}

# Async predictions. RunPod fronts pods with a proxy that closes long requests
# (observed: HTTP 524 at ~100s), so a synchronous 1280px/30-step run can never
# return over the wire. Cog's spec already covers this: `Prefer: respond-async`
# returns 202 immediately and the client polls GET /predictions/{id}.
PREDICTIONS = {}
_LOCK = threading.Lock()


def _store(pred):
    with _LOCK:
        PREDICTIONS[pred["id"]] = pred
    # Persist so a result outlives the process; the container can be restarted
    # or the server can die without losing a completed run.
    try:
        d = OUT_DIR / pred["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "prediction.json").write_text(json.dumps(pred))
    except OSError:
        pass


def _load(pid):
    with _LOCK:
        if pid in PREDICTIONS:
            return PREDICTIONS[pid]
    f = OUT_DIR / pid / "prediction.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def emit(severity, body, **attrs):
    """Append an OTel LogRecord, matching entrypoint.sh's format."""
    rec = {
        "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
        "SeverityText": severity,
        "SeverityNumber": {"INFO": 9, "WARN": 13, "ERROR": 17}.get(severity, 0),
        "Body": body,
        "Attributes": attrs,
        "Resource": {
            "service.name": os.environ.get(
                "OTEL_SERVICE_NAME", "interactor-seethrough-ggml"
            ),
            "service.version": os.environ.get("OTEL_SERVICE_VERSION", "unknown"),
            "runpod.pod.id": os.environ.get("RUNPOD_POD_ID", "unknown"),
        },
    }
    try:
        with LOGF.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass
    print(json.dumps(rec), flush=True)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def fetch_input(spec: str, dest: Path) -> Path:
    """Cog passes file inputs as URLs; data: URLs are accepted too."""
    if spec.startswith("data:"):
        dest.write_bytes(base64.b64decode(spec.split(",", 1)[1]))
    else:
        with urllib.request.urlopen(spec, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def run_prediction(inp: dict, pid=None) -> dict:
    pid = pid or str(uuid.uuid4())
    work = OUT_DIR / pid
    layers = work / "layers"
    layers.mkdir(parents=True, exist_ok=True)

    started = time.time()
    src = fetch_input(inp["image"], work / "in.png")
    psd = work / "out.psd"

    cmd = [BINARY, "-m", MODELS_DIR, "-i", str(src), "-o", str(psd),
           "--png-dir", str(layers)]
    for flag, key in (("--steps", "steps"), ("--res", "res"),
                      ("--seed", "seed"), ("--device", "device")):
        if inp.get(key) is not None:
            cmd += [flag, str(inp[key])]

    emit("INFO", "prediction start", **{"event.name": "prediction.start",
                                        "prediction.id": pid})
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        emit("INFO", line, **{"event.name": "seethrough"})

    elapsed = time.time() - started
    if proc.returncode != 0:
        emit("ERROR", "prediction failed",
             **{"event.name": "prediction.failed",
                "process.exit.code": proc.returncode})
        return {"id": pid, "input": inp, "status": "failed",
                "error": (proc.stderr or "see-through failed")[-4000:],
                "logs": (proc.stdout + proc.stderr)[-8000:],
                "metrics": {"predict_time": elapsed}}

    # One output entry per layer, so a client can view them individually.
    outputs = [data_url(p) for p in sorted(layers.glob("*.png"))]
    if psd.exists():
        outputs.append(data_url(psd))

    emit("INFO", "prediction complete",
         **{"event.name": "prediction.end", "prediction.id": pid,
            "output.layer.count": len(outputs), "predict_time": elapsed})
    return {"id": pid, "input": inp, "status": "succeeded",
            "output": outputs,
            "logs": (proc.stdout + proc.stderr)[-8000:],
            "metrics": {"predict_time": elapsed,
                        "layer_count": max(len(outputs) - 1, 0)}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):  # keep stdout as the OTel stream only
        pass

    def do_GET(self):
        if self.path.startswith("/health-check"):
            self._send(200, {"status": SETUP_STATUS["status"],
                             "setup": {"status": "succeeded", "logs": ""},
                             "version": {"cog": "wire-compatible",
                                         "python": sys.version.split()[0]}})
        elif self.path.startswith("/predictions/"):
            pid = self.path.split("/predictions/", 1)[1].split("?")[0]
            pred = _load(pid)
            self._send(200, pred) if pred else self._send(404, {"detail": "not found"})
        elif self.path in ("/", "/logs"):
            data = LOGF.read_bytes() if LOGF.exists() else b""
            self._send(200, data, "application/x-ndjson")
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        if not self.path.startswith("/predictions"):
            self._send(404, {"detail": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"detail": "invalid JSON"})
            return
        inp = req.get("input") or {}
        if "image" not in inp:
            self._send(422, {"detail": "input.image is required"})
            return

        if "respond-async" in (self.headers.get("Prefer") or "").lower():
            pid = req.get("prediction_id") or str(uuid.uuid4())
            _store({"id": pid, "input": inp, "status": "starting",
                    "output": None, "logs": "", "metrics": {}})

            def worker():
                _store({**_load(pid), "status": "processing"})
                try:
                    _store(run_prediction(inp, pid))
                except Exception as exc:
                    _store({"id": pid, "input": inp, "status": "failed",
                            "error": str(exc), "metrics": {}})

            threading.Thread(target=worker, daemon=True).start()
            self._send(202, _load(pid))
            return

        try:
            self._send(200, run_prediction(inp))
        except Exception as exc:  # surface as a failed prediction, per spec
            self._send(200, {"status": "failed", "error": str(exc)})


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emit("INFO", "cog-compatible server listening",
         **{"event.name": "server.start", "server.port": PORT})
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
