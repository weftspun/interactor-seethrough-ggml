#!/usr/bin/env python3
"""ETNF (Essential Tuple Normal Form) parquet data lake builder.

Reads OpenTelemetry-style spans from profiling/spans.jsonl and materializes
them as parquet in Essential Tuple Normal Form — flat, atomic tuples with no
nesting. Two tables:

  spans.parquet       — one row per span (scalar fields: trace_id, span_id,
                        parent_span_id, name, start/end time, duration_ms,
                        status_code)
  span_attrs.parquet  — one row per attribute key-value pair, linked by span_id

Join: SELECT * FROM spans JOIN span_attrs USING (span_id)

Usage:
  python3 tools/etnf_build.py                           # default paths
  python3 tools/etnf_build.py --input profiling/spans.jsonl --output profiling/lake

Can be called repeatedly — new runs' spans are appended (deduped by span_id).
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

# ── Schemas (Essential Tuple Normal Form — flat, no nesting) ────────────────

SPAN_SCHEMA = pa.schema([
    pa.field("trace_id",            pa.string(),    nullable=False),
    pa.field("span_id",             pa.string(),    nullable=False),
    pa.field("parent_span_id",      pa.string(),    nullable=True),
    pa.field("name",                pa.string(),    nullable=False),
    pa.field("start_time_unix_nano", pa.int64(),    nullable=False),
    pa.field("end_time_unix_nano",   pa.int64(),    nullable=False),
    pa.field("duration_ms",         pa.float64(),   nullable=False),
    pa.field("status_code",         pa.int32(),     nullable=False),
])

ATTR_SCHEMA = pa.schema([
    pa.field("span_id",             pa.string(),    nullable=False),
    pa.field("key",                 pa.string(),    nullable=False),
    pa.field("value",               pa.string(),    nullable=True),
])


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_spans(jsonl_path: str) -> tuple[list[dict], list[dict]]:
    """Read spans.jsonl, return (span_rows, attr_rows) in ETNF."""
    span_rows = []
    attr_rows = []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            span_rows.append({
                "trace_id":            obj.get("trace_id", ""),
                "span_id":             obj.get("span_id", ""),
                "parent_span_id":      obj.get("parent_span_id", "") or None,
                "name":                obj.get("name", ""),
                "start_time_unix_nano": int(obj.get("start_time_unix_nano", 0)),
                "end_time_unix_nano":   int(obj.get("end_time_unix_nano", 0)),
                "duration_ms":         float(obj.get("duration_ms", 0.0)),
                "status_code":         int(obj.get("status_code", 1)),
            })

            attrs = obj.get("attributes", {})
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    attr_rows.append({
                        "span_id": obj["span_id"],
                        "key":     k,
                        "value":   str(v) if v is not None else None,
                    })
            elif isinstance(attrs, list):
                for item in attrs:
                    attr_rows.append({
                        "span_id": obj["span_id"],
                        "key":     item.get("key", ""),
                        "value":   str(item.get("value", "")) if item.get("value") is not None else None,
                    })

    return span_rows, attr_rows


# ── Idempotent append (dedup by span_id) ────────────────────────────────────

def _existing_ids(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    try:
        t = pq.read_table(path, columns=["span_id"])
        return set(t.column("span_id").to_pylist())
    except Exception:
        return set()


def _append(path: str, rows: list[dict], schema: pa.Schema):
    if not rows:
        return
    known = _existing_ids(path)
    uniq = [r for r in rows if r["span_id"] not in known]
    if not uniq:
        return
    tbl = pa.Table.from_pylist(uniq, schema=schema)
    if os.path.isfile(path):
        tbl = pa.concat_tables([pq.read_table(path), tbl])
    pq.write_table(tbl, path)


# ── Build ───────────────────────────────────────────────────────────────────

def build_lake(jsonl_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    span_rows, attr_rows = parse_spans(jsonl_path)

    spans_path = os.path.join(output_dir, "spans.parquet")
    attrs_path = os.path.join(output_dir, "span_attrs.parquet")

    _append(spans_path, span_rows, SPAN_SCHEMA)
    _append(attrs_path, attr_rows, ATTR_SCHEMA)

    n_spans = len(span_rows)
    n_attrs = len(attr_rows)
    print(f"ETNF lake: {n_spans} spans, {n_attrs} attrs → {output_dir}/")
    print(f"  spans.parquet       ({_fmt_size(spans_path)})")
    print(f"  span_attrs.parquet  ({_fmt_size(attrs_path)})")

    if os.path.isfile(spans_path):
        preview = pq.read_table(spans_path)
        print(f"\nPreview: {preview.num_rows} total span rows")
        print(preview.to_string())


def _fmt_size(path):
    if not os.path.isfile(path):
        return "0 B"
    sz = os.path.getsize(path)
    for unit in ("B", "KiB", "MiB"):
        if sz < 1024:
            return f"{sz:.1f} {unit}"
        sz /= 1024.0
    return f"{sz:.1f} GiB"


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ETNF parquet data lake builder for OTel-style spans")
    parser.add_argument("--input",  default="profiling/spans.jsonl",
                        help="Path to spans.jsonl (default: profiling/spans.jsonl)")
    parser.add_argument("--output", default="profiling/lake",
                        help="Output directory (default: profiling/lake)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    build_lake(args.input, args.output)


if __name__ == "__main__":
    main()