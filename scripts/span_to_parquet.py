"""Convert profiling spans to ETNF parquet (zstd).
Reads all available span files, flattens to Essential Tuple Normal Form,
and writes as a single parquet with zstd compression."""
import json, os, pyarrow as pa, pyarrow.parquet as pq, pandas as pd

# Collect all span files with actual data
span_files = []
for f in ['profiling/spans.jsonl', '/tmp/spans_old.jsonl', '/tmp/spans_res64.jsonl']:
    if os.path.exists(f) and os.path.getsize(f) > 0:
        span_files.append(f)
        print(f"  {f}: {os.path.getsize(f)} bytes")
print(f"Reading {len(span_files)} span files")

rows = []
for sf in span_files:
    with open(sf) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            s = json.loads(line)
            rows.append({
                'trace_id': s.get('trace_id',''),
                'span_id': s.get('span_id',''),
                'parent_span_id': s.get('parent_span_id',''),
                'name': s.get('name',''),
                'start_time_unix_nano': s.get('start_time_unix_nano',0),
                'end_time_unix_nano': s.get('end_time_unix_nano',0),
                'duration_ms': s.get('duration_ms',0.0),
                'status_code': s.get('status_code',0),
                'attributes_json': json.dumps(s.get('attributes',{}), sort_keys=True),
                'source_file': os.path.basename(sf),
            })

df = pd.DataFrame(rows)
print(f"Total spans: {len(df)}")
for name, cnt in df['name'].value_counts().items():
    print(f"  {name:30s} {cnt:4d} spans")

# Schema: all primitive types, no nested structs -- ETNF
table = pa.Table.from_pandas(df, preserve_index=False)
out_path = 'profiling/spans.parquet'
pq.write_table(table, out_path, compression='zstd', flavor='spark')
sz = os.path.getsize(out_path)
print(f"Wrote {out_path} ({sz} bytes, {sz/1024:.1f} KB)")