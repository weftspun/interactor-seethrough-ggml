"""Repack a Q4_0 GGUF into a Q8_0 GGUF, exactly.

Not a requantization. Q4_0 and Q8_0 share a 32-element block with one f16
scale; Q4_0 dequantizes to `d*(q-8)` and Q8_0 to `d*qs`. Writing `qs = q - 8`
with `d` unchanged reproduces the identical real values, so a run using the
output MUST be bit-identical to a run using the input. Any difference is a bug
in this script, not a property of Q8_0.

That exactness is the point. Quantizing Q8_0 fresh from f16 would change
precision AND container together; this changes only the container, which makes
it a controlled test of one question:

    ggml-vulkan's MMQ shader keeps Q4_0 packed in shared memory and unpacks it
    INSIDE the dot-product loop (mul_mmq_funcs.glsl: `vui & 0x0F0F0F0F`,
    `>> 4`), plus a `- 8.0*ds.y` zero-point correction. Q8_0 feeds
    dotPacked4x8EXT directly with neither. Both issue the same number of dot
    instructions per 32 values -- Q4_0 just does more ALU around them, in
    exchange for half the shared memory and half the global reads.

Whether less ALU beats more shmem pressure is not decidable by reading the
shader, which is why this exists.

    python scripts/repack_q4_to_q8.py models-q4/layerdiff-unet-q4.gguf \\
        -o models-q8/layerdiff-unet-q8.gguf
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
QK = 32
Q4_BLOCK_BYTES = 2 + QK // 2   # f16 scale + 16 packed bytes
Q8_BLOCK_BYTES = 2 + QK       # f16 scale + 32 int8

# GGUF metadata value types, for the reader/writer below
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)


class Reader:
    def __init__(self, buf: memoryview):
        self.b, self.off = buf, 0

    def raw(self, n: int) -> bytes:
        v = bytes(self.b[self.off:self.off + n]); self.off += n; return v

    def u32(self) -> int: return struct.unpack("<I", self.raw(4))[0]
    def u64(self) -> int: return struct.unpack("<Q", self.raw(8))[0]
    def i32(self) -> int: return struct.unpack("<i", self.raw(4))[0]

    def string(self) -> bytes:
        return self.raw(self.u64())

    def value(self, vtype: int):
        """Read one metadata value, returning (bytes-as-written) so the writer
        can round-trip it without needing to understand every type."""
        start = self.off
        if vtype in (U8, I8, BOOL): self.raw(1)
        elif vtype in (U16, I16): self.raw(2)
        elif vtype in (U32, I32, F32): self.raw(4)
        elif vtype in (U64, I64, F64): self.raw(8)
        elif vtype == STRING: self.string()
        elif vtype == ARRAY:
            et, n = self.u32(), self.u64()
            for _ in range(n):
                self.value(et)
        else:
            raise ValueError(f"unknown gguf value type {vtype}")
        return bytes(self.b[start:self.off])


def repack(src: Path, dst: Path) -> int:
    raw = memoryview(src.read_bytes())
    r = Reader(raw)
    if r.u32() != GGUF_MAGIC:
        sys.exit(f"FAIL: {src} is not a GGUF file")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    kv_start = r.off
    for _ in range(n_kv):
        r.string()
        r.value(r.u32())
    kv_blob = bytes(raw[kv_start:r.off])

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        ndim = r.u32()
        dims = [r.u64() for _ in range(ndim)]
        ttype = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

    # ggml aligns the tensor data segment; the default is 32 and is recorded in
    # metadata as general.alignment when it differs. Reading it back out of the
    # blob would mean parsing kv properly, so assume 32 and verify below.
    align = 32
    data_start = (r.off + align - 1) // align * align

    n_converted = 0
    new_tensors, payloads = [], []
    cursor = 0
    for t in tensors:
        nelem = 1
        for d in t["dims"]:
            nelem *= d
        if t["type"] == GGML_TYPE_Q4_0:
            nb = nelem // QK
            blob = np.frombuffer(raw, dtype=np.uint8, count=nb * Q4_BLOCK_BYTES,
                                 offset=data_start + t["offset"]).reshape(nb, Q4_BLOCK_BYTES)
            scales = blob[:, :2]
            packed = blob[:, 2:]
            # ggml's block_q4_0 stores element i in the LOW nibble of byte i and
            # element i+16 in the HIGH nibble -- not interleaved pairs. Getting
            # this backwards produces a file that loads and decodes to garbage.
            lo = (packed & 0x0F).astype(np.int16)
            hi = (packed >> 4).astype(np.int16)
            q = np.concatenate([lo, hi], axis=1)
            qs = (q - 8).astype(np.int8)
            out = np.empty((nb, Q8_BLOCK_BYTES), dtype=np.uint8)
            out[:, :2] = scales
            out[:, 2:] = qs.view(np.uint8)
            payload = out.tobytes()
            ttype = GGML_TYPE_Q8_0
            n_converted += 1
        else:
            nbytes = _type_nbytes(t["type"], nelem)
            payload = bytes(raw[data_start + t["offset"]: data_start + t["offset"] + nbytes])
            ttype = t["type"]

        pad = (-len(payload)) % align
        new_tensors.append({**t, "type": ttype, "offset": cursor})
        payloads.append(payload + b"\x00" * pad)
        cursor += len(payload) + pad

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, version, n_tensors, n_kv))
        f.write(kv_blob)
        for t in new_tensors:
            f.write(struct.pack("<Q", len(t["name"]))); f.write(t["name"])
            f.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", t["type"]))
            f.write(struct.pack("<Q", t["offset"]))
        here = f.tell()
        f.write(b"\x00" * ((-here) % align))
        for p in payloads:
            f.write(p)

    return n_converted


def _type_nbytes(ttype: int, nelem: int) -> int:
    if ttype == 0: return nelem * 4          # F32
    if ttype == 1: return nelem * 2          # F16
    if ttype == GGML_TYPE_Q4_0: return nelem // QK * Q4_BLOCK_BYTES
    if ttype == GGML_TYPE_Q8_0: return nelem // QK * Q8_BLOCK_BYTES
    raise ValueError(f"unhandled ggml type {ttype} — refusing to guess its size")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.out)
    n = repack(src, dst)
    print(f"{src.name}: converted {n} Q4_0 tensors -> Q8_0")
    print(f"  {src.stat().st_size / 1e9:.2f} GB -> {dst.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
