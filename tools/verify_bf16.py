#!/usr/bin/env python3
"""Verify our bf16 GGUF against the q8_0 GGUF moshi.cpp wrote, tensor by tensor.
Q8_0 tensors: dequantize and compare with our BF16 -> cosine similarity and rel. error.
F32 tensors: must be bit-identical (both are exact bf16->f32 casts).
A wrong name mapping, split order or transpose shows up as cosine ~0, not ~1."""
import struct, sys
import numpy as np
A, B = sys.argv[1:3]   # ours (bf16), q8
def index(path):
    f = open(path, "rb")
    def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    def rs(): n = rd("Q"); return f.read(n).decode()
    assert f.read(4) == b"GGUF"; rd("I"); nt = rd("Q"); nkv = rd("Q"); assert nkv == 0
    ts = []
    for _ in range(nt):
        name = rs(); nd = rd("I"); ne = [rd("Q") for _ in range(nd)]; t = rd("I"); off = rd("Q"); ts.append((name, ne, t, off))
    base = (f.tell() + 31) // 32 * 32
    return f, base, ts
fa, ba, ta = index(A); fb, bb, tb = index(B)
assert [t[0] for t in ta] == [t[0] for t in tb], "name/order mismatch"
assert [t[1] for t in ta] == [t[1] for t in tb], "ne mismatch"
worst = (2.0, None); worst_rel = (0.0, None); n_f32 = 0
for (name, ne, t, oa), (_, _, tq, ob) in zip(ta, tb):
    nel = int(np.prod(ne))
    if t == 0:
        assert tq == 0
        fa.seek(ba + oa); fb.seek(bb + ob)
        assert fa.read(nel * 4) == fb.read(nel * 4), f"F32 mismatch {name}"
        n_f32 += 1; continue
    assert t == 30 and tq == 8, (name, t, tq)
    xy = xx = yy = dd = 0.0
    CH = 1 << 22   # elements per chunk (multiple of 32)
    for c0 in range(0, nel, CH):
        c = min(CH, nel - c0)
        fa.seek(ba + oa + c0 * 2)
        x = (np.frombuffer(fa.read(c * 2), np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        fb.seek(bb + ob + c0 // 32 * 34)
        raw = np.frombuffer(fb.read(c // 32 * 34), np.uint8).reshape(-1, 34)
        d = raw[:, :2].copy().view(np.float16).astype(np.float64)
        y = (d * raw[:, 2:].view(np.int8).astype(np.float64)).reshape(-1)
        xy += float(x @ y); xx += float(x @ x); yy += float(y @ y); dd += float((x - y) @ (x - y))
    cos = xy / (np.sqrt(xx * yy) + 1e-30)
    rel = float(np.sqrt(dd / (xx + 1e-30)))
    if cos < worst[0]: worst = (cos, name)
    if rel > worst_rel[0]: worst_rel = (rel, name)
print(f"tensors: {len(ta)}  F32 bit-identical: {n_f32}  BF16 vs Q8_0: {len(ta)-n_f32}")
print(f"worst cosine: {worst[0]:.6f} ({worst[1]})")
print(f"worst relative error: {worst_rel[0]:.4f} ({worst_rel[1]})")
print("PASS" if worst[0] > 0.999 else "FAIL")
