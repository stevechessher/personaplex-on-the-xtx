#!/usr/bin/env python3
"""Write moshi.cpp's GGUF cache for the UNQUANTIZED PersonaPlex model, streaming, low RAM.

moshi.cpp's own writer (`-g` without `-q`) needs ~27 GB host RAM for this model. This
script produces the file moshi.cpp would have written, one tensor at a time:

  * tensor order, names and ne come from the q8_0 cache moshi.cpp already wrote
    (same loader, same alloc order; only the dtypes differ)
  * dtype rule from src/loader.h: tensors that are F32 in the q8 cache
    (the rms_norm `alpha`s -> fetch(..., ggml_rms_norm)) are F32; every tensor
    quantized in the q8 cache keeps the source dtype, BF16, when not quantizing
  * name rule: GGUF name = "lm." + safetensors name, except attention projections
    (transformer.h get_weights): in_projs.K <- in_proj_weight, out_projs.K <-
    out_proj.weight, each split into N contiguous row blocks (ggml_view_2d over ne1)
  * GGUF v3, no KV pairs, alignment 32 (gguf_init_empty + gguf_write_to_file)

Output goes to <out>.partial and is renamed only after a size check, so a failed
run can never leave a file that `-g` would mistake for a valid cache.

usage: bf16gguf.py <model.safetensors> <q8_0 cache .gguf> <out .gguf>
"""
import json, os, re, struct, sys, time
from collections import Counter

import numpy as np

ST, Q8, OUT = sys.argv[1:4]
ALIGN = 32
T_F32, T_Q8_0, T_BF16 = 0, 8, 30


def pad(n, a=ALIGN):
    return (n + a - 1) // a * a


# --- read the q8 cache: order, names, ne, types -----------------------------------------
f = open(Q8, "rb")
def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rs(): n = rd("Q"); return f.read(n).decode()
assert f.read(4) == b"GGUF"
ver = rd("I"); nt = rd("Q"); nkv = rd("Q")
assert nkv == 0, f"expected no KV pairs, got {nkv}"
q8 = []
for _ in range(nt):
    name = rs(); nd = rd("I"); ne = [rd("Q") for _ in range(nd)]; t = rd("I"); off = rd("Q")
    q8.append((name, ne, t, off))
f.close()
assert {t for _, _, t, _ in q8} <= {T_Q8_0, T_F32}, "unexpected types in q8 cache"

# --- read the safetensors header -------------------------------------------------------
sf = open(ST, "rb")
hlen = struct.unpack("<Q", sf.read(8))[0]
hdr = json.loads(sf.read(hlen)); hdr.pop("__metadata__", None)
data0 = 8 + hlen
assert all(v["dtype"] == "BF16" for v in hdr.values())

# how many splits each projection has (from the q8 names)
nsplit = Counter()
for name, *_ in q8:
    m = re.fullmatch(r"lm\.(.*\.self_attn)\.(in|out)_projs\.(\d+)\.weight", name)
    if m: nsplit[(m.group(1), m.group(2))] += 1


def source(name, ne):
    """-> (safetensors key, byte offset within that tensor, byte length) for the BF16 bytes"""
    assert name.startswith("lm."), name
    m = re.fullmatch(r"lm\.(.*\.self_attn)\.(in|out)_projs\.(\d+)\.weight", name)
    if m:
        base, kind, k = m.group(1), m.group(2), int(m.group(3))
        key = f"{base}.in_proj_weight" if kind == "in" else f"{base}.out_proj.weight"
        rows, cols = hdr[key]["shape"]
        n = nsplit[(base, kind)]
        assert rows % n == 0 and ne == [cols, rows // n], (name, ne, hdr[key]["shape"], n)
        blk = (rows // n) * cols * 2
        return key, k * blk, blk
    key = name[3:]
    shape = hdr[key]["shape"]
    nel = 1
    for d in shape: nel *= d
    nel_ne = 1
    for d in ne: nel_ne *= d
    assert nel == nel_ne, (name, shape, ne)
    return key, 0, nel * 2


# --- plan the output -------------------------------------------------------------------
plan = []   # (name, ne, out_type, st_key, src_off, src_len, out_len)
used = set()
for name, ne, t, _ in q8:
    key, so, sl = source(name, ne)
    used.add(key)
    if t == T_F32:
        plan.append((name, ne, T_F32, key, so, sl, sl * 2))
    else:
        plan.append((name, ne, T_BF16, key, so, sl, sl))
unused = set(hdr) - used
assert not unused, f"safetensors tensors not mapped: {sorted(unused)[:5]}"

# header bytes
hb = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(plan), 0)
off = 0
for name, ne, t, key, so, sl, ol in plan:
    nb = name.encode()
    hb += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(ne))
    hb += b"".join(struct.pack("<Q", d) for d in ne) + struct.pack("<IQ", t, off)
    off = pad(off + ol)
hb += b"\0" * (pad(len(hb)) - len(hb))
total = len(hb) + off
print(f"plan: {len(plan)} tensors ({Counter(p[2] for p in plan)}), "
      f"{len(hdr)} safetensors keys all mapped, output {total:,} bytes", flush=True)

# --- write ----------------------------------------------------------------------------
tmp = OUT + ".partial"
t0 = time.time()
with open(tmp, "wb") as o:
    o.write(hb)
    written = 0
    for i, (name, ne, t, key, so, sl, ol) in enumerate(plan):
        a, _b = hdr[key]["data_offsets"]
        sf.seek(data0 + a + so)
        raw = sf.read(sl)
        assert len(raw) == sl
        if t == T_F32:
            raw = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).tobytes()
        assert len(raw) == ol
        o.write(raw)
        o.write(b"\0" * (pad(ol) - ol))
        written += ol
        if i % 100 == 0:
            print(f"  {i}/{len(plan)} {written/1e9:.1f} GB {time.time()-t0:.0f}s", flush=True)
    o.flush(); os.fsync(o.fileno())
size = os.path.getsize(tmp)
assert size == total, (size, total)
os.rename(tmp, OUT)
print(f"wrote {OUT}: {size:,} bytes in {time.time()-t0:.0f}s", flush=True)
