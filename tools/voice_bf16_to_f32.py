#!/usr/bin/env python3
"""Convert BF16 tensors in a PersonaPlex voice-embedding GGUF to F32 (lossless).

Why: ggml's Vulkan backend in moshi.cpp v0.8.0-beta has no bf16->f32 copy op on
devices that report bf16: 0 (RADV on the RX 7900 XTX), so PersonaPlex aborts at
the voice-prompt step. The only BF16 tensor in the whole model set is
voice.embeddings; F32 is an exact superset of BF16, so nothing is lost.

usage: voice_bf16_to_f32.py SRC.gguf DST.gguf
"""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)

arch_field = r.fields.get("general.architecture")
arch = arch_field.contents() if arch_field is not None else "personaplex-voice"
w = GGUFWriter(dst, arch)

# copy metadata verbatim (writer already adds general.architecture)
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vtype = f.types[0]
    if vtype == GGUFValueType.ARRAY:
        w.add_key_value(f.name, f.contents(), vtype, sub_type=f.types[-1])
    else:
        w.add_key_value(f.name, f.contents(), vtype)

def bf16_to_f32(raw: np.ndarray, gguf_shape) -> np.ndarray:
    u16 = np.ascontiguousarray(raw).view(np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    # numpy order is the reverse of GGUF's dimension order
    return f32.reshape(tuple(int(d) for d in reversed(list(gguf_shape))))

converted = []
for t in r.tensors:
    if t.tensor_type == GGMLQuantizationType.BF16:
        arr = bf16_to_f32(t.data, t.shape)
        converted.append(t.name)
    else:
        arr = np.ascontiguousarray(t.data).reshape(
            tuple(int(d) for d in reversed(list(t.shape))))
    w.add_tensor(t.name, arr)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

# ---- verify: same tensor set/shapes, converted values bit-exact ----
a, b = GGUFReader(src), GGUFReader(dst)
ta = {t.name: t for t in a.tensors}
tb = {t.name: t for t in b.tensors}
assert set(ta) == set(tb), f"tensor names differ: {set(ta) ^ set(tb)}"
for n in ta:
    assert list(ta[n].shape) == list(tb[n].shape), f"shape mismatch {n}"
    if n in converted:
        assert tb[n].tensor_type == GGMLQuantizationType.F32, f"{n} not F32"
        back = (np.asarray(tb[n].data, dtype=np.float32).ravel().view(np.uint32) >> 16).astype(np.uint16)
        orig = np.ascontiguousarray(ta[n].data).view(np.uint16).ravel()
        assert np.array_equal(back, orig), f"{n} values not bit-exact"
    else:
        assert ta[n].tensor_type == tb[n].tensor_type, f"type changed {n}"
        assert np.array_equal(np.asarray(ta[n].data).ravel(), np.asarray(tb[n].data).ravel()), f"data differs {n}"
ka = {f.name for f in a.fields.values() if not f.name.startswith("GGUF.")}
kb = {f.name for f in b.fields.values() if not f.name.startswith("GGUF.")}
missing = ka - kb
assert not missing, f"metadata keys lost: {missing}"
print(f"OK {src.split('/')[-1]}: converted {converted}; tensors={len(ta)}; "
      f"extra keys={sorted(kb - ka)}")
