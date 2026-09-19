#!/usr/bin/env python3
"""Minimal GGUF header reader: count tensors by ggml type (no deps)."""
import struct, sys, collections
T = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',30:'BF16'}
f = open(sys.argv[1], 'rb')
def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
def rstr(): n = rd('Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip_val(t):
    if t == 8: rstr()
    elif t == 9:
        at = rd('I'); n = rd('Q')
        for _ in range(n): skip_val(at)
    else: f.read(SZ[t])
assert f.read(4) == b'GGUF', 'not gguf'
ver = rd('I'); nt = rd('Q'); nkv = rd('Q')
for _ in range(nkv): rstr(); skip_val(rd('I'))
cnt = collections.Counter(); ex = {}
for _ in range(nt):
    name = rstr(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; t = rd('I'); rd('Q')
    tn = T.get(t, str(t)); cnt[tn] += 1; ex.setdefault(tn, (name, dims))
print(f'gguf v{ver}: {nt} tensors, {nkv} kv')
for k, v in cnt.most_common(): print(f'  {k:6s} {v:5d}   e.g. {ex[k][0]} {ex[k][1]}')
