"""Save the GPTQ 4-bit Llama as a new GGUF file (Q4_0 for the 112 layer matrices; everything else copied as is).
The original Ollama file is only read. Then the new file is loaded back and scored, to check it.
Usage: python -u save_q4.py <original gguf> <new gguf> [seqs]"""
import sys, io, os, time, struct
import numpy as np
from gguf import GGUF
from llama_np import Llama, LINEAR
from quant_test import TEXT, PROMPT
from gptq_test import run, kbit_q, SEQ

def q4_0(K, S):
    """K: (out, in) whole numbers -7..7; S: (out, in/32) scales -> Q4_0 bytes (18 per block of 32)."""
    q = (K + 8).astype(np.uint8).reshape(-1, 32)
    qs = q[:, :16] | (q[:, 16:] << 4)                                  # low nibble = value j, high = value j+16
    d = S.astype(np.float16).reshape(-1, 1).view(np.uint8)
    return np.concatenate([d, qs], 1).ravel()

def write(src, dst, new):                                             # new: tensor name -> (type, bytes)
    g = GGUF(src); g.pos = 24; ftpos = None
    for _ in range(struct.unpack("<QQ", bytes(g.mm[8:24]))[1]):
        k = g._str(); t, = struct.unpack("<I", g._read(4))
        if k == "general.file_type": ftpos = g.pos
        g._val(t)
    kv = bytearray(g.mm[24:g.pos])
    if ftpos: kv[ftpos - 24:ftpos - 20] = struct.pack("<I", 2)       # label the file "mostly Q4_0"
    align = g.meta.get("general.alignment", 32); infos = bytearray(); blobs = []; off = 0
    for name, info in g.tensors.items():
        t, data = new.get(name) or (info["type"], np.asarray(g.raw(name)[0]))
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(info["dims"])) + struct.pack(f"<{len(info['dims'])}Q", *info["dims"])
        infos += struct.pack("<IQ", t, off)
        blobs.append((off, data)); off = (off + len(data) + align - 1) // align * align
    with open(dst, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", g.version, len(g.tensors), struct.unpack("<QQ", bytes(g.mm[8:24]))[1]))
        f.write(kv); f.write(infos)
        f.write(b"\0" * ((-f.tell()) % align)); start = f.tell()
        for o, data in blobs:
            f.write(b"\0" * (start + o - f.tell())); f.write(data.tobytes())

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]; nseq = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    t0 = time.time(); m = Llama(src)
    calib = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration.txt"), encoding="utf-8").read()
    toks = m.tok.encode(calib, bos=False)
    ids = np.array([[128000] + toks[i * (SEQ - 1):(i + 1) * (SEQ - 1)] for i in range(min(nseq, len(toks) // (SEQ - 1)))])
    store = {}; run(m, ids, *kbit_q(4), store=store)
    new = {}
    for li, b in enumerate(m.blocks):
        for k in LINEAR:
            S = np.stack(store[(li, k)])                               # (in/32, out)
            K = np.clip(np.rint(b[k] / np.repeat(S, 32, 0)), -7, 7)    # (in, out)
            new[f"blk.{li}.{k}.weight"] = (2, q4_0(K.T, S.T))
    del m
    write(src, dst, new)
    print(f"saved {dst}: {os.path.getsize(dst) / 1e9:.2f} GB (original {os.path.getsize(src) / 1e9:.2f} GB) ({time.time() - t0:.0f}s)", flush=True)
    m = Llama(dst)
    ppl = m.perplexity(m.tok.encode(TEXT)); gen = m.generate(PROMPT, 8).replace("\n", " ")
    print(f"CHECK new file: perplexity {ppl:.2f}   {gen!r}", flush=True)
