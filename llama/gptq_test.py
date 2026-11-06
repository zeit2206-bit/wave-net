"""Test D: GPTQ-style rounding. Round each matrix one input at a time, and after each step nudge the
not-yet-rounded weights so the layer's OUTPUT on real sentences stays close to the original.
Layers are done in order, each one fed by the already-rounded layers before it.
Example sentences: calibration.txt (any English text you provide) (different from the bakery text used for scoring).
Usage: python -u gptq_test.py <gguf path> [seqs]"""
import sys, time, gc, io, os
import numpy as np
from llama_np import Llama
from quant_test import TEXT, PROMPT

SEQ = 256

def ternary_q(w, s): return np.clip(np.round(w / s), -1, 1) * s
def ternary_s(g):    return np.abs(g).mean(0) + 1e-12                  # per output, over a group of 32 inputs
def kbit_q(bits):
    top = 2 ** (bits - 1) - 1
    return (lambda w, s: np.clip(np.round(w / s), -top, top) * s), (lambda g: np.abs(g).max(0) / top + 1e-12)

def gptq(W, H, q, scale, block=128, group=32, scales=None):
    """W: (in, out) weights; H: (in, in) = sum of x x^T over example tokens. Returns rounded W."""
    W = W.copy(); n = W.shape[0]; H = H.copy()
    dead = np.diag(H) == 0; H[dead, dead] = 1; W[dead] = 0
    H[np.diag_indices(n)] += 0.01 * np.mean(np.diag(H))               # small damping for stability
    U = np.linalg.cholesky(np.linalg.inv(H)).T.astype(np.float32)     # upper factor of H^-1
    Q = np.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n); W1 = W[i1:i2]; E = np.zeros_like(W1)
        for i in range(i2 - i1):
            if i % group == 0:
                s = scale(W1[i:i + group])
                if scales is not None: scales.append(s)
            w = W1[i]; qw = q(w, s); Q[i1 + i] = qw
            e = (w - qw) / U[i1 + i, i1 + i]
            W1[i:] -= np.outer(U[i1 + i, i1 + i:i2], e)                # push this row's error onto later rows
            E[i] = e
        W[i2:] -= U[i1:i2, i2:].T @ E
    return Q

def attention(m, b, h):
    N, T, D = h.shape; pos = np.arange(T, dtype=np.float32); rep = m.H // m.KV
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1); out = np.empty_like(h)
    for n in range(N):
        q = m._rope((h[n] @ b["attn_q"]).reshape(T, m.H, m.hd), pos)
        k = np.repeat(m._rope((h[n] @ b["attn_k"]).reshape(T, m.KV, m.hd), pos), rep, 1)
        v = np.repeat((h[n] @ b["attn_v"]).reshape(T, m.KV, m.hd), rep, 1)
        a = np.einsum("thd,shd->hts", q, k) / np.sqrt(m.hd) + mask
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        out[n] = np.einsum("hts,shd->thd", a, v).reshape(T, D)
    return out

def hess(x):
    f = x.reshape(-1, x.shape[-1]).astype(np.float64); return f.T @ f

def run(m, ids, q, scale, store=None):
    def rq(li, k, W, H):                                               # store[(layer, kind)] = list of group scales
        return gptq(W, H, q, scale, scales=store.setdefault((li, k), []) if store is not None else None)
    x = m.emb[ids].astype(np.float32)                                  # (seqs, SEQ, D)
    for li, b in enumerate(m.blocks):
        t = time.time()
        h = m._norm(x, b["attn_norm"]); H = hess(h)
        for k in ("attn_q", "attn_k", "attn_v"): b[k] = rq(li, k, b[k], H)
        o = attention(m, b, h)
        b["attn_output"] = rq(li, "attn_output", b["attn_output"], hess(o))
        x = x + o @ b["attn_output"]
        h = m._norm(x, b["ffn_norm"]); H = hess(h)
        for k in ("ffn_gate", "ffn_up"): b[k] = rq(li, k, b[k], H)
        g = h @ b["ffn_gate"]; a = (g / (1 + np.exp(-g))) * (h @ b["ffn_up"])
        b["ffn_down"] = rq(li, "ffn_down", b["ffn_down"], hess(a))
        x = x + a @ b["ffn_down"]
        print(f"  layer {li:2d} done ({time.time() - t:.0f}s)", flush=True)

if __name__ == "__main__":
    path = sys.argv[1]; nseq = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    calib = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration.txt"), encoding="utf-8").read()
    SETTINGS = [("ternary + GPTQ", ternary_q, ternary_s, 2.08), ("3-bit + GPTQ", *kbit_q(3), 3.5), ("4-bit + GPTQ", *kbit_q(4), 4.5)]
    for name, q, scale, bits in SETTINGS:
        t0 = time.time()
        m = Llama(path)
        toks = m.tok.encode(calib, bos=False)
        ids = np.array([[128000] + toks[i * (SEQ - 1):(i + 1) * (SEQ - 1)] for i in range(min(nseq, len(toks) // (SEQ - 1)))])
        print(f"{name}: {len(ids)} example sequences x {SEQ} tokens", flush=True)
        run(m, ids, q, scale)
        ppl = m.perplexity(m.tok.encode(TEXT)); gen = m.generate(PROMPT, 8).replace("\n", " ")
        print(f"RESULT {name:<18}{bits:>6.2f} bits  perplexity {ppl:10.2f}   {gen!r}   ({time.time() - t0:.0f}s)", flush=True)
        del m; gc.collect()
