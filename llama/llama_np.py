"""Llama 3.2 run in plain NumPy from the Ollama GGUF file: tokenizer + forward pass + perplexity.
quantize: optional function (name, weight) -> new weight, applied to chosen weights before running."""
import re
import numpy as np
from gguf import GGUF

# ---------- tokenizer (byte-level BPE, approximating Llama 3's split rule with Python's re) ----------
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

SPLIT = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

class Tokenizer:
    def __init__(self, g):
        self.tokens = g.meta["tokenizer.ggml.tokens"]
        self.id = {t: i for i, t in enumerate(self.tokens)}
        self.rank = {tuple(m.split(" ", 1)): i for i, m in enumerate(g.meta["tokenizer.ggml.merges"])}
        self.b2u = _bytes_to_unicode(); self.u2b = {v: k for k, v in self.b2u.items()}
        self.bos = g.meta["tokenizer.ggml.bos_token_id"]

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.rank.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            r, i = min(pairs)
            if r == 1 << 30: break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return [self.id[p] for p in parts]

    def encode(self, text, bos=True):
        ids = [self.bos] if bos else []
        for w in SPLIT.findall(text):
            ids += self._bpe("".join(self.b2u[b] for b in w.encode("utf-8")))
        return ids

    def decode(self, ids):
        s = "".join(self.tokens[i] for i in ids)
        return bytes(self.u2b.get(c, 32) for c in s).decode("utf-8", "replace")

# ---------- model ----------
LINEAR = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]

class Llama:
    def __init__(self, path, quantize=None):
        g = GGUF(path); m = g.meta
        self.tok = Tokenizer(g)
        self.L = m["llama.block_count"]; self.H = m["llama.attention.head_count"]
        self.KV = m["llama.attention.head_count_kv"]; self.D = m["llama.embedding_length"]
        self.hd = self.D // self.H; self.eps = m["llama.attention.layer_norm_rms_epsilon"]
        base = m["llama.rope.freq_base"]
        inv = base ** (-np.arange(0, self.hd, 2) / self.hd)
        if "rope_freqs.weight" in g.tensors:                 # Llama 3 long-context frequency adjustment
            inv = inv / g.tensor("rope_freqs.weight")
        self.inv_freq = inv.astype(np.float32)
        self.emb = g.tensor("token_embd.weight")
        self.out_norm = g.tensor("output_norm.weight")
        self.blocks = []
        for i in range(self.L):
            b = {"attn_norm": g.tensor(f"blk.{i}.attn_norm.weight"), "ffn_norm": g.tensor(f"blk.{i}.ffn_norm.weight")}
            for k in LINEAR:
                w = g.tensor(f"blk.{i}.{k}.weight")
                if quantize: w = quantize(f"blk.{i}.{k}", w).astype(np.float32)
                b[k] = np.ascontiguousarray(w.T)             # (in, out) so that x @ w works
            self.blocks.append(b)

    def _norm(self, x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * w

    def _rope(self, x, pos):                                  # rotate adjacent pairs (llama.cpp "normal" mode)
        ang = pos[:, None] * self.inv_freq[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x); out[..., 0::2] = a * c - b * s; out[..., 1::2] = a * s + b * c
        return out

    def logits(self, ids):
        T = len(ids); pos = np.arange(T, dtype=np.float32)
        x = self.emb[ids].astype(np.float32)
        mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
        rep = self.H // self.KV
        for b in self.blocks:
            h = self._norm(x, b["attn_norm"])
            q = self._rope((h @ b["attn_q"]).reshape(T, self.H, self.hd), pos)
            k = self._rope((h @ b["attn_k"]).reshape(T, self.KV, self.hd), pos)
            v = (h @ b["attn_v"]).reshape(T, self.KV, self.hd)
            k, v = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
            att = np.einsum("thd,shd->hts", q, k) / np.sqrt(self.hd) + mask
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            x = x + np.einsum("hts,shd->thd", att, v).reshape(T, self.D) @ b["attn_output"]
            h = self._norm(x, b["ffn_norm"])
            gt = h @ b["ffn_gate"]
            x = x + ((gt / (1 + np.exp(-gt))) * (h @ b["ffn_up"])) @ b["ffn_down"]
        return self._norm(x, self.out_norm) @ self.emb.T

    def perplexity(self, ids):
        z = self.logits(ids)[:-1].astype(np.float64)
        z -= z.max(1, keepdims=True)
        lp = z[np.arange(len(ids) - 1), ids[1:]] - np.log(np.exp(z).sum(1))
        return float(np.exp(-lp.mean()))

    def generate(self, text, n=30):
        ids = self.tok.encode(text)
        for _ in range(n):                                     # greedy; recomputes everything (simple, slow)
            nxt = int(self.logits(ids)[-1].argmax())
            if nxt in (128001, 128009): break
            ids.append(nxt)
        return self.tok.decode(ids[len(self.tok.encode(text)):])
