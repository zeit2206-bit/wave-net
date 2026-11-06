"""Test C: different fast ways (no training) to turn Llama's weights into -1/0/+1. Same text and prompt as quant_test.py.
Bits/weight counts the ternary values (1.58), the scales (16 bits per block) and any extra kept weights.
Usage: python -u ternary_sweep.py <gguf path>"""
import sys, time, gc
import numpy as np
from llama_np import Llama
from quant_test import TEXT, PROMPT, ternary_block

def opt_ternary(w, block=32):          # per block: try several cut-offs, keep the one with the smallest error
    b = w.reshape(-1, block); a = np.abs(b); mean = a.mean(1, keepdims=True)
    best_err = np.full((len(b), 1), np.inf, np.float32); best_d = mean; best_s = mean
    for k in (0.3, 0.45, 0.6, 0.75, 0.9, 1.1, 1.3):
        m = a > k * mean
        s = (a * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)   # best scale for that cut-off
        err = (((a - s) * m) ** 2 + (a * ~m) ** 2).sum(1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err); best_d = np.where(better, k * mean, best_d); best_s = np.where(better, s, best_s)
    return (np.sign(b) * (a > best_d) * best_s).reshape(w.shape)

def twn(name, w):                      # classic "ternary weight network" rule: cut-off 0.7 x mean, scale = mean of kept
    b = w.reshape(-1, 32); a = np.abs(b); m = a > 0.7 * a.mean(1, keepdims=True)
    s = (a * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)
    return (np.sign(b) * m * s).reshape(w.shape)

def best(name, w): return opt_ternary(w)
def best8(name, w): return opt_ternary(w, 8)

def keep_top(frac):                    # keep the largest weights exactly, make the rest ternary
    def q(name, w):
        a = np.abs(w).ravel(); n = int(len(a) * frac)
        cut = np.partition(a, len(a) - n)[len(a) - n]
        big = np.abs(w) >= cut
        return np.where(big, w, opt_ternary(np.where(big, 0, w)))
    return q

def planes(n):                         # n stacked ternary copies: each one encodes what the previous ones got wrong
    def q(name, w):
        out = np.zeros_like(w)
        for _ in range(n): out += opt_ternary(w - out)
        return out
    return q

def by_layer(pred, yes, no):
    def q(name, w): return (yes if pred(int(name.split(".")[1])) else no)(name, w)
    return q

T1 = 1.58 + 16 / 32                    # one ternary plane with a 16-bit scale per 32 weights
TESTS = [
    ("plain ternary (what we had)",          ternary_block,                         T1),
    ("better cut-off (TWN rule)",            twn,                                   T1),
    ("best cut-off per block",               best,                                  T1),
    ("best cut-off, blocks of 8",            best8,                                 1.58 + 16 / 8),
    ("keep top 1% exact",                    keep_top(0.01),                        T1 + 0.01 * 24),
    ("keep top 5% exact",                    keep_top(0.05),                        T1 + 0.05 * 24),
    ("2 stacked copies",                     planes(2),                             2 * T1),
    ("2 copies layers 0-7, 1 copy 8-15",     by_layer(lambda i: i < 8, planes(2), best), 1.5 * T1),
    ("3 stacked copies",                     planes(3),                             3 * T1),
]

if __name__ == "__main__":
    path = sys.argv[1]
    print(f"{'version':<36}{'bits/w':>7}{'perplexity':>12}   continuation of '{PROMPT}'")
    for name, q, bits in TESTS:
        t = time.time()
        m = Llama(path, q)
        ids = m.tok.encode(TEXT)
        ppl = m.perplexity(ids)
        gen = m.generate(PROMPT, 8).replace("\n", " ")
        print(f"{name:<36}{bits:>7.2f}{ppl:>12.2f}   {gen!r}   ({time.time() - t:.0f}s)")
        del m; gc.collect()
