"""Tests A and B: how much does Llama 3.2 1B break when its weights get fewer bits, and which parts break worst?
Quality = perplexity on a plain English text (lower = better; how 'surprised' the model is by each next token).
Only the 112 layer matrices (attention + feed-forward) are changed; embeddings and norms stay as in the file.
Usage: python -u quant_test.py <gguf path>"""
import sys, time, gc
import numpy as np
from llama_np import Llama

TEXT = """Every morning the small bakery on the corner opened its doors at six. The owner, a quiet man who had learned
the trade from his grandmother, arrived an hour earlier to light the ovens and shape the first loaves. Neighbors said
they could tell the time by the smell of fresh bread drifting down the street. Over the years the town around the
bakery changed. A supermarket opened near the highway, and many of the old shops closed one after another. Yet the
bakery stayed busy. People came not only for the bread but for the conversation, the familiar faces, and the feeling
that some things could still be done slowly and well. When the owner finally decided to retire, he did not sell the
business to the highest bidder. Instead he offered it to the young woman who had worked beside him for five years,
asking only that she keep the ovens burning and the prices fair. She agreed, and on her first day as the new owner
she arrived even earlier than he used to, just to be sure everything would be ready when the doors opened at six."""
PROMPT = "The capital of France is"

def kbit(bits):                        # round each block of 32 weights to a few levels, one scale per block
    top = 2 ** (bits - 1) - 1
    def q(name, w):
        b = w.reshape(-1, 32); s = np.abs(b).max(1, keepdims=True) / top + 1e-12
        return (np.clip(np.round(b / s), -top, top) * s).reshape(w.shape)
    return q

def ternary_block(name, w):            # -1/0/+1 with one scale per block of 32 (mean |w|)
    b = w.reshape(-1, 32); s = np.abs(b).mean(1, keepdims=True) + 1e-12
    return (np.clip(np.round(b / s), -1, 1) * s).reshape(w.shape)

def ternary_tensor(name, w):           # BitNet-style: -1/0/+1 with ONE scale for the whole matrix
    s = np.abs(w).mean() + 1e-12
    return np.clip(np.round(w / s), -1, 1) * s

def only(pred, q):                     # apply q only where pred(layer index, kind) is true
    def f(name, w):
        i, kind = int(name.split(".")[1]), name.split(".")[2]
        return q(name, w) if pred(i, kind) else w
    return f

ATTN = {"attn_q", "attn_k", "attn_v", "attn_output"}
TESTS = [
    ("A", "original file (8-bit)",           None),
    ("A", "6-bit",                           kbit(6)),
    ("A", "4-bit",                           kbit(4)),
    ("A", "3-bit",                           kbit(3)),
    ("A", "ternary, scale per 32 weights",   ternary_block),
    ("A", "ternary, one scale per matrix",   ternary_tensor),
    ("B", "ternary: attention only",         only(lambda i, k: k in ATTN, ternary_block)),
    ("B", "ternary: feed-forward only",      only(lambda i, k: k not in ATTN, ternary_block)),
    ("B", "ternary: first 4 layers (0-3)",   only(lambda i, k: i < 4, ternary_block)),
    ("B", "ternary: middle 4 layers (6-9)",  only(lambda i, k: 6 <= i < 10, ternary_block)),
    ("B", "ternary: last 4 layers (12-15)",  only(lambda i, k: i >= 12, ternary_block)),
    ("B", "ternary: only layer 0",           only(lambda i, k: i == 0, ternary_block)),
    ("B", "ternary: only layer 15",          only(lambda i, k: i == 15, ternary_block)),
]

if __name__ == "__main__":
  path = sys.argv[1]
  print(f"{'test':<4}{'version':<34}{'perplexity':>11}   continuation of '{PROMPT}'")
  for grp, name, q in TESTS:
      t = time.time()
      m = Llama(path, q)
      ids = m.tok.encode(TEXT)
      ppl = m.perplexity(ids)
      gen = m.generate(PROMPT, 8).replace("\n", " ")
      print(f"{grp:<4}{name:<34}{ppl:>11.2f}   {gen!r}   ({time.time() - t:.0f}s)")
      del m; gc.collect()
