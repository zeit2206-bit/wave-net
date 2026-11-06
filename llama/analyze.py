"""Look inside the Llama 3.2 1B file that Ollama downloaded: settings, layers, parameter counts, weight values.
Usage: python analyze.py <path to the GGUF blob>"""
import sys, re
from collections import defaultdict
import numpy as np
from gguf import GGUF, TYPE_NAMES

g = GGUF(sys.argv[1])
m = g.meta
print("== Model settings ==")
for k, v in m.items():
    if isinstance(v, list):
        print(f"{k}: list of {len(v)}")
    elif not k.startswith("tokenizer.chat_template"):
        print(f"{k}: {v}")

print("\n== Tensors by kind ==")
groups = defaultdict(lambda: [0, 0, set()])
for name, t in g.tensors.items():
    kind = re.sub(r"blk\.\d+\.", "blk.N.", name)
    n = int(np.prod(t["dims"]))
    groups[kind][0] += 1; groups[kind][1] += n; groups[kind][2].add(TYPE_NAMES.get(t["type"], t["type"]))
total = sum(v[1] for v in groups.values())
for kind, (cnt, n, types) in sorted(groups.items(), key=lambda x: -x[1][1]):
    shape = next(t["dims"] for nm, t in g.tensors.items() if re.sub(r"blk\.\d+\.", "blk.N.", nm) == kind)
    print(f"{kind:<28} x{cnt:<3} shape {str(tuple(reversed(shape))):<16} {n:>13,} params ({n / total:5.1%})  {'/'.join(types)}")
print(f"{'TOTAL':<34}{'':<22}{total:>13,} params")

print("\n== Weight values (one feed-forward and one attention matrix, layer 7) ==")
for nm in ["blk.7.ffn_down.weight", "blk.7.attn_q.weight"]:
    w = g.tensor(nm).ravel()
    a = np.abs(w); s = a.mean()
    print(f"{nm}: mean |w| = {s:.4f}, std = {w.std():.4f}, max |w| = {a.max():.3f}")
    print(f"   share of weights below 0.5x mean |w|: {(a < .5 * s).mean():.0%}   above 3x: {(a > 3 * s).mean():.1%}")
