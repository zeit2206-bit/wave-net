"""Logic gate network: no neurons, no weights. Each unit is a 2-input logic gate; training picks which
of the 16 possible gates goes in each spot (differentiable relaxation, Petersen et al. 2022).
After training, it runs as pure bit logic. Same data as the other experiments.
Usage: python gate_net.py data          (small: 20k digits, 2 thresholds, 4 x 8,000 gates)
       python gate_net.py data big      (big: 60k digits, 4 thresholds, 6 x 12,000 gates, pass-through start)"""
import sys, time, os
import numpy as np

BIG = len(sys.argv) > 2 and sys.argv[2] == "big"
SAMPLES = 60000 if BIG else 20000
THRESH = [.2, .4, .6, .8] if BIG else [.25, .75]

d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xtr, ytr = d["x_train"][:SAMPLES].reshape(-1, 784) / 255.0, d["y_train"][:SAMPLES]
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]

# input bits: each pixel compared against each threshold
bits = lambda x: np.concatenate([x > t for t in THRESH], 1)
btr, bte = bits(xtr).astype(np.float32), bits(xte)

# the 16 two-input gates, each written as c0 + c1*a + c2*b + c3*a*b (exact on 0/1, smooth in between)
COEF = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,-1],[0,1,0,0],[0,0,1,-1],[0,0,1,0],[0,1,1,-2],[0,1,1,-1],
                 [1,-1,-1,1],[1,-1,-1,2],[1,0,-1,0],[1,0,-1,1],[1,-1,0,0],[1,-1,0,1],[1,0,0,-1],[1,0,0,0]], np.float32)
NAMES = ["0","AND","A>B","A","B>A","B","XOR","OR","NOR","XNOR","!B","A|!B","!A","!A|B","NAND","1"]

WIDTH, LAYERS, TAU, LR, EPOCHS = (12000, 6, 40.0, 0.01, 12) if BIG else (8000, 4, 30.0, 0.01, 10)
rng = np.random.default_rng(0)
layers = []
n_in = btr.shape[1]
for _ in range(LAYERS):
    ia, ib = rng.integers(0, n_in, WIDTH), rng.integers(0, n_in, WIDTH)
    theta = rng.normal(0, 1, (WIDTH, 16)).astype(np.float32)
    if BIG:
        theta[:, 3] += 5.0               # start mostly as pass-through ("A") so deep stacks can train
    layers.append(dict(ia=ia, ib=ib, oa=np.argsort(ia), ob=np.argsort(ib), theta=theta, n_in=n_in))
    n_in = WIDTH
for L_ in layers:
    L_["m"], L_["v"] = np.zeros_like(L_["theta"]), np.zeros_like(L_["theta"])

def scatter(g, idx, order, n):           # sum columns of g into positions idx (fast np.add.at)
    out = np.zeros((g.shape[0], n), np.float32)
    s = idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    out[:, s[starts]] = np.add.reduceat(g[:, order], starts, axis=1)
    return out

def forward(x):
    cache = []
    for L_ in layers:
        w = np.exp(L_["theta"] - L_["theta"].max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        C = w @ COEF
        a, b = x[:, L_["ia"]], x[:, L_["ib"]]
        cache.append((x, a, b, w, C))
        x = C[:, 0] + C[:, 1] * a + C[:, 2] * b + C[:, 3] * a * b
    return x, cache

def logits(out):                          # 10 groups of output gates; count of 1s in each group = class score
    return out.reshape(len(out), 10, -1).sum(2) / TAU

print(f"Logic gate network: {LAYERS} layers x {WIDTH:,} gates = {LAYERS*WIDTH:,} gates, {btr.shape[1]} input bits")
t0, step = time.time(), 0
for epoch in range(EPOCHS):
    idx = rng.permutation(len(btr))
    for i in range(0, len(idx), 100):
        xb, yb = btr[idx[i:i + 100]], ytr[idx[i:i + 100]]
        out, cache = forward(xb)
        z = logits(out)
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        p[np.arange(len(yb)), yb] -= 1
        g = np.repeat(p / len(yb) / TAU, WIDTH // 10, axis=1).astype(np.float32)
        step += 1
        for L_, (x, a, b, w, C) in zip(reversed(layers), reversed(cache)):
            G = np.stack([g.sum(0), (g * a).sum(0), (g * b).sum(0), (g * a * b).sum(0)], 1)
            dw = G @ COEF.T
            dtheta = w * (dw - (w * dw).sum(1, keepdims=True))
            ga, gb = g * (C[:, 1] + C[:, 3] * b), g * (C[:, 2] + C[:, 3] * a)
            g = scatter(ga, L_["ia"], L_["oa"], L_["n_in"]) + scatter(gb, L_["ib"], L_["ob"], L_["n_in"])
            L_["m"] = .9 * L_["m"] + .1 * dtheta; L_["v"] = .999 * L_["v"] + .001 * dtheta**2
            L_["theta"] -= LR * (L_["m"] / (1 - .9**step)) / (np.sqrt(L_["v"] / (1 - .999**step)) + 1e-8)
    soft = np.mean(np.concatenate([logits(forward(bte[j:j + 1000].astype(np.float32))[0]).argmax(1)
                                   for j in range(0, len(bte), 1000)]) == yte)
    print(f"epoch {epoch + 1}: test accuracy {soft:.1%}  ({time.time() - t0:.0f}s)")

# ---- Freeze: every spot becomes its single most likely gate. From here on: pure bit logic. ----
gate = [L_["theta"].argmax(1) for L_ in layers]
def run_bits(x):
    for L_, gt in zip(layers, gate):
        a, b = x[:, L_["ia"]], x[:, L_["ib"]]
        out = np.empty_like(a)
        for k in range(16):
            cols = gt == k
            if not cols.any(): continue
            A, B = a[:, cols], b[:, cols]
            out[:, cols] = [np.zeros_like(A), A & B, A & ~B, A, ~A & B, B, A ^ B, A | B, ~(A | B), ~(A ^ B),
                            ~B, A | ~B, ~A, ~A | B, ~(A & B), np.ones_like(A)][k]
        x = out
    return x.reshape(len(x), 10, -1).sum(2).argmax(1)

t1 = time.time()
pred = run_bits(bte)
dt = time.time() - t1
print(f"\nFrozen pure-logic network: test accuracy {(pred == yte).mean():.1%}")
print(f"speed: {len(bte) / dt:,.0f} images/s (NumPy, one bit per byte; packed bits on a chip would be far faster)")
used = np.bincount(np.concatenate(gate), minlength=16)
print("gates used:", ", ".join(f"{NAMES[k]} {used[k]}" for k in np.argsort(-used)[:8]))
np.savez(os.path.join(sys.argv[1], "gate_net_big.npz" if BIG else "gate_net.npz"), gates=np.array(gate),
         ia=np.array([L_["ia"] for L_ in layers]), ib=np.array([L_["ib"] for L_ in layers]))
mem = LAYERS * WIDTH * (4 + 2 * 14) / 8
print(f"model size: ~{mem / 1024:.0f} KB (4 bits for gate type + 2 wire indexes per gate)")
