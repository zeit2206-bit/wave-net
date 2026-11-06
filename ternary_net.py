"""The middle ground: the standard network, but every weight is only -1, 0 or +1 (BitNet b1.58 style).
During training a normal decimal copy of each weight is kept and nudged; the forward pass always uses its
rounded version (-1/0/+1 times one scale per layer). Activations are rounded to 8 bits.
Same data as standard_net.py (all 60k digits, 3 passes).  Usage: python ternary_net.py data"""
import sys, os, time
import numpy as np

d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xtr, ytr = d["x_train"].reshape(-1, 784) / 255.0, d["y_train"]
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]

def ternary(W):                                   # -> values in {-1, 0, +1} and one scale for the layer
    s = np.abs(W).mean() + 1e-8
    return np.clip(np.round(W / s), -1, 1), s

def q8(x):                                        # activations rounded to 8 bits (per image)
    s = np.abs(x).max(1, keepdims=True) / 127 + 1e-8
    return np.round(x / s) * s

def run(name, sizes, epochs=3):
    rng = np.random.default_rng(0)
    W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    params = W + bs
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]

    def forward(x):
        hs, Ts = [x], []
        for i in range(len(W)):
            t, s = ternary(W[i]); Ts.append((t, s))
            x = q8(x) @ (t * s) + bs[i]
            if i < len(W) - 1:
                x = np.maximum(x, 0)
            hs.append(x)
        return hs, Ts

    acc = lambda: np.mean(np.concatenate([forward(xte[i:i + 2000])[0][-1].argmax(1) for i in range(0, len(xte), 2000)]) == yte)
    print(f"\n{name}")
    t0, step = time.time(), 0
    for epoch in range(epochs):
        idx = rng.permutation(len(xtr))
        for i in range(0, len(idx), 64):
            xb, yb = xtr[idx[i:i + 64]], ytr[idx[i:i + 64]]
            hs, Ts = forward(xb)
            p = np.exp(hs[-1] - hs[-1].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            g = p; g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            gW, gb = [None] * len(W), [None] * len(W)
            for l in reversed(range(len(W))):
                gW[l], gb[l] = q8(hs[l]).T @ g, g.sum(0)     # gradient goes to the decimal copy ("straight-through")
                if l:
                    t, s = Ts[l]
                    g = (g @ (t * s).T) * (hs[l] > 0)
            step += 1
            for j, (p_, g_) in enumerate(zip(params, gW + gb)):
                m[j] = 0.9 * m[j] + 0.1 * g_; v[j] = 0.999 * v[j] + 0.001 * g_**2
                p_ -= 1e-3 * (m[j] / (1 - 0.9**step)) / (np.sqrt(v[j] / (1 - 0.999**step)) + 1e-8)
        print(f"epoch {epoch + 1}: test accuracy {acc():.1%}  ({time.time() - t0:.0f}s)")
    T = [ternary(Wl)[0] for Wl in W]
    n = sum(t.size for t in T)
    zeros = sum((t == 0).sum() for t in T) / n
    print(f"weights: {n:,}, all -1/0/+1 ({zeros:.0%} are 0)  |  size: ~{n * 2 / 8 / 1024:.0f} KB at 2 bits per weight")
    np.savez(os.path.join(sys.argv[1], f"ternary_{sizes[1]}.npz"),
             **{f"t{i}": T[i].astype(np.int8) for i in range(3)},
             **{f"s{i}": ternary(W[i])[1] for i in range(3)}, **{f"b{i}": bs[i] for i in range(3)})

run("Ternary network, same shape as standard (2 hidden layers of 128)", [784, 128, 128, 10])
run("Ternary network, wider (2 hidden layers of 512)", [784, 512, 512, 10])
