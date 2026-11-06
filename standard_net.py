"""Standard neural networks on the same data as wave_net.py (20k train digits, 10k test, 3 epochs, batch 64, Adam)."""
import sys, os, time
import numpy as np

SAMPLES = 60000 if len(sys.argv) > 2 and sys.argv[2] == "big" else 20000   # "big": all training digits
d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xtr, ytr = d["x_train"][:SAMPLES].reshape(-1, 784) / 255.0, d["y_train"][:SAMPLES]
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]

def run(name, sizes):
    rng = np.random.default_rng(0)
    W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    params = W + bs
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]

    def forward(x):
        hs = [x]
        for i in range(len(W)):
            x = x @ W[i] + bs[i]
            if i < len(W) - 1:
                x = np.maximum(x, 0)
            hs.append(x)
        return hs

    acc = lambda: (forward(xte)[-1].argmax(1) == yte).mean()
    n_params = sum(p.size for p in params)
    print(f"\n{name}  ({n_params:,} trainable numbers)")
    t0, step = time.time(), 0
    for epoch in range(3):
        idx = rng.permutation(len(xtr))
        for i in range(0, len(idx), 64):
            xb, yb = xtr[idx[i:i + 64]], ytr[idx[i:i + 64]]
            hs = forward(xb)
            p = np.exp(hs[-1] - hs[-1].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            g = p; g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            gW, gb = [None] * len(W), [None] * len(W)
            for l in reversed(range(len(W))):
                gW[l], gb[l] = hs[l].T @ g, g.sum(0)
                if l:
                    g = (g @ W[l].T) * (hs[l] > 0)
            step += 1
            for j, (p_, g_) in enumerate(zip(params, gW + gb)):
                m[j] = 0.9 * m[j] + 0.1 * g_; v[j] = 0.999 * v[j] + 0.001 * g_**2
                p_ -= 1e-3 * (m[j] / (1 - 0.9**step)) / (np.sqrt(v[j] / (1 - 0.999**step)) + 1e-8)
        print(f"epoch {epoch + 1}: test accuracy {acc():.1%}  ({time.time() - t0:.0f}s)")
    if len(W) == 3:
        tag = "_big" if SAMPLES == 60000 else ""
        np.savez(os.path.join(sys.argv[1], f"standard_net{tag}.npz"), *W, *bs)

run("Linear (no hidden layer, like light without nonlinearity)", [784, 10])
run("Standard network (2 hidden layers of 128 neurons)", [784, 128, 128, 10])
