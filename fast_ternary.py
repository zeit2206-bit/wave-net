"""Speed test for the ternary (-1/0/+1) networks, run with no multiplications in the layers:
each neuron keeps a list of inputs to ADD (+1) and a list to SUBTRACT (-1); zeros are skipped.
64 images are processed together so every add is one wide CPU instruction. Compiled with Numba, all cores.
Needs data/ternary_128.npz and ternary_512.npz from ternary_net.py.  Usage: python fast_ternary.py data"""
import sys, os, time
import numpy as np
import numba as nb

d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xtr = d["x_train"][:5000].reshape(-1, 784) / 255.0
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]
B = 64

def build(path):
    z = np.load(path)
    T = [z[f"t{i}"] for i in range(3)]; S = [float(z[f"s{i}"]) for i in range(3)]; Bs = [z[f"b{i}"] for i in range(3)]
    # calibrate one fixed 8-bit scale per layer's input, from training images
    h, scales = xtr, [1 / 127]
    for i in range(2):
        h = np.maximum(h @ (T[i] * S[i]) + Bs[i], 0); scales.append(np.percentile(h, 99.99) / 127)
    layers = []
    for i in range(3):
        t = T[i].T                                            # (out, in)
        plus = [np.flatnonzero(r == 1) for r in t]; minus = [np.flatnonzero(r == -1) for r in t]
        pp = np.r_[0, np.cumsum([len(p) for p in plus])]; mp = np.r_[0, np.cumsum([len(m) for m in minus])]
        nxt = scales[i + 1] if i < 2 else 1.0
        layers.append((pp, np.concatenate(plus).astype(np.int32), mp, np.concatenate(minus).astype(np.int32),
                       np.float32(scales[i] * S[i] / nxt), (Bs[i] / nxt).astype(np.float32)))
    nnz = sum(len(l[1]) + len(l[3]) for l in layers)
    return layers, nnz

@nb.njit(fastmath=True)
def layer(x, pp, pi, mp, mi, mult, bias, relu):
    out = np.empty((len(pp) - 1, x.shape[1]), np.int32)
    acc = np.empty(x.shape[1], np.int32)
    y = np.empty((len(pp) - 1, x.shape[1]), np.float32)
    for j in range(len(pp) - 1):
        acc[:] = 0
        for k in range(pp[j], pp[j + 1]):       # +1 weights: add
            acc += x[pi[k]]
        for k in range(mp[j], mp[j + 1]):       # -1 weights: subtract
            acc -= x[mi[k]]
        for q in range(x.shape[1]):
            v = acc[q] * mult + bias[j]
            y[j, q] = v
            out[j, q] = min(max(np.int32(round(v)), 0), 127) if relu else 0
    return out, y

@nb.njit(parallel=True, fastmath=True)
def run(X, l0, l1, l2):
    blocks = X.shape[0]
    pred = np.empty(blocks * X.shape[2], np.int64)
    for b in nb.prange(blocks):
        h, _ = layer(X[b], l0[0], l0[1], l0[2], l0[3], l0[4], l0[5], True)
        h, _ = layer(h, l1[0], l1[1], l1[2], l1[3], l1[4], l1[5], True)
        _, o = layer(h, l2[0], l2[1], l2[2], l2[3], l2[4], l2[5], False)
        for q in range(X.shape[2]):
            pred[b * X.shape[2] + q] = np.argmax(o[:, q])
    return pred

n = len(xte); pad = (-n) % B
Xi = np.round(np.concatenate([xte, np.zeros((pad, 784))]) * 127).astype(np.int32)
X = np.ascontiguousarray(Xi.reshape(-1, B, 784).transpose(0, 2, 1))       # (blocks, inputs, 64 images)

def best_speed(fn, reps=7):
    times = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t)
    return out, n / min(times)

for width in (128, 512):
    L, nnz = build(os.path.join(sys.argv[1], f"ternary_{width}.npz"))
    run(X, *L)                                                             # compile
    pred, sp = best_speed(lambda: run(X, *L))
    print(f"ternary {width:>3} wide: accuracy {(pred[:n] == yte).mean():.1%}   {sp:>12,.0f} images/s   "
          f"{nnz:,} adds/subtracts per image")

rng = np.random.default_rng(0)
Wt = [rng.random(s, np.float32) for s in [(784, 128), (128, 128), (128, 10)]]
xf = xte.astype(np.float32)
_, sp_s = best_speed(lambda: (np.maximum(np.maximum(xf @ Wt[0], 0) @ Wt[1], 0) @ Wt[2]).argmax(1))
print(f"standard 128 (NumPy, 32-bit):              {sp_s:>12,.0f} images/s   118,016 multiply-adds per image")
