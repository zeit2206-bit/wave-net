"""The standard network, made faster the usual industry way: 8-bit whole numbers instead of 32-bit decimals
(quantization), compiled with Numba on all cores. Fair counterpart to fast_gates.py.
Needs data/standard_net_big.npz from standard_net.py.  Usage: python int8_net.py data"""
import sys, time, os
import numpy as np
import numba as nb

d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xtr = d["x_train"][:5000].reshape(-1, 784) / 255.0
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]
w = np.load(os.path.join(sys.argv[1], "standard_net_big.npz"))
W = [w[f"arr_{i}"] for i in range(3)]; Bs = [w[f"arr_{i}"] for i in range(3, 6)]

def forward_float(x):
    h = np.maximum(x @ W[0] + Bs[0], 0); h = np.maximum(h @ W[1] + Bs[1], 0)
    return h @ W[2] + Bs[2]

# ---- quantize: every weight and every activation becomes a whole number from -127 to 127 ----
wscale = [np.abs(Wl).max(0) / 127 for Wl in W]                         # one scale per output neuron
Wq = [np.ascontiguousarray(np.round(Wl / s).astype(np.int8).T) for Wl, s in zip(W, wscale)]   # (out, in)
# activation scales measured on training images (calibration)
h1 = np.maximum(xtr @ W[0] + Bs[0], 0); h2 = np.maximum(h1 @ W[1] + Bs[1], 0)
ascale = np.array([1 / 127, np.percentile(h1, 99.99) / 127, np.percentile(h2, 99.99) / 127])
# fold everything into one float multiplier and bias per neuron
mult = [(ascale[l] * wscale[l] / (ascale[l + 1] if l < 2 else 1)).astype(np.float32) for l in range(3)]
bias = [(Bs[l] / (ascale[l + 1] if l < 2 else 1)).astype(np.float32) for l in range(3)]
Xq = np.round(xte * 127).astype(np.int8)

@nb.njit(fastmath=True)
def layer(x, Wq, mult, bias, relu):
    out_n, in_n = Wq.shape
    y = np.empty(out_n, np.float32)
    for j in range(out_n):
        acc = np.int32(0)
        for k in range(in_n):                          # int8 x int8 summed in int32
            acc += np.int32(x[k]) * np.int32(Wq[j, k])
        y[j] = acc * mult[j] + bias[j]
    if relu:
        q = np.empty(out_n, np.int8)
        for j in range(out_n):
            v = min(max(round(y[j]), 0), 127)
            q[j] = v
        return q, y
    return x, y

@nb.njit(parallel=True, fastmath=True)
def run(X, W0, W1, W2, m0, m1, m2, b0, b1, b2):
    n = X.shape[0]
    pred = np.empty(n, np.int64)
    for i in nb.prange(n):
        h, _ = layer(X[i], W0, m0, b0, True)
        h, _ = layer(h, W1, m1, b1, True)
        _, o = layer(h, W2, m2, b2, False)
        pred[i] = np.argmax(o)
    return pred

args = (Xq, *Wq, *mult, *bias)
run(*args)                                             # first call compiles; not timed

def best_speed(fn, reps=7):
    times = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t)
    return out, len(yte) / min(times)

xf = xte.astype(np.float32); Wf = [Wl.astype(np.float32) for Wl in W]; Bf = [b.astype(np.float32) for b in Bs]
pf, sp_f = best_speed(lambda: (np.maximum(np.maximum(xf @ Wf[0] + Bf[0], 0) @ Wf[1] + Bf[1], 0) @ Wf[2] + Bf[2]).argmax(1))
pq, sp_q = best_speed(lambda: run(*args))

print(f"{'version':<36}{'accuracy':>9}{'images/s':>12}{'size':>9}")
print(f"{'standard, 32-bit decimals (NumPy)':<36}{(pf == yte).mean():>9.1%}{sp_f:>12,.0f}{'~462 KB':>9}")
print(f"{'standard, 8-bit (compiled)':<36}{(pq == yte).mean():>9.1%}{sp_q:>12,.0f}{'~116 KB':>9}")
print(f"\n8-bit vs 32-bit: {sp_q / sp_f:.2f}x")
