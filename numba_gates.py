"""Compiled speed test: the trained logic gate network run as machine code (Numba), 64 images per
64-bit number, all CPU cores. Compared with the standard network on the same cores.
Needs data/gate_net.npz (or gate_net_big.npz) from gate_net.py.
Usage: python numba_gates.py data   |   python numba_gates.py data big"""
import sys, time, os
import numpy as np
import numba as nb

BIG = len(sys.argv) > 2 and sys.argv[2] == "big"
THRESH = [.2, .4, .6, .8] if BIG else [.25, .75]
d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]
bte = np.concatenate([xte > t for t in THRESH], 1)
g = np.load(os.path.join(sys.argv[1], "gate_net_big.npz" if BIG else "gate_net.npz"))
gates, IA, IB = g["gates"], g["ia"].astype(np.int32), g["ib"].astype(np.int32)
LAYERS, WIDTH = gates.shape

COEF = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,-1],[0,1,0,0],[0,0,1,-1],[0,0,1,0],[0,1,1,-2],[0,1,1,-1],
                 [1,-1,-1,1],[1,-1,-1,2],[1,0,-1,0],[1,0,-1,1],[1,-1,0,0],[1,-1,0,1],[1,0,0,-1],[1,0,0,0]])
TRUTH = np.array([[c[0] + c[1]*a + c[2]*b + c[3]*a*b for a, b in [(0,0),(0,1),(1,0),(1,1)]] for c in COEF])
MASKS = np.where(TRUTH[gates] == 1, np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(0))   # (layers, width, 4)

n = len(bte); pad = (-n) % 64
packed = np.packbits(np.concatenate([bte, np.zeros((pad, bte.shape[1]), bool)]).T, axis=1, bitorder="little")
X = np.ascontiguousarray(np.ascontiguousarray(packed).view(np.uint64).T)          # (words, input bits)

@nb.njit(parallel=True, fastmath=True)
def run(X, IA, IB, MASKS, n):
    words, nin = X.shape
    L, W = IA.shape
    G = W // 10
    pred = np.empty(words * 64, np.int64)
    for w in nb.prange(words):                       # each core takes its own block of 64 images
        cur = np.empty(max(nin, W), np.uint64)
        nxt = np.empty(W, np.uint64)
        cur[:nin] = X[w]
        for l in range(L):
            for j in range(W):
                a = cur[IA[l, j]]; b = cur[IB[l, j]]
                na = ~a; nb_ = ~b
                nxt[j] = (MASKS[l, j, 0] & na & nb_) | (MASKS[l, j, 1] & na & b) | \
                         (MASKS[l, j, 2] & a & nb_) | (MASKS[l, j, 3] & a & b)
            cur[:W] = nxt
        counts = np.zeros((64, 10), np.int64)        # count the 1s in each class group, per image
        for c in range(10):
            for j in range(c * G, (c + 1) * G):
                v = cur[j]
                for i in range(64):
                    counts[i, c] += (v >> np.uint64(i)) & np.uint64(1)
        for i in range(64):
            pred[w * 64 + i] = np.argmax(counts[i])
    return pred[:n]

def best_speed(fn, reps=7):
    times = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t)
    return out, len(yte) / min(times)

run(X, IA, IB, MASKS, n)                            # first call compiles; not timed
pred, sp_g = best_speed(lambda: run(X, IA, IB, MASKS, n))

rng = np.random.default_rng(0)
Wt = [rng.random(s, np.float32) for s in [(784, 128), (128, 128), (128, 10)]]
xf = xte.astype(np.float32)
_, sp_s = best_speed(lambda: (np.maximum(np.maximum(xf @ Wt[0], 0) @ Wt[1], 0) @ Wt[2]).argmax(1))

print(f"cores used: {nb.get_num_threads()}")
print(f"accuracy (compiled gates): {(pred == yte).mean():.1%}\n")
print(f"{'version':<36}{'images/s':>12}")
print(f"{'gates, compiled + packed':<36}{sp_g:>12,.0f}")
print(f"{'standard network (NumPy matrix)':<36}{sp_s:>12,.0f}")
print(f"\ncompiled gates vs standard: {sp_g / sp_s:.2f}x")
