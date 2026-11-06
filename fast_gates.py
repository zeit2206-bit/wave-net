"""The "better-organized factory" version of the trained gate network. Same answers, less work:
  1. pass-through gates (A, B) are removed: their readers are wired straight to the source
  2. gates whose output nobody uses are removed
  3. what's left is numbered in order, so memory is read mostly forwards
  4. 256 images per block (4 x 64-bit lanes) so the CPU can use its wide (AVX) instructions
  5. counting the 1s per class uses a bit-sliced counter: 64 images counted at once, not bit by bit
Usage: python fast_gates.py data big   (or without "big" for the small network)"""
import sys, time, os
import numpy as np
import numba as nb

BIG = len(sys.argv) > 2 and sys.argv[2] == "big"
THRESH = [.2, .4, .6, .8] if BIG else [.25, .75]
d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]
bte = np.concatenate([xte > t for t in THRESH], 1)
g = np.load(os.path.join(sys.argv[1], "gate_net_big.npz" if BIG else "gate_net.npz"))
gates, IA, IB = g["gates"], g["ia"], g["ib"]
LAYERS, WIDTH = gates.shape
NIN = bte.shape[1]

COEF = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,-1],[0,1,0,0],[0,0,1,-1],[0,0,1,0],[0,1,1,-2],[0,1,1,-1],
                 [1,-1,-1,1],[1,-1,-1,2],[1,0,-1,0],[1,0,-1,1],[1,-1,0,0],[1,-1,0,1],[1,0,0,-1],[1,0,0,0]])
TRUTH = np.array([[c[0] + c[1]*a + c[2]*b + c[3]*a*b for a, b in [(0,0),(0,1),(1,0),(1,1)]] for c in COEF])
PASS_A, PASS_B = 3, 5

# ---- 1. build a flat circuit, skipping pass-through gates ----
src = np.arange(NIN)                  # for each output of the current layer: which node really holds it
ops = []                              # (gate type, node a, node b) for every real gate
for l in range(LAYERS):
    new = np.empty(WIDTH, np.int64)
    for j in range(WIDTH):
        a, b, t = src[IA[l, j]], src[IB[l, j]], gates[l, j]
        if t == PASS_A: new[j] = a
        elif t == PASS_B: new[j] = b
        else:
            new[j] = NIN + len(ops); ops.append((t, a, b))
    src = new
outputs = src                         # final layer: which node holds each output bit
total = LAYERS * WIDTH

# ---- 2. keep only gates that feed the outputs ----
need = np.zeros(NIN + len(ops), bool); need[outputs] = True
for k in range(len(ops) - 1, -1, -1):
    if need[NIN + k]:
        need[ops[k][1]] = need[ops[k][2]] = True

# ---- 3. renumber what's left, in order ----
newid = -np.ones(NIN + len(ops), np.int64); newid[:NIN] = np.arange(NIN)
kept = []
for k in range(len(ops)):
    if need[NIN + k]:
        newid[NIN + k] = NIN + len(kept); kept.append(ops[k])
T = np.array([o[0] for o in kept]); A = newid[[o[1] for o in kept]]; B = newid[[o[2] for o in kept]]
ONES = np.uint64(0xFFFFFFFFFFFFFFFF)
M = np.where(TRUTH[T] == 1, ONES, np.uint64(0))                       # (gates, 4) truth-table masks
OUT = newid[outputs].reshape(10, -1)
print(f"gates: {total:,} trained -> {len(ops):,} after removing pass-through -> {len(kept):,} after removing unused "
      f"({len(kept) / total:.0%} of the original)")

# ---- 4. pack 256 images per block: (blocks, input bits, 4 lanes of 64) ----
LANES = 4
n = len(bte); pad = (-n) % (64 * LANES)
pk = np.packbits(np.concatenate([bte, np.zeros((pad, NIN), bool)]).T, axis=1, bitorder="little")
words = np.ascontiguousarray(pk).view(np.uint64)                      # (bits, words)
X = np.ascontiguousarray(words.reshape(NIN, -1, LANES).transpose(1, 0, 2))

@nb.njit(parallel=True, fastmath=True)
def run(X, A, B, M, OUT, n):
    blocks, nin, lanes = X.shape
    ng = A.shape[0]
    G = OUT.shape[1]
    pred = np.empty(blocks * lanes * 64, np.int64)
    for blk in nb.prange(blocks):
        node = np.empty((nin + ng, lanes), np.uint64)
        node[:nin] = X[blk]
        for k in range(ng):
            a = node[A[k]]; b = node[B[k]]
            m0, m1, m2, m3 = M[k, 0], M[k, 1], M[k, 2], M[k, 3]
            for q in range(lanes):                        # 4 lanes -> one wide AVX instruction
                av = a[q]; bv = b[q]
                node[nin + k, q] = (m0 & ~av & ~bv) | (m1 & ~av & bv) | (m2 & av & ~bv) | (m3 & av & bv)
        # ---- 5. bit-sliced counting: 64 images per lane counted together ----
        for q in range(lanes):
            counts = np.zeros((64, 10), np.int64)
            for c in range(10):
                cnt = np.zeros(12, np.uint64)             # 12 bit-planes: counts up to 4,095
                for j in range(G):
                    v = node[OUT[c, j], q]
                    for bit in range(12):                 # binary add of v into the counter
                        t = cnt[bit] & v
                        cnt[bit] ^= v
                        v = t
                        if v == 0: break
                for i in range(64):
                    s = 0
                    for bit in range(12):
                        s += ((cnt[bit] >> np.uint64(i)) & np.uint64(1)) << bit
                    counts[i, c] = s
            for i in range(64):
                pred[(blk * lanes + q) * 64 + i] = np.argmax(counts[i])
    return pred[:n]

def best_speed(fn, reps=7):
    times = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t)
    return out, len(yte) / min(times)

run(X, A, B, M, OUT, n)                                     # first call compiles; not timed
pred, sp_f = best_speed(lambda: run(X, A, B, M, OUT, n))

# reference: the straightforward frozen network, to prove the answers didn't change
def reference(x):
    for l in range(LAYERS):
        a, b = x[:, IA[l]], x[:, IB[l]]
        m = [TRUTH[gates[l], j].astype(bool) for j in range(4)]
        x = (m[0] & ~a & ~b) | (m[1] & ~a & b) | (m[2] & a & ~b) | (m[3] & a & b)
    return x.reshape(len(x), 10, -1).sum(2).argmax(1)
same = (reference(bte) == pred).all()

rng = np.random.default_rng(0)
Wt = [rng.random(s, np.float32) for s in [(784, 128), (128, 128), (128, 10)]]
xf = xte.astype(np.float32)
_, sp_s = best_speed(lambda: (np.maximum(np.maximum(xf @ Wt[0], 0) @ Wt[1], 0) @ Wt[2]).argmax(1))

print(f"identical answers to the original gate network: {bool(same)}")
print(f"accuracy: {(pred == yte).mean():.1%}\n")
print(f"{'version':<38}{'images/s':>12}")
print(f"{'gates, organized factory':<38}{sp_f:>12,.0f}")
print(f"{'standard network (NumPy matrix)':<38}{sp_s:>12,.0f}")
print(f"\norganized gates vs standard: {sp_f / sp_s:.2f}x")
print(f"model size: ~{len(kept) * (4 + 2 * 15) / 8 / 1024:.0f} KB")
