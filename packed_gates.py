"""Run the trained logic gate network with 64 images packed into each 64-bit number,
so every CPU bitwise operation evaluates one gate for 64 images at once.
Compares speed with the unpacked gate version and with the standard network.
Needs data/gate_net.npz from gate_net.py.  Usage: python packed_gates.py data"""
import sys, time, os
import numpy as np

d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xte, yte = d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]
bte = np.concatenate([xte > .25, xte > .75], 1)                      # same 1,568 input bits
g = np.load(os.path.join(sys.argv[1], "gate_net.npz"))
gates, IA, IB = g["gates"], g["ia"], g["ib"]
WIDTH = gates.shape[1]

# truth table of each of the 16 gates for inputs (a,b) = 00, 01, 10, 11 (same order as gate_net.py)
COEF = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,-1],[0,1,0,0],[0,0,1,-1],[0,0,1,0],[0,1,1,-2],[0,1,1,-1],
                 [1,-1,-1,1],[1,-1,-1,2],[1,0,-1,0],[1,0,-1,1],[1,-1,0,0],[1,-1,0,1],[1,0,0,-1],[1,0,0,0]])
TRUTH = np.array([[c[0] + c[1]*a + c[2]*b + c[3]*a*b for a, b in [(0,0),(0,1),(1,0),(1,1)]] for c in COEF])
ONES = np.uint64(0xFFFFFFFFFFFFFFFF)
# per gate: 4 masks (all ones or all zeros) saying which input combinations give 1
MASKS = [[np.where(TRUTH[gl, j] == 1, ONES, np.uint64(0))[:, None] for j in range(4)] for gl in gates]

def pack(b):                     # (images, bits) bool -> (bits, words) uint64; 64 images per word
    n = len(b); pad = (-n) % 64
    b = np.concatenate([b, np.zeros((pad, b.shape[1]), bool)]).T
    return np.ascontiguousarray(np.packbits(b, axis=1, bitorder="little")).view(np.uint64), n

def run_packed(x, n):
    for l in range(len(gates)):
        a, b = x[IA[l]], x[IB[l]]
        na, nb = ~a, ~b
        m00, m01, m10, m11 = MASKS[l]
        x = (m00 & na & nb) | (m01 & na & b) | (m10 & a & nb) | (m11 & a & b)   # any gate, branch-free
    bits = np.unpackbits(x.view(np.uint8), axis=1, bitorder="little")[:, :n]    # (8000, images)
    return bits.reshape(10, WIDTH // 10, n).sum(1).argmax(0)

def run_unpacked(x):             # one bit per byte (the gate_net.py way)
    for l in range(len(gates)):
        a, b = x[:, IA[l]], x[:, IB[l]]
        na, nb = ~a, ~b
        m = [TRUTH[gates[l], j].astype(bool) for j in range(4)]
        x = (m[0] & na & nb) | (m[1] & na & b) | (m[2] & a & nb) | (m[3] & a & b)
    return x.reshape(len(x), 10, -1).sum(2).argmax(1)

def best_speed(fn, reps=5):
    times = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t)
    return out, len(yte) / min(times)

xp, n = pack(bte)
pred_p, sp_p = best_speed(lambda: run_packed(xp, n))
pred_u, sp_u = best_speed(lambda: run_unpacked(bte))

rng = np.random.default_rng(0)
W = [rng.random(s, np.float32) for s in [(784, 128), (128, 128), (128, 10)]]
xf = xte.astype(np.float32)
_, sp_s = best_speed(lambda: (np.maximum(np.maximum(xf @ W[0], 0) @ W[1], 0) @ W[2]).argmax(1))

print(f"packed and unpacked give identical answers: {bool((pred_p == pred_u).all())}")
print(f"accuracy: {(pred_p == yte).mean():.1%}\n")
print(f"{'version':<34}{'images/s':>12}")
print(f"{'gates, 1 bit per byte':<34}{sp_u:>12,.0f}")
print(f"{'gates, 64 images per number':<34}{sp_p:>12,.0f}   ({sp_p / sp_u:.0f}x faster than unpacked)")
print(f"{'standard network (NumPy matrix)':<34}{sp_s:>12,.0f}")
print(f"\npacked gates vs standard: {sp_p / sp_s:.2f}x")
