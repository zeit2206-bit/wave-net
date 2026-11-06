"""Light + "camera" hybrid: same as wave_net.py, but after some plates a camera measures brightness
and re-emits it as new light. That measure-and-re-emit step is the nonlinearity plain light lacks.
Usage: python hybrid_net.py data            (runs: no camera, camera after plate 3, cameras after 2 and 4)"""
import sys, time, os
import numpy as np

d = np.load(os.path.join(sys.argv[1], "mnist.npz"), allow_pickle=False)
xtr, ytr, xte, yte = d["x_train"][:20000] / 255.0, d["y_train"][:20000], d["x_test"] / 255.0, d["y_test"]

N, PAD, L = 56, 14, 5
lam, z = 0.75, 30.0
f = np.fft.fftfreq(N)
FX, FY = np.meshgrid(f, f)
arg = 1 / lam**2 - FX**2 - FY**2
H = np.where(arg > 0, np.exp(2j * np.pi * z * np.sqrt(np.maximum(arg, 0))), 0)
prop = lambda u: np.fft.ifft2(np.fft.fft2(u) * H)
prop_adj = lambda g: np.fft.ifft2(np.fft.fft2(g) * np.conj(H))

mask = np.zeros((10, N, N))
centers = [(14, 14), (14, 28), (14, 42), (28, 10), (28, 22), (28, 34), (28, 46), (42, 14), (42, 28), (42, 42)]
for k, (r, c) in enumerate(centers):
    mask[k, r - 3:r + 3, c - 3:c + 3] = 1
T = 10.0

def run(name, cameras):
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, 2 * np.pi, (L, N, N))
    m_, v_ = np.zeros_like(phi), np.zeros_like(phi)

    def forward(x):
        a = np.zeros((len(x), N, N), complex)
        a[:, PAD:PAD + 28, PAD:PAD + 28] = x
        scale = (N * N / (x ** 2).sum((1, 2)))[:, None, None]   # keeps re-emitted light at a steady level
        acts = []
        for l in range(L):
            a = prop(a) * np.exp(1j * phi[l])
            acts.append(a)
            if l in cameras:                                    # camera: measure brightness, re-emit as light
                a = (np.abs(a) ** 2 * scale).astype(complex)
        c = prop(a)
        s = np.einsum("bij,kij->bk", np.abs(c) ** 2, mask)
        return acts, scale, c, s

    def accuracy(x, y):
        return np.mean(np.concatenate([forward(x[i:i + 500])[3].argmax(1) for i in range(0, len(x), 500)]) == y)

    print(f"\n{name}")
    t0, step = time.time(), 0
    for epoch in range(3):
        idx = rng.permutation(len(xtr))
        for i in range(0, len(idx), 64):
            xb, yb = xtr[idx[i:i + 64]], ytr[idx[i:i + 64]]
            acts, scale, c, s = forward(xb)
            S = s.sum(1, keepdims=True)
            zl = T * s / S
            p = np.exp(zl - zl.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            e = p.copy(); e[np.arange(len(yb)), yb] -= 1; e /= len(yb)
            ds = T * (e - (e * s / S).sum(1, keepdims=True)) / S
            g = prop_adj(2 * np.einsum("bk,kij->bij", ds, mask) * c)
            grad = np.zeros_like(phi)
            for l in reversed(range(L)):
                a = acts[l]
                if l in cameras:                                # back through the camera step
                    g = 2 * np.real(g) * scale * a
                grad[l] = np.real(np.conj(g) * 1j * a).sum(0)
                g = prop_adj(g * np.exp(-1j * phi[l]))
            step += 1
            m_ = 0.9 * m_ + 0.1 * grad; v_ = 0.999 * v_ + 0.001 * grad**2
            phi -= 0.05 * (m_ / (1 - 0.9**step)) / (np.sqrt(v_ / (1 - 0.999**step)) + 1e-8)
        print(f"epoch {epoch + 1}: test accuracy {accuracy(xte, yte):.1%}  ({time.time() - t0:.0f}s)")

run("Light only (no camera)", set())
run("Hybrid: camera after plate 3", {2})
run("Hybrid: cameras after plates 2 and 4", {1, 3})
