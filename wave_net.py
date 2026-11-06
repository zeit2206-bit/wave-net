"""Simulated diffractive optical network: light + phase plates classify MNIST digits.
The only trainable things are the phase delays of each plate pixel; computation = wave interference."""
import sys, time, urllib.request, os
import numpy as np

data_dir = sys.argv[1]
path = os.path.join(data_dir, "mnist.npz")
if not os.path.exists(path):
    urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", path)
d = np.load(path, allow_pickle=False)
xtr, ytr, xte, yte = d["x_train"][:20000] / 255.0, d["y_train"][:20000], d["x_test"] / 255.0, d["y_test"]

rng = np.random.default_rng(0)
N, PAD, L = 56, 14, 5            # grid size, padding around 28x28 digit, number of plates
lam, z = 0.75, 30.0              # wavelength and plate spacing (in pixel units)

# free-space propagation (angular spectrum)
f = np.fft.fftfreq(N)
FX, FY = np.meshgrid(f, f)
arg = 1 / lam**2 - FX**2 - FY**2
H = np.where(arg > 0, np.exp(2j * np.pi * z * np.sqrt(np.maximum(arg, 0))), 0)
prop = lambda u: np.fft.ifft2(np.fft.fft2(u) * H)
prop_adj = lambda g: np.fft.ifft2(np.fft.fft2(g) * np.conj(H))

# 10 detector squares on the output plane
mask = np.zeros((10, N, N))
centers = [(14, 14), (14, 28), (14, 42), (28, 10), (28, 22), (28, 34), (28, 46), (42, 14), (42, 28), (42, 42)]
for k, (r, c) in enumerate(centers):
    mask[k, r - 3:r + 3, c - 3:c + 3] = 1

phi = rng.uniform(0, 2 * np.pi, (L, N, N))
m_, v_ = np.zeros_like(phi), np.zeros_like(phi)
T = 10.0

def forward(x):
    a = np.zeros((len(x), N, N), complex)
    a[:, PAD:PAD + 28, PAD:PAD + 28] = x          # digit encoded as light amplitude
    acts = []
    for l in range(L):
        b = prop(a)
        a = b * np.exp(1j * phi[l])
        acts.append(a)
    c = prop(a)
    I = np.abs(c) ** 2
    s = np.einsum("bij,kij->bk", I, mask)
    return acts, c, s

def logits(s):
    S = s.sum(1, keepdims=True)
    return T * s / S, S

def accuracy(x, y):
    preds = []
    for i in range(0, len(x), 500):
        preds.append(forward(x[i:i + 500])[2].argmax(1))
    return (np.concatenate(preds) == y).mean()

print(f"before training: test accuracy {accuracy(xte, yte):.1%} (random = 10%)")
t0, step, lr, B = time.time(), 0, 0.05, 64
for epoch in range(3):
    idx = rng.permutation(len(xtr))
    for i in range(0, len(idx), B):
        xb, yb = xtr[idx[i:i + B]], ytr[idx[i:i + B]]
        acts, c, s = forward(xb)
        z_, S = logits(s)
        p = np.exp(z_ - z_.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        e = p.copy(); e[np.arange(len(yb)), yb] -= 1; e /= len(yb)
        n = s / S
        ds = T * (e - (e * n).sum(1, keepdims=True)) / S
        dI = np.einsum("bk,kij->bij", ds, mask)
        g = prop_adj(2 * dI * c)
        grad = np.zeros_like(phi)
        for l in reversed(range(L)):
            a = acts[l]
            grad[l] = np.real(np.conj(g) * 1j * a).sum(0)
            g = prop_adj(g * np.exp(-1j * phi[l]))
        step += 1
        m_ = 0.9 * m_ + 0.1 * grad; v_ = 0.999 * v_ + 0.001 * grad**2
        phi -= lr * (m_ / (1 - 0.9**step)) / (np.sqrt(v_ / (1 - 0.999**step)) + 1e-8)
    print(f"epoch {epoch + 1}: test accuracy {accuracy(xte, yte):.1%}  ({time.time() - t0:.0f}s)")

np.save(os.path.join(data_dir, "phase_plates.npy"), phi)
