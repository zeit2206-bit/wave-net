"""Draw the trained plates and the light passing through them into one PNG (numpy + zlib only)."""
import sys, os, zlib, struct
import numpy as np

data_dir = sys.argv[1]
phi = np.load(os.path.join(data_dir, "phase_plates.npy"))
d = np.load(os.path.join(data_dir, "mnist.npz"), allow_pickle=False)
xte, yte = d["x_test"] / 255.0, d["y_test"]

N, PAD, L, lam, z = 56, 14, 5, 0.75, 30.0
f = np.fft.fftfreq(N); FX, FY = np.meshgrid(f, f)
arg = 1 / lam**2 - FX**2 - FY**2
H = np.where(arg > 0, np.exp(2j * np.pi * z * np.sqrt(np.maximum(arg, 0))), 0)
prop = lambda u: np.fft.ifft2(np.fft.fft2(u) * H)
centers = [(14, 14), (14, 28), (14, 42), (28, 10), (28, 22), (28, 34), (28, 46), (42, 14), (42, 28), (42, 42)]

def heat(a):  # black -> purple -> orange -> yellow
    a = (a - a.min()) / (np.ptp(a) + 1e-12)
    stops = np.array([[0, 0, 0], [90, 20, 130], [230, 90, 40], [255, 240, 120]], float)
    x = a * 3; i = np.minimum(x.astype(int), 2); t = (x - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)

def wheel(p):  # phase as hue
    h = (p % (2 * np.pi)) / (2 * np.pi)
    return (255 * (0.5 + 0.5 * np.stack([np.cos(2*np.pi*(h + k/3)) for k in range(3)], -1))).astype(np.uint8)

def boxes(img, pred, true):
    img = img.copy()
    for k, (r, c) in enumerate(centers):
        col = [0, 255, 0] if k == true else [255, 60, 60] if k == pred else [120, 120, 120]
        img[r-3, c-3:c+3] = img[r+2, c-3:c+3] = img[r-3:r+3, c-3] = img[r-3:r+3, c+2] = col
    return img

S, gap = 4, 6
panel = lambda im: np.kron(im, np.ones((S, S, 1), np.uint8))
blank = np.full((N, N, 3), 255, np.uint8)
rows = [[blank] + [wheel(phi[l]) for l in range(L)] + [blank]]   # top row: the 5 plates
for idx in [0, 1, 2, 3]:
    a = np.zeros((N, N), complex); a[PAD:PAD+28, PAD:PAD+28] = xte[idx]
    row = [heat(np.abs(a)**2)]
    for l in range(L):
        a = prop(a) * np.exp(1j * phi[l]); row.append(heat(np.abs(a)**2))
    out = np.abs(prop(a))**2
    s = [out[r-3:r+3, c-3:c+3].sum() for r, c in centers]
    row.append(boxes(heat(out), int(np.argmax(s)), int(yte[idx])))
    rows.append(row)
    print(f"example {idx}: true digit {yte[idx]}, light picked {int(np.argmax(s))}")

W = len(rows[0]) * (N*S + gap) + gap
img = np.full((len(rows) * (N*S + gap) + gap, W, 3), 255, np.uint8)
for r, row in enumerate(rows):
    for c, p in enumerate(row):
        y, x = gap + r * (N*S + gap), gap + c * (N*S + gap)
        img[y:y+N*S, x:x+N*S] = panel(p)

raw = b"".join(b"\x00" + img[y].tobytes() for y in range(img.shape[0]))
chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, img.shape[0], 8, 2, 0, 0, 0)) \
      + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
out_path = os.path.join(data_dir, "wave_net.png")
open(out_path, "wb").write(png)
print(out_path)
