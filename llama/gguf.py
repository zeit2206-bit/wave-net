"""Minimal read-only GGUF reader (the file format Ollama stores models in), NumPy only.
Opens the file memory-mapped and read-only: the Ollama model file is never changed."""
import struct
import numpy as np

TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 12: (256, 144), 14: (256, 210), 30: (1, 2)}  # values, bytes per block

class GGUF:
    def __init__(self, path):
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.pos = 0
        assert self._read(4) == b"GGUF", "not a GGUF file"
        self.version, = struct.unpack("<I", self._read(4))
        n_tensors, n_kv = struct.unpack("<QQ", self._read(16))
        self.meta = {}
        for _ in range(n_kv):
            k = self._str(); t, = struct.unpack("<I", self._read(4))
            self.meta[k] = self._val(t)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str(); nd, = struct.unpack("<I", self._read(4))
            dims = struct.unpack(f"<{nd}Q", self._read(8 * nd))
            t, off = struct.unpack("<IQ", self._read(12))
            self.tensors[name] = dict(dims=dims, type=t, offset=off)
        align = self.meta.get("general.alignment", 32)
        self.data_start = (self.pos + align - 1) // align * align

    def _read(self, n):
        b = bytes(self.mm[self.pos:self.pos + n]); self.pos += n; return b

    def _str(self):
        n, = struct.unpack("<Q", self._read(8)); return self._read(n).decode("utf-8", "replace")

    def _val(self, t):
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in fmt:
            s = struct.calcsize(fmt[t]); return struct.unpack("<" + fmt[t], self._read(s))[0]
        if t == 8: return self._str()
        if t == 9:
            it, n = struct.unpack("<IQ", self._read(12))
            return [self._val(it) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")

    def raw(self, name):
        info = self.tensors[name]
        n = int(np.prod(info["dims"]))
        per, nbytes = BLOCK[info["type"]]
        start = self.data_start + info["offset"]
        return self.mm[start:start + n // per * nbytes], info

    def tensor(self, name):
        """Dequantized float32 array, shape (rows, cols) = (dims[1], dims[0]) for matrices."""
        buf, info = self.raw(name)
        t, dims = info["type"], info["dims"]
        if t == 0: x = buf.view(np.float32).astype(np.float32)
        elif t == 1: x = buf.view(np.float16).astype(np.float32)
        elif t == 30: x = (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif t == 8:                                   # Q8_0: blocks of 32 = 1 f16 scale + 32 int8
            b = buf.reshape(-1, 34)
            d = b[:, :2].copy().view(np.float16).astype(np.float32)
            q = b[:, 2:].copy().view(np.int8).astype(np.float32)
            x = (q * d).ravel()
        elif t == 2:                                   # Q4_0: blocks of 32 = 1 f16 scale + 16 bytes of 4-bit values (minus 8)
            b = buf.reshape(-1, 18)
            d = b[:, :2].copy().view(np.float16).astype(np.float32)
            qs = b[:, 2:]
            q = np.concatenate([qs & 15, qs >> 4], 1).astype(np.float32) - 8
            x = (q * d).ravel()
        elif t == 14: x = _q6_k(buf)
        elif t == 12: x = _q4_k(buf)
        else: raise NotImplementedError(TYPE_NAMES.get(t, t))
        return x.reshape(tuple(reversed(dims))) if len(dims) > 1 else x

def _q6_k(buf):
    b = buf.reshape(-1, 210)
    ql, qh, sc = b[:, :128], b[:, 128:192], b[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)
    out = np.empty((len(b), 256), np.float32)
    for half in range(2):
        l, h, s = ql[:, 64 * half:64 * half + 64], qh[:, 32 * half:32 * half + 32], sc[:, 8 * half:8 * half + 8]
        q1 = ((l[:, :32] & 15) | (((h >> 0) & 3) << 4)).astype(np.int32) - 32
        q2 = ((l[:, 32:] & 15) | (((h >> 2) & 3) << 4)).astype(np.int32) - 32
        q3 = ((l[:, :32] >> 4) | (((h >> 4) & 3) << 4)).astype(np.int32) - 32
        q4 = ((l[:, 32:] >> 4) | (((h >> 6) & 3) << 4)).astype(np.int32) - 32
        o = 128 * half
        for k, q in enumerate([q1, q2, q3, q4]):
            sidx = 2 * k + (np.arange(32) // 16)
            out[:, o + 32 * k:o + 32 * k + 32] = d * s[:, sidx] * q
    return out.ravel()

def _q4_k(buf):
    b = buf.reshape(-1, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    s = b[:, 4:16].astype(np.int32); q = b[:, 16:144]
    sc = np.empty((len(b), 8), np.int32); mn = np.empty((len(b), 8), np.int32)
    for j in range(8):
        if j < 4: sc[:, j] = s[:, j] & 63; mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 15) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    out = np.empty((len(b), 256), np.float32)
    for j in range(4):
        qq = q[:, 32 * j:32 * j + 32]
        out[:, 64 * j:64 * j + 32] = d * sc[:, 2 * j:2 * j + 1] * (qq & 15) - dmin * mn[:, 2 * j:2 * j + 1]
        out[:, 64 * j + 32:64 * j + 64] = d * sc[:, 2 * j + 1:2 * j + 2] * (qq >> 4) - dmin * mn[:, 2 * j + 1:2 * j + 2]
    return out.ravel()
