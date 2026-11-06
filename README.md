# wave-net

Simulated optical (light-based) neural network: light passes through 5 phase plates and
interference sorts handwritten digits onto 10 detectors. Only NumPy needed. ~91% on MNIST.

```
python wave_net.py data   # train (~2 min), saves data/phase_plates.npy
python show.py data       # draws data/wave_net.png
```

The `llama/` quantization scripts calibrate on `calibration.txt` in this folder: put any few pages of plain English text there.
