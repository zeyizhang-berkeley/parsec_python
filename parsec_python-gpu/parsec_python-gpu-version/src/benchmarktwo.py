import time
import numpy as np
import cupy as cp
from FermiDirac import FermiDirac as fd_gpu

# --- parameters ---
n = 5_000_000
EF = 0.5
Temp = 300
Nelec = 1e6

# --- arrays ---
lam_gpu = cp.random.randn(n, dtype=cp.float64)

print(f"Running GPU Fermi-Dirac benchmark... (n={n})")

# --- GPU timing ---
cp.cuda.runtime.deviceSynchronize()
start = time.perf_counter()
fe_gpu, occ_gpu = fd_gpu(lam_gpu, EF, Temp, Nelec)
cp.cuda.runtime.deviceSynchronize()
end = time.perf_counter()

print(f"GPU time: {end - start:.4f}s")
print(f"Deviation (fe): {float(fe_gpu):.4e}")
print(f"Occupancy mean: {float(cp.mean(occ_gpu)):.4f}, std: {float(cp.std(occ_gpu)):.4f}")
