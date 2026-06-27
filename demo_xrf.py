'''Demo: load an I14 XRF NeXus file, remap fields, and save to HDF5.

Run from the adapt_io project root:

    python3 demo_xrf.py
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from adapt_io import HDF5Loader
from adapt_io.cli import _write_hdf5, _write_manifest

INPUT   = Path("data/i14-394175_xrf.nxs")
MAPPING = Path("config/i14_xrf_mapping.yaml")
OUTPUT  = Path("data/i14-394175_xrf_remapped.h5")

# --- Load ---

print(f"Input:   {INPUT}")
print(f"Mapping: {MAPPING}\n")

loader = HDF5Loader(MAPPING)
data   = loader.load(INPUT)

# --- Summary ---

col = max(len(k) for k in data) + 2
print(f"{'Field':<{col}}  {'Shape / Value'}")
print("-" * (col + 40))
for name, value in data.items():
    arr = np.asarray(value)
    if arr.ndim == 0:
        print(f"{name:<{col}}  {arr.item()}")
    else:
        print(f"{name:<{col}}  shape={arr.shape}  dtype={arr.dtype}")

# --- Save ---

print(f"\nWriting → {OUTPUT}")
_write_hdf5(data, OUTPUT, MAPPING)
manifest = _write_manifest(OUTPUT, INPUT, MAPPING)
print(f"Manifest → {manifest}")
print("Done.")
