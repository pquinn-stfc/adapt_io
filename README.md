# nexus-io

Generic NeXus/HDF5 I/O for scanning X-ray nanoprobe beamlines.

Two complementary loaders cover the common cases:

| Loader | When to use |
|---|---|
| `NeXusLoader` | File follows NeXus conventions (`@signal`, `@axes`, Dimension Scales) — loader discovers structure automatically |
| `HDF5Loader` | File has a non-standard layout — a YAML mapping declares which paths to read, how to rename them, and what transforms to apply |

---

## Installation

```bash
pip install -e .
# or
uv pip install -e .
```

Requires Python ≥ 3.10, numpy, h5py, pyyaml.

---

## NeXusLoader — convention-based loading

Reads any file that follows the NeXus standard.  Axes, units, and navigate
flags are discovered automatically from `@signal`, `@axes`, and HDF5
Dimension Scales attributes.

```python
from adapt_io import NeXusLoader

loader = NeXusLoader("scan.nxs")

ds = loader.load()                        # follows the @default chain
ds = loader.load("/entry/instrument/merlin/data")  # specific NXdata group
all_ds = loader.load_all()               # every NXdata group in the file
```

Load a named selection from a YAML file:

```yaml
# selection.yaml
frames:
  path: /entry/instrument/merlin/data
  signal_type: DPC
  navigate:
    scan_y: true
    scan_x: true
```

```python
datasets = loader.load_from_yaml("selection.yaml")
frames   = datasets["frames"]
```

---

## HDF5Loader — YAML-driven field mapping

Declare a mapping from output field names to HDF5 paths, with optional unit
conversions, scaling, axis reordering, and shape transforms.

### Mapping syntax

```yaml
# config/i14_mapping.yaml

beam_energy:
  path: /entry/instrument/monochromator/energy
  default: 12.0                 # used if path is absent

detector_distance:
  path: /entry/instrument/merlin/distance
  transform: mm_to_m            # named unit conversion
  default: 2.0

pixel_size:
  path: /entry/instrument/merlin/pixel_size
  transform: um_to_m
  default: 55.0

scan_step_x:
  path: /entry/scan/sample_x/step_size
  transform: um_to_m

frames:
  path: /entry/instrument/merlin/data
  keep_array: true              # return as ndarray, not scalar
```

Each entry can also be a bare path string or a numeric literal:

```yaml
pixel_size: /entry/instrument/merlin/pixel_size   # bare path
wavelength: 1.54e-10                              # hard-coded value
```

### Field spec options

| Option | Type | Description |
|---|---|---|
| `path` | str | HDF5 dataset path |
| `value` | number | Hard-coded value (ignores file) |
| `default` | number | Fallback if `path` is absent |
| `scale` | float | Multiply raw value by this scalar |
| `transform` | str | Named unit conversion (see below) |
| `transpose` | `true` or `[axes]` | Transpose array axes |
| `flatten` | `true` or `[axes]` | Collapse all or a contiguous subset of axes |
| `keep_array` | bool | Return as ndarray; default raises if not scalar |

### Built-in transforms

| Name | Conversion |
|---|---|
| `mm_to_m`, `um_to_m`, `nm_to_m`, `pm_to_m` | Length → metres |
| `ev_to_kev`, `kev_to_ev` | Energy unit conversion |
| `deg_to_rad`, `rad_to_deg` | Angle conversion |
| `step_from_positions` | Mean step size from a 1-D position array |
| `step_from_positions_mm/um/nm` | Same, with unit conversion |

Register custom transforms at runtime:

```python
from adapt_io import register_transform
register_transform("mrad_to_rad", lambda x: np.asarray(x) * 1e-3)
```

### Loading

```python
from adapt_io import HDF5Loader, bind

loader = HDF5Loader("config/i14_mapping.yaml")
data   = loader.load("scan.nxs")    # plain dict: {field_name: value}

# Populate a dataclass from the dict
ds = bind(data, MyDataset)
```

---

## NXData — in-memory schema

Both loaders return (or can produce) `NXData` objects — the common currency
across analysis packages.

```python
from adapt_io import NXData, AxisInfo

ds.data          # ndarray, shape (*nav_shape, *sig_shape)
ds.axes          # list of AxisInfo, one per dimension
ds.signal_type   # e.g. "DPC", "XRF"
ds.metadata      # dict of scalar instrument fields
```

### Utility functions

```python
from adapt_io import coord_to_index, slice_by_coords, on_same_nav_grid, apply_nav

# Nearest index for a physical coordinate
i = coord_to_index(ds.axes[0], 250e-9)

# Slice by physical coordinate range
roi = slice_by_coords(ds, scan_x=(0, 500e-9), scan_y=(0, 500e-9))

# Check two datasets share the same scan grid
assert on_same_nav_grid(xrf_ds, dpc_ds)

# Apply a function at every navigation position
intensity = apply_nav(ds, lambda frame: frame.sum())
```

---

## Writing NeXus output

```python
from adapt_io import save_nxdata

save_nxdata("output.nxs", ds, compression="gzip")
```

Axes are written as HDF5 Dimension Scales so viewers such as
[silx](https://silx.org), [h5web](https://h5web.panosc.eu), and NeXpy assign
correct physical labels automatically.

---

## CLI — convert, inspect, clean

### Convert

Read an HDF5/NeXus file, apply the mapping, and write a remapped output file.
A sidecar manifest (`<output>.manifest.json`) is written alongside every
output so conversions can be cleaned up later.

```bash
adapt-io convert scan.nxs -m config/i14_mapping.yaml -o out.h5
```

### Inspect

Dry-run: print resolved field names, shapes, and values without writing
anything.

```bash
adapt-io inspect scan.nxs -m config/i14_mapping.yaml
```

```
Field                Shape / Value
-------------------------------------------
beam_energy          12.0
detector_distance    2.0
frames               array (512, 512, 256)  dtype=uint16
```

### Clean

Delete an output file and its manifest (use `--dry-run` to preview):

```bash
adapt-io clean out.h5 --dry-run   # preview
adapt-io clean out.h5             # delete out.h5 + out.h5.manifest.json
```

---

## Docker

Build the converter image from the repo root:

```bash
docker build -f docker/Dockerfile -t adapt-io .
```

Run with input data and config mounted at runtime:

```bash
# inspect
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/config:/config \
  adapt-io inspect /data/scan.nxs -m /config/i14_mapping.yaml

# convert
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/config:/config \
  adapt-io convert /data/scan.nxs -m /config/i14_mapping.yaml -o /data/out.h5

# clean up
docker run --rm \
  -v /path/to/data:/data \
  adapt-io clean /data/out.h5
```
