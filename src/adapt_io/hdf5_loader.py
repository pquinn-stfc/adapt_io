'''Generic HDF5/NeXus loader driven by a declarative field mapping.

The loader is completely decoupled from any specific data schema.  It
resolves a mapping into a plain ``dict`` of field names → values; a
separate :func:`bind` helper populates any dataclass from that dict.

Quick start
-----------
::

    from adapt_io import HDF5Loader, bind

    loader = HDF5Loader("config/i14_mapping.yaml")
    data   = loader.load("scan.nxs")      # plain dict
    ds     = bind(data, MyDataset)        # any dataclass

Field spec syntax
-----------------
Each entry in the mapping can be a bare HDF5 path string, a numeric literal,
or a dict with any combination of ``path``, ``value``, ``default``,
``scale``, ``transform``, ``transpose``, ``flatten``, and ``keep_array``::

    beam_energy:
      path: /entry/instrument/monochromator/energy

    detector_distance:
      path: /entry/instrument/merlin/distance
      transform: mm_to_m

    pixel_size:
      value: 55.0e-6

    scan_step_x:
      path: /entry/scan/sample_x/value
      transform: step_from_positions_um

    # Scalar multiply — applied after any named transform
    raw_counts:
      path: /entry/detector/data
      scale: 0.001
      keep_array: true

    # Transpose — reverse all axes
    diffraction_pattern:
      path: /entry/detector/data
      transpose: true
      keep_array: true

    # Transpose with explicit axis order
    volume:
      path: /entry/detector/data
      transpose: [2, 0, 1]
      keep_array: true

    # Flatten all axes to 1-D
    flat_signal:
      path: /entry/detector/data
      flatten: true
      keep_array: true

    # Collapse a subset of contiguous axes (e.g. [0,1] on shape (N,M,P) → (N*M,P))
    frames:
      path: /entry/detector/data
      flatten: [0, 1]
      keep_array: true

Built-in named transforms
-------------------------
``mm_to_m``, ``um_to_m``, ``nm_to_m``, ``pm_to_m``
    Length unit conversions to metres.
``ev_to_kev``, ``kev_to_ev``
    Energy unit conversions.
``deg_to_rad``, ``rad_to_deg``
    Angle conversions.
``step_from_positions``, ``step_from_positions_mm/um/nm``
    Reduce a 1-D position array to its mean step size.

Custom transforms can be registered via :func:`register_transform`.
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import h5py
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Named transform registry
# ---------------------------------------------------------------------------

_TRANSFORMS: dict[str, Callable] = {
    "mm_to_m":  lambda x: np.asarray(x) * 1e-3,
    "um_to_m":  lambda x: np.asarray(x) * 1e-6,
    "nm_to_m":  lambda x: np.asarray(x) * 1e-9,
    "pm_to_m":  lambda x: np.asarray(x) * 1e-12,
    "ev_to_kev":  lambda x: np.asarray(x) * 1e-3,
    "kev_to_ev":  lambda x: np.asarray(x) * 1e3,
    "deg_to_rad": lambda x: np.deg2rad(x),
    "rad_to_deg": lambda x: np.rad2deg(x),
    "step_from_positions":     lambda x: float(np.diff(np.asarray(x)).mean()),
    "step_from_positions_mm":  lambda x: float(np.diff(np.asarray(x)).mean()) * 1e-3,
    "step_from_positions_um":  lambda x: float(np.diff(np.asarray(x)).mean()) * 1e-6,
    "step_from_positions_nm":  lambda x: float(np.diff(np.asarray(x)).mean()) * 1e-9,
    "squeeze":                 lambda x: np.squeeze(np.asarray(x)),
}


def register_transform(name: str, fn: Callable) -> None:
    '''Register a named transform for use in YAML mapping files.

    Parameters
    ----------
    name : str
        Key to use in the YAML ``transform`` field.
    fn : callable
        ``fn(np.ndarray) -> np.ndarray | float``

    Example
    -------
    >>> register_transform("mrad_to_rad", lambda x: np.asarray(x) * 1e-3)
    '''
    _TRANSFORMS[name] = fn


# ---------------------------------------------------------------------------
# Helpers for FieldSpec transforms
# ---------------------------------------------------------------------------

def _flatten_axes(arr: np.ndarray, axes: list) -> np.ndarray:
    '''Collapse a set of contiguous axes into one.

    Parameters
    ----------
    arr : np.ndarray
    axes : list of int
        Must name a contiguous run of axes, e.g. [0, 1] or [1, 2, 3].

    Examples
    --------
    >>> a = np.zeros((2, 3, 4))
    >>> _flatten_axes(a, [0, 1]).shape   # (6, 4)
    >>> _flatten_axes(a, [1, 2]).shape   # (2, 12)
    '''
    axes = sorted(int(a) for a in axes)
    if axes != list(range(axes[0], axes[-1] + 1)):
        raise ValueError(
            f"flatten axes must be a contiguous run, got {axes}. "
            "Reorder with 'transpose' first if needed."
        )
    start, end = axes[0], axes[-1]
    new_shape = arr.shape[:start] + (-1,) + arr.shape[end + 1:]
    return arr.reshape(new_shape)


# ---------------------------------------------------------------------------
# FieldSpec
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    '''Specification for how to obtain a single value from an HDF5 file.'''

    path: Optional[str] = None
    value: Optional[Any] = None
    default: Optional[Any] = None
    scale: float = 1.0
    transform: Optional[str] = None
    transpose: Optional[Union[bool, list]] = None
    flatten: Optional[Union[bool, list]] = None
    keep_array: bool = False

    def resolve(self, f: h5py.File) -> Optional[Any]:
        if self.value is not None:
            raw = np.asarray(self.value)
        elif self.path is not None:
            if self.path in f:
                raw = np.asarray(f[self.path])
            elif self.default is not None:
                raw = np.asarray(self.default)
            else:
                return None
        elif self.default is not None:
            raw = np.asarray(self.default)
        else:
            return None

        if self.transform is not None:
            if self.transform not in _TRANSFORMS:
                raise KeyError(
                    f"Unknown transform {self.transform!r}. "
                    f"Available: {sorted(_TRANSFORMS)}"
                )
            raw = np.asarray(_TRANSFORMS[self.transform](raw))

        if self.transpose is not None and self.transpose is not False:
            axes = self.transpose if isinstance(self.transpose, list) else None
            raw = np.transpose(raw, axes)

        if self.flatten is not None and self.flatten is not False:
            if isinstance(self.flatten, list):
                raw = _flatten_axes(raw, self.flatten)
            else:
                raw = raw.ravel()

        if np.issubdtype(raw.dtype, np.number):
            result = raw * self.scale
        else:
            result = raw

        if self.keep_array:
            return result

        if result.ndim != 0:
            raise ValueError(
                f"Field at path {self.path!r} resolved to a non-scalar array "
                f"of shape {result.shape}. Add a reducing transform such as "
                f"'step_from_positions', or set keep_array=true in the mapping."
            )
        return result.item()


def _parse_field_spec(raw) -> FieldSpec:
    if isinstance(raw, str):
        return FieldSpec(path=raw)
    if isinstance(raw, (int, float)):
        return FieldSpec(value=float(raw))
    if isinstance(raw, dict):
        return FieldSpec(**raw)
    raise TypeError(f"Cannot parse field spec from {raw!r}")


# ---------------------------------------------------------------------------
# HDF5Loader
# ---------------------------------------------------------------------------

class HDF5Loader:
    '''Generic HDF5/NeXus loader driven by a declarative field mapping.

    Parameters
    ----------
    mapping : dict or path-like
        Field mapping as a dict or path to a YAML file.
    overrides : dict, optional
        Field specs merged on top of ``mapping``.

    Examples
    --------
    ::

        loader = HDF5Loader("config/i14_mapping.yaml")
        data = loader.load("scan.nxs")
        ds   = bind(data, MyDataset)
    '''

    def __init__(
        self,
        mapping: dict | str | Path,
        overrides: Optional[dict] = None,
    ):
        raw = mapping if isinstance(mapping, dict) else _load_yaml(mapping)
        if overrides:
            raw = {**raw, **overrides}
        self._specs: dict[str, FieldSpec] = {
            k: _parse_field_spec(v) for k, v in raw.items()
        }

    def load(
        self,
        hdf5_path: str | Path,
        lazy: bool = False,
        overrides: Optional[dict] = None,
    ) -> dict:
        '''Load fields from an HDF5 file into a plain dict.

        Parameters
        ----------
        hdf5_path : path-like
        lazy : bool
            When True, array fields with ``keep_array=True`` are returned as
            live ``h5py.Dataset`` objects.  The caller must keep the file open.
        overrides : dict, optional
        '''
        specs = self._specs
        if overrides:
            specs = {**specs, **{k: _parse_field_spec(v)
                                 for k, v in overrides.items()}}

        result: dict = {}
        f = h5py.File(hdf5_path, "r", locking=False)
        try:
            for field_name, spec in specs.items():
                if lazy and spec.keep_array and spec.path and spec.path in f:
                    result[field_name] = f[spec.path]
                else:
                    resolved = spec.resolve(f)
                    if resolved is not None:
                        result[field_name] = resolved
        finally:
            if not lazy:
                f.close()

        return result


# ---------------------------------------------------------------------------
# bind
# ---------------------------------------------------------------------------

def bind(data: dict, schema) -> Any:
    '''Populate a dataclass from a plain dict.

    Fields present in ``data`` that match a constructor parameter of
    ``schema`` are passed directly.  Any remaining fields are collected into
    ``schema.metadata`` if that field exists, otherwise they are silently
    dropped.

    Example
    -------
    ::

        ds = bind(loader.load("scan.nxs"), MyDataset)
    '''
    schema_fields = {f.name for f in dataclasses.fields(schema)}
    kwargs  = {k: v for k, v in data.items() if k in schema_fields
                                              and k != "metadata"}
    extra   = {k: v for k, v in data.items() if k not in schema_fields}

    if "metadata" in schema_fields:
        kwargs["metadata"] = extra

    return schema(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
