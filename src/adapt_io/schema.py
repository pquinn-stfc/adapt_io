'''In-memory data schemas for NeXus/HDF5 data.

:class:`AxisInfo` and :class:`NXData` mirror the NeXus NXdata model and are
the common currency across all beamline analysis packages (dpc, xrf, xanes, …).
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class AxisInfo(NamedTuple):
    '''Coordinate information for one dimension of an :class:`NXData` array.

    Parameters
    ----------
    name : str
        Axis name, e.g. ``"scan_x"``, ``"energy"``.
    values : ndarray, shape (N,)
        Physical coordinate for each point along this dimension.
    navigate : bool
        ``True`` for scan/map (navigation) dimensions; ``False`` for
        detector/spectrum (signal) dimensions.
    units : str
        Physical unit string, e.g. ``"m"``, ``"eV"``.
    '''
    name: str
    values: np.ndarray
    navigate: bool = False
    units: str = ""

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def is_uniform(self) -> bool:
        '''True if all coordinate steps agree within 0.1 %.'''
        if len(self.values) < 2:
            return True
        steps = np.diff(self.values)
        return bool(np.allclose(steps, steps[0], rtol=1e-3))

    @property
    def step_size(self) -> float:
        '''Mean spacing between coordinate values.'''
        if len(self.values) < 2:
            return 1.0
        return float(np.diff(self.values).mean())

    def __repr__(self) -> str:
        kind = "nav" if self.navigate else "sig"
        tag  = f"uniform step={self.step_size:.3g}" if self.is_uniform \
               else f"irregular mean_step={self.step_size:.3g}"
        return (f"AxisInfo({self.name!r}, size={self.size}, "
                f"{kind}, {tag}, units={self.units!r})")


@dataclass
class NXData:
    '''In-memory representation of a NeXus NXdata group.

    Mirrors the NeXus model directly: a primary ``data`` array, one
    :class:`AxisInfo` per dimension, a ``signal_type`` label, and
    free-form ``metadata``.

    Parameters
    ----------
    data : ndarray
        The measurement array, shape ``(*nav_shape, *sig_shape)``.
    axes : list of AxisInfo
        One entry per dimension of ``data``, in array order.  Navigation
        axes come first; signal axes come last.
    signal_type : str
        Free-form label: ``"DPC"``, ``"XRF"``, ``"XANES"``, etc.
    metadata : dict
        Scalar fields from the NeXus file (instrument settings, etc.).
    '''

    data: np.ndarray
    axes: list
    signal_type: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"len(axes)={len(self.axes)} must equal data.ndim={self.data.ndim}"
            )

    @property
    def nav_axes(self) -> list:
        return [ax for ax in self.axes if ax.navigate]

    @property
    def sig_axes(self) -> list:
        return [ax for ax in self.axes if not ax.navigate]

    @property
    def nav_shape(self) -> tuple:
        return tuple(ax.size for ax in self.nav_axes)

    @property
    def sig_shape(self) -> tuple:
        return tuple(ax.size for ax in self.sig_axes)

    @property
    def signal_dimension(self) -> int:
        return len(self.sig_axes)

    def get_axis(self, name: str) -> AxisInfo:
        for ax in self.axes:
            if ax.name == name:
                return ax
        raise KeyError(
            f"No axis {name!r}. Available: {[a.name for a in self.axes]}"
        )

    def __repr__(self) -> str:
        nav = " × ".join(str(ax.size) for ax in self.nav_axes) or "–"
        sig = " × ".join(str(ax.size) for ax in self.sig_axes) or "–"
        kind = {0: "ScalarMap", 1: "SpectrumImage", 2: "ImageStack"}.get(
            self.signal_dimension, "Dataset"
        )
        label = f" [{self.signal_type}]" if self.signal_type else ""
        return f"NXData{label} | {kind} | nav={nav} | sig={sig}"
