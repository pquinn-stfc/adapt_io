'''nexus-io — generic NeXus/HDF5 I/O for scanning X-ray nanoprobe beamlines.

Public API
----------
Schema:
    AxisInfo, NXData

NeXus file loading:
    NeXusLoader, coord_to_index, slice_by_coords, on_same_nav_grid, apply_nav

YAML-driven HDF5 field mapping:
    HDF5Loader, bind, FieldSpec, register_transform

Writing:
    save_nxdata
'''

from .schema import AxisInfo, NXData
from .loader import (
    NeXusLoader,
    coord_to_index,
    slice_by_coords,
    on_same_nav_grid,
    apply_nav,
)
from .hdf5_loader import HDF5Loader, bind, FieldSpec, register_transform
from .writer import save_nxdata

__all__ = [
    'AxisInfo', 'NXData',
    'NeXusLoader', 'coord_to_index', 'slice_by_coords',
    'on_same_nav_grid', 'apply_nav',
    'HDF5Loader', 'bind', 'FieldSpec', 'register_transform',
    'save_nxdata',
]
