'''Write NXData objects to NeXus/HDF5 files.

:func:`save_nxdata` is the generic writer: axes are written as 1-D datasets
and attached as HDF5 Dimension Scales so any NeXus-aware viewer (silx,
h5web, NeXpy) assigns correct physical axis labels automatically.
'''

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .schema import NXData


def save_nxdata(
    path: str | Path,
    ds: NXData,
    compression: str = 'gzip',
    program_name: str = 'nexus-io',
) -> None:
    '''Save an :class:`~adapt_io.schema.NXData` to a NeXus HDF5 file.

    Parameters
    ----------
    path : path-like
        Output file path.  Will be overwritten if it exists.
    ds : NXData
        Dataset to save.
    compression : str
        HDF5 compression filter.  Default ``'gzip'``.
    program_name : str
        Written to ``/entry/program_name``.  Override with the calling
        package name for provenance (e.g. ``'dpc'``, ``'xrf-linear-fit'``).
    '''
    with h5py.File(path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        entry.create_dataset('definition',   data='NXdata')
        entry.create_dataset('program_name', data=program_name)
        entry.create_dataset('start_time',
                             data=datetime.now(timezone.utc).isoformat())
        entry.create_dataset('signal_type',  data=ds.signal_type)

        if ds.metadata:
            meta = entry.create_group('metadata')
            meta.attrs['NX_class'] = 'NXcollection'
            for k, v in ds.metadata.items():
                try:
                    meta.create_dataset(k, data=v)
                except TypeError:
                    meta.create_dataset(k, data=str(v))

        nxdata = entry.create_group('data')
        nxdata.attrs['NX_class'] = 'NXdata'
        nxdata.attrs['signal']   = 'data'
        nxdata.attrs['axes']     = [ax.name for ax in ds.axes]

        ax_datasets = []
        for ax in ds.axes:
            ax_ds = nxdata.create_dataset(ax.name,
                                          data=ax.values.astype(np.float64))
            ax_ds.attrs['units']      = ax.units
            ax_ds.attrs['long_name']  = ax.name
            ax_ds.attrs['navigate']   = ax.navigate
            ax_ds.attrs['is_uniform'] = ax.is_uniform
            if ax.is_uniform:
                ax_ds.attrs['step_size'] = ax.step_size
            ax_ds.make_scale(ax.name)
            ax_datasets.append(ax_ds)

        data_ds = nxdata.create_dataset(
            'data', data=ds.data, compression=compression
        )
        for i, ax_ds in enumerate(ax_datasets):
            data_ds.dims[i].attach_scale(ax_ds)
