'''Command-line interface for adapt-io.

Commands
--------
convert
    Read an HDF5/NeXus file, apply a YAML field mapping, and write the
    remapped/transformed fields to a new file.  Writes a sidecar manifest
    (``<output>.manifest.json``) recording what was created.

inspect
    Dry-run: print what fields would be loaded without writing anything.

clean
    Delete output files listed in a manifest, then remove the manifest.

Usage examples
--------------
::

    adapt-io convert scan.nxs -m config/i14_mapping.yaml -o out.h5
    adapt-io inspect scan.nxs -m config/i14_mapping.yaml
    adapt-io clean out.h5                   # deletes out.h5 + manifest
    adapt-io clean out.h5 --dry-run         # shows what would be deleted
'''

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .hdf5_loader import HDF5Loader


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + '.manifest.json')


def _write_manifest(output_path: Path, input_path: Path, mapping_path: Path) -> Path:
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'input':      str(input_path.resolve()),
        'mapping':    str(mapping_path.resolve()),
        'outputs':    [str(output_path.resolve())],
    }
    mpath = _manifest_path(output_path)
    mpath.write_text(json.dumps(manifest, indent=2))
    return mpath


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_hdf5(data: dict[str, Any], output_path: Path, mapping_path: Path) -> None:
    with h5py.File(output_path, 'w') as f:
        f.attrs['NX_class']     = 'NXroot'
        f.attrs['created_by']   = 'adapt-io'
        f.attrs['mapping_file'] = str(mapping_path)
        f.attrs['created_at']   = datetime.now(timezone.utc).isoformat()

        for name, value in data.items():
            arr = np.asarray(value)
            try:
                f.create_dataset(name, data=arr)
            except TypeError:
                f.create_dataset(name, data=str(value))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_convert(args: argparse.Namespace) -> int:
    input_path   = Path(args.input)
    mapping_path = Path(args.mapping)
    output_path  = Path(args.output)

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1
    if not mapping_path.exists():
        print(f"error: mapping file not found: {mapping_path}", file=sys.stderr)
        return 1

    print(f"Loading  {input_path}")
    print(f"Mapping  {mapping_path}")

    loader = HDF5Loader(mapping_path)
    data   = loader.load(input_path)

    if not data:
        print("warning: no fields resolved — check your mapping paths", file=sys.stderr)
        return 1

    print(f"Resolved {len(data)} field(s): {', '.join(data)}")

    _write_hdf5(data, output_path, mapping_path)
    manifest = _write_manifest(output_path, input_path, mapping_path)

    print(f"Written  {output_path}")
    print(f"Manifest {manifest}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    input_path   = Path(args.input)
    mapping_path = Path(args.mapping)

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1
    if not mapping_path.exists():
        print(f"error: mapping file not found: {mapping_path}", file=sys.stderr)
        return 1

    loader = HDF5Loader(mapping_path)
    data   = loader.load(input_path)

    if not data:
        print("No fields resolved.")
        return 0

    col = max(len(k) for k in data) + 2
    print(f"\n{'Field':<{col}}  {'Shape / Value'}")
    print("-" * (col + 30))
    for name, value in data.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            summary = str(arr.item())
        else:
            summary = f"array {arr.shape}  dtype={arr.dtype}"
        print(f"{name:<{col}}  {summary}")
    print()
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    output_path   = Path(args.output)
    manifest_path = _manifest_path(output_path)
    dry_run       = args.dry_run

    if not manifest_path.exists():
        print(f"error: no manifest found at {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    to_delete = [Path(p) for p in manifest.get('outputs', [])] + [manifest_path]

    tag = "[dry-run] " if dry_run else ""
    for path in to_delete:
        if path.exists():
            print(f"{tag}delete {path}")
            if not dry_run:
                path.unlink()
        else:
            print(f"skip (not found) {path}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='adapt-io',
        description='Read, transform, and remap HDF5/NeXus fields via a YAML mapping.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # --- convert ---
    p_conv = sub.add_parser('convert', help='Read, transform, and write remapped fields.')
    p_conv.add_argument('input',   help='Input HDF5/NeXus file.')
    p_conv.add_argument('-m', '--mapping', required=True, help='YAML field mapping.')
    p_conv.add_argument('-o', '--output',  required=True, help='Output file path.')
    p_conv.add_argument(
        '-f', '--format',
        choices=['hdf5', 'nxs'],
        default='hdf5',
        help='Output format (default: hdf5).',
    )

    # --- inspect ---
    p_insp = sub.add_parser('inspect', help='Dry-run: print resolved fields without writing.')
    p_insp.add_argument('input',   help='Input HDF5/NeXus file.')
    p_insp.add_argument('-m', '--mapping', required=True, help='YAML field mapping.')

    # --- clean ---
    p_clean = sub.add_parser('clean', help='Delete outputs recorded in a manifest.')
    p_clean.add_argument('output', help='Output file passed to convert (used to locate manifest).')
    p_clean.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting.')

    return parser


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    handler = {'convert': cmd_convert, 'inspect': cmd_inspect, 'clean': cmd_clean}[args.command]
    sys.exit(handler(args))


if __name__ == '__main__':
    main()
