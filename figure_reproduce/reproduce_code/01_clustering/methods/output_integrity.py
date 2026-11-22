"""Atomic H5AD writing for clustering method outputs."""
from __future__ import annotations
import os
import uuid
from pathlib import Path
import h5py

def hdf5_read_error(path: Path) -> str:
    failed = ''
    try:
        with h5py.File(path, 'r') as handle:
            names = []
            handle.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
            for name in names:
                failed = f'/{name}'
                handle[name][()]
    except Exception as exc:
        return f'{failed}: {type(exc).__name__}: {exc}'
    return ''

def atomic_write_h5ad(data, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(f'.{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}')
    try:
        data.write_h5ad(temporary, compression='gzip')
        error = hdf5_read_error(temporary)
        if error:
            raise OSError(error)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
