"""Read a Zarr array and re-write it in another format via ``io.FileWriter``.

Needed because ``FileWriter``'s two-pass resize pipeline only supports
``zarr``/``ome-zarr`` targets (``_resizing_active`` in ``writer.py``
requires it) -- so any caller that needs to RESIZE data into a non-Zarr
final format (e.g. ``roi_extract``, which upsamples atlas-space region
masks to native resolution and wants a plain ``.tiff`` file) must resize
into an intermediate Zarr first, then convert. This module is that
conversion step, kept generic/reusable rather than one-off inside
``roi_extract``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .reader import FileReader
from .writer import FileWriter


def convert_zarr_to_format(
    zarr_path: str | Path,
    output_path: str | Path,
    output_name: str,
    output_type: str,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Convert a Zarr array to another ``io.FileWriter`` output format.

    Args:
        zarr_path: Path to the source Zarr store.
        output_path: Directory to write the converted output into.
        output_name: Base name for the converted output.
        output_type: Any ``io.FileWriter`` output type
            (``'single-tiff'``, ``'scroll-tiff'``, ``'single-nii'``,
            ``'scroll-nii'``, ``'zarr'``, ``'ome-zarr'``).
        chunk_size: Z-chunk size used for scroll-style outputs, to keep
            memory bounded; ``single-tiff``/``single-nii`` write the whole
            array in one call regardless (inherent to a single-file output).

    Returns:
        Path to the converted output.
    """
    reader = FileReader(zarr_path)
    shape = reader.volume_shape
    dtype = reader.volume_dtype

    file_name = None
    if output_type in ("scroll-tiff", "scroll-nii"):
        file_name = [Path(f"slice_{i:05d}") for i in range(shape[0])]

    writer = FileWriter(
        output_path=output_path,
        output_name=output_name,
        output_type=output_type,
        full_res_shape=shape,
        output_dtype=dtype,
        file_name=file_name,
        chunk_size=chunk_size,
    )

    if output_type in ("scroll-tiff", "scroll-nii"):
        total_z = shape[0]
        z_step = chunk_size[0]
        for z0 in range(0, total_z, z_step):
            z1 = min(z0 + z_step, total_z)
            writer.write(reader.read(z_start=z0, z_end=z1), z_start=z0, z_end=z1)
    else:
        writer.write(reader.read())

    # 'single-tiff'/'single-nii' embed the write() call's z-range in the
    # actual filename, only known after writing -- writer.output_path stays
    # the containing directory for these two types, so last_written_path is
    # the authoritative path. Every other output_type sets output_path to
    # the real final path at init time already.
    if writer.last_written_path is not None:
        return writer.last_written_path
    return writer.output_path
