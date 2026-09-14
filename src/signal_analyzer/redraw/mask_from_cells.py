"""Reconstruct a mask volume from filtered cell positions.

Two output modes, matching what MARS's x2xz.py actually does today:
  - ``redraw_native_mask``: native-space, no transform -- useful for QC
    against the original mask. Unlike MARS's x2xz.py (which writes one 2D
    TIFF per Y-layer via PIL + multiprocessing, applying a redundant
    hardcoded SCALE_X/Y/Z division), this writes a single chunked Zarr
    volume via ``io.FileWriter`` -- no second downsample step, since these
    cell coordinates are already native-resolution voxel coordinates
    (detection never downsamples; see the architecture plan's confirmed
    facts).
  - ``redraw_atlas_mask``: atlas-space (what x2xz.py actually produces) --
    converts each coordinate through ``regions.coord_transform.
    native_to_atlas_points`` before scattering, reproducing x2xz.py's
    downsample-and-reslice-into-annotation-space behavior via the same
    shared, tested coordinate-mapping utility used everywhere else in this
    codebase, instead of x2xz.py's standalone hardcoded SCALE_X/Y/Z.

Both are written in Z-slabs (not one big in-memory array), to stay
memory-bounded for full-size brain volumes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..common.cell_schema import read_cells
from ..io import FileWriter
from ..regions.coord_transform import native_to_atlas_points

logger = logging.getLogger(__name__)


def _scatter_points_chunked(
    points: pd.DataFrame,
    shape: tuple[int, int, int],
    output_path: str | Path,
    output_name: str,
    dtype,
    fill_value: int,
    chunk_size: tuple[int, int, int],
) -> Path:
    """Shared Z-chunked scatter-write used by both redraw modes.

    ``points`` must already be filtered to in-bounds (z,y,x) columns in the
    target ``shape``'s coordinate space.
    """
    writer = FileWriter(
        output_path=output_path,
        output_name=output_name,
        output_type="zarr",
        full_res_shape=shape,
        output_dtype=dtype,
        chunk_size=chunk_size,
    )

    total_z = shape[0]
    z_step = chunk_size[0]

    for z0 in range(0, total_z, z_step):
        z1 = min(z0 + z_step, total_z)
        slab = np.zeros((z1 - z0, shape[1], shape[2]), dtype=dtype)

        if not points.empty:
            chunk_points = points[(points["z"] >= z0) & (points["z"] < z1)]
            if not chunk_points.empty:
                local_z = chunk_points["z"].to_numpy() - z0
                slab[local_z, chunk_points["y"].to_numpy(), chunk_points["x"].to_numpy()] = fill_value

        writer.write(slab, z_start=z0, z_end=z1)

    return writer.output_path


def _drop_out_of_bounds(df: pd.DataFrame, shape: tuple[int, int, int], context: str) -> pd.DataFrame:
    if df.empty:
        return df
    in_bounds = (
        (df["z"] >= 0) & (df["z"] < shape[0])
        & (df["y"] >= 0) & (df["y"] < shape[1])
        & (df["x"] >= 0) & (df["x"] < shape[2])
    )
    n_dropped = int((~in_bounds).sum())
    if n_dropped:
        logger.warning("%s: dropping %d cell(s) outside shape=%s", context, n_dropped, shape)
    return df[in_bounds]


def redraw_native_mask(
    cells_path: str | Path,
    native_shape: tuple[int, int, int],
    output_path: str | Path,
    output_name: str,
    biomarker: Optional[str] = None,
    dtype=np.uint8,
    fill_value: int = 255,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Scatter passed-filter cell positions into a native-resolution mask volume.

    Args:
        cells_path: Path to the cell Parquet table.
        native_shape: (Z,Y,X) shape of the original mask this redraw should match.
        output_path: Directory to write the output Zarr store into.
        output_name: Base name for the output Zarr store.
        biomarker: If given, only redraw this biomarker's cells.
        dtype: Output voxel dtype.
        fill_value: Value written at each cell's voxel.
        chunk_size: Zarr chunking for the output; also the Z-slab size used
            while scattering, so memory use stays bounded regardless of
            ``native_shape``.

    Returns:
        Path to the written Zarr store.
    """
    df = read_cells(cells_path, biomarker=biomarker, passed_filter=True, columns=["z", "y", "x"])
    df = _drop_out_of_bounds(df, native_shape, "redraw_native_mask")
    return _scatter_points_chunked(df, native_shape, output_path, output_name, dtype, fill_value, chunk_size)


def redraw_atlas_mask(
    cells_path: str | Path,
    native_shape: tuple[int, int, int],
    atlas_shape: tuple[int, int, int],
    output_path: str | Path,
    output_name: str,
    biomarker: Optional[str] = None,
    dtype=np.uint8,
    fill_value: int = 255,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Scatter passed-filter cell positions into an atlas-space mask volume.

    Reproduces MARS's x2xz.py behavior (downsample + reslice filtered cells
    into the annotation's own orientation/resolution), via
    ``regions.coord_transform.native_to_atlas_points`` instead of x2xz.py's
    standalone hardcoded ``SCALE_X,Y,Z``.

    Args:
        cells_path: Path to the cell Parquet table (native-space coordinates).
        native_shape: The native mask's (Z,Y,X) shape these cells were
            detected against.
        atlas_shape: The annotation's logical (Z,Y,X)-matching shape (e.g.
            ``io.FileReader(annotation_path, transpose_order=...).volume_shape``).
        output_path: Directory to write the output Zarr store into.
        output_name: Base name for the output Zarr store.
        biomarker: If given, only redraw this biomarker's cells.
        dtype: Output voxel dtype.
        fill_value: Value written at each cell's voxel.
        chunk_size: Zarr chunking for the output.

    Returns:
        Path to the written Zarr store, in atlas space/orientation.
    """
    df = read_cells(cells_path, biomarker=biomarker, passed_filter=True, columns=["z", "y", "x"])

    if not df.empty:
        atlas_points = native_to_atlas_points(df[["z", "y", "x"]].to_numpy(), native_shape, atlas_shape)
        df = pd.DataFrame(atlas_points, columns=["z", "y", "x"])
        df = _drop_out_of_bounds(df, atlas_shape, "redraw_atlas_mask")

    return _scatter_points_chunked(df, atlas_shape, output_path, output_name, dtype, fill_value, chunk_size)
