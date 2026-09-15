"""Bidirectional coordinate mapping between native mask space and
annotation/atlas space.

Ported from MARS's 3Dfilter/filter_annotation.py (D:\\chu_lab\\MARS\\3Dfilter\\
filter_annotation.py -- see ``get_dimension``/``get_points``/``cal_points``),
generalized into an explicit, reusable, ``transpose_order``-driven utility
instead of MARS's variable-naming trick (it calls the raw annotation
volume's ``shape[0]`` "height" and ``shape[1]`` "depth" specifically to
encode a hardcoded Y/Z swap). This module makes that swap an explicit,
overridable parameter -- see the architecture plan's confirmed facts about
the BIRDS Y/Z-swap convention.

Key design choice, matching MARS's ACTUAL production algorithm (verified by
direct algebraic correspondence against filter_annotation.py, not a
simplification of it): the native<->atlas point mapping uses the two
volumes' real, measured shapes at runtime --
``native_point[i] * atlas_shape[i] // native_shape[i]`` -- not a nominal
``downsample_factor`` from config. This is strictly more robust: if BIRDS
registration's actual output resolution doesn't exactly match the nominal
downsample_factor (e.g. due to rounding during the external registration
step), the ratio-based mapping still gives correct results, while a naive
``coord // downsample_factor`` would silently drift. ``downsample_factor``
from config is only used by ``align.convert``'s FORWARD pass (deciding how
much to downsample when there's no existing atlas volume yet to measure
against) -- never here.

Once ``io.FileReader(annotation_path, transpose_order=transpose_order)``
reads the annotation, its ``.volume_shape`` and array indexing are already
in "logical" (Z,Y,X)-matching axis order (matching the fix made to
``io/reader.py`` in this same phase) -- so no further permutation logic is
needed in this module for the point-mapping direction; it reduces to a
plain elementwise ratio, which is exactly what the algebra above confirms
reproduces MARS's per-axis ``MarkerZ * anno_depth // mask_depth`` style
computation term for term.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..common.atomic_io import atomic_write_table
from ..common.backup import backup_cells_file
from ..common.cell_schema import CELL_SCHEMA
from ..io import FileReader, FileWriter
from ..io.convert import convert_zarr_to_format


def invert_transpose_order(transpose_order: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return the inverse of a 3-axis permutation.

    For a permutation ``p`` such that ``logical_shape[i] = native_shape[p[i]]``
    (the convention used throughout ``io.reader``/``io.writer``), the
    inverse ``q`` satisfies ``q[p[i]] = i`` -- i.e. ``logical[q[j]]`` maps
    back to native axis ``j``.
    """
    order = list(transpose_order)
    inverse = [0, 0, 0]
    for i, axis in enumerate(order):
        inverse[axis] = i
    return tuple(inverse)


def native_to_atlas_point(
    native_point: tuple[int, int, int],
    native_shape: tuple[int, int, int],
    atlas_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Map one (z,y,x) native-space point to atlas-space coordinates.

    ``atlas_shape`` must already be in logical (Z,Y,X)-matching axis order --
    i.e. the ``.volume_shape`` of an ``io.FileReader`` opened on the
    annotation with the run's ``transpose_order`` applied, NOT the
    annotation's raw on-disk shape.
    """
    return tuple(
        int(native_point[i]) * int(atlas_shape[i]) // int(native_shape[i])
        for i in range(3)
    )


def native_to_atlas_points(
    points: np.ndarray,
    native_shape: tuple[int, int, int],
    atlas_shape: tuple[int, int, int],
) -> np.ndarray:
    """Vectorized form of ``native_to_atlas_point`` for an (N,3) array of (z,y,x)."""
    points = np.asarray(points)
    scaled = np.empty(points.shape, dtype=np.int64)
    for i in range(3):
        scaled[:, i] = (points[:, i].astype(np.int64) * int(atlas_shape[i])) // int(native_shape[i])
    return scaled


def lookup_region_and_hemisphere(
    native_points: np.ndarray,
    native_shape: tuple[int, int, int],
    annotation: np.ndarray,
    hemisphere: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve region_id/hemisphere_id for a batch of native-space cell points.

    Args:
        native_points: (N,3) int array of (z,y,x) in native mask space.
        native_shape: The native mask's (Z,Y,X) shape.
        annotation: The annotation volume, already in logical (Z,Y,X)-matching
            axis order (e.g. read via ``io.FileReader(path, transpose_order=...)``).
        hemisphere: Optional hemisphere-label volume, same shape/order as
            ``annotation``. If omitted, all returned hemisphere ids are NA (0).

    Points that fall outside the annotation volume after scaling (possible
    at the extreme edge due to floor-division rounding) are clamped into
    bounds rather than raising, matching MARS's own tolerant boundary
    handling (``safe_z_end = min(anno_z_end + 1, self.anno_depth)``).

    Returns:
        (region_ids, hemisphere_ids): int32 and int8 arrays, length N.
    """
    if len(native_points) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8)

    atlas_shape = annotation.shape
    atlas_points = native_to_atlas_points(native_points, native_shape, atlas_shape)
    for i in range(3):
        atlas_points[:, i] = np.clip(atlas_points[:, i], 0, atlas_shape[i] - 1)

    region_ids = np.asarray(
        annotation[atlas_points[:, 0], atlas_points[:, 1], atlas_points[:, 2]]
    ).astype(np.int32)

    if hemisphere is not None:
        hemisphere_ids = np.asarray(
            hemisphere[atlas_points[:, 0], atlas_points[:, 1], atlas_points[:, 2]]
        ).astype(np.int8)
    else:
        hemisphere_ids = np.zeros(len(native_points), dtype=np.int8)

    return region_ids, hemisphere_ids


def assign_regions_in_cell_table(
    cells_path: str | Path,
    native_shape: tuple[int, int, int],
    annotation: np.ndarray,
    hemisphere: Optional[np.ndarray] = None,
    biomarker: Optional[str] = None,
    *,
    backup: bool = True,
    keep_backups: int = 3,
) -> dict:
    """Populate ``region_id``/``hemisphere_id`` for cells in a Parquet table.

    Realizes this phase's deliverable end-to-end: cell coordinates produced
    by ``detection.dask_runner.detect_biomarker`` (Phase 1, native space,
    ``region_id`` left null) can now be resolved against a real annotation
    volume. Rewrites the Parquet file in place (atomically), same pattern as
    ``filtering.volume_filter.apply_volume_filter``, including a pre-rewrite
    backup snapshot (see ``common.backup``).

    Args:
        cells_path: Path to the cell Parquet table.
        native_shape: The native mask's (Z,Y,X) shape these cells were
            detected against.
        annotation: Annotation volume in logical (Z,Y,X)-matching axis order.
        hemisphere: Optional hemisphere-label volume, same shape/order.
        biomarker: If given, only this biomarker's rows are (re)assigned.
        backup: Whether to snapshot the file before rewriting it.
        keep_backups: Number of most-recent "assign_regions" backups to retain.

    Returns:
        {"assigned": <number of rows updated>}
    """
    cells_path = Path(cells_path)
    if backup:
        backup_cells_file(cells_path, stage="assign_regions", keep=keep_backups)
    table = pq.read_table(cells_path)
    df = table.to_pandas()

    if biomarker is not None:
        selector = df["biomarker"] == biomarker
    else:
        selector = pd.Series(True, index=df.index)

    n_assigned = int(selector.sum())
    if n_assigned:
        points = df.loc[selector, ["z", "y", "x"]].to_numpy()
        region_ids, hemisphere_ids = lookup_region_and_hemisphere(points, native_shape, annotation, hemisphere)
        df.loc[selector, "region_id"] = region_ids
        df.loc[selector, "hemisphere_id"] = hemisphere_ids

    df["region_id"] = df["region_id"].astype("Int32")
    df["hemisphere_id"] = df["hemisphere_id"].astype("int8")

    new_table = pa.Table.from_pandas(df, schema=CELL_SCHEMA, preserve_index=False)
    atomic_write_table(new_table, cells_path)

    return {"assigned": n_assigned}


def resample_atlas_volume_to_native(
    atlas_path: str | Path,
    transpose_order: tuple[int, int, int],
    native_shape: tuple[int, int, int],
    output_path: str | Path,
    output_name: str,
    resize_order: int = 0,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    n_workers: int = 8,
    output_type: str = "single-tiff",
) -> Path:
    """Resample an atlas-space volume back to native mask resolution/orientation.

    Used by ``roi_extract`` (Phase 3) so an extracted region mask can be
    loaded directly alongside the original raw/native image (e.g. in
    Imaris) -- per the user's clarification of why that step exists at all.

    Composes two already-ported IO utilities rather than reinventing
    resize+transpose: reads the RAW (pre-transpose_order, as stored on
    disk -- e.g. an actual annotation.tif) atlas file with ``transpose_order``
    applied, exactly the same way ``lookup_region_and_hemisphere`` expects
    its ``annotation`` argument to have been produced -- this yields an
    array already in logical (native-axis-matching) order, just at
    atlas/downsampled RESOLUTION. From there it's a pure resize (no further
    transpose needed, since logical order by definition already matches
    native axis semantics) -- delegates to ``resize_atlas_array_to_native``.

    NOTE ON DIRECTION: this reads with ``transpose_order`` directly, not its
    inverse. An earlier version of this function used
    ``invert_transpose_order(transpose_order)`` here, which is wrong in
    general (only happened to work in testing because the only currently
    supported ``transpose_order`` values are self-inverse permutations --
    see ``common.config``'s validation, which now rejects any
    ``transpose_order`` that isn't self-inverse specifically so this
    forward/inverse direction can never silently diverge again).
    ``resize_order=0`` (nearest-neighbor) by default, since atlas volumes
    are integer region/label ids that must not be interpolated.

    Args:
        output_type: Final output format (default ``'single-tiff'``, so the
            result is directly openable in Fiji at the same dimensions as
            the original native image). See ``resize_atlas_array_to_native``
            for how non-Zarr formats are produced.
    """
    reader = FileReader(atlas_path, transpose_order=transpose_order)
    data = reader.read()
    return resize_atlas_array_to_native(
        data, native_shape, output_path, output_name,
        resize_order=resize_order, chunk_size=chunk_size, n_workers=n_workers,
        output_type=output_type,
    )


def resize_atlas_array_to_native(
    atlas_array: np.ndarray,
    native_shape: tuple[int, int, int],
    output_path: str | Path,
    output_name: str,
    resize_order: int = 0,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    n_workers: int = 8,
    output_type: str = "single-tiff",
) -> Path:
    """Upsample an already-logical-order atlas-space array to native resolution.

    No transpose is applied -- "logical order" (as produced by
    ``io.FileReader(path, transpose_order=transpose_order)``, or already
    held in memory e.g. by ``roi_extract`` right after building an extracted
    region mask) already matches native axis semantics; only the per-axis
    SIZE differs. This is the array-based counterpart to
    ``resample_atlas_volume_to_native`` (which additionally handles reading
    a raw on-disk file first) -- use this one directly when the logical-order
    array is already in memory.

    Args:
        output_type: Final output format. ``io.FileWriter``'s two-pass resize
            pipeline only supports ``'zarr'``/``'ome-zarr'`` targets, so for
            any other ``output_type`` (default ``'single-tiff'``) this
            resizes into a temporary intermediate Zarr store first, converts
            it to the requested format via ``io.convert_zarr_to_format``,
            then deletes the intermediate store.
    """
    zarr_needed = output_type not in ("zarr", "ome-zarr")
    zarr_output_path = Path(output_path) / f"_tmp_resize_{output_name}" if zarr_needed else output_path
    zarr_output_name = output_name

    writer = FileWriter(
        output_path=zarr_output_path,
        output_name=zarr_output_name,
        output_type="zarr",
        full_res_shape=native_shape,
        input_shape=atlas_array.shape,
        output_dtype=atlas_array.dtype,
        resize_order=resize_order,
        chunk_size=chunk_size,
        n_workers=n_workers,
    )
    writer.write(atlas_array)
    writer.complete_resize()
    zarr_path = writer.output_path

    if not zarr_needed:
        return zarr_path

    final_path = convert_zarr_to_format(
        zarr_path, output_path, output_name, output_type, chunk_size=chunk_size,
    )
    shutil.rmtree(zarr_path.parent, ignore_errors=True)
    return final_path
