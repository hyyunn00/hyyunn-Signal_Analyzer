"""Applies per-biomarker volume-filter bounds to an already-detected cell table.

Placement rationale (see the architecture plan's "Volume filtering design"
section): filtering is a distinct, cheap, idempotent step operating purely
on the already-written Parquet table -- no image data needed -- kept
separate from detection. Detection's ``volume_voxels`` is never
recomputed here; this only ever flips ``passed_filter`` based on config
thresholds, so detection runs once per biomarker and a threshold change is
a re-filter query, not a re-detection.

Bounds are inclusive (``min_volume_voxels <= volume_voxels <=
max_volume_voxels``), unlike MARS's exclusive ``25 < MarkerS < 10000``
comparisons -- and unlike MARS, there is exactly one place this bound is
applied, fixing the class of bug where MARS's ``filter_annotation.py``
(``25 < ... < 10000``) and ``x2xz.py`` (``10 < ... < 10000``) silently
diverged.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..common.atomic_io import atomic_write_table
from ..common.backup import backup_cells_file
from ..common.config import BiomarkerConfig


def apply_volume_filter(
    cells_path: str | Path,
    biomarker: BiomarkerConfig,
    *,
    backup: bool = True,
    keep_backups: int = 3,
) -> dict:
    """Recompute ``passed_filter`` for one biomarker's cells based on its configured bounds.

    Overwrites the Parquet file in place (read whole table, update the
    column, rewrite atomically) -- fine for a per-biomarker/per-brain cell
    table; not intended for a giant table merging many brains. By default, a
    snapshot of the pre-rewrite file is kept under ``cells_backups/`` (see
    ``common.backup``) so a threshold change can be compared against or
    recovered from.

    Args:
        cells_path: Path to a cell Parquet table (as written by
            ``detection.dask_runner.detect_biomarker``).
        biomarker: The biomarker's config, supplying
            ``filter.min_volume_voxels``/``filter.max_volume_voxels``.
        backup: Whether to snapshot the file before rewriting it.
        keep_backups: Number of most-recent "filter" backups to retain.

    Returns:
        {"total": <rows for this biomarker>, "passed": <rows now marked passed_filter>}
    """
    cells_path = Path(cells_path)
    if backup:
        backup_cells_file(cells_path, stage="filter", keep=keep_backups)
    table = pq.read_table(cells_path)

    is_this_biomarker = pc.equal(table.column("biomarker"), biomarker.name)
    volume = table.column("volume_voxels")
    within_bounds = pc.and_(
        pc.greater_equal(volume, biomarker.filter.min_volume_voxels),
        pc.less_equal(volume, biomarker.filter.max_volume_voxels),
    )

    # Only this biomarker's rows are touched; other biomarkers' passed_filter
    # values (if the table is ever shared across biomarkers) are preserved.
    existing = table.column("passed_filter")
    new_passed_filter = pc.if_else(is_this_biomarker, within_bounds, existing)

    table = table.set_column(
        table.schema.get_field_index("passed_filter"),
        "passed_filter",
        new_passed_filter,
    )
    atomic_write_table(table, cells_path)

    total = pc.sum(pc.cast(is_this_biomarker, "int64")).as_py() or 0
    passed = pc.sum(pc.cast(pc.and_(is_this_biomarker, within_bounds), "int64")).as_py() or 0
    return {"total": total, "passed": passed}
