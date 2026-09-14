"""Once-per-brain annotation geometry: voxel counts per Allen CCF region
(total/left/right), independent of any biomarker.

Computed once and cacheable per brain -- unlike Chulab's per-biomarker
full-volume Dask scan (``numba_unique_cell``, in
Chulab-Signal_Analyzer/utils/analyzer_count_tools.py), which recomputes
region geometry from scratch for every biomarker even though it's identical
across all of them for a given brain (region geometry depends only on
``annotation.tif`` + hemisphere, never on detected cells).

Ported from the per-region voxel-tally half of MARS's filter_annotation.py::
calculate_cells (the ``volume``/``volume_left``/``volume_right`` Counters) --
generalized to a single vectorized ``np.unique`` pass instead of MARS's
per-Z-slice multiprocessing Counter accumulation, since our atlas-space
annotation arrays are small enough (that's the point of downsampling for
registration) to process in one shot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..common.cell_schema import HEMISPHERE_LEFT, HEMISPHERE_RIGHT
from ..regions.structures import rollup_tiers


def compute_region_voxel_counts(
    annotation: np.ndarray,
    hemisphere: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Tally total/left/right voxel counts per Allen CCF region id.

    Args:
        annotation: Annotation volume (region id per voxel, 0 = background),
            in logical (Z,Y,X)-matching axis order.
        hemisphere: Optional hemisphere-label volume (HEMISPHERE_LEFT/RIGHT
            per voxel), same shape/order as ``annotation``. If omitted,
            falls back to MARS's own convention: an X-axis midline split
            directly on ``annotation`` (``annotation.shape[-1] // 2``).

    Returns:
        DataFrame with columns [id, total_voxels, left_voxels, right_voxels].
    """
    nonzero = annotation != 0
    ids, counts = np.unique(annotation[nonzero], return_counts=True)
    df = pd.DataFrame({"id": ids.astype(np.int64), "total_voxels": counts.astype(np.int64)})

    if hemisphere is not None:
        left_mask = nonzero & (hemisphere == HEMISPHERE_LEFT)
        right_mask = nonzero & (hemisphere == HEMISPHERE_RIGHT)
    else:
        midline = annotation.shape[-1] // 2
        left_region = np.zeros_like(nonzero)
        left_region[..., :midline] = True
        left_mask = nonzero & left_region
        right_mask = nonzero & ~left_region

    left_ids, left_counts = np.unique(annotation[left_mask], return_counts=True)
    right_ids, right_counts = np.unique(annotation[right_mask], return_counts=True)

    left_df = pd.DataFrame({"id": left_ids.astype(np.int64), "left_voxels": left_counts.astype(np.int64)})
    right_df = pd.DataFrame({"id": right_ids.astype(np.int64), "right_voxels": right_counts.astype(np.int64)})

    df = df.merge(left_df, on="id", how="left").merge(right_df, on="id", how="left")
    df[["left_voxels", "right_voxels"]] = df[["left_voxels", "right_voxels"]].fillna(0).astype(np.int64)
    return df


def build_region_geometry_report(
    annotation: np.ndarray,
    structure_df: pd.DataFrame,
    voxel_size_um: tuple[float, float, float],
    hemisphere: Optional[np.ndarray] = None,
) -> dict[int, pd.DataFrame]:
    """Build the tiered (leaf-to-root) per-region voxel-geometry report.

    This is the once-per-brain computation report.cell_report's type-A
    branch joins against for cell density -- see that module.

    Returns:
        {tier: DataFrame}, deepest tier first, each row a structure with
        columns [structure_id, structure_name, acronym, total_voxels,
        left_voxels, right_voxels, total_volume_mm3, left_volume_mm3,
        right_volume_mm3].
    """
    counts = compute_region_voxel_counts(annotation, hemisphere)

    summary = structure_df.merge(counts, on="id", how="left")
    summary[["total_voxels", "left_voxels", "right_voxels"]] = (
        summary[["total_voxels", "left_voxels", "right_voxels"]].fillna(0).astype(np.int64)
    )
    summary = summary.set_index("id", drop=False)

    sheets = rollup_tiers(summary, structure_df, ["total_voxels", "left_voxels", "right_voxels"])

    voxel_volume_um3 = voxel_size_um[0] * voxel_size_um[1] * voxel_size_um[2]
    voxel_volume_mm3 = voxel_volume_um3 * 1e-9

    for tier, sheet in sheets.items():
        sheet["total_volume_mm3"] = sheet["total_voxels"] * voxel_volume_mm3
        sheet["left_volume_mm3"] = sheet["left_voxels"] * voxel_volume_mm3
        sheet["right_volume_mm3"] = sheet["right_voxels"] * voxel_volume_mm3

    return sheets
