"""Colocalization analysis: for each of the SMALLER biomarker's filtered
cell centroids, test whether it falls inside the LARGER biomarker's
ORIGINAL, unfiltered mask.

Per the user's definition (repo_diff.md's "獨立需求：Colocalize 分析"): take
the centroid of the smaller biomarker's detected cells, and check whether it
falls inside the larger biomarker's original mask -- if so, the cell counts
as colocalized. Neither MARS nor Chulab-Signal_Analyzer has any code for
this (confirmed by exploration; the only prior art, MARS's untracked
``2D_multply_mytest.py``, does raw pixel-wise mask x intensity
multiplication between two channels -- not centroid-in-mask logic, not
reusable as-is).

Feasible as a direct point-membership test -- no spatial index, no
channel-to-channel alignment step -- because the user confirmed this lab's
multi-channel acquisitions are always co-registered onto the same native
voxel grid. Both the smaller biomarker's cell coordinates and the larger
biomarker's mask are already in that shared native space (detection never
downsamples or transposes -- see the architecture plan's confirmed facts),
so no coordinate mapping is needed here at all, unlike regions.coord_transform.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..common.cell_schema import CELL_SCHEMA, read_cells
from ..io import FileReader
from ..regions.structures import rollup_tiers


def points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vectorized point-in-mask membership test.

    Args:
        points: (N,3) int array of (z,y,x) native-space coordinates.
        mask: The larger biomarker's ORIGINAL, unfiltered mask volume
            (boolean or any nonzero-means-foreground dtype), native space,
            same voxel grid as ``points`` (co-registered acquisition assumed).

    Returns:
        Boolean array, length N: True where the point falls inside the mask
        (in bounds and nonzero). Out-of-bounds points are False rather than
        raising -- the two channels are expected to share a grid, but a
        point at a detection-algorithm edge case shouldn't crash the run.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return np.empty(0, dtype=bool)

    shape = np.array(mask.shape)
    in_bounds = np.all((points >= 0) & (points < shape), axis=1)

    result = np.zeros(len(points), dtype=bool)
    if in_bounds.any():
        valid = points[in_bounds]
        result[in_bounds] = mask[valid[:, 0], valid[:, 1], valid[:, 2]] > 0

    return result


def colocalize_pair(
    smaller_cells_path: str | Path,
    smaller_biomarker: str,
    larger_mask_path: str | Path,
    larger_biomarker: str,
) -> dict:
    """Test the smaller biomarker's filtered cells for colocalization with
    the larger biomarker's original, unfiltered mask.

    Rewrites ``smaller_cells_path`` in place, adding (or updating) a
    ``colocalized_with_<larger_biomarker>`` boolean column -- False for
    rows that weren't tested (a different biomarker, or ``passed_filter``
    is False; colocalization is only meaningful for cells that survived
    volume filtering).

    Args:
        smaller_cells_path: Path to the cell Parquet table containing the
            smaller biomarker's detected cells.
        smaller_biomarker: Name of the smaller biomarker.
        larger_mask_path: Path to the larger biomarker's ORIGINAL mask (the
            same one detection ran on -- not a filtered cell list, the
            actual mask volume).
        larger_biomarker: Name of the larger biomarker (used for the output
            column name and the summary).

    Returns:
        {"tested": n, "colocalized": n} for this biomarker's passed-filter cells.
    """
    smaller_cells_path = Path(smaller_cells_path)
    mask = FileReader(larger_mask_path).read()

    table = pq.read_table(smaller_cells_path)
    df = table.to_pandas()

    column_name = f"colocalized_with_{larger_biomarker}"
    if column_name not in df.columns:
        df[column_name] = False

    selector = (df["biomarker"] == smaller_biomarker) & df["passed_filter"]
    n_tested = int(selector.sum())
    if n_tested:
        points = df.loc[selector, ["z", "y", "x"]].to_numpy()
        df.loc[selector, column_name] = points_in_mask(points, mask)

    n_colocalized = int(df.loc[selector, column_name].sum())

    extended_schema = CELL_SCHEMA.append(pa.field(column_name, pa.bool_()))
    new_table = pa.Table.from_pandas(df, schema=extended_schema, preserve_index=False)
    pq.write_table(new_table, smaller_cells_path)

    return {"tested": n_tested, "colocalized": n_colocalized}


def whole_brain_colocalization_summary(
    cells_path: str | Path,
    biomarker: str,
    larger_biomarker: str,
) -> dict:
    """Whole-brain tested/colocalized counts for a biomarker pair (works
    for both brain types -- no region_id needed)."""
    column_name = f"colocalized_with_{larger_biomarker}"
    df = read_cells(cells_path, biomarker=biomarker, passed_filter=True)
    if df.empty or column_name not in df.columns:
        return {"tested": 0, "colocalized": 0}
    return {"tested": len(df), "colocalized": int(df[column_name].sum())}


def compute_colocalization_tallies(
    cells_path: str | Path,
    biomarker: str,
    larger_biomarker: str,
) -> pd.DataFrame:
    """Per-region tested/colocalized cell counts for one biomarker pair
    (brain type A only -- requires region_id to already be resolved).

    Returns:
        DataFrame with columns [id, tested_cells, colocalized_cells].
    """
    column_name = f"colocalized_with_{larger_biomarker}"
    df = read_cells(cells_path, biomarker=biomarker, passed_filter=True)
    df = df.dropna(subset=["region_id"])
    if df.empty or column_name not in df.columns:
        return pd.DataFrame(columns=["id", "tested_cells", "colocalized_cells"])

    df["region_id"] = df["region_id"].astype(np.int64)
    tested = df.groupby("region_id").size().rename("tested_cells")
    colocalized = df[df[column_name]].groupby("region_id").size().rename("colocalized_cells")

    out = pd.concat([tested, colocalized], axis=1).fillna(0).astype(np.int64).reset_index()
    return out.rename(columns={"region_id": "id"})


def tiered_colocalization_report(
    cells_path: str | Path,
    structure_df: pd.DataFrame,
    biomarker: str,
    larger_biomarker: str,
) -> dict[int, pd.DataFrame]:
    """Full tiered Allen-CCF colocalization-rate report for one biomarker
    pair (brain type A only).

    Reuses ``regions.structures.rollup_tiers`` -- the same shared
    leaf-to-root aggregation used by ``report.region_stats`` and
    ``report.cell_report`` -- per the architecture plan's summary design.

    Returns:
        {tier: DataFrame}, deepest tier first, each row a structure with
        [tested_cells, colocalized_cells, colocalization_rate].
    """
    tallies = compute_colocalization_tallies(cells_path, biomarker, larger_biomarker)

    summary = structure_df.merge(tallies, on="id", how="left")
    count_cols = ["tested_cells", "colocalized_cells"]
    summary[count_cols] = summary[count_cols].fillna(0).astype(np.int64)
    summary = summary.set_index("id", drop=False)

    sheets = rollup_tiers(summary, structure_df, count_cols)
    for sheet in sheets.values():
        if sheet.empty:
            continue
        sheet["colocalization_rate"] = sheet["colocalized_cells"] / sheet["tested_cells"].replace(0, np.nan)

    return sheets


def run_colocalization_pairs(
    pairs: list[tuple[str, str]],
    cells_paths: dict[str, str | Path],
    mask_paths: dict[str, str | Path],
) -> dict[tuple[str, str], dict]:
    """Convenience wrapper: run ``colocalize_pair`` for several (larger, smaller)
    biomarker pairs (e.g. from ``RunConfig.colocalization_pairs``) in one call.

    Args:
        pairs: List of (larger, smaller) biomarker name tuples.
        cells_paths: {biomarker: cell Parquet path} -- the ``smaller`` side
            of each pair is looked up here.
        mask_paths: {biomarker: mask path} -- the ``larger`` side of each
            pair is looked up here (the ORIGINAL mask, not a cell table).

    Returns:
        {(larger, smaller): {"tested": n, "colocalized": n}} for each pair.
    """
    results = {}
    for larger, smaller in pairs:
        results[(larger, smaller)] = colocalize_pair(
            smaller_cells_path=cells_paths[smaller],
            smaller_biomarker=smaller,
            larger_mask_path=mask_paths[larger],
            larger_biomarker=larger,
        )
    return results
