"""Cell-count reporting for both brain types.

Brain type B (no registration, see ``whole_brain_report``): a trivial
whole-brain total filtered-cell count per biomarker -- the only report
Phase 1 could implement, since these brains have no ``annotation.tif`` at
all and therefore no region/hemisphere breakdown is possible.

Brain type A (registered, see ``tiered_cell_report``): the full Allen-CCF
tiered rollup + asymmetry, merging MARS's ``filter_annotation.py::
xlsx_points`` (tiered rollup + density) and ``asym.py::
process_analysis_logic`` (``Diff_Percent``/``Dominant_Side``/
``Signif_Diff``) -- computed inline right after the L/R split, in the same
report pass, rather than asym.py's separate post-process script reading a
previously-written xlsx.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..common.cell_schema import HEMISPHERE_LEFT, HEMISPHERE_RIGHT, read_cells
from ..regions.structures import rollup_tiers


def whole_brain_report(cells_path: str | Path) -> pd.DataFrame:
    """Whole-brain total filtered-cell count per biomarker (brain type B).

    Returns:
        DataFrame with columns [biomarker, total_cells], one row per
        biomarker present in the cell table with at least one filtered cell.
        Empty (but correctly-columned) if the table has no passed cells.
    """
    df = read_cells(cells_path, passed_filter=True, columns=["biomarker"])
    if df.empty:
        return pd.DataFrame(columns=["biomarker", "total_cells"])
    report = (
        df.groupby("biomarker", observed=True)
        .size()
        .reset_index(name="total_cells")
        .sort_values("biomarker")
        .reset_index(drop=True)
    )
    return report


def write_whole_brain_report(cells_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Compute the whole-brain report and write it to a CSV file.

    Returns:
        The same DataFrame that was written, for callers that want to log
        results (e.g. via RunLogger.record_result) without re-reading the file.
    """
    report = whole_brain_report(cells_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def compute_cell_tallies(cells_path: str | Path, biomarker: str) -> pd.DataFrame:
    """Per-region total/left/right cell counts for one biomarker's filtered cells.

    Ported from MARS's filter_annotation.py::calculate_cells's point
    Counters, replaced with a Parquet groupby -- much cheaper than MARS's
    per-slice Counter accumulation or Chulab's full-volume voxel scan, since
    region_id/hemisphere_id are already resolved per cell (Phase 2's
    regions.coord_transform.assign_regions_in_cell_table).

    Returns:
        DataFrame with columns [id, total_cells, left_cells, right_cells].
        Empty (correctly columned) if there are no passed cells with a
        resolved region_id for this biomarker.
    """
    df = read_cells(cells_path, biomarker=biomarker, passed_filter=True, columns=["region_id", "hemisphere_id"])
    df = df.dropna(subset=["region_id"])
    if df.empty:
        return pd.DataFrame(columns=["id", "total_cells", "left_cells", "right_cells"])

    df["region_id"] = df["region_id"].astype(np.int64)
    total = df.groupby("region_id").size().rename("total_cells")
    left = df[df["hemisphere_id"] == HEMISPHERE_LEFT].groupby("region_id").size().rename("left_cells")
    right = df[df["hemisphere_id"] == HEMISPHERE_RIGHT].groupby("region_id").size().rename("right_cells")

    out = pd.concat([total, left, right], axis=1).fillna(0).astype(np.int64).reset_index()
    return out.rename(columns={"region_id": "id"})


def tiered_cell_report(
    cells_path: str | Path,
    structure_df: pd.DataFrame,
    region_geometry: dict[int, pd.DataFrame],
    biomarker: str,
) -> dict[int, pd.DataFrame]:
    """Full tiered Allen-CCF cell-count + density + asymmetry report for one
    biomarker (brain type A only -- requires registration).

    Args:
        cells_path: Path to the cell Parquet table (region_id/hemisphere_id
            already resolved, e.g. via
            regions.coord_transform.assign_regions_in_cell_table).
        structure_df: The Allen CCF structures table (regions.structures.load_structures).
        region_geometry: The once-per-brain voxel-geometry report
            (report.region_stats.build_region_geometry_report), for density.
        biomarker: Which biomarker's cells to report on.

    Returns:
        {tier: DataFrame}, deepest tier first, each row a structure with
        cell counts, densities (cells/mm3), and asymmetry columns
        (``Diff_Percent``, ``Abs_Diff_Percent``, ``Dominant_Side``,
        ``Signif_Diff_gt30pct``) -- the ``(L-R)/R*100`` formula and >=30%
        threshold are ported verbatim from asym.py::process_analysis_logic.
    """
    tallies = compute_cell_tallies(cells_path, biomarker)

    summary = structure_df.merge(tallies, on="id", how="left")
    count_cols = ["total_cells", "left_cells", "right_cells"]
    summary[count_cols] = summary[count_cols].fillna(0).astype(np.int64)
    summary = summary.set_index("id", drop=False)

    cell_sheets = rollup_tiers(summary, structure_df, count_cols)

    reports: dict[int, pd.DataFrame] = {}
    for tier, cell_sheet in cell_sheets.items():
        geometry_sheet = region_geometry.get(tier)
        if geometry_sheet is None or cell_sheet.empty:
            reports[tier] = cell_sheet
            continue

        merged = cell_sheet.merge(
            geometry_sheet[["structure_id", "total_volume_mm3", "left_volume_mm3", "right_volume_mm3"]],
            on="structure_id", how="left",
        )
        merged["total_cells_per_mm3"] = merged["total_cells"] / merged["total_volume_mm3"]
        merged["left_cells_per_mm3"] = merged["left_cells"] / merged["left_volume_mm3"]
        merged["right_cells_per_mm3"] = merged["right_cells"] / merged["right_volume_mm3"]

        merged = _add_asymmetry_columns(merged)
        reports[tier] = merged

    return reports


def _add_asymmetry_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ported verbatim from asym.py::process_analysis_logic's per-row formulas."""
    left = df["left_cells_per_mm3"]
    right = df["right_cells_per_mm3"]

    df["Diff_Percent"] = np.where(
        right != 0,
        ((left - right) / np.where(right == 0, 1, right)) * 100,
        np.where(left > 0, 100.0, 0.0),
    ).round(2)
    df["Abs_Diff_Percent"] = df["Diff_Percent"].abs()
    df["Dominant_Side"] = np.select([left > right, left < right], ["Left", "Right"], default="None")
    df["Signif_Diff_gt30pct"] = df["Abs_Diff_Percent"] >= 30.0
    return df


def tag_functional_system(
    df: pd.DataFrame,
    keywords: list[str],
    acronyms: list[str],
    exact_acronym_match: bool = False,
) -> pd.Series:
    """Ported from asym.py::process_analysis_logic's ``Is_Target_System`` logic.

    Custom mode (``exact_acronym_match=True``) matches ``acronym`` exactly
    against the given list; preset mode matches ``structure_name`` by
    case-insensitive keyword substring OR ``acronym`` by regex prefix.
    """
    if exact_acronym_match:
        return df["acronym"].astype(str).isin(acronyms)

    m_name = (
        df["structure_name"].astype(str).str.contains("|".join(keywords), case=False, na=False)
        if keywords else pd.Series(False, index=df.index)
    )
    m_acro = (
        df["acronym"].astype(str).str.contains("^(?:" + "|".join(acronyms) + ")", regex=True, na=False)
        if acronyms else pd.Series(False, index=df.index)
    )
    return m_name | m_acro


def build_target_summary(
    tiered_reports: dict[int, pd.DataFrame],
    keywords: list[str],
    acronyms: list[str],
    exact_acronym_match: bool = False,
    cell_threshold: int = 0,
) -> pd.DataFrame:
    """Concatenate all tiers, tag a functional-system subset, and return it
    sorted by |asymmetry| descending -- ported from asym.py::main's
    Target_Summary sheet construction.

    Args:
        tiered_reports: Output of ``tiered_cell_report``.
        keywords/acronyms/exact_acronym_match: See ``tag_functional_system``.
        cell_threshold: Drop rows with ``total_cells`` below this (0 = no filter).

    Returns:
        The target-system subset across all tiers, each row tagged with
        which tier it came from (``Source_Tier``), sorted by
        ``Abs_Diff_Percent`` descending.
    """
    parts = []
    for tier, df in tiered_reports.items():
        if df.empty or "total_cells" not in df.columns:
            continue
        part = df.copy()
        if cell_threshold > 0:
            part = part[part["total_cells"] >= cell_threshold]
        if part.empty:
            continue
        part.insert(0, "Source_Tier", tier)
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["Is_Target_System"] = tag_functional_system(combined, keywords, acronyms, exact_acronym_match)
    target = combined[combined["Is_Target_System"]].sort_values("Abs_Diff_Percent", ascending=False)
    return target.reset_index(drop=True)


def write_tiered_region_report(
    tiered_reports: dict[int, pd.DataFrame],
    output_path: str | Path,
    target_summary: Optional[pd.DataFrame] = None,
) -> None:
    """Write a tiered report to an xlsx workbook: one sheet per tier
    (``Tier {n}``), plus an optional ``Target_Summary`` sheet.

    Mirrors MARS's filter_annotation.py::xlsx_points / asym.py's sheet
    layout, generalized into one writer used by both.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        if target_summary is not None and not target_summary.empty:
            target_summary.to_excel(writer, sheet_name="Target_Summary", index=False)
        for tier in sorted(tiered_reports.keys()):
            sheet = tiered_reports[tier]
            sheet.to_excel(writer, sheet_name=f"Tier {tier}", index=False)
