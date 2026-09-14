"""Allen CCF structures-table lookup: region descendant queries and the
tier-depth grouping both MARS and Chulab independently reimplement for
their tiered region reports.

Ported from MARS's aba2roi.py::get_region_info
(D:\\chu_lab\\MARS\\aba2roi.py) -- the structure_id_path substring-containment
descendant lookup is correct and portable as-is: every descendant's path is
prefixed by its ancestor's path string, so a substring test is a cheap,
correct subtree query. Both source repos bundle an identical copy of this
table (data/structures.csv here is the single canonical copy, deduped from
MARS/structures.csv and Chulab-Signal_Analyzer/utils/structures.csv, which
were byte-identical aside from line endings).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_structures(structures_csv: str | Path) -> pd.DataFrame:
    """Load the Allen CCF structures table, coercing ``id`` to a clean int.

    Ported from MARS's aba2roi.py: force ``id`` through numeric coercion
    before casting to int, avoiding scientific-notation corruption of large
    Allen ids (e.g. HATA = 589508447) that a naive ``pd.read_csv`` dtype
    inference can produce.
    """
    structure_df = pd.read_csv(structures_csv)
    structure_df["id"] = pd.to_numeric(structure_df["id"], errors="coerce").fillna(0).astype(int)
    return structure_df


def get_region_info(structure_df: pd.DataFrame, target_acronym: str) -> tuple[int, list[int], str]:
    """Find the official id and all descendant sub-region ids for one acronym.

    Ported from MARS's aba2roi.py::get_region_info, unchanged: every
    descendant's structure_id_path is prefixed by its ancestor's path
    string, so a substring containment test is a correct, cheap subtree
    query -- includes the target region's own id.

    Returns:
        (original_id, all_descendant_ids_including_self, region_name)

    Raises:
        ValueError: if the acronym isn't found in the table.
    """
    target_row = structure_df[structure_df["acronym"] == target_acronym]
    if target_row.empty:
        raise ValueError(f"Acronym '{target_acronym}' not found in the structures table.")

    original_id = int(float(target_row.iloc[0]["id"]))
    base_path = target_row.iloc[0]["structure_id_path"]
    region_name = target_row.iloc[0]["name"]

    children_df = structure_df[structure_df["structure_id_path"].str.contains(base_path, regex=False)]
    all_ids = children_df["id"].unique().astype(int).tolist()

    return original_id, all_ids, region_name


def group_ids_by_tier(structure_df: pd.DataFrame) -> dict[int, list[int]]:
    """Group every structure id by its depth (tier) in structure_id_path.

    Ported from the tier-parsing logic MARS's filter_annotation.py::
    xlsx_points and Chulab's analyzer_report_tools.py independently
    reimplement: ``structure_id_path`` is a ``/``-delimited ancestor-id
    chain (e.g. ``/997/8/343/.../653/``); its number of path segments is
    that structure's depth. Returned tiers are sorted deepest-first, since
    tiered-report rollups (``rollup_tiers``, below) must aggregate children
    into parents leaf-to-root.
    """
    tiers: dict[int, list[int]] = defaultdict(list)
    for path_id in structure_df["structure_id_path"]:
        parts = str(path_id).strip("/").split("/")
        tiers[len(parts)].append(int(parts[-1]))
    return dict(sorted(tiers.items(), reverse=True))


def rollup_tiers(
    summary: pd.DataFrame,
    structure_df: pd.DataFrame,
    value_columns: list[str],
) -> dict[int, pd.DataFrame]:
    """Aggregate child structures' values into their parents, leaf-to-root.

    Ported from MARS's filter_annotation.py::xlsx_points's tier loop and
    Chulab's analyzer_report_tools.py's independent reimplementation of the
    same pattern (bucket by ``id``, then walk ``structure_id_path`` depth
    bottom-up summing child stats into ``parent_structure_id``) -- kept as
    exactly one implementation here, shared by ``report.region_stats``
    (per-brain voxel geometry) and ``report.cell_report`` (per-biomarker
    cell counts).

    Args:
        summary: DataFrame indexed by structure ``id``, with
            ``parent_structure_id``, ``name``, ``acronym``, and every column
            in ``value_columns`` already populated for LEAF-level rows
            (zero elsewhere is fine) -- e.g. the result of left-merging raw
            per-region counts onto the full structures table and filling NaN
            with 0. Mutated in place: by the time a shallower tier is
            processed, its already-updated deeper descendants are summed in,
            producing a proper cumulative (self + all descendants) rollup.
        structure_df: The full Allen CCF structures table.
        value_columns: Numeric columns to sum up the hierarchy.

    Returns:
        {tier: DataFrame}, deepest tier first -- one sheet-ready DataFrame
        per tier, each row a structure at that depth with its fully
        aggregated value_columns.
    """
    tiers = group_ids_by_tier(structure_df)
    sheets: dict[int, pd.DataFrame] = {}

    for tier, ids in tiers.items():
        rows = []
        for structure_id in ids:
            if structure_id not in summary.index:
                continue
            children = summary.loc[summary["parent_structure_id"] == structure_id]
            for col in value_columns:
                summary.at[structure_id, col] += children[col].sum()

            row = {
                "structure_id": structure_id,
                "structure_name": summary.at[structure_id, "name"],
                "acronym": summary.at[structure_id, "acronym"],
            }
            row.update({col: summary.at[structure_id, col] for col in value_columns})
            rows.append(row)

        sheets[tier] = pd.DataFrame(rows)

    return sheets
