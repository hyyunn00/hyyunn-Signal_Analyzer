"""Region-based and whole-brain cell-count reporting.

Phase 1 only implements the type-B (no registration) branch: a trivial
whole-brain total filtered-cell count per biomarker, since brain type B has
no annotation.tif at all (confirmed with the user) and therefore no
region/hemisphere breakdown is possible or needed. The type-A branch (full
Allen-CCF tiered rollup + asymmetry, merging MARS's asym.py/xlsx_points and
Chulab's create_cell_report) is built in Phase 3, once
regions/coord_transform.py resolves region_id/hemisphere_id for registered
brains.
"""
from .cell_report import (
    build_target_summary,
    compute_cell_tallies,
    tag_functional_system,
    tiered_cell_report,
    whole_brain_report,
    write_tiered_region_report,
    write_whole_brain_report,
)
from .region_stats import build_region_geometry_report, compute_region_voxel_counts

__all__ = [
    "whole_brain_report",
    "write_whole_brain_report",
    "compute_cell_tallies",
    "tiered_cell_report",
    "tag_functional_system",
    "build_target_summary",
    "write_tiered_region_report",
    "build_region_geometry_report",
    "compute_region_voxel_counts",
]
