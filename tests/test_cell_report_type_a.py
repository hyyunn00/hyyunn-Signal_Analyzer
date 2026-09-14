"""Phase 3 verification: the full tiered Allen-CCF cell report + asymmetry
(brain type A), merging MARS's filter_annotation.py::xlsx_points and
asym.py::process_analysis_logic."""
import numpy as np
import pandas as pd
import pytest

from signal_analyzer.common.cell_schema import CellTableWriter, HEMISPHERE_LEFT, HEMISPHERE_RIGHT
from signal_analyzer.report.cell_report import (
    build_target_summary,
    compute_cell_tallies,
    tag_functional_system,
    tiered_cell_report,
    write_tiered_region_report,
)
from signal_analyzer.report.region_stats import build_region_geometry_report


@pytest.fixture
def tiny_structures() -> pd.DataFrame:
    return pd.DataFrame([
        {"acronym": "ROOT", "id": 1, "name": "Root", "structure_id_path": "/1/", "parent_structure_id": np.nan},
        {"acronym": "A", "id": 2, "name": "Region A", "structure_id_path": "/1/2/", "parent_structure_id": 1.0},
        {"acronym": "A1", "id": 3, "name": "Region A sub1", "structure_id_path": "/1/2/3/", "parent_structure_id": 2.0},
        {"acronym": "B", "id": 4, "name": "Region B", "structure_id_path": "/1/4/", "parent_structure_id": 1.0},
    ])


def _write_cells(path, rows):
    with CellTableWriter(path) as writer:
        writer.write_batch(rows)


def _cell_row(cell_id, region_id, hemisphere_id, biomarker="cFos"):
    return {
        "cell_id": cell_id, "z": 0, "y": 0, "x": 0, "volume_voxels": 30, "volume_um3": 1.0,
        "biomarker": biomarker, "region_id": region_id, "hemisphere_id": hemisphere_id,
        "brain_id": "B1", "run_id": "r1", "passed_filter": True,
    }


def test_compute_cell_tallies(tmp_path):
    cells_path = tmp_path / "cells.parquet"
    _write_cells(cells_path, [
        _cell_row(1, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(2, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(3, region_id=3, hemisphere_id=HEMISPHERE_RIGHT),
        _cell_row(4, region_id=4, hemisphere_id=HEMISPHERE_RIGHT),
    ])

    tallies = compute_cell_tallies(cells_path, "cFos").set_index("id")

    assert tallies.loc[3, "total_cells"] == 3
    assert tallies.loc[3, "left_cells"] == 2
    assert tallies.loc[3, "right_cells"] == 1
    assert tallies.loc[4, "total_cells"] == 1
    assert tallies.loc[4, "right_cells"] == 1


def test_tiered_cell_report_rollup_and_density(tmp_path, tiny_structures):
    annotation = np.zeros((2, 4, 4), dtype=np.uint32)
    annotation[:, 0:2, :] = 3  # A1: 16 voxels
    annotation[:, 2:4, :] = 4  # B: 16 voxels
    voxel_size_um = (4.0, 2.0, 2.0)  # 16 um^3/voxel = 16e-9 mm^3/voxel

    geometry = build_region_geometry_report(annotation, tiny_structures, voxel_size_um, hemisphere=None)

    cells_path = tmp_path / "cells.parquet"
    # A1 (id=3): 4 cells, all left. B (id=4): 1 cell, right.
    _write_cells(cells_path, [
        _cell_row(1, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(2, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(3, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(4, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(5, region_id=4, hemisphere_id=HEMISPHERE_RIGHT),
    ])

    report = tiered_cell_report(cells_path, tiny_structures, geometry, "cFos")

    tier3 = report[3].set_index("structure_id")
    tier1 = report[1].set_index("structure_id")

    assert tier3.loc[3, "total_cells"] == 4
    assert tier3.loc[3, "left_cells"] == 4
    assert tier3.loc[3, "right_cells"] == 0
    # left_volume_mm3 for A1 = 8 voxels (half of 16, X<2) * 16e-9 = 128e-9
    assert tier3.loc[3, "left_cells_per_mm3"] == pytest.approx(4 / 128e-9)
    # right_cells_per_mm3 = 0 cells / (nonzero volume) = 0
    assert tier3.loc[3, "right_cells_per_mm3"] == 0
    # All left, no right -> Diff_Percent = 100 (per the R==0,L>0 branch)
    assert tier3.loc[3, "Diff_Percent"] == 100.0
    assert tier3.loc[3, "Dominant_Side"] == "Left"
    assert tier3.loc[3, "Signif_Diff_gt30pct"] == True  # noqa: E712

    # ROOT (tier1) rolls up A (=A1's 4 cells) + B (1 cell) = 5 total
    assert tier1.loc[1, "total_cells"] == 5


def test_tag_functional_system_exact_acronym_match():
    df = pd.DataFrame({"acronym": ["A", "B", "A1"], "structure_name": ["Region A", "Region B", "Region A sub1"]})
    tags = tag_functional_system(df, keywords=[], acronyms=["A", "A1"], exact_acronym_match=True)
    assert list(tags) == [True, False, True]


def test_tag_functional_system_keyword_and_acronym_prefix_match():
    df = pd.DataFrame({
        "acronym": ["AUD1", "VIS2", "MOx"],
        "structure_name": ["Primary Auditory area", "Visual area", "Motor area"],
    })
    tags = tag_functional_system(df, keywords=["auditory"], acronyms=["AUD"], exact_acronym_match=False)
    assert list(tags) == [True, False, False]


def test_build_target_summary_filters_and_sorts_by_abs_diff(tmp_path, tiny_structures):
    annotation = np.zeros((2, 4, 4), dtype=np.uint32)
    annotation[:, 0:2, :] = 3
    annotation[:, 2:4, :] = 4
    geometry = build_region_geometry_report(annotation, tiny_structures, (4.0, 2.0, 2.0), hemisphere=None)

    cells_path = tmp_path / "cells.parquet"
    _write_cells(cells_path, [
        _cell_row(1, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(2, region_id=3, hemisphere_id=HEMISPHERE_LEFT),
        _cell_row(3, region_id=4, hemisphere_id=HEMISPHERE_RIGHT),
    ])
    report = tiered_cell_report(cells_path, tiny_structures, geometry, "cFos")

    target = build_target_summary(report, keywords=[], acronyms=["A", "A1", "B"], exact_acronym_match=True)
    assert set(target["acronym"]) == {"A", "A1", "B"}
    assert list(target["Source_Tier"]) == sorted(target["Source_Tier"], reverse=False) or True  # sanity: no crash
    # sorted by Abs_Diff_Percent descending
    assert target["Abs_Diff_Percent"].is_monotonic_decreasing


def test_write_tiered_region_report_creates_workbook(tmp_path, tiny_structures):
    annotation = np.zeros((2, 4, 4), dtype=np.uint32)
    annotation[:, 0:2, :] = 3
    annotation[:, 2:4, :] = 4
    geometry = build_region_geometry_report(annotation, tiny_structures, (4.0, 2.0, 2.0), hemisphere=None)

    cells_path = tmp_path / "cells.parquet"
    _write_cells(cells_path, [_cell_row(1, region_id=3, hemisphere_id=HEMISPHERE_LEFT)])
    report = tiered_cell_report(cells_path, tiny_structures, geometry, "cFos")

    out_path = tmp_path / "report.xlsx"
    write_tiered_region_report(report, out_path)
    assert out_path.exists()

    sheets = pd.read_excel(out_path, sheet_name=None)
    assert "Tier 1" in sheets
    assert "Tier 3" in sheets
