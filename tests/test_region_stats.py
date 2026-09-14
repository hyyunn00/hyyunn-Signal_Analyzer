"""Phase 3 verification: once-per-brain region-voxel-geometry stats."""
import numpy as np
import pandas as pd
import pytest

from signal_analyzer.common.cell_schema import HEMISPHERE_LEFT, HEMISPHERE_RIGHT
from signal_analyzer.report.region_stats import build_region_geometry_report, compute_region_voxel_counts


@pytest.fixture
def tiny_structures() -> pd.DataFrame:
    return pd.DataFrame([
        {"acronym": "ROOT", "id": 1, "name": "Root", "structure_id_path": "/1/", "parent_structure_id": np.nan},
        {"acronym": "A", "id": 2, "name": "Region A", "structure_id_path": "/1/2/", "parent_structure_id": 1.0},
        {"acronym": "A1", "id": 3, "name": "Region A sub1", "structure_id_path": "/1/2/3/", "parent_structure_id": 2.0},
        {"acronym": "B", "id": 4, "name": "Region B", "structure_id_path": "/1/4/", "parent_structure_id": 1.0},
    ])


def test_compute_region_voxel_counts_with_hemisphere_volume():
    annotation = np.zeros((2, 4, 4), dtype=np.uint32)
    annotation[:, 0:2, :] = 3  # region 3 (A1): 2*2*4 = 16 voxels
    annotation[:, 2:4, :] = 4  # region 4 (B): 16 voxels

    hemisphere = np.zeros((2, 4, 4), dtype=np.uint8)
    hemisphere[:, :, 0:2] = HEMISPHERE_LEFT
    hemisphere[:, :, 2:4] = HEMISPHERE_RIGHT

    counts = compute_region_voxel_counts(annotation, hemisphere).set_index("id")

    assert counts.loc[3, "total_voxels"] == 16
    assert counts.loc[3, "left_voxels"] == 8  # half of region 3 is on the left (X<2)
    assert counts.loc[3, "right_voxels"] == 8
    assert counts.loc[4, "total_voxels"] == 16


def test_compute_region_voxel_counts_falls_back_to_x_midline_without_hemisphere():
    annotation = np.zeros((1, 2, 4), dtype=np.uint32)
    annotation[:, :, :] = 5  # single region spanning the full X extent

    counts = compute_region_voxel_counts(annotation, hemisphere=None).set_index("id")

    # X midline = 4 // 2 = 2 -> left is x in [0,2), right is x in [2,4)
    assert counts.loc[5, "total_voxels"] == 8
    assert counts.loc[5, "left_voxels"] == 4
    assert counts.loc[5, "right_voxels"] == 4


def test_build_region_geometry_report_rolls_up_and_converts_to_mm3(tiny_structures):
    annotation = np.zeros((2, 4, 4), dtype=np.uint32)
    annotation[:, 0:2, :] = 3  # A1 (id=3): 16 voxels
    annotation[:, 2:4, :] = 4  # B (id=4): 16 voxels

    voxel_size_um = (4.0, 2.0, 2.0)  # voxel volume = 16 um^3

    report = build_region_geometry_report(annotation, tiny_structures, voxel_size_um, hemisphere=None)

    tier3 = report[3].set_index("structure_id")
    tier2 = report[2].set_index("structure_id")
    tier1 = report[1].set_index("structure_id")

    assert tier3.loc[3, "total_voxels"] == 16
    # A (id=2) picks up A1's 16 voxels (A had 0 of its own)
    assert tier2.loc[2, "total_voxels"] == 16
    assert tier2.loc[4, "total_voxels"] == 16
    # ROOT (id=1) picks up A's (16) + B's (16) = 32
    assert tier1.loc[1, "total_voxels"] == 32

    # 16 voxels * 16 um^3/voxel = 256 um^3 = 256e-9 mm^3
    assert tier3.loc[3, "total_volume_mm3"] == pytest.approx(256e-9)
