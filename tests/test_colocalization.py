"""Phase 4 verification: colocalization -- smaller biomarker's filtered
cells tested against the larger biomarker's original, unfiltered mask."""
import numpy as np
import pandas as pd
import pytest

from signal_analyzer.colocalization.coloc import (
    colocalize_pair,
    compute_colocalization_tallies,
    points_in_mask,
    tiered_colocalization_report,
    whole_brain_colocalization_summary,
)
from signal_analyzer.common.cell_schema import CellTableWriter, HEMISPHERE_NA, read_cells
from signal_analyzer.common.config import BiomarkerConfig, BrainConfig, DetectionConfig, FilterConfig
from signal_analyzer.detection.dask_runner import detect_biomarker
from signal_analyzer.filtering.volume_filter import apply_volume_filter
from signal_analyzer.io import FileWriter


def _cell_row(cell_id, z, y, x, biomarker, passed_filter=True, region_id=None):
    return {
        "cell_id": cell_id, "z": z, "y": y, "x": x, "volume_voxels": 10, "volume_um3": 1.0,
        "biomarker": biomarker, "region_id": region_id, "hemisphere_id": HEMISPHERE_NA,
        "brain_id": "B1", "run_id": "r1", "passed_filter": passed_filter,
    }


def test_points_in_mask_basic():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    points = np.array([[2, 2, 2], [0, 0, 0], [4, 4, 4]])
    result = points_in_mask(points, mask)
    np.testing.assert_array_equal(result, [True, False, False])


def test_points_in_mask_out_of_bounds_returns_false_not_error():
    mask = np.ones((5, 5, 5), dtype=np.uint8)
    points = np.array([[10, 10, 10], [-1, 0, 0], [2, 2, 2]])
    result = points_in_mask(points, mask)
    np.testing.assert_array_equal(result, [False, False, True])


def test_points_in_mask_empty_input():
    mask = np.ones((5, 5, 5), dtype=np.uint8)
    result = points_in_mask(np.empty((0, 3), dtype=int), mask)
    assert result.shape == (0,)


def test_colocalize_pair_marks_cells_inside_larger_mask(tmp_path):
    # Larger biomarker's original mask: a block occupying x in [0,5).
    larger_mask = np.zeros((10, 10, 10), dtype=np.uint8)
    larger_mask[:, :, 0:5] = 1

    mask_writer = FileWriter(
        output_path=tmp_path, output_name="TH_mask", output_type="zarr",
        full_res_shape=larger_mask.shape, output_dtype=np.uint8, chunk_size=larger_mask.shape,
    )
    mask_writer.write(larger_mask)

    cells_path = tmp_path / "cfos_cells.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([
            _cell_row(1, z=1, y=1, x=1, biomarker="cFos"),               # inside TH mask -> colocalized
            _cell_row(2, z=1, y=1, x=8, biomarker="cFos"),               # outside TH mask
            _cell_row(3, z=1, y=1, x=1, biomarker="cFos", passed_filter=False),  # not passed -> not tested
            _cell_row(4, z=1, y=1, x=1, biomarker="TH"),                  # different biomarker -> not tested
        ])

    summary = colocalize_pair(cells_path, "cFos", mask_writer.output_path, "TH")
    assert summary == {"tested": 2, "colocalized": 1}

    df = read_cells(cells_path)
    col = "colocalized_with_TH"
    row1 = df[df["cell_id"] == 1].iloc[0]
    row2 = df[df["cell_id"] == 2].iloc[0]
    row3 = df[df["cell_id"] == 3].iloc[0]
    row4 = df[df["cell_id"] == 4].iloc[0]

    assert row1[col] == True  # noqa: E712
    assert row2[col] == False  # noqa: E712
    assert row3[col] == False  # not tested (didn't pass filter)
    assert row4[col] == False  # not tested (different biomarker)


def test_colocalize_pair_with_no_passed_cells_is_a_noop(tmp_path):
    larger_mask = np.ones((5, 5, 5), dtype=np.uint8)
    mask_writer = FileWriter(
        output_path=tmp_path, output_name="TH_mask2", output_type="zarr",
        full_res_shape=larger_mask.shape, output_dtype=np.uint8, chunk_size=larger_mask.shape,
    )
    mask_writer.write(larger_mask)

    cells_path = tmp_path / "cfos_cells2.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([_cell_row(1, z=1, y=1, x=1, biomarker="cFos", passed_filter=False)])

    summary = colocalize_pair(cells_path, "cFos", mask_writer.output_path, "TH")
    assert summary == {"tested": 0, "colocalized": 0}


def test_whole_brain_colocalization_summary(tmp_path):
    larger_mask = np.zeros((5, 5, 5), dtype=np.uint8)
    larger_mask[:, :, 0:3] = 1
    mask_writer = FileWriter(
        output_path=tmp_path, output_name="TH_mask3", output_type="zarr",
        full_res_shape=larger_mask.shape, output_dtype=np.uint8, chunk_size=larger_mask.shape,
    )
    mask_writer.write(larger_mask)

    cells_path = tmp_path / "cfos_cells3.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([
            _cell_row(1, z=1, y=1, x=1, biomarker="cFos"),  # inside
            _cell_row(2, z=1, y=1, x=4, biomarker="cFos"),  # outside
        ])
    colocalize_pair(cells_path, "cFos", mask_writer.output_path, "TH")

    summary = whole_brain_colocalization_summary(cells_path, "cFos", "TH")
    assert summary == {"tested": 2, "colocalized": 1}


def test_whole_brain_colocalization_summary_before_colocalize_pair_runs(tmp_path):
    cells_path = tmp_path / "cfos_cells4.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([_cell_row(1, z=1, y=1, x=1, biomarker="cFos")])

    # No colocalized_with_TH column exists yet -- must not raise.
    summary = whole_brain_colocalization_summary(cells_path, "cFos", "TH")
    assert summary == {"tested": 0, "colocalized": 0}


@pytest.fixture
def tiny_structures() -> pd.DataFrame:
    return pd.DataFrame([
        {"acronym": "ROOT", "id": 1, "name": "Root", "structure_id_path": "/1/", "parent_structure_id": np.nan},
        {"acronym": "A", "id": 10, "name": "Region A", "structure_id_path": "/1/10/", "parent_structure_id": 1.0},
        {"acronym": "B", "id": 20, "name": "Region B", "structure_id_path": "/1/20/", "parent_structure_id": 1.0},
    ])


def test_tiered_colocalization_report(tmp_path, tiny_structures):
    larger_mask = np.zeros((5, 5, 5), dtype=np.uint8)
    larger_mask[:, :, 0:3] = 1  # covers region A's area
    mask_writer = FileWriter(
        output_path=tmp_path, output_name="TH_mask5", output_type="zarr",
        full_res_shape=larger_mask.shape, output_dtype=np.uint8, chunk_size=larger_mask.shape,
    )
    mask_writer.write(larger_mask)

    cells_path = tmp_path / "cfos_cells5.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([
            _cell_row(1, z=1, y=1, x=1, biomarker="cFos", region_id=10),  # region A, inside mask
            _cell_row(2, z=1, y=1, x=4, biomarker="cFos", region_id=20),  # region B, outside mask
        ])
    colocalize_pair(cells_path, "cFos", mask_writer.output_path, "TH")

    tallies = compute_colocalization_tallies(cells_path, "cFos", "TH").set_index("id")
    assert tallies.loc[10, "tested_cells"] == 1
    assert tallies.loc[10, "colocalized_cells"] == 1
    assert tallies.loc[20, "tested_cells"] == 1
    assert tallies.loc[20, "colocalized_cells"] == 0

    report = tiered_colocalization_report(cells_path, tiny_structures, "cFos", "TH")
    tier2 = report[2].set_index("structure_id")
    assert tier2.loc[10, "colocalization_rate"] == 1.0
    assert tier2.loc[20, "colocalization_rate"] == 0.0

    tier1 = report[1].set_index("structure_id")
    assert tier1.loc[1, "tested_cells"] == 2
    assert tier1.loc[1, "colocalized_cells"] == 1


def test_full_detect_filter_colocalize_pipeline(tmp_path):
    """Runs real detection (not hand-constructed Parquet rows) for two
    biomarkers, then colocalizes -- confirms the whole chain composes
    correctly, not just colocalize_pair in isolation."""
    native_shape = (10, 20, 20)

    # TH (larger biomarker): occupies the left half of the volume (x < 10).
    th_mask = np.zeros(native_shape, dtype=np.uint8)
    th_mask[:, :, 0:10] = 1
    th_writer = FileWriter(
        output_path=tmp_path, output_name="th_mask", output_type="zarr",
        full_res_shape=native_shape, output_dtype=np.uint8, chunk_size=(5, 20, 20),
    )
    th_writer.write(th_mask)

    # cFos (smaller biomarker): three blobs.
    #   - blob_in (64 voxels, centroid x~5): inside TH's region -> colocalized
    #   - blob_out (64 voxels, centroid x~15): outside TH's region -> not colocalized
    #   - blob_tiny (8 voxels): below the volume filter, never tested at all
    cfos_mask = np.zeros(native_shape, dtype=np.uint8)
    cfos_mask[2:6, 2:6, 3:7] = 1
    cfos_mask[2:6, 2:6, 13:17] = 1
    cfos_mask[0:2, 0:2, 0:2] = 1
    cfos_writer = FileWriter(
        output_path=tmp_path, output_name="cfos_mask", output_type="zarr",
        full_res_shape=native_shape, output_dtype=np.uint8, chunk_size=(5, 20, 20),
    )
    cfos_writer.write(cfos_mask)

    brain = BrainConfig(id="ColocBrain", voxel_size_um=(4.0, 1.82, 1.82), needs_registration=False)
    cfos_biomarker = BiomarkerConfig(
        name="cFos", mask_path=str(cfos_writer.output_path),
        detection=DetectionConfig(connectivity=18, z_chunk=5),
        filter=FilterConfig(min_volume_voxels=30, max_volume_voxels=100),
    )

    cells_path = tmp_path / "cfos_cells_full.parquet"
    n_detected = detect_biomarker(cfos_writer.output_path, brain, cfos_biomarker, cells_path, run_id="coloc-1")
    assert n_detected == 3

    filter_summary = apply_volume_filter(cells_path, cfos_biomarker)
    assert filter_summary == {"total": 3, "passed": 2}  # blob_tiny excluded

    coloc_summary = colocalize_pair(cells_path, "cFos", th_writer.output_path, "TH")
    assert coloc_summary == {"tested": 2, "colocalized": 1}  # only blob_in overlaps TH

    df = read_cells(cells_path, passed_filter=True)
    colocalized_rows = df[df["colocalized_with_TH"]]
    assert len(colocalized_rows) == 1
    assert colocalized_rows.iloc[0]["x"] < 10  # the colocalized cell is indeed in TH's region
