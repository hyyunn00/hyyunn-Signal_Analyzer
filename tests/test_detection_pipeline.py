"""End-to-end Phase 1 verification: the full "brain type B" pipeline
(detect -> filter -> whole-brain report -> native redraw) on a synthetic
mask with known blobs, run entirely through the public module APIs the
same way a real (unregistered) brain would be processed.
"""
import numpy as np

from signal_analyzer.common.cell_schema import read_cells
from signal_analyzer.common.config import BiomarkerConfig, BrainConfig, DetectionConfig, FilterConfig
from signal_analyzer.detection.dask_runner import detect_biomarker
from signal_analyzer.filtering.volume_filter import apply_volume_filter
from signal_analyzer.io import FileReader, FileWriter
from signal_analyzer.redraw.mask_from_cells import redraw_native_mask
from signal_analyzer.report.cell_report import whole_brain_report, write_whole_brain_report

NATIVE_SHAPE = (12, 20, 20)


def _write_synthetic_mask(tmp_path):
    """Three solid-cube blobs of distinct sizes:
      - blob_small: 3x3x3 = 27 voxels, z[0:3] y[0:3] x[0:3] -- below the
        volume filter's min bound, should be excluded.
      - blob_medium: 4x4x4 = 64 voxels, z[5:9] y[5:9] x[5:9] -- within
        bounds, should be the only one that passes.
      - blob_boundary: 3x3x3 = 27 voxels, z[9:12] y[10:13] x[10:13] --
        touches the volume's final Z-slice (z=11), exercising the
        final-flush fix at the full pipeline level; same size as
        blob_small, so also below the min bound (excluded).
    """
    volume = np.zeros(NATIVE_SHAPE, dtype=np.uint8)
    volume[0:3, 0:3, 0:3] = 1
    volume[5:9, 5:9, 5:9] = 1
    volume[9:12, 10:13, 10:13] = 1

    writer = FileWriter(
        output_path=tmp_path,
        output_name="synthetic_mask",
        output_type="zarr",
        full_res_shape=NATIVE_SHAPE,
        output_dtype=np.uint8,
        chunk_size=(5, 20, 20),
    )
    writer.write(volume)
    return writer.output_path


def _make_configs(mask_path):
    brain = BrainConfig(id="TestBrain-B", voxel_size_um=(4.0, 1.82, 1.82), needs_registration=False)
    biomarker = BiomarkerConfig(
        name="cFos",
        mask_path=str(mask_path),
        detection=DetectionConfig(connectivity=18, z_chunk=5),
        filter=FilterConfig(min_volume_voxels=30, max_volume_voxels=100),
    )
    return brain, biomarker


def test_full_brain_type_b_pipeline(tmp_path):
    mask_path = _write_synthetic_mask(tmp_path)
    brain, biomarker = _make_configs(mask_path)

    cells_path = tmp_path / "cells.parquet"

    # --- detect ---
    n_detected = detect_biomarker(mask_path, brain, biomarker, cells_path, run_id="run-1")
    assert n_detected == 3  # all three blobs found, including the one touching the final slice

    raw = read_cells(cells_path)
    assert len(raw) == 3
    assert set(raw["volume_voxels"]) == {27, 27, 64}
    # region/hemisphere are correctly left null/NA for a non-registered brain
    assert raw["region_id"].isna().all()
    assert (raw["hemisphere_id"] == 0).all()
    # nothing has been evaluated by the filter yet -- fails safe
    assert (raw["passed_filter"] == False).all()  # noqa: E712

    # --- filter ---
    summary = apply_volume_filter(cells_path, biomarker)
    assert summary == {"total": 3, "passed": 1}

    filtered = read_cells(cells_path, passed_filter=True)
    assert len(filtered) == 1
    assert filtered.iloc[0]["volume_voxels"] == 64

    # --- report ---
    report = whole_brain_report(cells_path)
    assert report.to_dict("records") == [{"biomarker": "cFos", "total_cells": 1}]

    report_path = tmp_path / "report.csv"
    written = write_whole_brain_report(cells_path, report_path)
    assert report_path.exists()
    assert written.equals(report)

    # --- redraw (native space, default scroll-tiff so it's directly
    # openable in Fiji next to the original raw image) ---
    redraw_path = redraw_native_mask(
        cells_path,
        native_shape=NATIVE_SHAPE,
        output_path=tmp_path,
        output_name="redrawn_mask",
        chunk_size=(5, 20, 20),
    )
    assert redraw_path.is_dir()
    assert len(list(redraw_path.glob("*.tiff"))) == NATIVE_SHAPE[0]
    redrawn = FileReader(redraw_path).read()
    assert redrawn.shape == NATIVE_SHAPE

    nonzero = np.argwhere(redrawn > 0)
    assert len(nonzero) == 1  # exactly one cell passed the filter
    z, y, x = nonzero[0]
    # blob_medium spans indices [5,6,7,8] on every axis; integer-division
    # centroid of a symmetric 4-wide span floors to 6 on every axis.
    assert (z, y, x) == (6, 6, 6)


def test_filter_thresholds_are_independent_of_detection(tmp_path):
    """Re-filtering with different bounds must not require re-detection --
    this is the whole point of keeping passed_filter as a column (see the
    architecture plan's cell schema rationale)."""
    mask_path = _write_synthetic_mask(tmp_path)
    brain, biomarker = _make_configs(mask_path)
    cells_path = tmp_path / "cells.parquet"

    detect_biomarker(mask_path, brain, biomarker, cells_path, run_id="run-1")

    # Loosen the filter to include everything.
    loose = BiomarkerConfig(
        name="cFos", mask_path=str(mask_path),
        detection=biomarker.detection,
        filter=FilterConfig(min_volume_voxels=1, max_volume_voxels=1000),
    )
    summary = apply_volume_filter(cells_path, loose)
    assert summary == {"total": 3, "passed": 3}
    assert whole_brain_report(cells_path).iloc[0]["total_cells"] == 3

    # Tighten it to exclude everything, still without re-detecting.
    strict = BiomarkerConfig(
        name="cFos", mask_path=str(mask_path),
        detection=biomarker.detection,
        filter=FilterConfig(min_volume_voxels=1000, max_volume_voxels=2000),
    )
    summary = apply_volume_filter(cells_path, strict)
    assert summary == {"total": 3, "passed": 0}
    assert whole_brain_report(cells_path).empty
