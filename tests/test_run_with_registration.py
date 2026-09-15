"""Phase 5 verification: the run_with_registration orchestrator wires
detect -> filter -> assign_regions -> report -> roi_extract -> redraw
together correctly for a registered brain."""
import numpy as np
import pytest

from signal_analyzer.common.config import (
    BiomarkerConfig, BrainConfig, DetectionConfig, FilterConfig,
    PathsConfig, RegistrationConfig, RunConfig,
)
from signal_analyzer.io import FileReader, FileWriter
from signal_analyzer.pipelines.run_with_registration import run_with_registration

NATIVE_SHAPE = (12, 20, 20)
TRANSPOSE_ORDER = (1, 0, 2)
RAW_ANNOTATION_SHAPE = (5, 4, 5)


def _write_synthetic_data(tmp_path):
    volume = np.zeros(NATIVE_SHAPE, dtype=np.uint8)
    volume[5:9, 5:9, 1:5] = 1    # region A side
    volume[5:9, 5:9, 15:19] = 1  # region B side

    mask_writer = FileWriter(
        output_path=tmp_path, output_name="mask", output_type="zarr",
        full_res_shape=NATIVE_SHAPE, output_dtype=np.uint8, chunk_size=(4, 20, 20),
    )
    mask_writer.write(volume)

    raw_annotation = np.zeros(RAW_ANNOTATION_SHAPE, dtype=np.uint32)
    raw_annotation[:, :, 0:2] = 10  # region A
    raw_annotation[:, :, 2:5] = 20  # region B
    anno_writer = FileWriter(
        output_path=tmp_path, output_name="raw_annotation", output_type="zarr",
        full_res_shape=RAW_ANNOTATION_SHAPE, output_dtype=np.uint32, chunk_size=RAW_ANNOTATION_SHAPE,
    )
    anno_writer.write(raw_annotation)

    return mask_writer.output_path, anno_writer.output_path


def _write_structures_csv(tmp_path):
    path = tmp_path / "structures.csv"
    path.write_text(
        "acronym,id,name,structure_id_path,parent_structure_id\n"
        "ROOT,1,Root,/1/,\n"
        "A,10,Region A,/1/10/,1.0\n"
        "B,20,Region B,/1/20/,1.0\n",
        encoding="utf-8",
    )
    return path


def _make_config(mask_path, annotation_path, structures_csv, output_dir) -> RunConfig:
    # Deliberately just ONE biomarker, no colocalization section -- this
    # doubles as proof the pipeline works fine for single-marker runs (the
    # user flagged this as a requirement); nothing here or in
    # run_with_registration requires a second biomarker or colocalization.
    return RunConfig(
        brain=BrainConfig(id="PipelineTestA", voxel_size_um=(4.0, 1.82, 1.82), needs_registration=True),
        biomarkers={
            "cFos": BiomarkerConfig(
                name="cFos", mask_path=str(mask_path),
                detection=DetectionConfig(connectivity=18, z_chunk=5),
                filter=FilterConfig(min_volume_voxels=30, max_volume_voxels=100),
            ),
        },
        registration=RegistrationConfig(
            annotation_path=str(annotation_path),
            downsample_factor=(3, 4, 4),
            transpose_order=TRANSPOSE_ORDER,
        ),
        paths=PathsConfig(structures_csv=str(structures_csv), output_dir=str(output_dir)),
    )


def test_run_with_registration_full_pipeline(tmp_path):
    mask_path, annotation_path = _write_synthetic_data(tmp_path)
    structures_csv = _write_structures_csv(tmp_path)
    output_dir = tmp_path / "out"
    config = _make_config(mask_path, annotation_path, structures_csv, output_dir)

    functional_system = {"keywords": [], "acronyms": ["A", "B"], "exact_acronym_match": True}
    results = run_with_registration(config, output_dir=output_dir, roi_acronyms=["A"], functional_system=functional_system)

    stage = results["cFos"]
    assert stage["cells_path"].exists()
    assert stage["filter"]["total"] == 2
    assert stage["assign_regions"] == {"assigned": 2}
    assert stage["report_path"].exists()

    assert stage["roi_extract"] is not None
    assert "A" in stage["roi_extract"]
    roi_out_path = stage["roi_extract"]["A"]["output_path"]
    assert roi_out_path is not None
    roi_result = FileReader(roi_out_path).read()
    assert roi_result.shape == NATIVE_SHAPE  # matches the ORIGINAL image's dimensions

    atlas_redraw = FileReader(stage["redraw"]["atlas"]).read()
    native_redraw = FileReader(stage["redraw"]["native"]).read()
    assert (atlas_redraw > 0).sum() == 2
    assert (native_redraw > 0).sum() == 2

    import pandas as pd
    sheets = pd.read_excel(stage["report_path"], sheet_name=None)
    assert "Target_Summary" in sheets
    assert "Tier 2" in sheets


def test_run_with_registration_redetects_after_truncated_cells_file(tmp_path):
    """A crash-truncated cells.parquet must not be mistaken for a completed
    detection result (same fix/behavior as run_without_registration)."""
    mask_path, annotation_path = _write_synthetic_data(tmp_path)
    structures_csv = _write_structures_csv(tmp_path)
    output_dir = tmp_path / "out"
    config = _make_config(mask_path, annotation_path, structures_csv, output_dir)

    run_with_registration(config, output_dir=output_dir)
    log_dir = output_dir / "cFos" / "logs"
    assert len(list(log_dir.glob("detect_*.log"))) == 1

    cells_path = output_dir / "cFos" / "cells.parquet"
    cells_path.write_bytes(cells_path.read_bytes()[:20])

    run_with_registration(config, output_dir=output_dir)
    assert len(list(log_dir.glob("detect_*.log"))) == 2
    assert cells_path.exists()


def test_run_with_registration_roi_extract_skips_on_second_call(tmp_path):
    """roi_extract is a genuinely expensive full-volume resample, same cost
    class as detect -- it should skip re-extracting a region whose output
    file already exists, unlike before this repo added skip-if-exists there."""
    mask_path, annotation_path = _write_synthetic_data(tmp_path)
    structures_csv = _write_structures_csv(tmp_path)
    output_dir = tmp_path / "out"
    config = _make_config(mask_path, annotation_path, structures_csv, output_dir)

    run_with_registration(config, output_dir=output_dir, roi_acronyms=["A"])
    log_dir = output_dir / "cFos" / "logs"
    assert len(list(log_dir.glob("roi_extract_*.log"))) == 1
    roi_path = output_dir / "cFos" / "roi_extract" / "A_atlas.tiff"
    assert roi_path.exists()
    mtime_after_first = roi_path.stat().st_mtime_ns

    run_with_registration(config, output_dir=output_dir, roi_acronyms=["A"])
    assert len(list(log_dir.glob("roi_extract_*.log"))) == 2  # stage itself still runs/logs...
    assert roi_path.stat().st_mtime_ns == mtime_after_first  # ...but didn't rewrite the file


def test_run_with_registration_rejects_non_registered_config(tmp_path):
    mask_path, annotation_path = _write_synthetic_data(tmp_path)
    structures_csv = _write_structures_csv(tmp_path)
    output_dir = tmp_path / "out"
    good_config = _make_config(mask_path, annotation_path, structures_csv, output_dir)

    bad_config = RunConfig(
        brain=BrainConfig(id="Bad", voxel_size_um=good_config.brain.voxel_size_um, needs_registration=False),
        biomarkers=good_config.biomarkers, paths=good_config.paths,
    )
    with pytest.raises(ValueError, match="run_with_registration"):
        run_with_registration(bad_config, output_dir=output_dir)
