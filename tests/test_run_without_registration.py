"""Phase 5 verification: the run_without_registration orchestrator wires
detect -> filter -> report -> redraw together correctly, with skip-if-exists
on the expensive detect stage and per-stage RunLogger records."""
import numpy as np
import pytest

from signal_analyzer.common.config import BiomarkerConfig, BrainConfig, DetectionConfig, FilterConfig, RunConfig, PathsConfig
from signal_analyzer.io import FileReader, FileWriter
from signal_analyzer.pipelines.run_without_registration import run_without_registration

NATIVE_SHAPE = (12, 20, 20)


def _write_synthetic_mask(tmp_path, name="synthetic_mask"):
    volume = np.zeros(NATIVE_SHAPE, dtype=np.uint8)
    volume[5:9, 5:9, 5:9] = 1   # 64 voxels, passes filter
    volume[0:2, 0:2, 0:2] = 1   # 8 voxels, filtered out

    writer = FileWriter(
        output_path=tmp_path, output_name=name, output_type="zarr",
        full_res_shape=NATIVE_SHAPE, output_dtype=np.uint8, chunk_size=(4, 20, 20),
    )
    writer.write(volume)
    return writer.output_path


def _make_config(tmp_path, mask_path, output_dir) -> RunConfig:
    # Deliberately just ONE biomarker, no colocalization section -- this
    # doubles as proof the pipeline works fine for single-marker runs (the
    # user flagged this as a requirement); nothing here or in
    # run_without_registration requires a second biomarker or colocalization.
    return RunConfig(
        brain=BrainConfig(id="PipelineTestB", voxel_size_um=(4.0, 1.82, 1.82), needs_registration=False),
        biomarkers={
            "cFos": BiomarkerConfig(
                name="cFos", mask_path=str(mask_path),
                detection=DetectionConfig(connectivity=18, z_chunk=5),
                filter=FilterConfig(min_volume_voxels=30, max_volume_voxels=100),
            ),
        },
        paths=PathsConfig(structures_csv="data/structures.csv", output_dir=str(output_dir)),
    )


def test_run_without_registration_full_pipeline(tmp_path):
    mask_path = _write_synthetic_mask(tmp_path)
    output_dir = tmp_path / "out"
    config = _make_config(tmp_path, mask_path, output_dir)

    results = run_without_registration(config, output_dir=output_dir)

    assert set(results.keys()) == {"cFos"}
    stage = results["cFos"]
    assert stage["cells_path"].exists()
    assert stage["filter"] == {"total": 2, "passed": 1}
    assert stage["report_path"].exists()
    assert stage["redraw_path"] is not None

    redrawn = FileReader(stage["redraw_path"]).read()
    assert (redrawn > 0).sum() == 1  # only the passing blob was redrawn

    # Per-stage log files were written.
    log_dir = output_dir / "cFos" / "logs"
    log_files = list(log_dir.glob("*.log"))
    stages_logged = {f.name.split("_")[0] for f in log_files}
    assert stages_logged == {"detect", "filter", "report", "redraw"}


def test_run_without_registration_skips_detect_on_second_call(tmp_path):
    mask_path = _write_synthetic_mask(tmp_path)
    output_dir = tmp_path / "out"
    config = _make_config(tmp_path, mask_path, output_dir)

    run_without_registration(config, output_dir=output_dir)
    log_dir = output_dir / "cFos" / "logs"
    n_detect_logs_after_first = len(list(log_dir.glob("detect_*.log")))
    assert n_detect_logs_after_first == 1

    run_without_registration(config, output_dir=output_dir)
    n_detect_logs_after_second = len(list(log_dir.glob("detect_*.log")))
    assert n_detect_logs_after_second == 1  # detect was skipped, no new log

    # But filter/report/redraw always re-run (cheap, and how threshold
    # changes take effect without re-detecting).
    n_filter_logs = len(list(log_dir.glob("filter_*.log")))
    assert n_filter_logs == 2


def test_run_without_registration_rejects_registered_brain_config(tmp_path):
    mask_path = _write_synthetic_mask(tmp_path)
    output_dir = tmp_path / "out"
    config = _make_config(tmp_path, mask_path, output_dir)
    # Manually flip to a registered-brain config (bypassing load_config's
    # own invariant check, to test run_without_registration's own guard).
    bad_config = RunConfig(
        brain=BrainConfig(id="Bad", voxel_size_um=config.brain.voxel_size_um, needs_registration=True),
        biomarkers=config.biomarkers, paths=config.paths,
    )
    with pytest.raises(ValueError, match="run_without_registration"):
        run_without_registration(bad_config, output_dir=output_dir)
