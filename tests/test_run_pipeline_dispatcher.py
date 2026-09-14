"""Phase 5 verification: run_pipeline dispatches to the correct orchestrator
based on config.brain.needs_registration, and the CLI loads a real YAML
config end-to-end."""
import logging

import numpy as np
import yaml

from signal_analyzer.common.config import (
    BiomarkerConfig, BrainConfig, DetectionConfig, FilterConfig, PathsConfig, RunConfig,
)
from signal_analyzer.io import FileWriter
from signal_analyzer.pipelines.run import main, run_pipeline

NATIVE_SHAPE = (6, 10, 10)


def _write_mask(tmp_path, name="mask"):
    volume = np.zeros(NATIVE_SHAPE, dtype=np.uint8)
    volume[1:5, 1:5, 1:5] = 1  # 4x4x4 = 64 voxels
    writer = FileWriter(
        output_path=tmp_path, output_name=name, output_type="zarr",
        full_res_shape=NATIVE_SHAPE, output_dtype=np.uint8, chunk_size=NATIVE_SHAPE,
    )
    writer.write(volume)
    return writer.output_path


def test_run_pipeline_dispatches_to_without_registration(tmp_path):
    mask_path = _write_mask(tmp_path)
    output_dir = tmp_path / "out"
    config = RunConfig(
        brain=BrainConfig(id="Dispatch-B", voxel_size_um=(4.0, 1.82, 1.82), needs_registration=False),
        biomarkers={
            "cFos": BiomarkerConfig(
                name="cFos", mask_path=str(mask_path),
                detection=DetectionConfig(connectivity=18, z_chunk=3),
                filter=FilterConfig(min_volume_voxels=30, max_volume_voxels=100),
            ),
        },
        paths=PathsConfig(structures_csv="data/structures.csv", output_dir=str(output_dir)),
    )

    results = run_pipeline(config, output_dir=output_dir)
    assert "report_path" in results["cFos"]
    assert "assign_regions" not in results["cFos"]  # type-B result shape, not type-A's


def test_run_pipeline_warns_and_ignores_roi_for_type_b(tmp_path, caplog):
    mask_path = _write_mask(tmp_path)
    output_dir = tmp_path / "out"
    config = RunConfig(
        brain=BrainConfig(id="Dispatch-B2", voxel_size_um=(4.0, 1.82, 1.82), needs_registration=False),
        biomarkers={
            "cFos": BiomarkerConfig(
                name="cFos", mask_path=str(mask_path),
                detection=DetectionConfig(connectivity=18, z_chunk=3),
                filter=FilterConfig(min_volume_voxels=30, max_volume_voxels=100),
            ),
        },
        paths=PathsConfig(structures_csv="data/structures.csv", output_dir=str(output_dir)),
    )

    with caplog.at_level(logging.WARNING):
        results = run_pipeline(config, output_dir=output_dir, roi_acronyms=["A"])

    assert "ignoring roi_acronyms" in caplog.text
    assert "report_path" in results["cFos"]  # pipeline still ran, roi just skipped


def test_cli_main_runs_without_registration_end_to_end(tmp_path, capsys):
    mask_path = _write_mask(tmp_path)
    output_dir = tmp_path / "out"

    config_data = {
        "brain": {"id": "CLI-Test-B", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": False},
        "biomarkers": {
            "cFos": {
                "mask_path": str(mask_path),
                "detection": {"connectivity": 18, "z_chunk": 3},
                "filter": {"min_volume_voxels": 30, "max_volume_voxels": 100},
            },
        },
        "paths": {"structures_csv": "data/structures.csv", "output_dir": str(output_dir)},
    }
    config_path = tmp_path / "cli_config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    exit_code = main([str(config_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "cFos" in captured.out
    assert "report_path" in captured.out
    assert (output_dir / "cFos" / "cFos_whole_brain_report.csv").exists()
