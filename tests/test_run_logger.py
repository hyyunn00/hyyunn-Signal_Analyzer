"""Phase 0 verification: RunLogger produces a readable .log + .params.json
for each stage invocation, capturing config, inputs, timing, and results."""
import json
import logging

import pytest

from signal_analyzer.common.run_logger import RunLogger


def test_run_logger_writes_log_and_params_on_success(tmp_path):
    with RunLogger("detect", tmp_path, run_id="abc123", config={"voxel_size_um": [4.0, 1.82, 1.82]}) as run:
        run.record_input("mask_path", tmp_path / "mask.tif")
        logging.getLogger("some.stage.module").info("processing chunk 1")
        run.record_result("cells_detected", 42)

    assert run.log_path.exists()
    assert run.params_path.exists()

    log_text = run.log_path.read_text(encoding="utf-8")
    assert "processing chunk 1" in log_text
    assert "Run started" in log_text
    assert "Run completed" in log_text

    record = json.loads(run.params_path.read_text(encoding="utf-8"))
    assert record["stage"] == "detect"
    assert record["run_id"] == "abc123"
    assert record["status"] == "completed"
    assert record["error"] is None
    assert record["config"] == {"voxel_size_um": [4.0, 1.82, 1.82]}
    assert record["results"] == {"cells_detected": 42}
    assert record["inputs"]["mask_path"].endswith("mask.tif")
    assert record["duration_s"] >= 0


def test_run_logger_records_failure_without_swallowing_exception(tmp_path):
    with pytest.raises(ValueError, match="boom"):
        with RunLogger("filter", tmp_path, run_id="fail1") as run:
            raise ValueError("boom")

    record = json.loads(run.params_path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "boom" in record["error"]


def test_run_logger_file_handler_is_detached_after_exit(tmp_path):
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)

    with RunLogger("report", tmp_path, run_id="detach1"):
        assert len(root_logger.handlers) == len(handlers_before) + 1

    assert root_logger.handlers == handlers_before


def test_two_runs_get_distinct_log_files(tmp_path):
    with RunLogger("detect", tmp_path, run_id="run1") as run1:
        pass
    with RunLogger("detect", tmp_path, run_id="run2") as run2:
        pass

    assert run1.log_path != run2.log_path
    assert run1.log_path.exists()
    assert run2.log_path.exists()
