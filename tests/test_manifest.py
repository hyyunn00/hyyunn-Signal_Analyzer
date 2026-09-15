"""Reproducibility verification: manifest.json summarizes each stage's
last-known-good run so a user can tell whether a biomarker output directory
is complete/current without globbing timestamped logs/*.params.json files."""
from signal_analyzer.common.manifest import (
    fingerprint_matches,
    read_manifest,
    stage_fingerprint,
    update,
)


def test_read_manifest_returns_empty_dict_when_missing(tmp_path):
    assert read_manifest(tmp_path) == {}


def test_read_manifest_tolerates_corrupt_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{not valid json", encoding="utf-8")
    assert read_manifest(tmp_path) == {}


def test_update_records_successful_stage(tmp_path):
    record = {"stage": "detect", "status": "completed", "timestamp": "20260101-000000"}
    update(tmp_path, "detect", record, success=True)

    manifest = read_manifest(tmp_path)
    assert manifest["stages"]["detect"] == record
    assert manifest["last_attempt"]["detect"] == record


def test_update_failed_run_does_not_clobber_last_known_good(tmp_path):
    good_record = {"stage": "redraw", "status": "completed", "timestamp": "1"}
    update(tmp_path, "redraw", good_record, success=True)

    failed_record = {"stage": "redraw", "status": "failed", "timestamp": "2", "error": "boom"}
    update(tmp_path, "redraw", failed_record, success=False)

    manifest = read_manifest(tmp_path)
    assert manifest["stages"]["redraw"] == good_record  # untouched by the failure
    assert manifest["last_attempt"]["redraw"] == failed_record  # but visible here


def test_update_preserves_other_stages(tmp_path):
    update(tmp_path, "detect", {"status": "completed", "timestamp": "1"}, success=True)
    update(tmp_path, "filter", {"status": "completed", "timestamp": "2"}, success=True)

    manifest = read_manifest(tmp_path)
    assert set(manifest["stages"].keys()) == {"detect", "filter"}


def test_stage_fingerprint_none_when_missing(tmp_path):
    assert stage_fingerprint(tmp_path / "nope.parquet") is None


def test_stage_fingerprint_reflects_mtime_and_size(tmp_path):
    path = tmp_path / "cells.parquet"
    path.write_bytes(b"hello")
    fp = stage_fingerprint(path)
    assert fp == {"mtime_ns": path.stat().st_mtime_ns, "size": 5}


def test_fingerprint_matches_false_when_recorded_is_none(tmp_path):
    path = tmp_path / "cells.parquet"
    path.write_bytes(b"hello")
    assert fingerprint_matches(None, path) is False


def test_fingerprint_matches_false_when_file_missing(tmp_path):
    fp = {"mtime_ns": 123, "size": 5}
    assert fingerprint_matches(fp, tmp_path / "gone.parquet") is False


def test_fingerprint_matches_true_when_unchanged(tmp_path):
    path = tmp_path / "cells.parquet"
    path.write_bytes(b"hello")
    fp = stage_fingerprint(path)
    assert fingerprint_matches(fp, path) is True


def test_fingerprint_matches_false_after_rewrite(tmp_path):
    path = tmp_path / "cells.parquet"
    path.write_bytes(b"hello")
    fp = stage_fingerprint(path)

    tmp = tmp_path / ".tmp-cells.parquet"
    tmp.write_bytes(b"hello, but different")
    tmp.replace(path)

    assert fingerprint_matches(fp, path) is False
