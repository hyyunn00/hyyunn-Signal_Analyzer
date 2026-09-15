"""Phase 0 verification: the Parquet cell schema round-trips incremental
per-chunk writes and supports the filtered reads later stages depend on
(colocalization joins, per-biomarker report aggregation)."""
import pytest

from signal_analyzer.common.cell_schema import (
    HEMISPHERE_LEFT,
    HEMISPHERE_NA,
    CellTableWriter,
    is_complete_cells_file,
    read_cells,
)


def _record(cell_id, z, y, x, biomarker="cFos", passed_filter=True, region_id=None):
    return {
        "cell_id": cell_id,
        "z": z,
        "y": y,
        "x": x,
        "volume_voxels": 30,
        "volume_um3": 1.23,
        "biomarker": biomarker,
        "region_id": region_id,
        "hemisphere_id": HEMISPHERE_LEFT if region_id is not None else HEMISPHERE_NA,
        "brain_id": "TestBrain-1",
        "run_id": "run-xyz",
        "passed_filter": passed_filter,
    }


def test_incremental_write_then_read_all(tmp_path):
    out_path = tmp_path / "cells.parquet"

    with CellTableWriter(out_path) as writer:
        n1 = writer.write_batch([_record(1, 0, 0, 0), _record(2, 0, 1, 1)])
        n2 = writer.write_batch([_record(3, 1, 0, 0)])

    assert n1 == 2
    assert n2 == 1
    assert writer._n_written == 3

    df = read_cells(out_path)
    assert len(df) == 3
    assert set(df["cell_id"]) == {1, 2, 3}


def test_brain_type_b_rows_have_null_region_id(tmp_path):
    """Brain type B has no annotation.tif at all -- region_id must stay null,
    never a placeholder, matching the corrected pipeline understanding."""
    out_path = tmp_path / "cells_no_reg.parquet"

    with CellTableWriter(out_path) as writer:
        writer.write_batch([_record(1, 0, 0, 0, region_id=None)])

    df = read_cells(out_path)
    assert df.loc[0, "region_id"] is None or df["region_id"].isna().all()
    assert df.loc[0, "hemisphere_id"] == HEMISPHERE_NA


def test_read_cells_filters_by_biomarker_and_passed_filter(tmp_path):
    out_path = tmp_path / "cells_mixed.parquet"

    with CellTableWriter(out_path) as writer:
        writer.write_batch([
            _record(1, 0, 0, 0, biomarker="cFos", passed_filter=True),
            _record(2, 0, 1, 1, biomarker="cFos", passed_filter=False),
            _record(3, 0, 2, 2, biomarker="TH", passed_filter=True),
        ])

    cfos_passed = read_cells(out_path, biomarker="cFos", passed_filter=True)
    assert list(cfos_passed["cell_id"]) == [1]

    th_only = read_cells(out_path, biomarker="TH")
    assert list(th_only["cell_id"]) == [3]

    all_passed = read_cells(out_path, passed_filter=True)
    assert set(all_passed["cell_id"]) == {1, 3}


def test_write_batch_rejects_missing_required_field(tmp_path):
    out_path = tmp_path / "cells_bad.parquet"
    bad_record = _record(1, 0, 0, 0)
    del bad_record["volume_voxels"]

    writer = CellTableWriter(out_path)
    try:
        try:
            writer.write_batch([bad_record])
            assert False, "expected ValueError for missing field"
        except ValueError as e:
            assert "volume_voxels" in str(e)
    finally:
        writer.close()


def test_empty_batch_is_a_noop(tmp_path):
    out_path = tmp_path / "cells_empty.parquet"
    writer = CellTableWriter(out_path)
    n = writer.write_batch([])
    assert n == 0
    assert writer.close() == 0


def test_write_batch_exception_mid_loop_leaves_no_final_file(tmp_path):
    """A crash mid-write must never leave a truncated file at the real
    ``output_path`` -- writes go to a temp sibling, renamed onto the final
    path only on a clean ``close()``/``__exit__``. Otherwise a later run's
    skip-if-exists check (bare ``.exists()``) would mistake a truncated file
    for a completed detection result."""
    out_path = tmp_path / "cells.parquet"

    with pytest.raises(ValueError):
        with CellTableWriter(out_path) as writer:
            writer.write_batch([_record(1, 0, 0, 0)])
            bad_record = _record(2, 0, 1, 1)
            del bad_record["volume_voxels"]
            writer.write_batch([bad_record])

    assert not out_path.exists()
    assert list(tmp_path.glob(".tmp-*")) == []


def test_is_complete_cells_file(tmp_path):
    out_path = tmp_path / "cells.parquet"

    assert is_complete_cells_file(out_path) is False  # doesn't exist yet

    with CellTableWriter(out_path) as writer:
        writer.write_batch([_record(1, 0, 0, 0)])
    assert is_complete_cells_file(out_path) is True

    # Simulate a pre-existing truncated file (e.g. from before this repo's
    # write-then-rename fix, or any other truncation source).
    truncated_path = tmp_path / "truncated.parquet"
    truncated_path.write_bytes(out_path.read_bytes()[:20])
    assert is_complete_cells_file(truncated_path) is False
