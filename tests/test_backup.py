"""Reproducibility verification: cells.parquet snapshots taken before an
in-place-rewriting stage (filter/assign_regions/colocalization) mutates it,
so a threshold change or a bad rewrite can be inspected/recovered from."""
from signal_analyzer.common.backup import backup_cells_file


def _atomic_overwrite(path, content: bytes) -> None:
    """Rewrite ``path`` via a fresh temp file + rename -- mirroring
    ``common.atomic_io.atomic_write_table``'s real write pattern (a new
    inode, swapped in by rename) rather than an in-place truncate+rewrite.
    Backups are hardlinks to the pre-rewrite inode; only a rename-based
    rewrite (not an in-place one) leaves an existing hardlink's content
    untouched, so tests exercising multiple rewrites must overwrite this way
    to match how the real mutating stages (filter/assign_regions/
    colocalization) behave."""
    tmp = path.with_name(f".tmp-{path.name}")
    tmp.write_bytes(content)
    tmp.replace(path)


def test_backup_is_noop_when_cells_file_does_not_exist_yet(tmp_path):
    cells_path = tmp_path / "cells.parquet"
    assert backup_cells_file(cells_path, stage="filter") is None
    assert not (tmp_path / "cells_backups").exists()


def test_backup_creates_a_snapshot_with_matching_content(tmp_path):
    cells_path = tmp_path / "cells.parquet"
    cells_path.write_bytes(b"original-bytes")

    backup_path = backup_cells_file(cells_path, stage="filter")

    assert backup_path is not None
    assert backup_path.parent.name == "cells_backups"
    assert backup_path.name.startswith("cells_before_filter_")
    assert backup_path.read_bytes() == b"original-bytes"

    # The live file is untouched by taking a snapshot of it.
    assert cells_path.read_bytes() == b"original-bytes"


def test_backup_pruning_keeps_only_newest_n_per_stage(tmp_path):
    cells_path = tmp_path / "cells.parquet"

    for i in range(5):
        _atomic_overwrite(cells_path, f"version-{i}".encode())
        backup_cells_file(cells_path, stage="filter", keep=2)

    backups = sorted((tmp_path / "cells_backups").glob("cells_before_filter_*.parquet"))
    assert len(backups) == 2
    # The two newest snapshots survive; their content is versions 3 and 4
    # (taken before the 4th and 5th rewrites respectively).
    contents = {p.read_bytes() for p in backups}
    assert contents == {b"version-3", b"version-4"}


def test_backup_pruning_is_scoped_per_stage(tmp_path):
    cells_path = tmp_path / "cells.parquet"

    cells_path.write_bytes(b"v0")
    backup_cells_file(cells_path, stage="filter", keep=1)
    _atomic_overwrite(cells_path, b"v1")
    backup_cells_file(cells_path, stage="assign_regions", keep=1)

    backups = list((tmp_path / "cells_backups").glob("cells_before_*.parquet"))
    # Both stages' single backup survive -- pruning for one stage must not
    # touch another stage's backups.
    assert len(backups) == 2
