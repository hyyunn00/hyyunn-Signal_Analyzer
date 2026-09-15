"""Shared atomic-write primitives, used everywhere a pipeline stage writes a
final output (a Parquet cell table, a redraw/roi_extract volume, the run
manifest) so a crash mid-write can never leave a partial file/directory at
the final path for a later run's existence-based skip check to mistake for
a completed result.

Both helpers rely on ``os.replace``, a single filesystem rename operation
(cannot itself be "half done" the way a multi-batch streaming write can) --
this works for both files and whole directory trees (e.g. a scroll-tiff
folder or a zarr store) on the same filesystem. On Windows, ``os.replace``
can only target a directory destination that does not already exist yet;
every caller here only enters the write path when the final path is
confirmed absent (the "not skipping" branch of a skip-if-exists check), so
this constraint is never actually hit in practice.
"""
from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq


def _temp_sibling(path: Path) -> Path:
    return path.with_name(f".tmp-{uuid.uuid4().hex[:8]}-{path.name}")


def atomic_write_table(table: pa.Table, path: str | Path) -> None:
    """Write a Parquet table to ``path`` via a same-directory temp file + rename.

    On any exception during the write, the temp file is removed and the
    exception is re-raised -- ``path`` is left untouched (either its prior
    contents, if any, or absent).
    """
    path = Path(path)
    tmp_path = _temp_sibling(path)
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_output(final_path: str | Path) -> Iterator[Path]:
    """Yield a temp sibling path to write into; rename it onto ``final_path`` on clean exit.

    Works for a single file or a whole directory tree. On exception, the
    temp path is removed (best-effort) and the exception propagates --
    nothing is left at ``final_path``.

    If ``final_path`` already exists as a directory, it is removed just
    before the rename (``os.replace`` cannot target an existing directory on
    Windows) -- callers only reach this when they've already decided to
    (re)write ``final_path``, so this is expected, not a surprise overwrite.
    This narrows, but can't fully close, the atomicity window for a
    directory-shaped output.
    """
    final_path = Path(final_path)
    tmp_path = _temp_sibling(final_path)
    try:
        yield tmp_path
        if final_path.is_dir():
            shutil.rmtree(final_path)
        elif final_path.exists():
            final_path.unlink()
        tmp_path.replace(final_path)
    except BaseException:
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path, ignore_errors=True)
        else:
            tmp_path.unlink(missing_ok=True)
        raise
