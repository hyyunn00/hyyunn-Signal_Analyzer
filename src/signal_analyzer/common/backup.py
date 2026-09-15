"""Lightweight snapshotting of a biomarker's ``cells.parquet`` before an
in-place-rewriting stage (filter / assign_regions / colocalization) mutates
it, so a threshold change (or a bad rewrite) can be inspected or recovered
from -- not a full data-versioning system, just enough for a small lab
pipeline where each cell table is a single per-biomarker Parquet file.

Snapshots are hardlinks, not copies, whenever the filesystem supports them:
every mutating stage now rewrites ``cells.parquet`` via
``common.atomic_io.atomic_write_table`` (temp file + rename), so the file is
always *replaced*, never edited byte-in-place -- a hardlink taken right
before the rewrite keeps pointing at the pre-rewrite inode's data at
essentially zero extra disk cost, even for a multi-million-row table. Falls
back to a real copy if hardlinking isn't supported (e.g. across filesystems).

Recovery is manual: copy the desired ``cells_backups/cells_before_<stage>_
<timestamp>_<id>.parquet`` file back over ``cells.parquet``. No restore CLI
-- disproportionate for this scope.
"""
from __future__ import annotations

import itertools
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BACKUP_DIRNAME = "cells_backups"

# A process-wide monotonic counter, not a timestamp, decides backup
# creation order for pruning -- see backup_cells_file's docstring note.
_sequence = itertools.count()


def backup_cells_file(cells_path: str | Path, stage: str, keep: int = 3) -> Optional[Path]:
    """Snapshot ``cells_path`` before ``stage`` mutates it. No-op if it doesn't exist yet.

    Args:
        cells_path: Path to the cell Parquet table about to be rewritten.
        stage: Name of the stage about to mutate it (e.g. ``"filter"``,
            ``"assign_regions"``, ``"colocalization"``) -- used in the
            backup filename and to scope retention pruning per-stage.
        keep: Number of most-recent backups to retain per stage; older ones
            for the same stage are pruned after this snapshot is taken.

    Returns:
        Path to the new backup file, or ``None`` if ``cells_path`` didn't
        exist yet (nothing to snapshot -- e.g. the very first filter run).
    """
    cells_path = Path(cells_path)
    if not cells_path.exists():
        return None

    backup_dir = cells_path.parent / _BACKUP_DIRNAME
    backup_dir.mkdir(parents=True, exist_ok=True)

    # A hardlinked backup shares its source file's inode -- and therefore its
    # mtime, not the moment the link was created -- so pruning can't sort by
    # filesystem mtime (rapid successive calls can tie, or even sort
    # "backwards" relative to call order). A wall-clock timestamp has the
    # same problem under rapid successive calls, since the clock's actual
    # resolution can be coarser than the time between calls. A process-wide
    # monotonic counter, zero-padded into the filename, has neither issue --
    # it sorts lexicographically in exact creation order, always.
    human_ts = time.strftime("%Y%m%d-%H%M%S")
    seq = next(_sequence)
    backup_path = backup_dir / f"cells_before_{stage}_{human_ts}_{seq:012d}_{uuid.uuid4().hex[:8]}.parquet"

    try:
        os.link(cells_path, backup_path)
    except OSError:
        shutil.copy2(cells_path, backup_path)

    _prune_backups(backup_dir, stage, keep)
    return backup_path


def _prune_backups(backup_dir: Path, stage: str, keep: int) -> None:
    # Filenames sort chronologically (see the zero-padded ns timestamp
    # above), so a plain name sort -- not filesystem mtime -- is what's
    # reliable here.
    backups = sorted(backup_dir.glob(f"cells_before_{stage}_*.parquet"), reverse=True)
    for stale in backups[keep:]:
        try:
            stale.unlink()
        except OSError:
            logger.warning("Failed to prune stale backup %s", stale)
