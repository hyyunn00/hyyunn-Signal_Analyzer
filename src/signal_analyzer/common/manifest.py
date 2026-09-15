"""Per-biomarker run manifest: a single ``manifest.json`` summarizing, for
each pipeline stage, the config/inputs/results/status from its last
invocation -- so a user can tell at a glance whether an output directory's
results are current, without globbing timestamped ``logs/*.params.json``
files (which accumulate one new pair per invocation for every stage that
reruns on every pipeline call).

Written by ``common.run_logger.RunLogger`` as part of its existing
``__exit__`` lifecycle (one update per stage invocation, from the exact same
in-memory record already persisted to that invocation's ``params.json``) --
not a separate re-derivation from the logs directory, so the manifest and a
stage's own log record can never diverge.

Also supplies the staleness-detection primitives used by ``redraw``'s
skip-if-exists check: a downstream output can exist and still need
recomputing if the ``cells.parquet`` it was built from has since been
rewritten (e.g. a changed volume-filter threshold).
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


def _json_default(obj: Any) -> Any:
    """Best-effort JSON coercion for common non-serializable types.

    Mirrors ``common.run_logger``'s ``_json_default`` (not imported from
    there to avoid a circular import, since ``run_logger`` imports this
    module) -- manifest records are exactly the records ``RunLogger`` builds
    for ``params.json``, so they need the same coercions (e.g. ``Path``
    values in ``inputs``/``results``).
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def stage_fingerprint(path: str | Path) -> Optional[dict]:
    """A cheap ``(mtime_ns, size)`` fingerprint of a file, or ``None`` if it doesn't exist.

    Deliberately not a content hash -- avoids reading a multi-million-row
    Parquet file just to decide whether a downstream stage's cached output
    is still valid. This is reliable here because every stage that mutates
    ``cells.parquet`` now does so via ``common.atomic_io`` (temp file, then
    rename), which always produces a fresh inode/mtime -- never an in-place
    byte edit that could leave the mtime unchanged.
    """
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def fingerprint_matches(recorded: Optional[dict], path: str | Path) -> bool:
    """Whether ``path``'s current fingerprint matches a previously recorded one.

    False whenever either side is missing or the values differ -- treating
    an output as stale is the safe default whenever there's any doubt.
    """
    if not recorded:
        return False
    return stage_fingerprint(path) == recorded


def _manifest_path(biomarker_dir: str | Path) -> Path:
    return Path(biomarker_dir) / MANIFEST_FILENAME


def read_manifest(biomarker_dir: str | Path) -> dict:
    """Read a biomarker directory's ``manifest.json``, or ``{}`` if missing/corrupt.

    Never raises -- a manifest is a convenience/staleness-check input, not
    authoritative data; a corrupt manifest should degrade to "no cached
    info", not break the pipeline.
    """
    path = _manifest_path(biomarker_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read manifest %s (%s); treating as empty.", path, e)
        return {}


def update(biomarker_dir: str | Path, stage_name: str, record: dict, success: bool) -> None:
    """Read-merge-write ``record`` into the biomarker directory's manifest.

    Always updates ``last_attempt[stage_name]``. Only updates
    ``stages[stage_name]`` -- the "last known good" record consulted by
    skip-if-exists/staleness checks -- when ``success`` is True, so a failed
    run never clobbers the last successful record for that stage.

    Written atomically (temp file + rename, matching ``common.atomic_io``'s
    pattern) so a crash mid-write never leaves a corrupt manifest for
    ``read_manifest`` to choke on later.
    """
    biomarker_dir = Path(biomarker_dir)
    biomarker_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(biomarker_dir)
    manifest.setdefault("manifest_version", 1)
    manifest.setdefault("stages", {})
    manifest.setdefault("last_attempt", {})

    manifest["last_attempt"][stage_name] = record
    if success:
        manifest["stages"][stage_name] = record
    manifest["updated_at"] = record.get("timestamp")

    path = _manifest_path(biomarker_dir)
    tmp_path = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}-{path.name}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=_json_default)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
