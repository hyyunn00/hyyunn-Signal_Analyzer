"""Shared per-run logging: a console+file log plus a structured params/results
JSON sidecar, used identically by every pipeline stage/CLI.

Neither MARS (click.echo/tqdm only) nor Chulab-Signal_Analyzer
(console-only logging.basicConfig) persists a per-run record of what was
run, with what parameters, and what it produced. This generalizes the one
partial precedent that does exist -- MARS's aba2roi.py lookup-table txt
(job name + timestamp + atlas path) -- into a uniform utility for every stage.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from . import manifest as _manifest


def _json_default(obj: Any) -> Any:
    """Best-effort JSON coercion for common non-serializable types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


class RunLogger:
    """Attach a per-run file log + params/results JSON sidecar to a pipeline stage.

    Usage:
        with RunLogger("detect", output_dir, config=cfg_dict) as run:
            run.record_input("mask_path", mask_path)
            ...
            run.record_result("cells_detected", n)

    On exit, writes:
      - ``<output_dir>/logs/<stage>_<timestamp>_<run_id>.log``: a FileHandler
        attached to the root logger (alongside any existing console handler),
        so every ``logging.info(...)`` call made during the ``with`` block is
        captured, not just calls this class makes itself.
      - ``<output_dir>/logs/<stage>_<timestamp>_<run_id>.params.json``: the
        resolved config, recorded inputs, timing, status, and a ``results``
        dict populated via ``record_result``.
    """

    def __init__(
        self,
        stage_name: str,
        output_dir: str | Path,
        run_id: Optional[str] = None,
        config: Optional[dict] = None,
        level: int = logging.INFO,
    ) -> None:
        self.stage_name = stage_name
        self.output_dir = Path(output_dir)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.level = level
        self.config = config or {}

        self._timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._base_name = f"{stage_name}_{self._timestamp}_{self.run_id}"

        self.logs_dir = self.output_dir / "logs"
        self.log_path = self.logs_dir / f"{self._base_name}.log"
        self.params_path = self.logs_dir / f"{self._base_name}.params.json"

        self._inputs: dict[str, Any] = {}
        self._results: dict[str, Any] = {}
        self._fingerprints: dict[str, Any] = {}
        self._file_handler: Optional[logging.Handler] = None
        self._start_time: Optional[float] = None
        self._status = "not_started"
        self._error: Optional[str] = None

    def record_input(self, key: str, value: Any) -> None:
        """Record an input path/parameter to be captured in the params sidecar."""
        self._inputs[key] = value

    def record_result(self, key: str, value: Any) -> None:
        """Record a result value (e.g. cell counts, output paths) in the params sidecar."""
        self._results[key] = value

    def record_input_fingerprint(self, key: str, path: str | Path) -> None:
        """Record a cheap (mtime, size) fingerprint of an input file, for later
        staleness checks (see ``common.manifest.fingerprint_matches``) --
        e.g. so a downstream stage can tell whether ``cells.parquet`` has
        changed since this stage last ran successfully.
        """
        self._fingerprints[key] = _manifest.stage_fingerprint(path)

    def __enter__(self) -> "RunLogger":
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._file_handler.setLevel(self.level)
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(self._file_handler)
        if root_logger.level == logging.NOTSET or root_logger.level > self.level:
            root_logger.setLevel(self.level)

        self._start_time = time.time()
        self._status = "running"
        logging.getLogger(__name__).info(
            "Run started: stage=%s run_id=%s log=%s", self.stage_name, self.run_id, self.log_path
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration_s = time.time() - self._start_time if self._start_time is not None else None

        if exc_type is not None:
            self._status = "failed"
            self._error = f"{exc_type.__name__}: {exc_val}"
            logging.getLogger(__name__).error(
                "Run failed: stage=%s run_id=%s error=%s", self.stage_name, self.run_id, self._error
            )
        else:
            self._status = "completed"
            logging.getLogger(__name__).info(
                "Run completed: stage=%s run_id=%s duration_s=%.2f",
                self.stage_name, self.run_id, duration_s or 0.0,
            )

        record = {
            "stage": self.stage_name,
            "run_id": self.run_id,
            "timestamp": self._timestamp,
            "status": self._status,
            "duration_s": duration_s,
            "error": self._error,
            "config": self.config,
            "inputs": self._inputs,
            "results": self._results,
            "fingerprints": self._fingerprints,
            "log_path": str(self.log_path),
        }
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=_json_default)

        try:
            _manifest.update(self.output_dir, self.stage_name, record, success=(self._status == "completed"))
        except OSError as e:
            logging.getLogger(__name__).warning("Failed to update manifest for stage=%s: %s", self.stage_name, e)

        if self._file_handler is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

        return False  # never swallow exceptions
