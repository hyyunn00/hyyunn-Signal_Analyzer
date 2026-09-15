"""End-to-end orchestration for brains NOT needing atlas registration
("brain type B"): detect -> filter -> whole-brain report -> native redraw.

No ``annotation.tif`` exists for these brains at all (confirmed with the
user), so no region assignment, ``roi_extract``, or atlas-space redraw
applies -- see the architecture plan's confirmed facts and
``report.cell_report``'s module docstring.

Resumable / skip-if-exists for the expensive stage (detection), matching
MARS's and Chulab's own pattern (MARS's ``3Dfilter/__main__.py``'s
"Skipping: <<...>> (File exists)" checks): filtering and reporting always
re-run (they're cheap Parquet operations, and re-running them is how a
changed volume-filter threshold takes effect without re-detecting).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..common.cell_schema import is_complete_cells_file
from ..common.config import RunConfig
from ..common.manifest import fingerprint_matches, read_manifest
from ..common.run_logger import RunLogger
from ..detection.dask_runner import detect_biomarker
from ..filtering.volume_filter import apply_volume_filter
from ..io import FileReader, expected_output_path
from ..redraw.mask_from_cells import redraw_native_mask
from ..report.cell_report import write_whole_brain_report

logger = logging.getLogger(__name__)


def run_without_registration(config: RunConfig, output_dir: Optional[str | Path] = None) -> dict:
    """Run the full non-registered-brain pipeline for every configured biomarker.

    Args:
        config: A validated RunConfig with ``needs_registration=False``.
        output_dir: Where to write outputs; defaults to ``config.paths.output_dir``.

    Returns:
        {biomarker_name: {"cells_path", "filter", "report_path",
        "redraw_path"}} summary for every biomarker.
    """
    if config.brain.needs_registration or config.registration is not None:
        raise ValueError(
            "run_without_registration requires needs_registration=False and no registration section "
            "-- use run_with_registration for registered brains"
        )

    output_dir = Path(output_dir or config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, biomarker in config.biomarkers.items():
        biomarker_dir = output_dir / name
        biomarker_dir.mkdir(parents=True, exist_ok=True)
        cells_path = biomarker_dir / "cells.parquet"
        native_shape = FileReader(biomarker.mask_path).volume_shape

        if cells_path.exists() and is_complete_cells_file(cells_path):
            logger.info("Skipping detect for biomarker=%s: %s already exists", name, cells_path)
        else:
            with RunLogger("detect", biomarker_dir, config={"biomarker": name, "brain_id": config.brain.id}) as run:
                run.record_input("mask_path", biomarker.mask_path)
                n_detected = detect_biomarker(biomarker.mask_path, config.brain, biomarker, cells_path, run_id=run.run_id)
                run.record_result("cells_detected", n_detected)

        with RunLogger("filter", biomarker_dir, config={"biomarker": name}) as run:
            filter_summary = apply_volume_filter(cells_path, biomarker)
            run.record_result("filter_summary", filter_summary)

        report_path = biomarker_dir / f"{name}_whole_brain_report.csv"
        with RunLogger("report", biomarker_dir, config={"biomarker": name}) as run:
            report_df = write_whole_brain_report(cells_path, report_path)
            total_cells = int(report_df["total_cells"].sum()) if not report_df.empty else 0
            run.record_result("report_path", str(report_path))
            run.record_result("total_cells", total_cells)

        redraw_output_name = f"{name}_redraw_native"
        expected_redraw = expected_output_path(biomarker_dir, redraw_output_name, "scroll-tiff")
        recorded_fp = read_manifest(biomarker_dir).get("stages", {}).get("redraw", {}).get("fingerprints", {}).get("cells_parquet")
        if expected_redraw.exists() and fingerprint_matches(recorded_fp, cells_path):
            logger.info(
                "Skipping redraw for biomarker=%s: %s exists and cells.parquet unchanged", name, expected_redraw,
            )
            redraw_path = expected_redraw
        else:
            with RunLogger("redraw", biomarker_dir, config={"biomarker": name}) as run:
                run.record_input_fingerprint("cells_parquet", cells_path)
                redraw_path = redraw_native_mask(
                    cells_path, native_shape, biomarker_dir, redraw_output_name, biomarker=name,
                )
                run.record_result("redraw_path", str(redraw_path))

        results[name] = {
            "cells_path": cells_path,
            "filter": filter_summary,
            "report_path": report_path,
            "redraw_path": redraw_path,
        }

    return results
