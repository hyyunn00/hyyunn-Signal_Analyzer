"""End-to-end orchestration for brains needing atlas registration
("brain type A"): detect -> filter -> assign regions -> tiered region
report + asymmetry (+ optional functional-system Target_Summary) ->
optional roi_extract -> redraw (atlas-space + native-space).

Does NOT run ``align.convert.prepare_registration_input`` or BIRDS itself
-- BIRDS registration is explicitly out of this repo's scope (see the
architecture plan). This orchestrator picks up from an already-registered
brain: a ``RunConfig`` whose ``registration.annotation_path`` already
points at BIRDS's output.

Resumable / skip-if-exists for the expensive stage (detection), matching
MARS's and Chulab's own pattern (MARS's ``3Dfilter/__main__.py``'s
"Skipping: <<...>> (File exists)" checks): every other stage always
re-runs (they're cheap Parquet/array operations, and re-running them is how
a changed volume-filter threshold or functional-system selection takes
effect without re-detecting).

Region geometry (``report.region_stats.build_region_geometry_report``) is
computed exactly once per run and reused across every biomarker, per the
architecture plan's refinement over both source repos (region geometry
depends only on the annotation, never on a specific biomarker's cells).
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
from ..redraw.mask_from_cells import redraw_atlas_mask, redraw_native_mask
from ..regions.coord_transform import assign_regions_in_cell_table
from ..regions.structures import load_structures
from ..report.cell_report import build_target_summary, tiered_cell_report, write_tiered_region_report
from ..report.region_stats import build_region_geometry_report
from ..roi_extract.extract import extract_regions_to_native

logger = logging.getLogger(__name__)


def run_with_registration(
    config: RunConfig,
    output_dir: Optional[str | Path] = None,
    roi_acronyms: Optional[list[str]] = None,
    functional_system: Optional[dict] = None,
) -> dict:
    """Run the full registered-brain pipeline for every configured biomarker.

    Args:
        config: A validated RunConfig with ``needs_registration=True``.
        output_dir: Where to write outputs; defaults to ``config.paths.output_dir``.
        roi_acronyms: Optional list of Allen CCF acronyms to extract (via
            ``roi_extract``) for every biomarker; skipped entirely if
            None/empty.
        functional_system: Optional ``{"keywords": [...], "acronyms": [...],
            "exact_acronym_match": bool, "cell_threshold": int}`` (e.g. one
            entry from ``configs/functional_systems.yaml``) to additionally
            build a ``Target_Summary`` sheet in each biomarker's report.

    Returns:
        {biomarker_name: {"cells_path", "filter", "assign_regions",
        "report_path", "roi_extract", "redraw"}} summary for every biomarker.
    """
    if not config.brain.needs_registration or config.registration is None:
        raise ValueError(
            "run_with_registration requires a RunConfig with needs_registration=True and a "
            "'registration' section -- use run_without_registration for brain type B"
        )

    output_dir = Path(output_dir or config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_df = load_structures(config.paths.structures_csv)

    reg = config.registration
    annotation = FileReader(reg.annotation_path, transpose_order=reg.transpose_order).read()
    hemisphere = (
        FileReader(reg.hemisphere_path, transpose_order=reg.transpose_order).read()
        if reg.hemisphere_path else None
    )

    # Computed once, reused for every biomarker below -- depends only on the
    # annotation, never on a specific biomarker's detected cells.
    geometry = build_region_geometry_report(annotation, structure_df, config.brain.voxel_size_um, hemisphere)

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

        with RunLogger("assign_regions", biomarker_dir, config={"biomarker": name}) as run:
            assign_summary = assign_regions_in_cell_table(
                cells_path, native_shape, annotation, hemisphere, biomarker=name,
            )
            run.record_result("assign_summary", assign_summary)

        report_path = biomarker_dir / f"{name}_region_report.xlsx"
        with RunLogger("report", biomarker_dir, config={"biomarker": name, "functional_system": functional_system}) as run:
            report = tiered_cell_report(cells_path, structure_df, geometry, name)
            target_summary = None
            if functional_system:
                target_summary = build_target_summary(
                    report,
                    keywords=functional_system.get("keywords", []),
                    acronyms=functional_system.get("acronyms", []),
                    exact_acronym_match=functional_system.get("exact_acronym_match", False),
                    cell_threshold=functional_system.get("cell_threshold", 0),
                )
            write_tiered_region_report(report, report_path, target_summary)
            run.record_result("report_path", str(report_path))

        roi_results = None
        if roi_acronyms:
            roi_dir = biomarker_dir / "roi_extract"
            with RunLogger("roi_extract", biomarker_dir, config={"biomarker": name, "acronyms": roi_acronyms}) as run:
                roi_results = extract_regions_to_native(annotation, structure_df, roi_acronyms, native_shape, roi_dir)
                run.record_result(
                    "roi_results",
                    {k: str(v["output_path"]) if v["output_path"] else None for k, v in roi_results.items()},
                )

        atlas_redraw_name = f"{name}_redraw_atlas"
        native_redraw_name = f"{name}_redraw_native"
        expected_atlas_redraw = expected_output_path(biomarker_dir, atlas_redraw_name, "zarr")
        expected_native_redraw = expected_output_path(biomarker_dir, native_redraw_name, "scroll-tiff")
        recorded_fp = read_manifest(biomarker_dir).get("stages", {}).get("redraw", {}).get("fingerprints", {}).get("cells_parquet")
        if (
            expected_atlas_redraw.exists()
            and expected_native_redraw.exists()
            and fingerprint_matches(recorded_fp, cells_path)
        ):
            logger.info(
                "Skipping redraw for biomarker=%s: outputs exist and cells.parquet unchanged", name,
            )
            atlas_redraw_path = expected_atlas_redraw
            native_redraw_path = expected_native_redraw
        else:
            with RunLogger("redraw", biomarker_dir, config={"biomarker": name}) as run:
                run.record_input_fingerprint("cells_parquet", cells_path)
                atlas_redraw_path = redraw_atlas_mask(
                    cells_path, native_shape, annotation.shape, biomarker_dir, atlas_redraw_name, biomarker=name,
                )
                native_redraw_path = redraw_native_mask(
                    cells_path, native_shape, biomarker_dir, native_redraw_name, biomarker=name,
                )
                run.record_result("atlas_redraw_path", str(atlas_redraw_path))
                run.record_result("native_redraw_path", str(native_redraw_path))

        results[name] = {
            "cells_path": cells_path,
            "filter": filter_summary,
            "assign_regions": assign_summary,
            "report_path": report_path,
            "roi_extract": roi_results,
            "redraw": {"atlas": atlas_redraw_path, "native": native_redraw_path},
        }

    return results
