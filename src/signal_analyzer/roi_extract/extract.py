"""Brain-region extraction: pull one or more Allen CCF regions (+ all
descendants) out of an annotation volume, resampled to native mask
resolution so the output is directly usable alongside the source image.

Ported from MARS's aba2roi.py (D:\\chu_lab\\MARS\\aba2roi.py). Two
differences from MARS's original, both enabled by regions.structures and
regions.coord_transform (Phase 2/3):
  1. Operates directly on an already-loaded, logical-order annotation array
     (typically small -- it's the downsampled BIRDS-registration output) --
     no behavioral difference for realistic annotation sizes, just
     decoupled from a per-call ``tifffile.imread`` of the whole atlas.
  2. Unlike MARS's aba2roi.py, the extracted region mask is then resampled
     atlas->native (``regions.coord_transform.resize_atlas_array_to_native``)
     before being written out, so its dimensions match the original raw/
     native image -- this is what makes the output directly usable alongside
     the source image (e.g. Imaris "Classify by Intensity"), per the user's
     clarification of why this step exists at all.

``extract_region_to_native``'s ``skip_existing`` (default True) reuses an
already-extracted region's output file instead of re-running the relabel +
upsample-to-native resample (an expensive full-volume operation). This is
existence-based only, deliberately NOT staleness-aware against the
``annotation``/``structures_csv`` inputs: unlike ``cells.parquet`` (which
this pipeline itself rewrites every run, so its mtime is a reliable
staleness signal), the annotation/structures inputs are external and loaded
once per invocation, never mutated by this pipeline. A genuine atlas-input
change is treated the same way a genuine mask/detection-parameter change is
for ``detect`` (see ``pipelines.run_with_registration``'s module docstring):
the user points at a fresh ``output_dir`` or clears the stale
``roi_extract/`` folder, rather than a heuristic silently re-triggering a
costly resample on a false positive.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..io import expected_output_path
from ..regions.coord_transform import resize_atlas_array_to_native
from ..regions.structures import get_region_info

logger = logging.getLogger(__name__)


def extract_region_mask_atlas_space(
    annotation: np.ndarray,
    region_ids: list[int],
    target_id: int,
) -> np.ndarray:
    """Build a relabeled mask (atlas space) for one region + its descendants.

    All matched voxels are relabeled to ``target_id`` (not kept as their
    individual descendant ids) -- matches MARS's aba2roi.py behavior.
    """
    mask = np.isin(annotation, region_ids)
    region_image = np.zeros(annotation.shape, dtype=np.uint32)
    region_image[mask] = target_id
    return region_image


def extract_region_to_native(
    annotation: np.ndarray,
    structure_df: pd.DataFrame,
    acronym: str,
    native_shape: tuple[int, int, int],
    output_path: str | Path,
    output_name: Optional[str] = None,
    resize_order: int = 0,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    n_workers: int = 8,
    output_type: str = "single-tiff",
    skip_existing: bool = True,
) -> Optional[Path]:
    """Extract one region (+ descendants) and resample it to native resolution.

    Args:
        annotation: Annotation volume, already in logical (Z,Y,X)-matching
            axis order (e.g. via ``io.FileReader(path, transpose_order=...)``).
        structure_df: The Allen CCF structures table.
        acronym: Target region acronym.
        native_shape: The native mask's (Z,Y,X) shape to resample into.
        output_path: Directory to write the output into.
        output_name: Base name for the output; defaults to ``f"{acronym}_atlas"``.
        resize_order: skimage interpolation order; 0 (nearest-neighbor) by
            default since this is a region-id/label mask, not intensity data.
        output_type: Output format, default ``'single-tiff'`` -- a single
            multi-page ``.tiff`` at native resolution, directly openable in
            Fiji alongside the original raw image (they share dimensions).
        skip_existing: If the expected output already exists, skip
            re-extracting/re-resampling this region entirely and return its
            path. This is a genuinely expensive full-volume operation (atlas
            relabel + upsample-to-native resample), the same cost class as
            detection, so it gets the same kind of existence-based skip.
            Not staleness-aware (see ``roi_extract.extract`` module note):
            a changed ``annotation``/``structures_csv`` between runs is not
            auto-detected here, by design.

    Returns:
        Path to the native-resolution, relabeled region mask, or ``None``
        if the region has no voxels in this annotation volume (matches
        MARS's aba2roi.py, which skips writing a file in that case too).
    """
    output_name = output_name or f"{acronym}_atlas"
    expected = expected_output_path(output_path, output_name, output_type)
    if skip_existing and expected.exists():
        logger.info("Skipping roi_extract for '%s': %s already exists", acronym, expected)
        return expected

    original_id, all_ids, region_name = get_region_info(structure_df, acronym)

    if not np.any(np.isin(annotation, all_ids)):
        logger.warning(
            "Region '%s' (id=%d, %s) has no voxels in this annotation volume; skipping.",
            acronym, original_id, region_name,
        )
        return None

    atlas_mask = extract_region_mask_atlas_space(annotation, all_ids, original_id)

    tmp_output_name = f".tmp-{uuid.uuid4().hex[:8]}-{output_name}"
    try:
        tmp_result = resize_atlas_array_to_native(
            atlas_mask, native_shape, output_path, tmp_output_name,
            resize_order=resize_order, chunk_size=chunk_size, n_workers=n_workers,
            output_type=output_type,
        )
        # `expected` normally doesn't exist yet here (the skip_existing check
        # above already returned early if it did) -- guarded anyway for the
        # skip_existing=False case, since os.replace can't target an
        # existing directory on Windows (e.g. a 'zarr' output_type).
        if expected.is_dir():
            shutil.rmtree(expected)
        elif expected.exists():
            expected.unlink()
        tmp_result.replace(expected)
    except BaseException:
        _cleanup_stray_tmp_output(output_path, tmp_output_name)
        raise
    return expected


def _cleanup_stray_tmp_output(output_path: str | Path, tmp_output_name: str) -> None:
    """Best-effort removal of any partial output left under a mangled temp name.

    Matches anywhere in the name (not just as a prefix): ``resize_atlas_
    array_to_native``'s own intermediate Zarr directory is named
    ``_tmp_resize_<tmp_output_name>``, prefixed rather than starting with
    ``tmp_output_name`` itself.
    """
    for candidate in Path(output_path).glob(f"*{tmp_output_name}*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        else:
            candidate.unlink(missing_ok=True)


def extract_regions_to_native(
    annotation: np.ndarray,
    structure_df: pd.DataFrame,
    acronyms: list[str],
    native_shape: tuple[int, int, int],
    output_path: str | Path,
    resize_order: int = 0,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    n_workers: int = 8,
    output_type: str = "single-tiff",
    skip_existing: bool = True,
) -> dict[str, dict]:
    """Extract several regions in one call.

    Ported from MARS's aba2roi.py::run_separate loop -- generalized to a
    plain returned dict here (callers using common.run_logger get an
    equivalent, uniform run record for free via ``record_result``, rather
    than MARS's bespoke per-tool lookup-table txt file).

    Returns:
        ``{acronym: {"id": int, "name": str, "output_path": Path | None}}``
        for each requested acronym; entries with ``output_path=None`` had no
        voxels in this annotation volume (skipped, matches MARS's behavior).
        Unknown acronyms are logged and omitted, matching MARS's per-acronym
        try/except that doesn't abort the whole batch.
    """
    results: dict[str, dict] = {}
    for acronym in acronyms:
        try:
            original_id, all_ids, region_name = get_region_info(structure_df, acronym)
        except ValueError as e:
            logger.warning("Skipping '%s': %s", acronym, e)
            continue

        out_path = extract_region_to_native(
            annotation, structure_df, acronym, native_shape, output_path,
            resize_order=resize_order, chunk_size=chunk_size, n_workers=n_workers,
            output_type=output_type, skip_existing=skip_existing,
        )
        results[acronym] = {"id": original_id, "name": region_name, "output_path": out_path}

    return results
