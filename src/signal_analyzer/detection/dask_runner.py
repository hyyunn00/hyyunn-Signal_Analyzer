"""Streams a biomarker's native-resolution mask through StreamingCC3DDetector
and writes finalized cells incrementally to the canonical cell Parquet table.

Named for the architecture plan's role ("dask_runner.py"), though the
cross-chunk connected-component merge is inherently sequential -- each
Z-chunk's result depends on the previous chunk's boundary state, which does
not admit a parallel dask.array task graph. "Dask/Zarr streaming" here means
the chunked, memory-bounded I/O path (via io.FileReader, which is
Zarr-capable), matching how MARS's own DetectStructure.run() already
processes Z-chunks sequentially. If a later phase needs the detection loop
itself to run in a Dask worker pool (e.g. one worker per biomarker/brain
processed concurrently), that's a coarser-grained parallelism this function
can be wrapped in without changing its internals.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..common.cell_schema import HEMISPHERE_NA, CellTableWriter
from ..common.config import BiomarkerConfig, BrainConfig
from ..io import FileReader
from .cc3d_detector import StreamingCC3DDetector

logger = logging.getLogger(__name__)

_WRITE_BATCH_SIZE = 50_000


def detect_biomarker(
    mask_path: str | Path,
    brain: BrainConfig,
    biomarker: BiomarkerConfig,
    output_path: str | Path,
    run_id: str,
) -> int:
    """Detect cells for one biomarker's mask and write them to a Parquet cell table.

    Runs entirely in native mask space -- detection never touches
    downsampled or axis-transposed data (per the architecture plan's
    confirmed fact that only the BIRDS registration-input volume /
    annotation.tif are ever resampled). ``region_id`` is left null and
    ``hemisphere_id`` left NA here; Phase 2/3's coordinate mapping fills
    them in for registered brains. ``passed_filter`` is left False here --
    it only becomes meaningful after ``filtering.volume_filter`` runs, and
    defaulting to False (rather than True) means a forgotten filter step
    fails safe (nothing shows up downstream) rather than silently treating
    every unfiltered cell as if it passed.

    Args:
        mask_path: Path to this biomarker's native-resolution mask (any
            format ``io.FileReader`` supports).
        brain: The brain's config (used for ``voxel_size_um`` and ``id``).
        biomarker: This biomarker's config (mask path is passed separately
            so callers can override/resolve it; ``biomarker.detection``
            supplies ``connectivity``/``z_chunk``).
        output_path: Path to the Parquet file to write cells into.
        run_id: Identifier tying these rows back to the run's log record.

    Returns:
        The total number of detected cells written.
    """
    reader = FileReader(mask_path)
    z_chunk = biomarker.detection.z_chunk
    total_z = reader.volume_shape[0]

    detector = StreamingCC3DDetector(connectivity=biomarker.detection.connectivity)

    logger.info(
        "Detecting biomarker=%s mask=%s shape=%s z_chunk=%d connectivity=%d",
        biomarker.name, mask_path, reader.volume_shape, z_chunk, biomarker.detection.connectivity,
    )

    for z_start in range(0, total_z, z_chunk):
        z_end = min(z_start + z_chunk, total_z)
        chunk = reader.read(z_start=z_start, z_end=z_end)
        detector.process_chunk(chunk, z_offset=z_start)
        logger.info("Processed z chunk [%d, %d) of %d", z_start, z_end, total_z)

    cells = detector.finalize()
    logger.info("Detection finalized: %d cells for biomarker=%s", len(cells), biomarker.name)

    voxel_volume_um3 = brain.voxel_size_um[0] * brain.voxel_size_um[1] * brain.voxel_size_um[2]

    writer = CellTableWriter(output_path)
    n_written = 0
    batch: list[dict] = []
    for cell_id, (label, coords) in enumerate(cells.items(), start=1):
        z, y, x, vol = coords
        batch.append({
            "cell_id": cell_id,
            "z": int(z),
            "y": int(y),
            "x": int(x),
            "volume_voxels": int(vol),
            "volume_um3": float(vol) * voxel_volume_um3,
            "biomarker": biomarker.name,
            "region_id": None,
            "hemisphere_id": HEMISPHERE_NA,
            "brain_id": brain.id,
            "run_id": run_id,
            "passed_filter": False,
        })
        if len(batch) >= _WRITE_BATCH_SIZE:
            n_written += writer.write_batch(batch)
            batch = []
    n_written += writer.write_batch(batch)
    writer.close()

    return n_written
