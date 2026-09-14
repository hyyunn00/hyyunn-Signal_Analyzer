"""Regression tests for the ported cc3d-based detector.

Two bugs were found in MARS's original filter_tools.py::connect_calculate_plane
while writing these tests (see signal_analyzer/detection/cc3d_detector.py's
module docstring for the full explanation):

1. Cells still "open" (touching the very last Z-slice of the whole volume)
   are silently dropped -- MARS's DetectStructure.run() never flushes them.
2. MARS's new-label counter (``previous_plane.max() + 1``) is NOT globally
   unique across the run: whenever a chunk boundary has no active
   components (a common "gap" between cells), the counter resets low enough
   to collide with an earlier, already-finalized cell's label. When the
   colliding cell later finalizes, it silently OVERWRITES the earlier
   cell's record in ``registered_cell_xyz_vol`` -- not a missing cell, but
   silently wrong data for a cell that looks present.

These tests load MARS's actual, unmodified filter_tools.py by file path (not
via sys.path package import, since MARS's own modules use bare
same-directory imports) so the comparison is against the real source, and
they reproduce bug #2 concretely against it before checking that the ported
version avoids both.
"""
import importlib.util
from pathlib import Path

import cc3d
import numpy as np
import pytest
from numba import types
from numba.typed import Dict

from signal_analyzer.detection.cc3d_detector import (
    StreamingCC3DDetector,
    connect_calculate_plane as ported_connect_calculate_plane,
    finalize_open_cells,
)

MARS_FILTER_TOOLS_PATH = Path(r"D:\chu_lab\MARS\3Dfilter\filter_tools.py")


def _load_mars_connect_calculate_plane():
    if not MARS_FILTER_TOOLS_PATH.exists():
        pytest.skip(f"MARS repo not available at {MARS_FILTER_TOOLS_PATH}; skipping cross-validation")
    spec = importlib.util.spec_from_file_location("mars_filter_tools", MARS_FILTER_TOOLS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.connect_calculate_plane


def _make_synthetic_volume(total_z=10):
    """(total_z, 8, 8) volume with three solid-cube blobs and a deliberate
    "gap" (all-zero chunk boundary) between B and C:
      - A: z[0:2], y[0:2], x[0:2] (8 voxels) -- fully inside the first chunk.
      - B: z[3:6], y[4:6], x[4:6] (12 voxels) -- spans the chunk boundary at
        z=4 (when z_chunk=4), exercising the cross-chunk label merge.
      - C: z[8:10], y[0:2], x[6:8] (8 voxels) -- starts right after an
        all-zero chunk boundary (z=7 has nothing), exercising the label-
        collision bug; with total_z=10, C also touches the volume's final
        slice, additionally exercising the final-flush bug.
    """
    vol = np.zeros((total_z, 8, 8), dtype=np.bool_)
    vol[0:2, 0:2, 0:2] = True
    vol[3:6, 4:6, 4:6] = True
    vol[8:10, 0:2, 6:8] = True
    return vol


def _run_streaming_mars(connect_fn, volume, z_chunk, connectivity=18):
    """Replays DetectStructure.run()'s chunking loop with MARS's original
    (previous_plane.max()+1 based) signature."""
    detected = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    registered = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    previous_plane = None

    total_z = volume.shape[0]
    for z0 in range(0, total_z, z_chunk):
        z1 = min(z0 + z_chunk, total_z)
        labeled = cc3d.connected_components(volume[z0:z1], connectivity=connectivity, out_dtype=np.uint32)
        detected, registered, previous_plane = connect_fn(
            current_plane=labeled,
            previous_plane=previous_plane,
            detected_cell_xyz_vol=detected,
            registered_cell_xyz_vol=registered,
            z_layer=z0,
        )

    return dict(registered), dict(detected)


def _run_streaming_ported(volume, z_chunk, connectivity=18):
    detector = StreamingCC3DDetector(connectivity=connectivity)
    total_z = volume.shape[0]
    for z0 in range(0, total_z, z_chunk):
        z1 = min(z0 + z_chunk, total_z)
        detector.process_chunk(volume[z0:z1], z_offset=z0)
    return detector


def test_mars_original_silently_drops_the_final_boundary_cell():
    mars_connect = _load_mars_connect_calculate_plane()
    volume = _make_synthetic_volume(total_z=10)

    registered, still_open = _run_streaming_mars(mars_connect, volume, z_chunk=4)

    # Blob C never finalizes: it's left dangling in `still_open` and does
    # NOT appear in `registered` at all -- silent data loss, reproduced
    # against MARS's actual unmodified code.
    assert len(registered) == 2
    assert len(still_open) == 1


def test_mars_original_would_corrupt_data_under_naive_flush():
    """Confirms bug #2: even if you patch in a final flush without also
    fixing the label-numbering scheme, MARS's original algorithm corrupts
    data via a label collision rather than merely dropping a cell."""
    mars_connect = _load_mars_connect_calculate_plane()
    volume = _make_synthetic_volume(total_z=10)

    registered, still_open = _run_streaming_mars(mars_connect, volume, z_chunk=4)
    assert len(registered) == 2  # blobs A and B, correctly finalized so far

    # Reuses MARS's own centroid formula (see finalize_open_cells) to flush
    # the still-open dict the same way this port's fix #1 would.
    for label, (z, y, x, vol) in still_open.items():
        registered[label] = np.array([z // vol, y // vol, x // vol, vol], dtype=np.int64)

    # A naive flush does NOT recover 3 distinct cells: blob C's label
    # collides with blob A's (both assigned global label 1, because the
    # chunk boundary between B and C was all-zero, resetting MARS's
    # `previous_plane.max()+1` counter back down to 1) -- so C's data
    # silently overwrote A's registered entry instead of adding a third one.
    assert len(registered) == 2
    volumes = sorted(int(coords[3]) for coords in registered.values())
    # Blob A (8 voxels) is GONE -- overwritten by blob C (also 8 voxels,
    # coincidentally the same size, making the corruption doubly silent).
    assert volumes == [8, 12]


def test_ported_detector_avoids_both_bugs():
    volume = _make_synthetic_volume(total_z=10)
    detector = _run_streaming_ported(volume, z_chunk=4)
    cells = detector.finalize()

    # All three blobs present, each with its own distinct label -- no drop,
    # no collision/overwrite.
    assert len(cells) == 3
    volumes = sorted(int(coords[3]) for coords in cells.values())
    assert volumes == [8, 8, 12]


def test_ported_algorithm_matches_mars_original_when_no_collision_is_possible():
    """Sanity check that the port isn't just "different" -- for a volume with
    no label-reset gap (so MARS's original scheme would have been globally
    unique anyway), the two algorithms must produce IDENTICAL registered
    cells, confirming the per-voxel accumulation/centroid logic itself is
    an unchanged, faithful port."""
    mars_connect = _load_mars_connect_calculate_plane()

    # Single blob spanning a chunk boundary, no gap chunk, single chunk size
    # covering the whole volume in two pieces -- previous_plane.max() stays
    # meaningfully monotonic throughout since there's only ever one component.
    vol = np.zeros((6, 6, 6), dtype=np.bool_)
    vol[1:4, 1:4, 1:4] = True  # one 3x3x3=27 voxel blob, spanning z=1..3

    mars_registered, mars_open = _run_streaming_mars(mars_connect, vol, z_chunk=3)
    detector = _run_streaming_ported(vol, z_chunk=3)
    ported_registered = dict(detector.registered_cell_xyz_vol)
    ported_open = dict(detector.detected_cell_xyz_vol)

    # The single blob finalizes within chunk 2 (it doesn't reach chunk 2's
    # last slice), so nothing should be left open by either implementation.
    assert mars_open == {}
    assert ported_open == {}
    assert len(mars_registered) == len(ported_registered) == 1
    mars_coords = next(iter(mars_registered.values()))
    ported_coords = next(iter(ported_registered.values()))
    np.testing.assert_array_equal(mars_coords, ported_coords)


def test_chunking_does_not_change_result_once_finalized():
    """Whether processed as one chunk or several, the ported algorithm must
    produce the same finalized cells once flushed."""
    volume = _make_synthetic_volume(total_z=12)  # extra empty slices after C
    # so it finalizes mid-run rather than needing the flush, exercising a
    # different code path than test_ported_detector_avoids_both_bugs.

    detector_single = _run_streaming_ported(volume, z_chunk=12)
    cells_single = detector_single.finalize()

    detector_multi = _run_streaming_ported(volume, z_chunk=3)
    cells_multi = detector_multi.finalize()

    def _sorted_coords(cells):
        return sorted(tuple(int(v) for v in coords) for coords in cells.values())

    assert _sorted_coords(cells_single) == _sorted_coords(cells_multi)
    assert len(cells_single) == 3
