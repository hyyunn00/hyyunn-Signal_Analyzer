"""3D connected-component cell detection with streaming, memory-bounded
cross-Z-chunk merging.

Ported from MARS's 3Dfilter/filter_structure.py + filter_tools.py
(D:\\chu_lab\\MARS\\3Dfilter\\filter_structure.py,
D:\\chu_lab\\MARS\\3Dfilter\\filter_tools.py). This is the only one of the two
source repos' detectors that computes true per-cell voxel volume (Chulab's
local-maxima detector never does), which is why it was chosen as the new
pipeline's detection engine -- see the architecture plan's "Architecture
Decision" section.

Two deliberate correctness fixes over the original MARS algorithm, both
found while writing this port's regression tests (tests/test_cc3d_detector.py,
which runs MARS's own unmodified connect_calculate_plane side-by-side):

1. MARS's DetectStructure.run() never flushes cells still "open" (touching
   the very last processed Z-slice of the whole volume) once the chunk loop
   ends, so those cells are silently dropped from the XML output. This port
   adds an explicit final flush (``finalize_open_cells``, called from
   ``StreamingCC3DDetector.finalize()``).

2. MARS's original ``new_label = previous_plane.max() + 1`` scheme is NOT
   globally unique across the whole run -- it resets relative to whatever
   the immediately preceding chunk's boundary happened to contain. Whenever
   a chunk boundary has no components touching it (a common "gap" between
   cells), ``previous_plane.max()`` drops back to 0, and the next chunk's
   freshly-assigned labels can start from 1 again -- colliding with an
   earlier chunk's already-finalized global label. When that collision
   later gets finalized (whether via fix #1's flush or a later chunk
   boundary), ``registered_cell_xyz_vol[label] = ...`` silently
   **overwrites** the earlier, unrelated cell's record with the colliding
   cell's data, permanently losing it -- not just a missing cell, but
   silently wrong data for a cell that appeared to be present. This port
   fixes it by threading a single monotonically-increasing
   ``next_global_label`` counter through every chunk call (passed in,
   incremented, and returned) instead of deriving new labels from local
   boundary state, guaranteeing every label is unique for the life of the run.
"""
from __future__ import annotations

from typing import Optional

import cc3d
import numpy as np
from numba import njit, types
from numba.typed import Dict


@njit
def connect_calculate_plane(
    current_plane: np.ndarray,
    previous_plane,
    detected_cell_xyz_vol,
    registered_cell_xyz_vol,
    z_layer: int,
    next_global_label: int,
):
    """Merge one cc3d-labeled Z-chunk into the running cross-chunk cell accumulators.

    Adapted from MARS's filter_tools.py::connect_calculate_plane. The core
    per-voxel accumulation and finalization logic (running centroid sums,
    "still open at this chunk's last slice" test, integer-division centroid)
    is unchanged. What changed is how a brand-new label is assigned: MARS
    derived it from ``previous_plane.max() + 1``, which is not globally
    unique across the run (see this module's docstring, fix #2); this
    version always draws new labels from ``next_global_label``, a counter
    the caller threads through every chunk call.

    Args:
        current_plane: cc3d-labeled (uint32) block for this chunk, shape (dz, Y, X).
            Mutated in place: its last Z-slice is relabeled to global label ids
            so it can be passed back in as the next call's ``previous_plane``.
        previous_plane: the previous chunk's (already globally-relabeled) last
            Z-slice, or None for the first chunk.
        detected_cell_xyz_vol: numba Dict[int64, int64[:]] of "open" labels
            (still touching a chunk boundary) -> running
            [z_sum, y_sum, x_sum, voxel_count].
        registered_cell_xyz_vol: numba Dict[int64, int64[:]] of finalized
            cells -> [z_centroid, y_centroid, x_centroid, voxel_count].
            Mutated in place and returned; grows across the whole run.
        z_layer: global Z offset of this chunk's first slice.
        next_global_label: the next unused global label id (start at 1;
            0 is background). Threaded through calls so labels never repeat.

    Returns:
        Tuple of (new_detected_cell_xyz_vol, registered_cell_xyz_vol,
        last_slice, next_global_label) where ``last_slice``
        (current_plane[-1], globally relabeled) should be passed as
        ``previous_plane`` and ``next_global_label`` as-is to the next call.
    """
    index_map = Dict.empty(
        key_type=types.uint32,
        value_type=types.uint32,
    )

    new_detected_cell_xyz_vol = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:],
    )

    if previous_plane is not None:
        for y in range(previous_plane.shape[0]):
            for x in range(previous_plane.shape[1]):
                if current_plane[0, y, x] in index_map:
                    continue

                if previous_plane[y, x] != 0 and current_plane[0, y, x] != 0:
                    index_map[np.uint32(current_plane[0, y, x])] = np.uint32(previous_plane[y, x])

    for z in range(current_plane.shape[0]):
        for y in range(current_plane.shape[1]):
            for x in range(current_plane.shape[2]):
                local_label = current_plane[z, y, x]

                if local_label == 0:
                    continue

                if local_label in index_map:
                    current_label = index_map[local_label]
                else:
                    current_label = np.uint32(next_global_label)
                    index_map[np.uint32(local_label)] = current_label
                    next_global_label += 1

                corrected_z = z + z_layer

                if current_label in detected_cell_xyz_vol:
                    coords = detected_cell_xyz_vol[current_label]
                    coords[0] += corrected_z
                    coords[1] += y
                    coords[2] += x
                    coords[3] += 1
                else:
                    detected_cell_xyz_vol[current_label] = np.array([corrected_z, y, x, 1], dtype=np.int64)

                if z == current_plane.shape[0] - 1:
                    current_plane[z, y, x] = current_label

    unique_last_first_plane = set(np.unique(current_plane[-1]))

    for label, (z, y, x, vol) in detected_cell_xyz_vol.items():
        if label in unique_last_first_plane:
            new_detected_cell_xyz_vol[np.int64(label)] = np.array([z, y, x, vol], dtype=np.int64)
        else:
            registered_cell_xyz_vol[np.int64(label)] = np.array([z // vol, y // vol, x // vol, vol], dtype=np.int64)

    return new_detected_cell_xyz_vol, registered_cell_xyz_vol, current_plane[-1], next_global_label


def finalize_open_cells(detected_cell_xyz_vol, registered_cell_xyz_vol):
    """Flush any cells still "open" after the last chunk into ``registered_cell_xyz_vol``.

    Fixes the data-loss bug described in this module's docstring: MARS's
    original ``DetectStructure.run()`` never calls the equivalent of this,
    silently dropping cells still touching the very last Z-slice of the
    whole volume.
    """
    for label, (z, y, x, vol) in detected_cell_xyz_vol.items():
        registered_cell_xyz_vol[np.int64(label)] = np.array(
            [z // vol, y // vol, x // vol, vol], dtype=np.int64
        )
    return registered_cell_xyz_vol


class StreamingCC3DDetector:
    """Streaming 3D connected-component detector over Z-chunks.

    Runs ``cc3d.connected_components`` per chunk and merges cross-chunk
    labels via ``connect_calculate_plane``, matching MARS's
    ``DetectStructure`` algorithm (plus the final-flush fix above). Chunks
    are fed in one at a time via ``process_chunk`` -- callers are expected to
    stream them from ``io.FileReader`` against the native-resolution mask
    (detection never touches downsampled/axis-transposed data; see the
    architecture plan's confirmed facts).

    Unlike MARS's ``DetectStructure``, this does not offer the inner
    per-chunk multiprocessing split (``parallel`` > 1) -- deferred until a
    real need for it shows up; the cross-chunk merge itself is inherently
    sequential regardless.
    """

    def __init__(self, connectivity: int = 18):
        self.connectivity = connectivity
        self.detected_cell_xyz_vol = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        self.registered_cell_xyz_vol = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        self._previous_plane: Optional[np.ndarray] = None
        self._next_global_label = 1  # 0 is background

    def process_chunk(self, mask_chunk: np.ndarray, z_offset: int) -> None:
        """Process one (dz, Y, X) mask chunk starting at global Z index ``z_offset``."""
        labeled = cc3d.connected_components(
            np.asarray(mask_chunk) > 0, connectivity=self.connectivity, out_dtype=np.uint32
        )
        (
            self.detected_cell_xyz_vol,
            self.registered_cell_xyz_vol,
            self._previous_plane,
            self._next_global_label,
        ) = connect_calculate_plane(
            current_plane=labeled,
            previous_plane=self._previous_plane,
            detected_cell_xyz_vol=self.detected_cell_xyz_vol,
            registered_cell_xyz_vol=self.registered_cell_xyz_vol,
            z_layer=z_offset,
            next_global_label=self._next_global_label,
        )

    def finalize(self) -> dict[int, np.ndarray]:
        """Flush any still-open cells and return the complete {label: [z,y,x,vol]} dict.

        Safe to call at most once per detector instance (subsequent calls
        return an empty flush since ``detected_cell_xyz_vol`` is drained).
        """
        finalize_open_cells(self.detected_cell_xyz_vol, self.registered_cell_xyz_vol)
        self.detected_cell_xyz_vol = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        return dict(self.registered_cell_xyz_vol)
