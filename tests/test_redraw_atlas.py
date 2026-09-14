"""Phase 3 verification: atlas-space redraw, reproducing MARS's x2xz.py
behavior via the shared native_to_atlas_points utility."""
import numpy as np

from signal_analyzer.common.cell_schema import CellTableWriter
from signal_analyzer.io import FileReader
from signal_analyzer.redraw.mask_from_cells import redraw_atlas_mask


def _cell_row(cell_id, z, y, x, biomarker="cFos"):
    return {
        "cell_id": cell_id, "z": z, "y": y, "x": x, "volume_voxels": 30, "volume_um3": 1.0,
        "biomarker": biomarker, "region_id": None, "hemisphere_id": 0,
        "brain_id": "B1", "run_id": "r1", "passed_filter": True,
    }


def test_redraw_atlas_mask_maps_native_points_into_atlas_space(tmp_path):
    native_shape = (20, 20, 20)
    atlas_shape = (10, 10, 10)  # 2x downsample every axis

    cells_path = tmp_path / "cells.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([
            _cell_row(1, z=4, y=4, x=4),    # -> atlas (2,2,2)
            _cell_row(2, z=18, y=18, x=18),  # -> atlas (9,9,9)
        ])

    out_path = redraw_atlas_mask(
        cells_path, native_shape, atlas_shape, tmp_path, "atlas_redraw",
        chunk_size=atlas_shape,
    )
    result = FileReader(out_path).read()

    assert result.shape == atlas_shape
    assert result[2, 2, 2] > 0
    assert result[9, 9, 9] > 0
    assert (result > 0).sum() == 2


def test_redraw_atlas_mask_drops_out_of_bounds_points(tmp_path):
    native_shape = (10, 10, 10)
    atlas_shape = (5, 5, 5)

    cells_path = tmp_path / "cells.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([
            _cell_row(1, z=0, y=0, x=0),      # valid: -> atlas (0,0,0)
            _cell_row(2, z=15, y=0, x=0),      # z outside native_shape[0]=10 -> maps
                                                # to atlas z=(15*5)//10=7, >= atlas_shape[0]=5 -> dropped
        ])

    out_path = redraw_atlas_mask(
        cells_path, native_shape, atlas_shape, tmp_path, "atlas_redraw_bounds",
        chunk_size=atlas_shape,
    )
    result = FileReader(out_path).read()
    assert result.shape == atlas_shape
    assert (result > 0).sum() == 1  # only the in-bounds point survives


def test_redraw_atlas_mask_empty_cells_produces_zero_volume(tmp_path):
    native_shape = (10, 10, 10)
    atlas_shape = (5, 5, 5)

    # A different biomarker's row, so the file exists but filtering by
    # biomarker="cFos" below yields a genuinely empty selection.
    cells_path = tmp_path / "cells.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([_cell_row(1, z=0, y=0, x=0, biomarker="TH")])

    out_path = redraw_atlas_mask(
        cells_path, native_shape, atlas_shape, tmp_path, "atlas_redraw_empty",
        biomarker="cFos", chunk_size=atlas_shape,
    )
    result = FileReader(out_path).read()
    assert result.shape == atlas_shape
    assert (result > 0).sum() == 0
