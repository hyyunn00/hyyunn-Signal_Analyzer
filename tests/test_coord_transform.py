"""Phase 2 verification: native<->atlas coordinate mapping, region/hemisphere
lookup, and atlas->native volume resampling.

test_native_to_atlas_matches_mars_formula cross-checks the point-mapping
math directly against MARS's inline formula from filter_annotation.py::
get_points (``MarkerZ * anno_depth // mask_depth`` etc.), confirming the
generalized, transpose_order-driven version reproduces it exactly when used
with the BIRDS (1,0,2) convention.
"""
import numpy as np

from signal_analyzer.common.cell_schema import CellTableWriter, HEMISPHERE_LEFT, HEMISPHERE_NA, read_cells
from signal_analyzer.io import FileReader, FileWriter
from signal_analyzer.regions.coord_transform import (
    assign_regions_in_cell_table,
    invert_transpose_order,
    lookup_region_and_hemisphere,
    native_to_atlas_point,
    native_to_atlas_points,
    resample_atlas_volume_to_native,
    resize_atlas_array_to_native,
)


def test_invert_transpose_order():
    assert invert_transpose_order((0, 1, 2)) == (0, 1, 2)
    assert invert_transpose_order((1, 0, 2)) == (1, 0, 2)  # self-inverse (swap)
    # A 3-cycle: order=(2,0,1) means logical[i]=native[order[i]]; its inverse
    # must satisfy inverse[order[i]] = i for all i.
    order = (2, 0, 1)
    inverse = invert_transpose_order(order)
    for i in range(3):
        assert inverse[order[i]] == i


def test_native_to_atlas_matches_mars_formula():
    """Reproduces MARS's filter_annotation.py::get_points formula directly:
    z = MarkerZ * anno_depth // mask_depth (and analogously for y, x), where
    anno_depth/anno_height/anno_width are the raw annotation.tif's shape
    components under the `transform=True` (Y/Z-swap) convention."""
    mask_depth, mask_height, mask_width = 100, 200, 300
    anno_height, anno_depth, anno_width = 40, 20, 60  # raw annotation.tif shape (transform=True naming)

    marker_z, marker_y, marker_x = 37, 123, 210

    mars_z = marker_z * anno_depth // mask_depth
    mars_y = marker_y * anno_height // mask_height
    mars_x = marker_x * anno_width // mask_width

    # logical/atlas_shape as io.FileReader(transpose_order=(1,0,2)) would report it:
    # atlas_shape[i] = raw_shape[transpose_order[i]] = raw_shape[(1,0,2)[i]]
    raw_shape = (anno_height, anno_depth, anno_width)
    transpose_order = (1, 0, 2)
    atlas_shape = tuple(raw_shape[axis] for axis in transpose_order)
    assert atlas_shape == (anno_depth, anno_height, anno_width)

    native_shape = (mask_depth, mask_height, mask_width)
    result = native_to_atlas_point((marker_z, marker_y, marker_x), native_shape, atlas_shape)

    assert result == (mars_z, mars_y, mars_x)


def test_native_to_atlas_points_vectorized_matches_scalar():
    native_shape = (100, 200, 300)
    atlas_shape = (20, 40, 60)
    points = np.array([[0, 0, 0], [50, 100, 150], [99, 199, 299]])

    vectorized = native_to_atlas_points(points, native_shape, atlas_shape)
    for i, p in enumerate(points):
        scalar = native_to_atlas_point(tuple(p), native_shape, atlas_shape)
        np.testing.assert_array_equal(vectorized[i], scalar)


def test_lookup_region_and_hemisphere_end_to_end(tmp_path):
    """Builds a small annotation.tif at the RAW (anno_height, anno_depth,
    anno_width) shape (BIRDS convention), reads it back through
    io.FileReader with transpose_order=(1,0,2) (now fixed in Phase 2), and
    confirms native-space points resolve to the correct region/hemisphere."""
    native_shape = (10, 10, 10)
    transpose_order = (1, 0, 2)

    # Raw annotation.tif shape: (anno_height=5, anno_depth=5, anno_width=5) --
    # a 2x downsample of the native shape in every axis, BIRDS-style naming.
    raw_annotation = np.zeros((5, 5, 5), dtype=np.uint32)
    # Region 42 occupies the "left" half (low X), region 99 the "right" half,
    # in the RAW file's own axis order (height, depth, width).
    raw_annotation[:, :, 0:2] = 42
    raw_annotation[:, :, 2:5] = 99

    hemisphere_raw = np.zeros((5, 5, 5), dtype=np.uint8)
    hemisphere_raw[:, :, 0:2] = HEMISPHERE_LEFT

    def _write_and_reload(raw_array, name, dtype):
        writer = FileWriter(
            output_path=tmp_path, output_name=name, output_type="zarr",
            full_res_shape=raw_array.shape, output_dtype=dtype, chunk_size=raw_array.shape,
        )
        writer.write(raw_array)
        return FileReader(writer.output_path, transpose_order=transpose_order)

    anno_reader = _write_and_reload(raw_annotation, "anno", np.uint32)
    hema_reader = _write_and_reload(hemisphere_raw, "hema", np.uint8)

    annotation_logical = anno_reader.read()
    hemisphere_logical = hema_reader.read()
    assert annotation_logical.shape == (5, 5, 5)  # isotropic here, but now correctly indexed

    native_points = np.array([
        [1, 1, 1],  # -> atlas ~(0,0,0) region 42 (X<2 in raw), left
        [9, 9, 9],  # -> atlas ~(4,4,4) region 99 (X>=2 in raw), right
    ])
    region_ids, hemisphere_ids = lookup_region_and_hemisphere(
        native_points, native_shape, annotation_logical, hemisphere_logical
    )

    assert region_ids[0] == 42
    assert hemisphere_ids[0] == HEMISPHERE_LEFT
    assert region_ids[1] == 99
    assert hemisphere_ids[1] == HEMISPHERE_NA  # not explicitly set to RIGHT above


def test_lookup_region_without_hemisphere_returns_na():
    native_shape = (10, 10, 10)
    annotation = np.full((5, 5, 5), 7, dtype=np.uint32)
    points = np.array([[0, 0, 0], [9, 9, 9]])
    region_ids, hemisphere_ids = lookup_region_and_hemisphere(points, native_shape, annotation, hemisphere=None)
    assert (region_ids == 7).all()
    assert (hemisphere_ids == HEMISPHERE_NA).all()


def test_lookup_clamps_out_of_bounds_points():
    native_shape = (10, 10, 10)
    annotation = np.arange(1, 1 + 5 * 5 * 5, dtype=np.uint32).reshape(5, 5, 5)
    # A point right at the native edge should clamp into bounds, not raise.
    points = np.array([[9, 9, 9]])
    region_ids, _ = lookup_region_and_hemisphere(points, native_shape, annotation)
    assert region_ids[0] == annotation[-1, -1, -1]


def test_assign_regions_in_cell_table(tmp_path):
    native_shape = (10, 10, 10)
    annotation = np.zeros((5, 5, 5), dtype=np.uint32)
    annotation[:, :, 0:2] = 42
    annotation[:, :, 2:5] = 99

    cells_path = tmp_path / "cells.parquet"
    with CellTableWriter(cells_path) as writer:
        writer.write_batch([
            {
                "cell_id": 1, "z": 1, "y": 1, "x": 1, "volume_voxels": 10, "volume_um3": 1.0,
                "biomarker": "cFos", "region_id": None, "hemisphere_id": HEMISPHERE_NA,
                "brain_id": "B1", "run_id": "r1", "passed_filter": True,
            },
            {
                "cell_id": 2, "z": 9, "y": 9, "x": 9, "volume_voxels": 10, "volume_um3": 1.0,
                "biomarker": "cFos", "region_id": None, "hemisphere_id": HEMISPHERE_NA,
                "brain_id": "B1", "run_id": "r1", "passed_filter": True,
            },
            {
                # Different biomarker: must be left untouched when biomarker="cFos" is passed.
                "cell_id": 3, "z": 1, "y": 1, "x": 1, "volume_voxels": 10, "volume_um3": 1.0,
                "biomarker": "TH", "region_id": None, "hemisphere_id": HEMISPHERE_NA,
                "brain_id": "B1", "run_id": "r1", "passed_filter": True,
            },
        ])

    summary = assign_regions_in_cell_table(cells_path, native_shape, annotation, biomarker="cFos")
    assert summary == {"assigned": 2}

    df = read_cells(cells_path)
    row1 = df[df["cell_id"] == 1].iloc[0]
    row2 = df[df["cell_id"] == 2].iloc[0]
    row3 = df[df["cell_id"] == 3].iloc[0]

    assert row1["region_id"] == 42
    assert row2["region_id"] == 99
    assert pd_isna(row3["region_id"])  # untouched: still null


def pd_isna(value):
    import pandas as pd
    return pd.isna(value)


def test_resample_atlas_volume_to_native_uses_transpose_order_not_its_inverse(tmp_path):
    """Direct proof the fixed direction is correct, using a genuinely
    non-self-inverse permutation (a 3-cycle) -- config validation now
    rejects such a transpose_order for real runs (see test_config.py::
    test_transpose_order_must_be_self_inverse), but the underlying function
    should still be mathematically self-consistent with the rest of the
    module's established convention (io.FileReader(path,
    transpose_order=transpose_order) -> logical/native-matching order,
    used identically by lookup_region_and_hemisphere) regardless.

    An earlier, buggy version of resample_atlas_volume_to_native read with
    invert_transpose_order(transpose_order) instead -- for this 3-cycle,
    that produces a DIFFERENT (wrong) logical array, so this test would
    have failed against that version."""
    transpose_order = (2, 0, 1)  # a genuine 3-cycle, not self-inverse
    raw = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)

    writer = FileWriter(
        output_path=tmp_path, output_name="raw_3cycle", output_type="zarr",
        full_res_shape=raw.shape, output_dtype=np.uint32, chunk_size=raw.shape,
    )
    writer.write(raw)

    # The established convention (used by lookup_region_and_hemisphere and
    # everywhere else): reading with transpose_order directly gives the
    # correct logical/native-matching array.
    logical = FileReader(writer.output_path, transpose_order=transpose_order).read()
    assert logical.shape == (4, 2, 3)
    np.testing.assert_array_equal(logical, np.transpose(raw, transpose_order))

    native_shape = (8, 4, 6)  # 2x upsample of the logical shape on every axis
    direct_path = resize_atlas_array_to_native(
        logical, native_shape, tmp_path, "direct_resize", resize_order=0, chunk_size=native_shape,
    )
    via_resample_path = resample_atlas_volume_to_native(
        atlas_path=writer.output_path, transpose_order=transpose_order, native_shape=native_shape,
        output_path=tmp_path, output_name="via_resample", resize_order=0, chunk_size=native_shape,
    )

    direct_result = FileReader(direct_path).read()
    via_resample_result = FileReader(via_resample_path).read()
    np.testing.assert_array_equal(direct_result, via_resample_result)


def test_resample_atlas_volume_to_native(tmp_path):
    """Round-trips an atlas-space labeled volume back to native resolution
    via the inverse-transpose + upsample composition."""
    transpose_order = (1, 0, 2)
    native_shape = (10, 10, 10)

    # Atlas-space volume (already in logical Z,Y,X-matching order, as if it
    # were read from a raw annotation.tif via FileReader(transpose_order=...)),
    # half labeled 5, half labeled 6, split along the (atlas) X axis.
    atlas_volume = np.zeros((5, 5, 5), dtype=np.uint32)
    atlas_volume[:, :, 0:2] = 5
    atlas_volume[:, :, 2:5] = 6

    # To round-trip through resample_atlas_volume_to_native (which expects a
    # path to the RAW, transpose_order-oriented atlas file), first write the
    # atlas volume out in its RAW (pre-transpose_order) orientation.
    raw_atlas = np.transpose(atlas_volume, invert_transpose_order(transpose_order))
    writer = FileWriter(
        output_path=tmp_path, output_name="raw_atlas", output_type="zarr",
        full_res_shape=raw_atlas.shape, output_dtype=np.uint32, chunk_size=raw_atlas.shape,
    )
    writer.write(raw_atlas)

    out_path = resample_atlas_volume_to_native(
        atlas_path=writer.output_path,
        transpose_order=transpose_order,
        native_shape=native_shape,
        output_path=tmp_path,
        output_name="resampled_native",
        resize_order=0,
        chunk_size=native_shape,
    )

    result = FileReader(out_path).read()
    assert result.shape == native_shape
    # Nearest-neighbor upsample: only the original labels {0,5,6} should appear.
    assert set(np.unique(result)).issubset({0, 5, 6})
    # The X-axis split should still roughly separate labels 5 (low X) and 6 (high X).
    assert result[5, 5, 1] == 5
    assert result[5, 5, 8] == 6
