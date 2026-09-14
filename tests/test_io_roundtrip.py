"""Phase 0 verification: the ported io.FileWriter/FileReader round-trip volumes
correctly, including the two-pass resize path used later by align/convert.py.
"""
import numpy as np
import pytest

from signal_analyzer.io import FileReader, FileWriter


def _make_volume(shape=(4, 16, 16), dtype=np.uint16, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    return rng.random(shape).astype(dtype)


def test_zarr_roundtrip_no_resize(tmp_path):
    volume = _make_volume(shape=(4, 16, 16), dtype=np.uint16)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="test_vol",
        output_type="zarr",
        full_res_shape=volume.shape,
        output_dtype=volume.dtype,
        chunk_size=(2, 8, 8),
    )
    writer.write(volume)

    reader = FileReader(writer.output_path)
    assert reader.volume_shape == volume.shape
    result = reader.read()
    np.testing.assert_array_equal(result, volume)


def test_zarr_roundtrip_chunked_write(tmp_path):
    """Write a volume in two Z-slabs (as detection/align stages will), then read it back whole."""
    volume = _make_volume(shape=(6, 8, 8), dtype=np.uint8)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="chunked_vol",
        output_type="zarr",
        full_res_shape=volume.shape,
        output_dtype=volume.dtype,
        chunk_size=(3, 8, 8),
    )
    writer.write(volume[0:3], z_start=0, z_end=3)
    writer.write(volume[3:6], z_start=3, z_end=6)

    reader = FileReader(writer.output_path)
    result = reader.read()
    np.testing.assert_array_equal(result, volume)


def test_single_tiff_roundtrip(tmp_path):
    volume = _make_volume(shape=(3, 12, 12), dtype=np.uint16)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="test_tiff",
        output_type="single-tiff",
        full_res_shape=volume.shape,
        output_dtype=volume.dtype,
    )
    writer.write(volume, z_start=0, z_end=3)

    tiff_path = tmp_path / "test_tiff_z0-3.tiff"
    assert tiff_path.exists()

    reader = FileReader(tiff_path)
    result = reader.read()
    np.testing.assert_array_equal(result, volume)


def test_zarr_two_pass_resize_nearest_neighbor(tmp_path):
    """Downsample by a factor of 2 per axis (order=0, nearest-neighbor) -- the same
    resampling mode align/convert.py and roi_extract's atlas<->native mapping will use
    for label-preserving volumes (annotation.tif, region masks)."""
    source_shape = (4, 8, 8)
    target_shape = (4, 4, 4)
    volume = _make_volume(shape=source_shape, dtype=np.uint8)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="resized_vol",
        output_type="zarr",
        full_res_shape=target_shape,
        input_shape=source_shape,
        output_dtype=volume.dtype,
        chunk_size=(4, 4, 4),
        resize_order=0,
        n_workers=2,
    )
    writer.write(volume, z_start=0, z_end=source_shape[0])
    writer.complete_resize()

    reader = FileReader(writer.output_path)
    assert reader.volume_shape == target_shape
    result = reader.read()
    assert result.shape == target_shape
    # Nearest-neighbor downsample must only ever produce values that were present
    # in the source volume (no interpolation artifacts).
    assert set(np.unique(result)).issubset(set(np.unique(volume)))


def test_transpose_order_swaps_axes_on_read_tiff(tmp_path):
    """Confirms the Y/Z-swap mechanism (transpose_order) works end-to-end through
    FileReader for a TIFF source -- this is the primitive regions/coord_transform.py
    will build on for the real native<->atlas mapping in Phase 2/3."""
    volume = _make_volume(shape=(2, 4, 6), dtype=np.uint8)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="transpose_tiff",
        output_type="single-tiff",
        full_res_shape=volume.shape,
        output_dtype=volume.dtype,
    )
    writer.write(volume, z_start=0, z_end=2)
    tiff_path = tmp_path / "transpose_tiff_z0-2.tiff"

    reader = FileReader(tiff_path, transpose_order=(1, 0, 2))
    assert reader.volume_shape == (4, 2, 6)
    result = reader.read()
    np.testing.assert_array_equal(result, np.transpose(volume, (1, 0, 2)))


def test_transpose_order_on_zarr_source_full_read(tmp_path):
    """Regression test for a bug found in Phase 0 (documented there via xfail,
    now fixed in Phase 2 for regions/coord_transform.py's sake): reading a
    Zarr source with an anisotropic transpose_order used to produce a shape
    mismatch / silently wrong data, because FileReader.read() sliced the raw
    (untransposed) zarr array using already-transposed logical-space bounds.
    Fixed in io/reader.py by mapping logical bounds back to native axis order
    before indexing, then transposing the (small) result."""
    volume = _make_volume(shape=(2, 4, 6), dtype=np.uint8)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="transpose_zarr",
        output_type="zarr",
        full_res_shape=volume.shape,
        output_dtype=volume.dtype,
        chunk_size=(2, 4, 6),
    )
    writer.write(volume)

    reader = FileReader(writer.output_path, transpose_order=(1, 0, 2))
    assert reader.volume_shape == (4, 2, 6)
    result = reader.read()
    np.testing.assert_array_equal(result, np.transpose(volume, (1, 0, 2)))


def test_transpose_order_on_zarr_source_partial_read(tmp_path):
    """Same as above, but reading a sub-range rather than the whole volume --
    exercises the native-bounds mapping with non-trivial (non-full) bounds."""
    volume = _make_volume(shape=(3, 5, 7), dtype=np.uint16)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="transpose_zarr_partial",
        output_type="zarr",
        full_res_shape=volume.shape,
        output_dtype=volume.dtype,
        chunk_size=(3, 5, 7),
    )
    writer.write(volume)

    reader = FileReader(writer.output_path, transpose_order=(1, 0, 2))
    assert reader.volume_shape == (5, 3, 7)

    # Read a sub-range entirely in logical (post-transpose) space.
    result = reader.read(z_start=1, z_end=4, y_start=0, y_end=2, x_start=2, x_end=6)
    expected = np.transpose(volume, (1, 0, 2))[1:4, 0:2, 2:6]
    np.testing.assert_array_equal(result, expected)


def test_transpose_order_identity_on_zarr_is_a_noop(tmp_path):
    """transpose_order=(0,1,2) (identity) must behave exactly like no transpose."""
    volume = _make_volume(shape=(3, 5, 7), dtype=np.uint8)

    writer = FileWriter(
        output_path=tmp_path,
        output_name="identity_transpose_zarr",
        output_type="zarr",
        full_res_shape=volume.shape,
        output_dtype=volume.dtype,
        chunk_size=(3, 5, 7),
    )
    writer.write(volume)

    reader = FileReader(writer.output_path, transpose_order=(0, 1, 2))
    result = reader.read()
    np.testing.assert_array_equal(result, volume)
