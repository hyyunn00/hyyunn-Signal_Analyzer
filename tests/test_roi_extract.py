"""Phase 3 verification: region extraction, resampled to native resolution
so the output is directly usable alongside the source image (the user's
stated reason this step exists at all)."""
import numpy as np
import pandas as pd
import pytest

from signal_analyzer.io import FileReader
from signal_analyzer.roi_extract.extract import (
    extract_region_mask_atlas_space,
    extract_region_to_native,
    extract_regions_to_native,
)


@pytest.fixture
def tiny_structures() -> pd.DataFrame:
    return pd.DataFrame([
        {"acronym": "ROOT", "id": 1, "name": "Root", "structure_id_path": "/1/", "parent_structure_id": np.nan},
        {"acronym": "A", "id": 2, "name": "Region A", "structure_id_path": "/1/2/", "parent_structure_id": 1.0},
        {"acronym": "A1", "id": 3, "name": "Region A sub1", "structure_id_path": "/1/2/3/", "parent_structure_id": 2.0},
        {"acronym": "B", "id": 4, "name": "Region B", "structure_id_path": "/1/4/", "parent_structure_id": 1.0},
    ])


def test_extract_region_mask_atlas_space_relabels_descendants_to_target_id():
    annotation = np.zeros((2, 4, 4), dtype=np.uint32)
    annotation[:, 0:2, :] = 3  # A1
    annotation[:, 2:4, :] = 4  # B (not in region_ids below)

    mask = extract_region_mask_atlas_space(annotation, region_ids=[2, 3], target_id=2)

    assert (mask[:, 0:2, :] == 2).all()  # A1 relabeled to A's id (2)
    assert (mask[:, 2:4, :] == 0).all()  # B excluded, background


def test_extract_region_to_native_upsamples_to_native_shape(tmp_path, tiny_structures):
    atlas_shape = (2, 4, 4)
    native_shape = (4, 8, 8)  # 2x upsample every axis

    annotation = np.zeros(atlas_shape, dtype=np.uint32)
    annotation[:, 0:2, :] = 3  # A1

    out_path = extract_region_to_native(
        annotation, tiny_structures, "A", native_shape, tmp_path,
        resize_order=0, chunk_size=native_shape,
    )
    assert out_path is not None

    result = FileReader(out_path).read()
    assert result.shape == native_shape  # matches the ORIGINAL/native image's dimensions
    assert set(np.unique(result)).issubset({0, 2})  # relabeled to A's id (2), nearest-neighbor only
    assert (result > 0).any()


def test_extract_region_to_native_returns_none_for_empty_region(tmp_path, tiny_structures):
    annotation = np.zeros((2, 4, 4), dtype=np.uint32)
    annotation[:, :, :] = 4  # only B present, no A/A1 voxels

    out_path = extract_region_to_native(
        annotation, tiny_structures, "A", (4, 8, 8), tmp_path, chunk_size=(4, 8, 8),
    )
    assert out_path is None


def test_extract_regions_to_native_batch(tmp_path, tiny_structures):
    atlas_shape = (2, 4, 4)
    native_shape = (2, 4, 4)
    annotation = np.zeros(atlas_shape, dtype=np.uint32)
    annotation[:, 0:2, :] = 3  # A1
    annotation[:, 2:4, :] = 4  # B

    results = extract_regions_to_native(
        annotation, tiny_structures, ["A", "B", "NOT_REAL"], native_shape, tmp_path,
        chunk_size=native_shape,
    )

    assert set(results.keys()) == {"A", "B"}  # unknown acronym skipped, not raised
    assert results["A"]["id"] == 2
    assert results["A"]["output_path"] is not None
    assert results["B"]["id"] == 4
    assert results["B"]["output_path"] is not None
