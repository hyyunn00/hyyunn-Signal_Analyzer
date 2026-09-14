"""End-to-end Phase 3 verification: the full "brain type A" (registered)
pipeline -- detect (native, unchanged from Phase 1) -> filter -> assign
regions (Phase 2's coord_transform) -> tiered region report + asymmetry ->
roi_extract -> atlas-space + native-space redraw -- run entirely through
the public module APIs, the same way a real registered brain would be
processed once BIRDS (external, out of scope) has already produced
annotation.tif.
"""
import numpy as np
import pandas as pd

from signal_analyzer.common.cell_schema import read_cells
from signal_analyzer.common.config import BiomarkerConfig, BrainConfig, DetectionConfig, FilterConfig
from signal_analyzer.detection.dask_runner import detect_biomarker
from signal_analyzer.filtering.volume_filter import apply_volume_filter
from signal_analyzer.io import FileReader, FileWriter
from signal_analyzer.redraw.mask_from_cells import redraw_atlas_mask, redraw_native_mask
from signal_analyzer.regions.coord_transform import assign_regions_in_cell_table
from signal_analyzer.report.cell_report import tiered_cell_report
from signal_analyzer.report.region_stats import build_region_geometry_report
from signal_analyzer.roi_extract.extract import extract_region_to_native

NATIVE_SHAPE = (12, 20, 20)
TRANSPOSE_ORDER = (1, 0, 2)
# RAW (on-disk, BIRDS-convention) annotation shape: a downsampled native
# shape (4,5,5) with its first two axes swapped -> (5,4,5).
RAW_ANNOTATION_SHAPE = (5, 4, 5)


def _structures() -> pd.DataFrame:
    return pd.DataFrame([
        {"acronym": "ROOT", "id": 1, "name": "Root", "structure_id_path": "/1/", "parent_structure_id": np.nan},
        {"acronym": "A", "id": 10, "name": "Region A", "structure_id_path": "/1/10/", "parent_structure_id": 1.0},
        {"acronym": "B", "id": 20, "name": "Region B", "structure_id_path": "/1/20/", "parent_structure_id": 1.0},
    ])


def _write_synthetic_mask_and_annotation(tmp_path):
    """Three blobs (biomarker cFos):
      - blob_A: z[5:9] y[5:9] x[1:5] (64 voxels) -- centroid x~2, maps into
        atlas x=0 -> region A.
      - blob_B: z[5:9] y[5:9] x[15:19] (64 voxels) -- centroid x~16, maps
        into atlas x=4 -> region B.
      - blob_small: z[0:2] y[0:2] x[0:2] (8 voxels) -- below the volume
        filter's min bound, excluded regardless of region.
    """
    volume = np.zeros(NATIVE_SHAPE, dtype=np.uint8)
    volume[5:9, 5:9, 1:5] = 1
    volume[5:9, 5:9, 15:19] = 1
    volume[0:2, 0:2, 0:2] = 1

    mask_writer = FileWriter(
        output_path=tmp_path, output_name="synthetic_mask", output_type="zarr",
        full_res_shape=NATIVE_SHAPE, output_dtype=np.uint8, chunk_size=(4, 20, 20),
    )
    mask_writer.write(volume)

    # RAW annotation, BIRDS convention: X untouched by the (1,0,2) swap, so
    # splitting by raw X directly gives the same region split logical-space
    # points will resolve to after FileReader(transpose_order=(1,0,2)).
    raw_annotation = np.zeros(RAW_ANNOTATION_SHAPE, dtype=np.uint32)
    raw_annotation[:, :, 0:2] = 10  # region A: atlas x in [0,2)
    raw_annotation[:, :, 2:5] = 20  # region B: atlas x in [2,5)

    anno_writer = FileWriter(
        output_path=tmp_path, output_name="raw_annotation", output_type="zarr",
        full_res_shape=RAW_ANNOTATION_SHAPE, output_dtype=np.uint32, chunk_size=RAW_ANNOTATION_SHAPE,
    )
    anno_writer.write(raw_annotation)

    return mask_writer.output_path, anno_writer.output_path


def test_full_brain_type_a_pipeline(tmp_path):
    mask_path, raw_annotation_path = _write_synthetic_mask_and_annotation(tmp_path)
    structure_df = _structures()

    brain = BrainConfig(id="TestBrain-A", voxel_size_um=(4.0, 1.82, 1.82), needs_registration=True)
    biomarker = BiomarkerConfig(
        name="cFos", mask_path=str(mask_path),
        detection=DetectionConfig(connectivity=18, z_chunk=5),
        filter=FilterConfig(min_volume_voxels=30, max_volume_voxels=100),
    )

    cells_path = tmp_path / "cells.parquet"

    # --- detect (native space, exactly like brain type B -- Phase 1 unchanged) ---
    n_detected = detect_biomarker(mask_path, brain, biomarker, cells_path, run_id="run-a1")
    assert n_detected == 3
    assert read_cells(cells_path)["region_id"].isna().all()  # not yet assigned

    # --- filter ---
    summary = apply_volume_filter(cells_path, biomarker)
    assert summary == {"total": 3, "passed": 2}

    # --- coordinate mapping: resolve region_id/hemisphere_id (Phase 2) ---
    annotation_logical = FileReader(raw_annotation_path, transpose_order=TRANSPOSE_ORDER).read()
    assert annotation_logical.shape == (4, 5, 5)  # downsampled native shape, matches NATIVE_SHAPE // (3,4,4)

    assign_summary = assign_regions_in_cell_table(
        cells_path, NATIVE_SHAPE, annotation_logical, hemisphere=None, biomarker="cFos",
    )
    assert assign_summary == {"assigned": 3}  # all cFos rows get a region_id, filtered or not

    cells = read_cells(cells_path)
    passed = cells[cells["passed_filter"]]
    assert set(passed["region_id"]) == {10, 20}  # blob_A -> region A, blob_B -> region B

    # --- full tiered region report + asymmetry ---
    geometry = build_region_geometry_report(annotation_logical, structure_df, brain.voxel_size_um, hemisphere=None)
    report = tiered_cell_report(cells_path, structure_df, geometry, "cFos")

    tier2 = report[2].set_index("structure_id")
    assert tier2.loc[10, "total_cells"] == 1  # region A: blob_A only
    assert tier2.loc[20, "total_cells"] == 1  # region B: blob_B only

    tier1 = report[1].set_index("structure_id")
    assert tier1.loc[1, "total_cells"] == 2  # ROOT rolls up A + B

    # --- roi_extract: region A resampled to NATIVE resolution ---
    region_a_path = extract_region_to_native(
        annotation_logical, structure_df, "A", NATIVE_SHAPE, tmp_path,
        resize_order=0, chunk_size=(4, 20, 20),
    )
    assert region_a_path is not None
    region_a_result = FileReader(region_a_path).read()
    assert region_a_result.shape == NATIVE_SHAPE  # matches the ORIGINAL raw image's dimensions
    assert set(np.unique(region_a_result)).issubset({0, 10})

    # --- redraw: atlas-space (x2xz.py-equivalent) ---
    atlas_redraw_path = redraw_atlas_mask(
        cells_path, NATIVE_SHAPE, annotation_logical.shape, tmp_path, "atlas_redraw",
        biomarker="cFos", chunk_size=annotation_logical.shape,
    )
    atlas_redraw_result = FileReader(atlas_redraw_path).read()
    assert atlas_redraw_result.shape == annotation_logical.shape
    assert (atlas_redraw_result > 0).sum() == 2  # only the 2 passed cells

    # --- redraw: native-space (QC against the original mask) ---
    native_redraw_path = redraw_native_mask(
        cells_path, NATIVE_SHAPE, tmp_path, "native_redraw",
        biomarker="cFos", chunk_size=(4, 20, 20),
    )
    native_redraw_result = FileReader(native_redraw_path).read()
    assert native_redraw_result.shape == NATIVE_SHAPE
    assert (native_redraw_result > 0).sum() == 2
