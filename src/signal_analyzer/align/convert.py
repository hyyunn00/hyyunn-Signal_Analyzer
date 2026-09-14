"""Registration-input preparation: downsample + axis-transpose a reference
volume, producing what gets submitted to external BIRDS registration.

Per the architecture plan's confirmed fact (user-verified): this step NEVER
touches the mask/signal volume used for cell detection -- detection always
runs at native, full resolution (see detection/dask_runner.py). This module
only prepares a SEPARATE, smaller, axis-swapped volume for the external
registration tool; BIRDS returns annotation.tif already living in that same
downsampled/transposed space, which regions.coord_transform later maps
native-space cell coordinates into (using the annotation's actual measured
shape at that point, not this step's nominal downsample_factor -- see that
module's docstring for why).
"""
from __future__ import annotations

from pathlib import Path

from ..io import FileReader, FileWriter


def prepare_registration_input(
    source_path: str | Path,
    downsample_factor: tuple[int, int, int],
    transpose_order: tuple[int, int, int],
    output_path: str | Path,
    output_name: str,
    resize_order: int = 1,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    n_workers: int = 8,
) -> Path:
    """Downsample then axis-transpose ``source_path``, writing the result
    for submission to BIRDS.

    Two explicit steps, not fused: ``io.FileWriter``'s two-pass resize
    pipeline only changes per-axis SIZE, not axis ORDER, so downsampling and
    transposing can't be expressed as one pipeline call.
      1. Downsample in native axis order via ``FileWriter``'s resize
         pipeline, to an intermediate Zarr store.
      2. Read that intermediate back with ``transpose_order`` applied (cheap
         at this point: the data is already small after downsampling) and
         write the final, axis-transposed output.

    Note: this reads the full source volume into memory for step 1 (as does
    Chulab's own ``converter.py``, which this is adapted from) -- reasonable
    for the volumes practical registration pipelines feed BIRDS today; a
    fully streamed version can be revisited if a brain's reference channel
    turns out too large to hold in memory.

    Args:
        source_path: Path to the native-resolution reference volume (e.g. a
            raw/autofluorescence channel commonly used for atlas
            registration -- NOT the detection mask).
        downsample_factor: Per-native-axis (Z,Y,X) integer downsample factor.
        transpose_order: Axis permutation applied after downsampling (e.g.
            ``(1,0,2)`` for the BIRDS Y/Z-swap convention).
        output_path: Directory to write the final output into.
        output_name: Base name for the final output Zarr store.
        resize_order: skimage interpolation order for the downsample pass
            (1=linear, appropriate for intensity images; use 0 for
            label-preserving volumes).

    Returns:
        Path to the final, downsampled+transposed Zarr store.
    """
    reader = FileReader(source_path)
    native_shape = reader.volume_shape
    downsampled_shape = tuple(
        max(1, native_shape[i] // downsample_factor[i]) for i in range(3)
    )

    data = reader.read()

    intermediate_writer = FileWriter(
        output_path=Path(output_path) / "_tmp_downsampled",
        output_name=f"{output_name}_downsampled",
        output_type="zarr",
        full_res_shape=downsampled_shape,
        input_shape=native_shape,
        output_dtype=data.dtype,
        resize_order=resize_order,
        chunk_size=chunk_size,
        n_workers=n_workers,
    )
    intermediate_writer.write(data)
    intermediate_writer.complete_resize()
    del data

    transposed_reader = FileReader(intermediate_writer.output_path, transpose_order=transpose_order)
    transposed_data = transposed_reader.read()

    final_writer = FileWriter(
        output_path=output_path,
        output_name=output_name,
        output_type="zarr",
        full_res_shape=transposed_reader.volume_shape,
        output_dtype=transposed_data.dtype,
        chunk_size=chunk_size,
    )
    final_writer.write(transposed_data)

    return final_writer.output_path
