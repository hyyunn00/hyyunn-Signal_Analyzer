"""Utilities for writing volume data to Zarr, TIFF, and NIfTI outputs.

Ported from Chulab-Signal_Analyzer's IO/writer.py
(D:\\chu_lab\\Chulab-Signal_Analyzer\\IO\\writer.py), unchanged aside from the
relative import path.
"""
import logging
import zarr
import numpy as np
import tifffile
import nibabel as nib
import dask.array as da

from pathlib import Path
from numcodecs import Blosc

from .writer_tools import (
    resize_xy_block_to_temp,
    resize_xy_volume_to_temp,
    collapse_xz_from_temp,
    write_chunk_to_zarr,
)

# Set up module-level logger
logger = logging.getLogger(__name__)


def expected_output_path(output_path, output_name, output_type) -> Path:
    """The final path ``FileWriter(output_path, output_name, output_type, ...)``
    will produce for a full-volume (single, non-partial) write, without
    instantiating ``FileWriter`` -- instantiating would be destructive for a
    ``'zarr'``/``'ome-zarr'`` target (its initializer opens the store with
    ``mode='w'``, truncating any existing data). Used by skip-if-exists
    checks (``redraw``, ``roi_extract``) that need to know a stage's output
    path *before* deciding whether to run it at all.

    Mirrors ``FileWriter``'s own per-``output_type`` naming
    (``_initialize_zarr``, ``_initialize_scroll_tiff``, etc.) and, for
    ``'single-tiff'``/``'single-nii'``, ``_single_output_name``'s
    full-volume-write case (no ``z<start>-<end>`` suffix).
    """
    output_path = Path(output_path)
    suffix = {
        "zarr": ".zarr",
        "ome-zarr": ".ome.zarr",
        "single-tiff": ".tiff",
        "scroll-tiff": ".scroll-tif",
        "single-nii": ".nii.gz",
        "scroll-nii": ".scroll-nii",
    }.get(output_type)
    if suffix is None:
        raise ValueError(f"Unknown output_type: {output_type}")
    return output_path / f"{output_name}{suffix}"


class FileWriter:
    """Handle writing datasets to the supported output formats.

    Supports:
      - Flat Zarr arrays
      - OME-Zarr multiscale pyramids (with on-the-fly level generation)
      - Single TIFF and NIfTI volumes
      - Per-slice "scroll" exports for TIFF and NIfTI
    """

    def __init__(
        self,
        output_path,
        output_name,
        output_type,
        full_res_shape,
        output_dtype,
        file_name=None,
        chunk_size=(128, 128, 128),
        n_level=5,
        resize_factor=2,
        resize_order=0,
        input_shape=None,
        n_workers=8,
    ):
        """Create and initialize a writer for the desired output.

        Args:
            output_path (str | Path): Directory where outputs are written.
            output_name (str): Base name used to form output file/store names.
            output_type (str): One of: 'zarr', 'ome-zarr', 'single-tiff',
                'scroll-tiff', 'single-nii', 'scroll-nii'.
            full_res_shape (tuple[int,int,int]): Target (Z, Y, X) shape.
            output_dtype (np.dtype | str): Output dtype.
            file_name (list[Path] | None): For scroll outputs, the per-slice base names.
            chunk_size (tuple[int,int,int]): Chunking used for Zarr IO.
            n_level (int): Number of pyramid levels for OME-Zarr.
            resize_factor (int): Downscale factor per level for OME-Zarr.
            resize_order (int): Interpolation order used during resizing.
            input_shape (tuple[int,int,int] | None): Source shape; if it differs
                from ``full_res_shape`` and the output is a Zarr target, a two-pass
                resize pipeline is enabled.
            n_workers (int): Number of parallel workers for resizing tasks.
        """
        self.output_path = Path(output_path)
        self.output_name = output_name
        self.output_type = output_type
        self.full_res_shape = tuple(full_res_shape)
        self.output_dtype = np.dtype(output_dtype)
        self.source_shape = tuple(full_res_shape) if input_shape is None else tuple(input_shape)
        self.file_name = file_name
        self.chunk_size = tuple(chunk_size)
        self.resize_order = int(resize_order)
        self.resize_factor = int(resize_factor)
        self.n_level = int(n_level)

        # For 'single-tiff'/'single-nii', the actual written filename embeds
        # the z-range of the write() call (only known at write time, unlike
        # every other output_type where the final path is fixed at init) --
        # _write_single_tiff/_write_single_nii populate this so callers have
        # a reliable path to the file that was actually written, instead of
        # self.output_path (which stays the containing directory for these
        # two types).
        self.last_written_path: Path | None = None

        self.n_workers = int(n_workers)
        logger.info(f"Initialized FileWriter with output: {self.output_path}")

        handler = self._output_initializers().get(self.output_type)
        if handler is None:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        else:
            handler()

        # Internal state for on-the-fly resizing into Zarr targets
        self._resizing_active = tuple(self.source_shape) != tuple(self.full_res_shape) and self.output_type in ("zarr", "ome-zarr")
        self._resize_temp_path: Path | None = None

    def write(self, array: np.ndarray, z_start=0, z_end=None, y_start=0, y_end=None, x_start=0, x_end=None) -> None:
        """Write a chunk of data to the configured destination.

        Args:
            array (np.ndarray): Input block shaped ``(dz, dy, dx)``.
            z_start (int): Inclusive Z start in the destination space.
            z_end (int | None): Exclusive Z end; defaults to Z size.
            y_start (int): Inclusive Y start.
            y_end (int | None): Exclusive Y end.
            x_start (int): Inclusive X start.
            x_end (int | None): Exclusive X end.
        """
        z0, z1 = z_start, self.full_res_shape[0] if z_end is None else z_end
        y0, y1 = y_start, self.full_res_shape[1] if y_end is None else y_end
        x0, x1 = x_start, self.full_res_shape[2] if x_end is None else x_end

        if not self._resizing_active:
            logger.info(f"Writing volume z: {z0} - {z1}, y: {y0} - {y1}, x: {x0} - {x1}")

            handler = self._write_handlers().get(self.output_type)
            if handler is None:
                raise ValueError(f"Unknown output_type: {self.output_type}")

            handler(array, z0, z1, y0, y1, x0, x1)

        else:
            z0 = z_start
            z1 = self.source_shape[0] if z_end is None else z_end

            temp_path = resize_xy_block_to_temp(
                block=array,
                z_range=(z0, z1),
                current_shape=self.source_shape if self._resize_temp_path is None else None,
                target_shape=self.full_res_shape,
                dtype=self.output_dtype,
                order=self.resize_order,
                chunk_size=self.chunk_size,
                temp_store_path=self._resize_temp_path,
                n_workers=self.n_workers,
            )
            # Store the path on first use for subsequent blocks
            if self._resize_temp_path is None:
                self._resize_temp_path = Path(temp_path)
            return

    def complete_resize(self) -> None:
        """Finish two-pass resize into the configured Zarr destination and cleanup temp.

        Only valid when the writer is configured for Zarr or OME-Zarr and a
        resize is active (``source_shape != full_res_shape``).

        Side Effects:
            Writes the final resized data into the target Zarr dataset and
            removes any temporary working directory created during the XY pass.
        """
        if not self._resizing_active:
            return
        if self._resize_temp_path is None:
            # Nothing was staged
            return

        if self.output_type == 'zarr':
            target = self.store_array
        elif self.output_type == 'ome-zarr':
            target = self.store_group['0']
        else:
            raise ValueError('finalize_resize_to_output is only valid for zarr/ome-zarr outputs')

        collapse_xz_from_temp(
            temp_store_path=self._resize_temp_path,
            output_source=target,
            target_shape=self.full_res_shape,
            dtype=self.output_dtype,
            order=self.resize_order,
            chunk_size=self.chunk_size,
            n_workers=self.n_workers,
        )
        self._resize_temp_path = None

    def complete_ome(self) -> None:
        """Generate downsampled OME-Zarr pyramid levels in-place.

        Uses the level-0 dataset as a source, generates lower-resolution levels
        according to ``resize_factor`` and ``n_level``, and writes OME-Zarr
        multiscales metadata.
        """
        if self.output_type != 'ome-zarr':
            raise ValueError('write_ome_levels is only valid for ome-zarr outputs')

        if not hasattr(self, 'store_group'):
            raise ValueError('OME-Zarr store is not initialized.')

        for i in range(1, self.n_level):
            prev_arr = da.from_zarr(self.store_group[str(i - 1)].store, component=str(i - 1))
            target_shape = self.store_group[str(i)].shape
            logger.info(f"Generating level {i} with shape {target_shape} from level {i - 1}...")

            temp_store_path = resize_xy_volume_to_temp(
                input_source=self.store_group[str(i - 1)],
                current_shape=prev_arr.shape,
                target_shape=target_shape,
                dtype=self.output_dtype,
                order=self.resize_order,
                chunk_size=self.chunk_size,
                n_workers=self.n_workers,
            )

            collapse_xz_from_temp(
                temp_store_path=temp_store_path,
                output_source=self.store_group[str(i)],
                target_shape=target_shape,
                dtype=self.output_dtype,
                order=self.resize_order,
                chunk_size=self.chunk_size,
                n_workers=self.n_workers,
            )

        self.store_group.attrs['multiscales'] = self._ome_metadata()

    def _write_handlers(self):
        """Dispatch table matching output types to write implementations."""
        return {
            'ome-zarr': self._write_ome_zarr,
            'zarr': self._write_zarr,
            'single-tiff': self._write_single_tiff,
            'scroll-tiff': self._write_scroll_tiff,
            'single-nii': self._write_single_nii,
            'scroll-nii': self._write_scroll_nii,
        }

    def _write_ome_zarr(self, array: np.ndarray, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> None:
        """Write data into the root OME-Zarr level."""
        target = self.store_group['0']
        write_chunk_to_zarr(array, self.chunk_size, target,
            (
                slice(z0, z1),
                slice(y0, y1),
                slice(x0, x1),
            ),
        )

    def _write_zarr(self, array: np.ndarray, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> None:
        """Write data into a flat Zarr array target."""
        write_chunk_to_zarr(array, self.chunk_size, self.store_array,
            (
                slice(z0, z1),
                slice(y0, y1),
                slice(x0, x1),
            ),
        )

    def _single_output_name(self, z0: int, z1: int, suffix: str) -> str:
        """Filename for a 'single' (non-scroll) output covering z0:z1.

        A z0:z1 range covering the whole volume -- the common case, e.g.
        roi_extract's one-shot resize->convert write -- gets a clean name
        with no z-range suffix (matching MARS's own aba2roi.py convention,
        e.g. ``{acronym}_atlas.tif``). A genuine partial write (multiple
        write() calls, each covering a sub-range) still gets its z-range
        embedded, since otherwise those files would collide.
        """
        if z0 == 0 and z1 == self.full_res_shape[0]:
            return f"{self.output_name}{suffix}"
        return f"{self.output_name}_z{z0}-{z1}{suffix}"

    def _write_single_tiff(self, array: np.ndarray, z0: int, z1: int, *_: int) -> None:
        """Persist the supplied block as a multi-page TIFF file."""
        output_path = self.output_path / self._single_output_name(z0, z1, ".tiff")
        tifffile.imwrite(output_path, array.astype(self.output_dtype), **self._tiff_write_kwargs())
        self.last_written_path = output_path

    def _write_scroll_tiff(self, array: np.ndarray, z0: int, z1: int, *_: int) -> None:
        """Write per-slice TIFF files for scroll outputs."""
        kwargs = self._tiff_write_kwargs()
        for idx, file_path in enumerate(self.output_file_path[z0:z1]):
            tifffile.imwrite(file_path, array[idx].astype(self.output_dtype), **kwargs)

    def _imagej_safe(self) -> bool:
        """Whether ``self.output_dtype`` is safe to write with ``imagej=True``.

        ImageJ-compatible TIFFs only support bool/uint8/uint16/float32 sample
        types. Label/region-id data (e.g. roi_extract's uint32 Allen CCF ids,
        which can exceed 65535) must be written as a plain multi-page TIFF
        instead -- still directly openable in Fiji as an image stack, just
        without the ImageJ hyperstack metadata tag, which tifffile would
        otherwise reject or write incorrectly for an unsupported dtype.
        """
        return self.output_dtype in (np.dtype(np.bool_), np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))

    def _tiff_write_kwargs(self) -> dict:
        """tifffile.imwrite kwargs for the current output_dtype.

        For imagej-safe dtypes, ``imagej=True`` gives Fiji its hyperstack
        metadata. Otherwise (e.g. uint32 label/region-id data), explicitly
        pass ``photometric='minisblack'`` -- without it, tifffile's shape/
        dtype-based guessing can misinterpret a plain (Z,Y,X) grayscale
        label volume as RGB-with-separate-planes (observed for uint32
        arrays), which would silently corrupt how the file is interpreted
        by Fiji or any other reader, even though the file writes without error.
        """
        if self._imagej_safe():
            return {"imagej": True}
        return {"photometric": "minisblack"}

    def _write_single_nii(self, array: np.ndarray, z0: int, z1: int, *_: int) -> None:
        """Persist a full NIfTI volume covering the requested range."""
        arr_xyz = self._volume_to_nii_axes(array).astype(self.output_dtype)
        nii_img = nib.Nifti1Image(arr_xyz, affine=np.eye(4))
        output_path = self.output_path / self._single_output_name(z0, z1, ".nii.gz")
        nib.save(nii_img, output_path)
        self.last_written_path = output_path

    def _write_scroll_nii(self, array: np.ndarray, z0: int, z1: int, *_: int) -> None:
        """Write individual NIfTI files for scroll-style outputs."""
        for idx, file_path in enumerate(self.output_file_path[z0:z1]):
            slice_xyz = self._volume_to_nii_axes(array[idx]).astype(self.output_dtype)
            nii_img = nib.Nifti1Image(slice_xyz, affine=np.eye(4))
            nib.save(nii_img, file_path)

    def _output_initializers(self):
        """Return a mapping from output type to initialisation callable."""
        return {
            'zarr': self._initialize_zarr,
            'ome-zarr': self._initialize_ome,
            'single-tiff': self._initialize_single_tiff,
            'scroll-tiff': self._initialize_scroll_tiff,
            'single-nii': self._initialize_single_nii,
            'scroll-nii': self._initialize_scroll_nii,
        }

    def _initialize_ome(self) -> None:
        """Create the multiscale OME-Zarr hierarchy and allocate datasets."""
        self.output_path = self.output_path / f"{self.output_name}.ome.zarr"

        store = zarr.DirectoryStore(self.output_path)
        self.store_group = zarr.group(store=store)

        for level in range(self.n_level):
            if level == 0:
                target_z, target_y, target_x = self.full_res_shape
            else:
                prev_shape = self.store_group[str(level - 1)].shape
                current_z, current_y, current_x = prev_shape
                target_z = int(current_z) // self.resize_factor
                target_y = int(current_y) // self.resize_factor
                target_x = int(current_x) // self.resize_factor

            if target_z < 0 or target_y < 0 or target_x < 0:
                logger.info(
                    'Skipping level %s due to insufficient shape: %s',
                    level,
                    (target_z, target_y, target_x),
                )
                break

            self.store_group.create_dataset(
                str(level),
                shape=(target_z, target_y, target_x),
                chunks=self.chunk_size,
                dtype=self.output_dtype,
                compression=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
                overwrite=True,
            )

    def _initialize_zarr(self) -> None:
        """Initialise a flat Zarr array with the configured chunking."""
        self.output_path = self.output_path / f"{self.output_name}.zarr"

        store = zarr.DirectoryStore(self.output_path)
        self.store_array = zarr.open_array(
            store=store,
            mode='w',
            shape=self.full_res_shape,
            chunks=self.chunk_size,
            dtype=self.output_dtype,
            compressor=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
        )

    def _initialize_single_tiff(self) -> None:
        """Ensure the output directory exists for single TIFF exports."""
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _initialize_scroll_tiff(self) -> None:
        """Prepare paths for per-slice TIFF scroll outputs."""
        self.output_path = self.output_path / f"{self.output_name}.scroll-tif"
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.output_file_path = [self.output_path / f"{name.stem}.tiff" for name in self.file_name]

    def _initialize_single_nii(self) -> None:
        """Ensure the output directory exists for single NIfTI exports."""
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _initialize_scroll_nii(self) -> None:
        """Prepare paths for scroll-style NIfTI outputs."""
        self.output_path = self.output_path / f"{self.output_name}.scroll-nii"
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.output_file_path = [self.output_path / f"{name.stem}.nii.gz" for name in self.file_name]

    @staticmethod
    def _volume_to_nii_axes(array: np.ndarray) -> np.ndarray:
        """Reorder YXZ data into the X,Y,Z layout expected by NIfTI."""
        if array.ndim == 3:
            return np.transpose(array, (2, 1, 0))
        if array.ndim == 2:
            return np.transpose(array, (1, 0))[..., np.newaxis]
        raise ValueError(f"Unsupported array shape for NIfTI: {array.shape}")

    def _ome_metadata(self):
        datasets = []
        for level in range(self.n_level):
            scale_factor = self.resize_factor ** level
            datasets.append(
                {
                    'path': str(level),
                    'coordinateTransformations': [
                        {
                            'type': 'scale',
                            'scale': [scale_factor] * 3,
                        }
                    ],
                }
            )

        multiscales = [
            {
                'version': '0.4',
                'name': 'image',
                'axes': [
                    {'name': 'z', 'type': 'space'},
                    {'name': 'y', 'type': 'space'},
                    {'name': 'x', 'type': 'space'},
                ],
                'datasets': datasets,
            }
        ]

        return multiscales
