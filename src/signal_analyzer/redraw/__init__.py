"""Reconstruct a mask volume from filtered cell positions.

Phase 1 only implements the native-space output mode: scatter each passed
cell's native-space (z,y,x) coordinate directly into a zero-initialized
volume matching the mask's own resolution/orientation -- no coordinate
transform, useful for QC against the original mask. The atlas-space mode
(reproducing MARS's x2xz.py XZ-reslice-into-annotation-space behavior via
regions/coord_transform.py) is added in Phase 3 for registered brains.
"""
from .mask_from_cells import redraw_atlas_mask, redraw_native_mask

__all__ = ["redraw_native_mask", "redraw_atlas_mask"]
