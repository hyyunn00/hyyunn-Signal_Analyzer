"""Public IO package exports for readers, writers, and shared metadata.

Ported from Chulab-Signal_Analyzer's IO/__init__.py
(D:\\chu_lab\\Chulab-Signal_Analyzer\\IO\\__init__.py). Exposes the high-level
``FileReader``/``FileWriter`` classes and common constants. Also re-exports
``read_image`` from ``io.reader_tools`` for direct, single-file metadata/array
opening.
"""
from .reader import FileReader
from .reader_tools import read_image
from .writer import FileWriter
from .types import OUTPUT_CHOICES, TYPE_MAP, VALID_SUFFIXES, VolumeMetadata

__all__ = [
    "FileReader",
    "FileWriter",
    "read_image",
    "OUTPUT_CHOICES",
    "TYPE_MAP",
    "VALID_SUFFIXES",
    "VolumeMetadata",
]
