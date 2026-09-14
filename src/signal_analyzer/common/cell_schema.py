"""Parquet schema and read/write helpers for per-cell detection records.

See the architecture plan's "Cell-Position Data Schema (Parquet)" section.
Parquet was chosen (over MARS's XML or Chulab's JSONL) as the sole canonical
cell-data format: it streams well per Z-chunk (matching how both source
pipelines already accumulate results incrementally), supports predicate/
column pushdown for the filtered re-queries every later stage needs
(colocalization joins, per-region/per-biomarker report aggregation, redraw),
and enforces a typed schema -- eliminating the class of bug where MARS's
volume-filter threshold silently diverged between files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Hemisphere id convention shared across the whole pipeline.
HEMISPHERE_NA = 0
HEMISPHERE_LEFT = 1
HEMISPHERE_RIGHT = 2

CELL_SCHEMA = pa.schema([
    pa.field("cell_id", pa.int64()),
    pa.field("z", pa.int32()),
    pa.field("y", pa.int32()),
    pa.field("x", pa.int32()),
    pa.field("volume_voxels", pa.int32()),
    pa.field("volume_um3", pa.float32()),
    pa.field("biomarker", pa.dictionary(pa.int8(), pa.string())),
    pa.field("region_id", pa.int32(), nullable=True),
    pa.field("hemisphere_id", pa.int8()),
    pa.field("brain_id", pa.string()),
    pa.field("run_id", pa.string()),
    pa.field("passed_filter", pa.bool_()),
])
"""Canonical per-cell record schema.

``region_id`` is null whenever no ``annotation.tif`` exists for the brain
(brain type B, per the pipeline plan) -- never a placeholder region id.
``passed_filter`` is kept as a column rather than the row being dropped, so
volume-filter threshold changes are a re-filter query, not a re-detection,
and colocalization can still see a biomarker's un-filtered cell set if needed.
"""

_REQUIRED_FIELDS = set(CELL_SCHEMA.names)


class CellTableWriter:
    """Incrementally append cell records (e.g. one call per Z-chunk) to a Parquet file.

    Mirrors how both source pipelines already stream detection output --
    MARS's per-chunk numba dict accumulation, Chulab's per-slab JSONL
    append -- just targeting a single Parquet file via row-group writes.
    """

    def __init__(self, output_path: str | Path, schema: pa.Schema = CELL_SCHEMA) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self._writer: Optional[pq.ParquetWriter] = None
        self._n_written = 0

    def write_batch(self, records: Iterable[dict]) -> int:
        """Append a batch of cell-record dicts. Returns the number of rows written.

        Each dict must contain every field in ``self.schema`` (missing
        optional fields like ``region_id`` should be passed explicitly as
        ``None``, not omitted, since ``pa.Table.from_pylist`` requires a
        consistent key set for schema validation to raise a useful error).
        """
        records = list(records)
        if not records:
            return 0

        missing = _REQUIRED_FIELDS - set(records[0].keys())
        if missing:
            raise ValueError(f"Cell record batch missing required field(s): {sorted(missing)}")

        table = pa.Table.from_pylist(records, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.output_path, self.schema)
        self._writer.write_table(table)
        self._n_written += table.num_rows
        return table.num_rows

    def close(self) -> int:
        """Finalize the Parquet file. Returns the total number of rows written."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._n_written

    def __enter__(self) -> "CellTableWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def read_cells(
    path: str | Path,
    *,
    biomarker: Optional[str] = None,
    passed_filter: Optional[bool] = None,
    columns: Optional[list[str]] = None,
):
    """Read cell records with optional predicate pushdown filters.

    Args:
        path: Path to a cell-records Parquet file (or a directory of them).
        biomarker: If given, only rows for this biomarker are read.
        passed_filter: If given, only rows with this ``passed_filter`` value are read.
        columns: If given, only these columns are read from disk.

    Returns:
        A pandas DataFrame. Filtering/column selection is pushed down via
        ``pyarrow.dataset``, so unmatched row-groups/columns are never
        actually read from disk.
    """
    dataset = ds.dataset(str(path), format="parquet")

    filter_expr = None
    if biomarker is not None:
        expr = ds.field("biomarker") == biomarker
        filter_expr = expr if filter_expr is None else filter_expr & expr
    if passed_filter is not None:
        expr = ds.field("passed_filter") == passed_filter
        filter_expr = expr if filter_expr is None else filter_expr & expr

    table = dataset.to_table(columns=columns, filter=filter_expr)
    return table.to_pandas()
