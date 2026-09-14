# Signal_Analyzer

Unified brain-imaging signal-analysis pipeline: 3D cell detection with volume
filtering, brain-region statistics (with asymmetry analysis), region
extraction, mask reconstruction, and cross-biomarker colocalization.

It replaces two earlier lab tools (`MARS` and `Chulab-Signal_Analyzer`),
combining Chulab's Dask/Zarr I/O layer with MARS's volume-aware cell
detector. See `repo_diff.md` for the original requirements and the
architecture plan referenced in that discussion for full design rationale.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10. This installs the `signal-analyzer` command and the
`signal_analyzer` Python package. Dependencies (numpy, dask, zarr, numba,
`connected-components-3d`, pyarrow, pandas, etc.) install automatically.

## Two kinds of brain

Every run is described by one YAML config with a `brain.needs_registration`
flag:

- **`true` (brain type A)** — the brain has gone through atlas registration
  (an external tool, BIRDS, not part of this repo) and has an
  `annotation.tif`. Full pipeline: cell detection → volume filtering →
  region/hemisphere assignment → tiered Allen-CCF region report with
  asymmetry → optional region extraction (`roi_extract`) → mask redraw in
  both native and atlas space.
- **`false` (brain type B)** — no registration, no `annotation.tif` at all.
  Pipeline: cell detection → volume filtering → whole-brain cell count →
  native-space mask redraw. No region/hemisphere breakdown is possible for
  these brains.

Cell detection itself is identical either way and always runs on the
native-resolution mask — registration only ever affects a separate,
downsampled/axis-transposed copy sent to BIRDS, never the mask detection
runs on.

## Config file

Start from the two example files in `configs/`:

- `configs/default_biomarkers.yaml` — lab-wide defaults per biomarker
  (detection/filter parameters). A per-brain config is deep-merged on top of
  this; anything the brain config doesn't override falls back here.
- `configs/example_brain.yaml` — a full per-brain config, annotated.

Minimal example for a **type B** (non-registered) brain:

```yaml
brain:
  id: "my-brain-01"
  voxel_size_um: [4.0, 1.82, 1.82]   # native (Z, Y, X)
  needs_registration: false

biomarkers:
  cFos:
    mask_path: "/path/to/cFos_mask.tif"
    filter:
      min_volume_voxels: 25
      max_volume_voxels: 10000

paths:
  structures_csv: "data/structures.csv"
  output_dir: "/path/to/output"
```

For a **type A** (registered) brain, add a `registration` section
(see `configs/example_brain.yaml` for the full annotated version):

```yaml
registration:
  annotation_path: "/path/to/annotation.tif"   # BIRDS output
  hemisphere_path: "/path/to/hemisphere.tif"    # optional
  downsample_factor: [5, 5, 2]        # (Z,Y,X), used when preparing BIRDS input
  transpose_order: [1, 0, 2]          # BIRDS Y/Z-swap convention
```

`transpose_order` must be self-inverse (identity or a single axis swap) —
config loading rejects anything else, since the coordinate math only works
for that case.

Config validation enforces: `needs_registration: true` requires a
`registration` section; `false` forbids one (a type-B brain genuinely has no
`annotation.tif`, never a placeholder).

## Run it

```bash
signal-analyzer configs/example_brain.yaml --defaults configs/default_biomarkers.yaml
```

Options:

| Flag | Meaning |
|---|---|
| `--defaults PATH` | Lab-wide defaults YAML, deep-merged under the brain config |
| `--output-dir PATH` | Override `paths.output_dir` from the config |
| `--roi ACRONYM` | Extract this Allen CCF region via `roi_extract` (repeatable; type A only) |
| `--functional-system PATH` | YAML/JSON file with one preset's `{keywords, acronyms, exact_acronym_match, cell_threshold}`, added as a `Target_Summary` sheet (type A only) |
| `--log-level LEVEL` | `DEBUG`/`INFO`/`WARNING`/`ERROR` (default `INFO`) |

`--roi`/`--functional-system` are silently ignored (with a warning) for
type-B brains, since neither is possible without an `annotation.tif`.

`configs/functional_systems.yaml` holds several ready-made presets (ported
from the lab's asymmetry-analysis tool) keyed by system name — pick one
entry's fields out into its own file to pass via `--functional-system`,
e.g.:

```yaml
keywords: [auditory, cochlear, "inferior colliculus", ...]
acronyms: [AUD, IC, MG, SOC, CN, LL]
exact_acronym_match: false
cell_threshold: 30
```

Re-running is safe: cell detection is skipped if its output already exists;
every other stage (filtering, region assignment, reporting, redraw) always
re-runs, so a changed volume threshold or a new `--roi`/`--functional-system`
takes effect without redoing detection.

### From Python

```python
from signal_analyzer.common.config import load_config
from signal_analyzer.pipelines import run_pipeline

config = load_config("configs/example_brain.yaml", defaults_path="configs/default_biomarkers.yaml")
results = run_pipeline(config, roi_acronyms=["HIP", "STR"])
```

`run_pipeline` dispatches to `run_with_registration` / `run_without_registration`
based on `config.brain.needs_registration` — call either directly if you
already know the brain type.

## Output layout

Under `paths.output_dir` (or `--output-dir`), one subdirectory per biomarker:

```
<output_dir>/<biomarker>/
  cells.parquet                     # detected cells: position, volume, region/hemisphere, pass/fail
  logs/                             # one .log + .params.json per stage per run
  <biomarker>_whole_brain_report.csv        # type B only
  <biomarker>_region_report.xlsx            # type A only (tiered Allen-CCF + asymmetry + Target_Summary)
  <biomarker>_redraw_native.zarr            # filtered cells drawn back into native-space mask
  <biomarker>_redraw_atlas.zarr             # type A only: same, in atlas space
  roi_extract/<acronym>_atlas.zarr          # type A only, if --roi was given
```

`cells.parquet` is the canonical record — every other output is derived from
it and can be regenerated from it directly (see `signal_analyzer.common.cell_schema.read_cells`).

## Colocalization

Colocalization (does biomarker A's cell fall inside biomarker B's mask?) is
not yet wired into the automatic CLI run — call it directly once both
biomarkers have been detected and filtered:

```python
from signal_analyzer.colocalization.coloc import colocalize_pair

summary = colocalize_pair(
    smaller_cells_path="output/cFos/cells.parquet",
    smaller_biomarker="cFos",
    larger_mask_path="/path/to/TH_mask.tif",   # the ORIGINAL, unfiltered mask
    larger_biomarker="TH",
)
# {"tested": ..., "colocalized": ...}
```

This adds a `colocalized_with_TH` column to `cFos/cells.parquet`. For a
region-broken-down report (type A brains only), use
`colocalization.coloc.tiered_colocalization_report`. A config's
`colocalization.pairs` list names which `(larger, smaller)` pairs are
intended for a brain, but running them is a manual step for now.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

86 tests as of this writing, including regression tests that run MARS's own
original detection algorithm side-by-side to prove behavioral parity (and
to prove two real bugs found in it are fixed here — see
`tests/test_cc3d_detector.py`).

## Known limitations

- BIRDS registration itself is out of scope — this repo only prepares its
  input (`signal_analyzer.align.convert.prepare_registration_input`) and
  consumes its output.
- `transpose_order` only supports self-inverse permutations (identity or a
  single axis swap) — the only convention actually used by BIRDS today.
- Colocalization isn't wired into the CLI/orchestrators yet (see above).
- Not yet validated against real archived MARS/Chulab run outputs — no real
  brain imaging data has been available while building this; correctness was
  validated with synthetic data and direct cross-checks against MARS's
  actual source code instead.
