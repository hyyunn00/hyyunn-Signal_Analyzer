# Config examples

Every file here is a complete, independently loadable config demonstrating
one point in the schema's scenario space. Each file's header comment states
exactly how to load it and which other file to compare it against for the
field that varies.

| File | Brain type | Biomarkers | `hemisphere_path` | Colocalization | Loaded with `--defaults`? |
|---|---|---|---|---|---|
| [`01_type_a_full_featured.yaml`](01_type_a_full_featured.yaml) | A (registered) | 2 (cFos, TH) | yes | yes | yes |
| [`02_type_a_single_marker.yaml`](02_type_a_single_marker.yaml) | A (registered) | 1 (cFos) | no | no | yes |
| [`03_type_a_no_hemisphere.yaml`](03_type_a_no_hemisphere.yaml) | A (registered) | 1 (cFos) | no (X-midline fallback) | no | yes |
| [`04_type_b_single_marker.yaml`](04_type_b_single_marker.yaml) | B (not registered) | 1 (cFos) | n/a | no | yes |
| [`05_type_b_with_colocalization.yaml`](05_type_b_with_colocalization.yaml) | B (not registered) | 2 (cFos, TH) | n/a | yes | yes |
| [`06_standalone_no_defaults.yaml`](06_standalone_no_defaults.yaml) | B (not registered) | 1 (cFos) | n/a | no | **no** (fully self-contained) |

Points this matrix is meant to make explicit:

- **`hemisphere_path` is always optional**, registered or not. Omitting it
  (`03`) falls back to an X-axis midline split instead of a real hemisphere
  segmentation (`01`) -- see `report/region_stats.py`.
- **A single biomarker is fully supported** (`02`, `03`, `04`, `06`) --
  nothing in this pipeline requires two, and `colocalization:` is simply
  absent from those files rather than present-but-empty.
- **Colocalization is independent of `needs_registration`** (`05`) -- it's a
  direct point-in-mask test between two biomarkers, never touches
  `annotation.tif`. It's still a manual Python-API step either way (see the
  main README's "Colocalization" section), not something the CLI runs on its
  own.
- **`--defaults`/`defaults_path` is optional** (`06`) -- a config can supply
  every `detection`/`filter` field inline and load standalone. Every other
  file here uses `--defaults configs/default_biomarkers.yaml` because that's
  the realistic multi-brain lab workflow, not because the schema requires it.
- **Type B (`needs_registration: false`) forbids a `registration:` section
  outright** (`04`, `05`, `06`) -- config loading rejects one if present,
  since these brains genuinely have no `annotation.tif`, ever.

`../default_biomarkers.yaml` (lab-wide biomarker defaults) and
`../functional_systems.yaml` (named region-set presets for `--functional-system`)
are shared support files, not per-scenario examples, so they stay one level up.
