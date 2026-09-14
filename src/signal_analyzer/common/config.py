"""YAML configuration schema and loader/validator for pipeline runs.

Replaces the scattered hardcoded literals in both source repos (MARS's
inconsistent volume-filter bounds across files, its baked-in voxel-volume
and SCALE_X/Y/Z constants; Chulab's single-global-invocation CLI flags with
no per-biomarker mechanism) with one per-brain, per-biomarker config file.
See the architecture plan's "Config Schema (YAML)" section for the full
field reference and rationale.

Encodes one structural invariant directly: a brain with
``needs_registration: true`` MUST have a ``registration`` section (it will
have an ``annotation.tif``), and a brain with ``needs_registration: false``
MUST NOT have one (brain type B has no annotation.tif at all -- confirmed
with the user; region/hemisphere stats simply don't exist for these brains).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration is invalid, incomplete, or inconsistent."""


@dataclass(frozen=True)
class DetectionConfig:
    """Per-biomarker detection parameters for the connected-component detector."""
    connectivity: int = 18
    z_chunk: int = 100


@dataclass(frozen=True)
class FilterConfig:
    """Per-biomarker volume-filter bounds, in raw voxel count."""
    min_volume_voxels: int
    max_volume_voxels: int


@dataclass(frozen=True)
class BiomarkerConfig:
    """One biomarker/channel's mask path, detection params, and filter bounds."""
    name: str
    mask_path: str
    detection: DetectionConfig
    filter: FilterConfig


@dataclass(frozen=True)
class RegistrationConfig:
    """Atlas-registration parameters. Only present for brains needing registration.

    ``downsample_factor``/``transpose_order`` describe the transform applied
    when preparing the volume submitted to BIRDS -- NOT applied to the mask
    used for detection, which always stays native-resolution. The same values
    are required later to map native-space cell coordinates into the
    downsampled/transposed space that the resulting ``annotation.tif`` lives in.
    """
    annotation_path: str
    downsample_factor: tuple[int, int, int]
    transpose_order: tuple[int, int, int] = (1, 0, 2)
    hemisphere_path: Optional[str] = None


@dataclass(frozen=True)
class ColocalizationPair:
    """One (larger, smaller) biomarker pair to test for colocalization.

    The smaller biomarker's filtered cell centroids are tested against the
    larger biomarker's ORIGINAL, unfiltered mask.
    """
    larger: str
    smaller: str


@dataclass(frozen=True)
class BrainConfig:
    """Per-brain identity, physical voxel size, and registration status."""
    id: str
    voxel_size_um: tuple[float, float, float]
    needs_registration: bool


@dataclass(frozen=True)
class PathsConfig:
    """Shared reference-data and output paths for the run."""
    structures_csv: str
    output_dir: str


@dataclass(frozen=True)
class RunConfig:
    """A fully-resolved, validated configuration for one brain's pipeline run."""
    brain: BrainConfig
    biomarkers: dict[str, BiomarkerConfig]
    paths: PathsConfig
    registration: Optional[RegistrationConfig] = None
    colocalization_pairs: tuple[ColocalizationPair, ...] = ()
    source_path: Optional[Path] = None

    def biomarker(self, name: str) -> BiomarkerConfig:
        """Look up a configured biomarker by name, raising a clear error if unknown."""
        try:
            return self.biomarkers[name]
        except KeyError:
            raise ConfigError(
                f"Unknown biomarker '{name}'; configured biomarkers: {sorted(self.biomarkers)}"
            )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_config(defaults: dict, override: dict) -> dict:
    """Merge a lab-wide defaults dict under a per-brain config dict.

    Like ``_deep_merge``, except ``biomarkers`` is handled specially: only
    biomarker names actually present in ``override["biomarkers"]`` are kept
    in the result (each one's fields still deep-merged against the matching
    default entry, if any). A plain ``_deep_merge`` would instead union the
    two dicts' keys, silently pulling in every OTHER biomarker
    ``defaults`` happens to define even when the per-brain config only
    lists one -- confirmed with the user that a config naming exactly one
    biomarker must produce a run with exactly that one, regardless of how
    many the shared defaults file defines.
    """
    merged = _deep_merge(defaults, override)

    override_biomarkers = override.get("biomarkers")
    if isinstance(override_biomarkers, dict):
        merged["biomarkers"] = {
            name: merged["biomarkers"][name]
            for name in override_biomarkers
            if name in merged.get("biomarkers", {})
        }

    return merged


def load_raw_yaml(path: str | Path) -> dict:
    """Load a YAML file into a plain dict, without validation."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(brain_config_path: str | Path, defaults_path: str | Path | None = None) -> RunConfig:
    """Load a per-brain YAML config, optionally deep-merged over lab-wide defaults.

    Args:
        brain_config_path: Path to the per-brain/run YAML config.
        defaults_path: Optional path to a defaults YAML (e.g.
            ``configs/default_biomarkers.yaml``) that ``brain_config_path`` is
            deep-merged on top of -- values in the brain config always win.

    Returns:
        A validated, immutable RunConfig.
    """
    brain_config_path = Path(brain_config_path)
    raw = load_raw_yaml(brain_config_path)

    if defaults_path is not None:
        defaults = load_raw_yaml(defaults_path)
        raw = _merge_config(defaults, raw)

    return _build_run_config(raw, source_path=brain_config_path)


def _require(data: dict, key: str, context: str):
    if key not in data or data[key] is None:
        raise ConfigError(f"Missing required field '{key}' in {context}")
    return data[key]


def _as_tuple3(value, field_name: str, cast=float) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"'{field_name}' must be a 3-element list, got: {value!r}")
    return tuple(cast(v) for v in value)


def _is_self_inverse_permutation(order: tuple[int, int, int]) -> bool:
    """True if applying ``order`` twice returns to the original axis order."""
    return all(order[order[i]] == i for i in range(len(order)))


def _build_run_config(raw: dict, source_path: Path) -> RunConfig:
    brain_raw = _require(raw, "brain", "config")
    brain = BrainConfig(
        id=str(_require(brain_raw, "id", "brain")),
        voxel_size_um=_as_tuple3(_require(brain_raw, "voxel_size_um", "brain"), "brain.voxel_size_um", float),
        needs_registration=bool(_require(brain_raw, "needs_registration", "brain")),
    )

    biomarkers_raw = _require(raw, "biomarkers", "config")
    if not biomarkers_raw:
        raise ConfigError("At least one biomarker must be configured under 'biomarkers'")

    biomarkers: dict[str, BiomarkerConfig] = {}
    for name, bm_raw in biomarkers_raw.items():
        context = f"biomarkers.{name}"
        detection_raw = bm_raw.get("detection", {}) or {}
        filter_raw = _require(bm_raw, "filter", context)

        detection = DetectionConfig(
            connectivity=int(detection_raw.get("connectivity", 18)),
            z_chunk=int(detection_raw.get("z_chunk", 100)),
        )
        filter_cfg = FilterConfig(
            min_volume_voxels=int(_require(filter_raw, "min_volume_voxels", f"{context}.filter")),
            max_volume_voxels=int(_require(filter_raw, "max_volume_voxels", f"{context}.filter")),
        )
        if filter_cfg.min_volume_voxels >= filter_cfg.max_volume_voxels:
            raise ConfigError(
                f"{context}.filter: min_volume_voxels ({filter_cfg.min_volume_voxels}) "
                f"must be < max_volume_voxels ({filter_cfg.max_volume_voxels})"
            )

        biomarkers[name] = BiomarkerConfig(
            name=name,
            mask_path=str(_require(bm_raw, "mask_path", context)),
            detection=detection,
            filter=filter_cfg,
        )

    registration_raw = raw.get("registration")
    registration: Optional[RegistrationConfig] = None
    if registration_raw is not None:
        transpose_order = _as_tuple3(
            registration_raw.get("transpose_order", [1, 0, 2]),
            "registration.transpose_order", int,
        )
        if sorted(transpose_order) != [0, 1, 2]:
            raise ConfigError(f"registration.transpose_order must be a permutation of 0,1,2, got {transpose_order}")
        if not _is_self_inverse_permutation(transpose_order):
            # regions.coord_transform applies transpose_order in BOTH
            # directions (reading a raw annotation file into native-matching
            # order, and -- via align.convert -- producing that raw file in
            # the first place). That's only correct if applying the same
            # permutation twice returns to the original, i.e. it's an
            # involution (identity, or a single axis swap). A genuine 3-cycle
            # would need the forward and reverse passes to use different
            # (inverse) permutations, which nothing in this codebase does
            # today -- reject it here rather than silently produce wrong
            # coordinates, matching a bug found and fixed in Phase 2/3.
            raise ConfigError(
                f"registration.transpose_order={transpose_order} is not self-inverse "
                "(applying it twice must return to the original axis order -- e.g. "
                "identity or a single axis swap like (1,0,2)). A true 3-cycle permutation "
                "is not currently supported: the coordinate-mapping code applies the same "
                "transpose_order in both directions (native->atlas prep and atlas->native "
                "reads), which is only correct for self-inverse permutations."
            )
        registration = RegistrationConfig(
            annotation_path=str(_require(registration_raw, "annotation_path", "registration")),
            hemisphere_path=registration_raw.get("hemisphere_path"),
            downsample_factor=_as_tuple3(
                _require(registration_raw, "downsample_factor", "registration"),
                "registration.downsample_factor", int,
            ),
            transpose_order=transpose_order,
        )

    # The invariant confirmed with the user: type A always has annotation.tif,
    # type B never does -- not "an annotation that happens to be native-space".
    if brain.needs_registration and registration is None:
        raise ConfigError(
            f"brain '{brain.id}': needs_registration is true but no 'registration' "
            "section was provided (a registered brain must have an annotation_path)"
        )
    if not brain.needs_registration and registration is not None:
        raise ConfigError(
            f"brain '{brain.id}': needs_registration is false but a 'registration' "
            "section was provided. Brain type B has no annotation.tif at all -- "
            "remove the 'registration' section or set needs_registration: true."
        )

    coloc_raw = (raw.get("colocalization") or {}).get("pairs", []) or []
    coloc_pairs = tuple(
        ColocalizationPair(larger=pair["larger"], smaller=pair["smaller"])
        for pair in coloc_raw
    )
    for pair in coloc_pairs:
        for bm_name in (pair.larger, pair.smaller):
            if bm_name not in biomarkers:
                raise ConfigError(
                    f"colocalization pair references unknown biomarker '{bm_name}'; "
                    f"configured biomarkers: {sorted(biomarkers)}"
                )

    paths_raw = _require(raw, "paths", "config")
    paths = PathsConfig(
        structures_csv=str(_require(paths_raw, "structures_csv", "paths")),
        output_dir=str(_require(paths_raw, "output_dir", "paths")),
    )

    return RunConfig(
        brain=brain,
        biomarkers=biomarkers,
        paths=paths,
        registration=registration,
        colocalization_pairs=coloc_pairs,
        source_path=source_path,
    )
