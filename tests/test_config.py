"""Phase 0 verification: YAML config loading/validation, including the
needs_registration <-> registration-section invariant confirmed with the user
(brain type A always has annotation.tif, brain type B never does)."""
import pytest
import yaml

from signal_analyzer.common.config import ConfigError, load_config


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _base_biomarkers():
    return {
        "cFos": {
            "mask_path": "cfos_mask.tif",
            "detection": {"connectivity": 18, "z_chunk": 100},
            "filter": {"min_volume_voxels": 25, "max_volume_voxels": 10000},
        },
        "TH": {
            "mask_path": "th_mask.tif",
            "filter": {"min_volume_voxels": 10, "max_volume_voxels": 5000},
        },
    }


def test_load_type_a_config_with_registration(tmp_path):
    cfg_path = _write_yaml(tmp_path / "brain_a.yaml", {
        "brain": {"id": "Hoa_PD-2", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": True},
        "biomarkers": _base_biomarkers(),
        "registration": {
            "annotation_path": "annotation.tif",
            "hemisphere_path": "hemisphere.tif",
            "downsample_factor": [5, 5, 2],
            "transpose_order": [1, 0, 2],
        },
        "colocalization": {"pairs": [{"larger": "TH", "smaller": "cFos"}]},
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    cfg = load_config(cfg_path)

    assert cfg.brain.id == "Hoa_PD-2"
    assert cfg.brain.needs_registration is True
    assert cfg.brain.voxel_size_um == (4.0, 1.82, 1.82)
    assert cfg.registration is not None
    assert cfg.registration.downsample_factor == (5, 5, 2)
    assert cfg.registration.transpose_order == (1, 0, 2)
    assert cfg.biomarker("cFos").filter.min_volume_voxels == 25
    # detection defaults apply when omitted (TH has no "detection" block)
    assert cfg.biomarker("TH").detection.connectivity == 18
    assert len(cfg.colocalization_pairs) == 1
    assert cfg.colocalization_pairs[0].larger == "TH"


def test_load_type_b_config_without_registration(tmp_path):
    cfg_path = _write_yaml(tmp_path / "brain_b.yaml", {
        "brain": {"id": "Simple-1", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": False},
        "biomarkers": _base_biomarkers(),
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    cfg = load_config(cfg_path)

    assert cfg.brain.needs_registration is False
    assert cfg.registration is None


def test_needs_registration_true_requires_registration_section(tmp_path):
    cfg_path = _write_yaml(tmp_path / "bad_a.yaml", {
        "brain": {"id": "Missing-Reg", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": True},
        "biomarkers": _base_biomarkers(),
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    with pytest.raises(ConfigError, match="no 'registration' section"):
        load_config(cfg_path)


def test_needs_registration_false_forbids_registration_section(tmp_path):
    """Confirms the user's correction is enforced: brain type B has no annotation.tif
    at all, so a registration section on a non-registered brain is rejected outright
    rather than silently accepted as 'an annotation that happens to be native-space'."""
    cfg_path = _write_yaml(tmp_path / "bad_b.yaml", {
        "brain": {"id": "Spurious-Reg", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": False},
        "biomarkers": _base_biomarkers(),
        "registration": {
            "annotation_path": "annotation.tif",
            "downsample_factor": [5, 5, 2],
        },
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    with pytest.raises(ConfigError, match="has no annotation.tif at all"):
        load_config(cfg_path)


def test_min_volume_must_be_less_than_max(tmp_path):
    biomarkers = _base_biomarkers()
    biomarkers["cFos"]["filter"] = {"min_volume_voxels": 100, "max_volume_voxels": 50}
    cfg_path = _write_yaml(tmp_path / "bad_filter.yaml", {
        "brain": {"id": "Bad-Filter", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": False},
        "biomarkers": biomarkers,
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    with pytest.raises(ConfigError, match="must be <"):
        load_config(cfg_path)


def test_colocalization_pair_must_reference_known_biomarker(tmp_path):
    cfg_path = _write_yaml(tmp_path / "bad_coloc.yaml", {
        "brain": {"id": "Bad-Coloc", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": False},
        "biomarkers": _base_biomarkers(),
        "colocalization": {"pairs": [{"larger": "TH", "smaller": "NotConfigured"}]},
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    with pytest.raises(ConfigError, match="unknown biomarker"):
        load_config(cfg_path)


def test_defaults_are_deep_merged_and_overridable(tmp_path):
    defaults_path = _write_yaml(tmp_path / "defaults.yaml", {
        "biomarkers": {
            "cFos": {
                "mask_path": "PLACEHOLDER",
                "filter": {"min_volume_voxels": 25, "max_volume_voxels": 10000},
            },
        },
    })
    brain_path = _write_yaml(tmp_path / "brain.yaml", {
        "brain": {"id": "Merged", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": False},
        "biomarkers": {
            "cFos": {
                "mask_path": "actual_mask.tif",  # overrides the default placeholder
            },
        },
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    cfg = load_config(brain_path, defaults_path=defaults_path)

    assert cfg.biomarker("cFos").mask_path == "actual_mask.tif"
    # filter bounds came from defaults, since brain.yaml didn't override them
    assert cfg.biomarker("cFos").filter.min_volume_voxels == 25


def test_transpose_order_must_be_self_inverse(tmp_path):
    """regions.coord_transform applies transpose_order the same way in both
    the native->atlas-prep direction (align.convert) and the atlas->native
    direction (resample_atlas_volume_to_native) -- correct only when
    applying it twice returns to the original order. A genuine 3-cycle
    (not just a swap) would silently produce wrong coordinates, so config
    loading rejects it outright. This is a real bug found while building
    Phase 3's region extraction: an earlier version of
    resample_atlas_volume_to_native used the WRONG direction and only
    happened to test correctly because (1,0,2) is self-inverse."""
    biomarkers = _base_biomarkers()
    cfg_path = _write_yaml(tmp_path / "bad_transpose.yaml", {
        "brain": {"id": "Bad-Transpose", "voxel_size_um": [4.0, 1.82, 1.82], "needs_registration": True},
        "biomarkers": biomarkers,
        "registration": {
            "annotation_path": "annotation.tif",
            "downsample_factor": [5, 5, 2],
            "transpose_order": [2, 0, 1],  # a genuine 3-cycle, NOT self-inverse
        },
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    with pytest.raises(ConfigError, match="not self-inverse"):
        load_config(cfg_path)


def test_missing_required_field_raises(tmp_path):
    cfg_path = _write_yaml(tmp_path / "missing.yaml", {
        "brain": {"id": "NoVoxelSize", "needs_registration": False},
        "biomarkers": _base_biomarkers(),
        "paths": {"structures_csv": "data/structures.csv", "output_dir": "out/"},
    })

    with pytest.raises(ConfigError, match="voxel_size_um"):
        load_config(cfg_path)
