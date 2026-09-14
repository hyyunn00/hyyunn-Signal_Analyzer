"""Phase 3 verification: structures-table loading, descendant lookup, and
the shared tier-rollup utility."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_analyzer.regions.structures import get_region_info, group_ids_by_tier, load_structures, rollup_tiers

REAL_STRUCTURES_CSV = Path(__file__).resolve().parents[1] / "data" / "structures.csv"


@pytest.fixture
def tiny_structures() -> pd.DataFrame:
    """A small, hand-verifiable 3-tier hierarchy: ROOT -> {A, B}, A -> A1."""
    return pd.DataFrame([
        {"acronym": "ROOT", "id": 1, "name": "Root", "structure_id_path": "/1/", "parent_structure_id": np.nan},
        {"acronym": "A", "id": 2, "name": "Region A", "structure_id_path": "/1/2/", "parent_structure_id": 1.0},
        {"acronym": "A1", "id": 3, "name": "Region A sub1", "structure_id_path": "/1/2/3/", "parent_structure_id": 2.0},
        {"acronym": "B", "id": 4, "name": "Region B", "structure_id_path": "/1/4/", "parent_structure_id": 1.0},
    ])


def test_load_real_structures_csv():
    df = load_structures(REAL_STRUCTURES_CSV)
    assert len(df) == 840  # header + 840 data rows per the file's line count
    assert {"acronym", "id", "name", "structure_id_path", "parent_structure_id"}.issubset(df.columns)
    assert df["id"].dtype.kind == "i"  # coerced to clean int, not float/scientific-notation


def test_get_region_info_on_real_acronym():
    df = load_structures(REAL_STRUCTURES_CSV)
    original_id, all_ids, name = get_region_info(df, "VI")
    assert original_id == 653
    assert name == "Abducens nucleus"
    assert 653 in all_ids


def test_get_region_info_unknown_acronym_raises():
    df = load_structures(REAL_STRUCTURES_CSV)
    with pytest.raises(ValueError, match="not found"):
        get_region_info(df, "NOT_A_REAL_ACRONYM")


def test_get_region_info_includes_descendants(tiny_structures):
    original_id, all_ids, name = get_region_info(tiny_structures, "A")
    assert original_id == 2
    assert name == "Region A"
    assert set(all_ids) == {2, 3}  # A itself + its descendant A1, not B


def test_get_region_info_leaf_has_only_itself(tiny_structures):
    original_id, all_ids, _ = get_region_info(tiny_structures, "A1")
    assert original_id == 3
    assert all_ids == [3]


def test_group_ids_by_tier(tiny_structures):
    tiers = group_ids_by_tier(tiny_structures)
    assert tiers == {3: [3], 2: [2, 4], 1: [1]}  # deepest first
    assert list(tiers.keys()) == [3, 2, 1]  # order: deepest-first


def test_rollup_tiers_aggregates_leaf_to_root(tiny_structures):
    # Leaf-level counts: A1 has 5, B has 7, A and ROOT start at 0 directly.
    summary = tiny_structures.copy()
    summary["count"] = [0, 0, 5, 7]  # ROOT, A, A1, B
    summary = summary.set_index("id", drop=False)

    sheets = rollup_tiers(summary, tiny_structures, ["count"])

    tier3 = sheets[3].set_index("structure_id")
    tier2 = sheets[2].set_index("structure_id")
    tier1 = sheets[1].set_index("structure_id")

    assert tier3.loc[3, "count"] == 5  # A1 unchanged, no children
    assert tier2.loc[2, "count"] == 5  # A picks up A1's 5 (A itself had 0)
    assert tier2.loc[4, "count"] == 7  # B unchanged, no children
    assert tier1.loc[1, "count"] == 12  # ROOT = A's (now 5) + B's (7)
