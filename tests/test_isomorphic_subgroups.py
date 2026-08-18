"""Tests for the vendored isomorphic-subgroup dataset and declared-supercell collapse.

The ``isomorphic_subgroups_std`` dataset tabulates same-IT-number subgroup transforms for all
230 groups (indices up to 9) -- including the non-maximal composites (e.g. cubic index-8
``2x2x2``) that the Bärnighausen maximal-subgroup tables rightly exclude.  The entry path uses
them to collapse a declared-symmetry (SG >= 2) exact supercell before the upward search, the
declared-group counterpart of the P1 primitive reduction.
"""

from fractions import Fraction as F
from typing import Any

import pytest
from httk.core import FracVector, SurdVector

import httk.atomistic.symmetry.lift as lift_module
import httk.atomistic.symmetry.subgroups as subgroups_module
from httk.atomistic import (
    ASUStructure,
    Cell,
    Species,
    WyckoffSite,
    canonicalize,
    data,
)
from httk.atomistic.symmetry import isomorphic_subgroup_transforms, subgroup_transforms
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.subgroups import SubgroupTransform, _child_sites


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _descend(parent: ASUStructure, transform: SubgroupTransform) -> ASUStructure:
    """Build the declared-supercell description of ``parent`` through one self-transform."""
    sites = _child_sites(parent, transform)
    matrix = transform.operation.matrix.T()
    cell = Cell(SurdVector(matrix) * SurdVector(parent.cell.basis), periodicity=parent.cell.periodicity)
    return ASUStructure(cell, transform.subgroup, list(sites), parent.species, transform=SettingTransform.identity())


def _result_key(result: Any) -> tuple[Any, ...]:
    return (
        result.spacegroup.it_number,
        tuple(
            sorted(
                (site.species, site.wyckoff, tuple(site.free_params.to_fractions()))
                for site in result.asu.wyckoff_sites
            )
        ),
        tuple(tuple(value) for value in result.asu.cell.basis.to_floats()),
    )


def _by_index(it_number: int) -> dict[int, SubgroupTransform]:
    transforms: dict[int, SubgroupTransform] = {}
    for transform in isomorphic_subgroup_transforms(it_number):
        transforms.setdefault(transform.index, transform)
    return transforms


_TRICLINIC = ASUStructure(
    Cell(((5, 0, 0), (F(1, 2), 6, 0), (F(-1, 3), F(1, 4), 7))),
    2,
    [WyckoffSite("i", FracVector((F(1, 5), F(1, 7), F(1, 9))), "Si")],
    _species("Si"),
)
_RUTILE = ASUStructure(
    Cell(((F(459, 100), 0, 0), (0, F(459, 100), 0), (0, 0, F(296, 100)))),
    136,
    [WyckoffSite("a", FracVector(()), "Ti"), WyckoffSite("f", FracVector((F(61, 200),)), "O")],
    _species("Ti", "O"),
)


def test_dataset_accessor_schema_and_pinned_entry() -> None:
    record = data.isomorphic_subgroup_record(2)
    assert record["it_number"] == 2
    items = record["isomorphic_subgroups"]["items"]
    assert {"index", "wyckoff_splitting", "affine_transformation"} <= set(items[0])
    # Pinned: the SG 2 index-2 tabulation is the c-doubling.
    index_two = next(item for item in items if item["index"] == 2)
    assert index_two["affine_transformation"]["matrix"] == [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "2"]]
    # Every group is covered, and unknown numbers raise.
    for it_number in (1, 14, 166, 221, 230):
        assert data.isomorphic_subgroup_record(it_number)["it_number"] == it_number
    with pytest.raises(KeyError):
        data.isomorphic_subgroup_record(0)


def test_transforms_carry_the_baernighausen_conventions_and_skip_index_one() -> None:
    transforms = isomorphic_subgroup_transforms(2)
    assert transforms and all(transform.index > 1 for transform in transforms)
    assert all(
        transform.parent is transform.subgroup or transform.parent == transform.subgroup for transform in transforms
    )
    assert all(
        (transform.subgroup_type, transform.k_subtype) == ("k", "enlarged_unit_cell") for transform in transforms
    )
    # Cubic groups carry only the non-maximal 2x2x2 composite -- the entry class absent from the
    # Bärnighausen tables because cubic isomorphic subgroups are never maximal.
    assert sorted({transform.index for transform in isomorphic_subgroup_transforms(221)}) == [8]


def test_sg2_dataset_and_vendored_self_entries_differ_only_in_tabulation_choice() -> None:
    # Both sources tabulate an SG 2 index-2 isomorphic transform with identical typing; their
    # affine matrices are different family members (vendored: a-doubling with shear; dataset:
    # c-doubling).  Both must descend the same crystal to a genuine 2x supercell.
    vendored = next(t for t in subgroup_transforms(2, 2) if t.index == 2)
    dataset = _by_index(2)[2]
    assert (vendored.subgroup_type, vendored.k_subtype) == (dataset.subgroup_type, dataset.k_subtype)
    assert vendored.operation.matrix != dataset.operation.matrix  # tabulation choice differs
    for transform in (vendored, dataset):
        supercell = _descend(_TRICLINIC, transform)
        assert len(supercell.expand_sites()) == 2 * len(_TRICLINIC.expand_sites())


@pytest.mark.parametrize(
    ("reference", "index"),
    [(_TRICLINIC, 2), (_RUTILE, 3)],
    ids=["P-1-index-2", "rutile-136-index-3"],
)
def test_declared_supercell_recovers_the_exact_representative(reference: ASUStructure, index: int) -> None:
    base = canonicalize(reference, tolerance=1e-3)
    supercell = _descend(reference, _by_index(reference.spacegroup.it_number)[index])
    assert len(supercell.expand_sites()) == index * len(reference.expand_sites())
    assert _result_key(canonicalize(supercell, tolerance=1e-3)) == _result_key(base)


def test_cubic_declared_supercell_collapses_with_and_without_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cubic isomorphic subgroups are never maximal, so a declared 221 doubled-cell description can
    # collapse through the plain cross-group chain (221(2a) -> 225(2a) -> 221(a)) even without the
    # self-lift loop; with the loop it collapses at entry.  Both routes must land the same key.
    polonium = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), 221, [WyckoffSite("a", FracVector(()), "Po")], _species("Po")
    )
    base = canonicalize(polonium, tolerance=1e-3)
    supercell = _descend(polonium, _by_index(221)[8])
    assert len(supercell.expand_sites()) == 8
    assert _result_key(canonicalize(supercell, tolerance=1e-3)) == _result_key(base)
    monkeypatch.setattr(lift_module, "_isomorphic_reduced_entry", lambda structure: structure)
    assert _result_key(canonicalize(supercell, tolerance=1e-3)) == _result_key(base)


def test_non_supercell_entry_never_reaches_the_self_lift_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guard (one expansion + exact translation check) is the only cost for ordinary input: the
    # loop body -- which is what consults the isomorphic dataset -- must never fire.
    calls = [0]
    real = subgroups_module.isomorphic_subgroup_transforms

    def counting(spacegroup: Any) -> Any:
        calls[0] += 1
        return real(spacegroup)

    monkeypatch.setattr(subgroups_module, "isomorphic_subgroup_transforms", counting)
    result = canonicalize(_RUTILE, tolerance=1e-3)
    assert result.spacegroup.it_number == 136
    assert calls[0] == 0


def test_near_supercell_is_not_reduced() -> None:
    # Exact-only contract: a supercell description with one orbit nudged off the exact translation
    # relation has no pseudo-translations and passes through the reduction untouched.
    supercell = _descend(_RUTILE, _by_index(136)[3])
    nudge = next(position for position, site in enumerate(supercell.wyckoff_sites) if site.free_params.to_fractions())
    nudged_sites = [
        WyckoffSite(
            site.wyckoff,
            FracVector(tuple(value + F(1, 10000) for value in site.free_params.to_fractions()))
            if position == nudge
            else site.free_params,
            site.species,
        )
        for position, site in enumerate(supercell.wyckoff_sites)
    ]
    nudged = ASUStructure(
        supercell.cell, supercell.spacegroup, nudged_sites, supercell.species, transform=SettingTransform.identity()
    )
    assert lift_module._pseudo_translations(nudged) == []
    reduced = lift_module._isomorphic_reduced_entry(nudged)
    assert len(reduced.expand_sites()) == len(nudged.expand_sites())


def test_composite_supercell_collapses_stepwise_within_budget() -> None:
    # A 4x supercell built as index-2 twice has no single tabulated index-4 route in general; the
    # loop must reach the base by iterating divisor-filtered index-2 steps.  The implied-index
    # filter is what keeps this fast: without it, table order burns tens of seconds on
    # non-dividing indices before every landing (measured 65 s on one failing index-5 attempt).
    import time

    transform = _by_index(2)[2]
    quadruple = _descend(_descend(_TRICLINIC, transform), transform)
    assert len(quadruple.expand_sites()) == 4 * len(_TRICLINIC.expand_sites())
    base = canonicalize(_TRICLINIC, tolerance=1e-3)
    start = time.monotonic()
    result = canonicalize(quadruple, tolerance=1e-3)
    elapsed = time.monotonic() - start
    assert _result_key(result) == _result_key(base)
    assert elapsed < 30.0, f"composite collapse took {elapsed:.1f}s"


def test_centred_declared_structures_have_no_pseudo_translations(monkeypatch: pytest.MonkeyPatch) -> None:
    # Centring-filter regression tripwire: a centred group's full-cell atom set is invariant under
    # its own centring translations, which must never count as pseudosymmetry -- otherwise every
    # centred structure would pay a multi-second self-lift detour on entry.
    calls = [0]
    real = subgroups_module.isomorphic_subgroup_transforms

    def counting(spacegroup: Any) -> Any:
        calls[0] += 1
        return real(spacegroup)

    monkeypatch.setattr(subgroups_module, "isomorphic_subgroup_transforms", counting)
    nacl = ASUStructure(
        Cell(((F(564, 100), 0, 0), (0, F(564, 100), 0), (0, 0, F(564, 100)))),
        225,
        [WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")],
        _species("Na", "Cl"),
    )
    c_centred = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (F(-2), 0, 7))),
        12,
        [WyckoffSite("i", FracVector((F(1, 5), F(1, 7))), "Bi")],
        _species("Bi"),
    )
    assert lift_module._pseudo_translations(nacl) == []
    assert lift_module._pseudo_translations(c_centred) == []
    # The loop body (which is what consults the dataset) never fires for either.
    assert lift_module._isomorphic_reduced_entry(c_centred) is c_centred
    result = canonicalize(nacl, tolerance=1e-3)
    assert result.spacegroup.it_number == 225
    assert calls[0] == 0
