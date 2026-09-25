"""Behavioral checks for the experimental protostructure-first canonicalizer."""

from collections import defaultdict
from fractions import Fraction as F

import pytest
from httk.core import FracVector
from test_lift_p1 import _SCRAMBLE_BATTERY, _scrambled_p1

import httk.atomistic.symmetry.canonical_protostructure as protostructure_module
import httk.atomistic.symmetry.lift as lift_module
from httk.atomistic import (
    Assembly,
    ASUStructure,
    Cell,
    Spacegroup,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    canonical_asu_protostructure,
    data,
)
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.lift import (
    _apply_normalizer_operation,
    _canonical_without_bfs,
    _discrete_normalizer_translations,
)


def _species(*names: str) -> tuple[Species, ...]:
    return tuple(Species(name, (name,), (1,)) for name in names)


def _key(asu: ASUStructure) -> tuple[object, ...]:
    """The full exact canonical output, excluding only non-representational caches."""
    return asu.spacegroup, asu.cell.basis, asu.wyckoff_sites, asu.species, asu.charge


def _anonymous_pattern(asu: ASUStructure) -> tuple[tuple[str, ...], ...]:
    """Return the species-name-free Wyckoff occupation pattern used by the first comparison tier."""
    letters: defaultdict[str, list[str]] = defaultdict(list)
    for site in asu.wyckoff_sites:
        letters[site.species].append(site.wyckoff)
    return tuple(sorted(tuple(sorted(value)) for value in letters.values()))


def _fixed() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        225,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")),
        _species("Na", "Cl"),
        charge=F(1),
    )


def _free() -> ASUStructure:
    return ASUStructure(
        Cell(((F(46, 10), 0, 0), (0, F(46, 10), 0), (0, 0, 3))),
        136,
        (WyckoffSite("a", FracVector(()), "Ti"), WyckoffSite("f", FracVector((F(3, 10),)), "O")),
        _species("Ti", "O"),
    )


def _mixed_repeated() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        15,
        (
            WyckoffSite("a", FracVector(()), "Na"),
            WyckoffSite("e", FracVector((F(1, 7),)), "O"),
            WyckoffSite("e", FracVector((F(1, 3),)), "O"),
        ),
        _species("Na", "O"),
    )


@pytest.mark.parametrize("build", (_fixed, _free, _mixed_repeated), ids=("fixed", "free", "mixed-repeated"))
def test_exact_seam_is_idempotent_for_fixed_free_and_repeated_occupations(build: object) -> None:
    first = protostructure_module._canonical_protostructure_asu(build())  # type: ignore[operator]
    second = protostructure_module._canonical_protostructure_asu(first)

    assert _key(second) == _key(first)


def test_discrete_pattern_precedes_metric_tie_breaking() -> None:
    """SG 47 exposes the deliberate difference from the old metric-first terminal form."""
    source = ASUStructure(
        Cell(((3, 0, 0), (0, 4, 0), (0, 0, 5))),
        47,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("c", FracVector(()), "O")),
        _species("Na", "O"),
    )

    # This is an independent finite enumeration of the documented normalizer domain.  The generic
    # exact rematcher is an auxiliary oracle for applying tabulated affine actions; it does not rank
    # candidates and therefore cannot supply the discrete-first decision under test.
    group = source.spacegroup
    representatives = [AffineOperation.identity()]
    representatives.extend(
        AffineOperation.from_record(record)
        for record in data.affine_normalizer_coset_record(group.hall_entry).get("affine_normalizer_cosets", ())
    )
    images: list[ASUStructure] = []
    identity = FracVector.eye((3, 3))
    for representative in representatives:
        image = _apply_normalizer_operation(source, representative, trusted=True)
        if image is None:
            continue
        for translation in _discrete_normalizer_translations(group):
            shifted = _apply_normalizer_operation(
                image,
                AffineOperation(identity, FracVector(translation)),
                trusted=True,
            )
            if shifted is not None:
                images.append(shifted)
    assert images
    expected = min(_anonymous_pattern(image) for image in images)

    old = _canonical_without_bfs(source)
    result = protostructure_module._canonical_protostructure_asu(source)
    assert _anonymous_pattern(old) == (("a",), ("c",))
    assert expected == (("a",), ("b",))
    assert _anonymous_pattern(result) == expected


def test_same_orbit_free_parameter_and_nonstandard_setting_normalize_identically() -> None:
    free = _free()
    # The second branch of 136:4f expresses the same O orbit with x -> -x.
    alternate = ASUStructure(
        free.cell,
        free.spacegroup,
        (WyckoffSite("a", FracVector(()), "Ti"), WyckoffSite("f", FracVector((F(7, 10),)), "O")),
        free.species,
    )
    transform = Spacegroup.from_setting("15:c1").transform_from_standard
    nonstandard = ASUStructure(
        Cell(transform.basis_to_setting(((5, 0, 0), (0, 6, 0), (0, 0, 7)))),
        15,
        (WyckoffSite("e", FracVector((F(1, 3),)), "Si"),),
        _species("Si"),
        transform=transform,
    )
    standard = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        15,
        (WyckoffSite("e", FracVector((F(1, 3),)), "Si"),),
        _species("Si"),
    )

    assert _key(protostructure_module._canonical_protostructure_asu(alternate)) == _key(
        protostructure_module._canonical_protostructure_asu(free)
    )
    assert _key(protostructure_module._canonical_protostructure_asu(nonstandard)) == _key(
        protostructure_module._canonical_protostructure_asu(standard)
    )


def test_continuous_and_centred_origin_translations_have_one_exact_result() -> None:
    continuous = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        4,
        (WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Si"),),
        _species("Si"),
    )
    centred = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        216,
        (WyckoffSite("b", FracVector(()), "Na"), WyckoffSite("c", FracVector(()), "O")),
        _species("Na", "O"),
    )
    identity = FracVector.eye((3, 3))
    continuous_shifted = _apply_normalizer_operation(
        continuous, AffineOperation(identity, FracVector((0, F(1, 5), 0))), trusted=True
    )
    centred_shifted = _apply_normalizer_operation(
        centred,
        AffineOperation(identity, FracVector((F(1, 4), F(1, 4), F(1, 4)))),
        trusted=True,
    )
    assert continuous_shifted is not None
    assert centred_shifted is not None
    assert continuous_shifted.wyckoff_sites != continuous.wyckoff_sites
    assert centred_shifted.wyckoff_sites != centred.wyckoff_sites

    assert _key(protostructure_module._canonical_protostructure_asu(continuous_shifted)) == _key(
        protostructure_module._canonical_protostructure_asu(continuous)
    )
    assert _key(protostructure_module._canonical_protostructure_asu(centred_shifted)) == _key(
        protostructure_module._canonical_protostructure_asu(centred)
    )


@pytest.mark.parametrize("build", (_fixed, _free, _mixed_repeated), ids=("fixed", "free", "mixed-repeated"))
def test_exact_terminal_oracle_confirms_crystal_preservation(build: object) -> None:
    """The old exact form independently sees the input and new result as one conventional crystal."""
    source = build()  # type: ignore[operator]
    result = protostructure_module._canonical_protostructure_asu(source)
    assert _key(_canonical_without_bfs(result)) == _key(_canonical_without_bfs(source))


@pytest.mark.parametrize(
    "cell",
    (
        ((3, 0, 0), (F(1, 2), 4, 0), (F(1, 3), F(2, 5), 5)),
        ((3, 0, 0), (0, 3, 0), (F(1, 2), F(1, 2), 4)),
    ),
    ids=("triclinic", "niggli-boundary"),
)
def test_triclinic_and_boundary_cells_are_exactly_idempotent(cell: object) -> None:
    source = ASUStructure(
        Cell(cell),  # type: ignore[arg-type]
        1,
        (
            WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "C"),
            WyckoffSite("a", FracVector((F(2, 7), F(3, 11), F(5, 13))), "O"),
        ),
        _species("C", "O"),
    )
    first = protostructure_module._canonical_protostructure_asu(source)
    assert _key(protostructure_module._canonical_protostructure_asu(first)) == _key(first)


def test_chirality_policy_and_old_terminal_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ASUStructure(
        Cell(((7, 0, 0), (0, 7, 0), (0, 0, 7))),
        213,
        (WyckoffSite("c", FracVector((F(1, 13),)), "Si"),),
        _species("Si"),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the protostructure-first path must not delegate to the old terminal canonicalizer")

    monkeypatch.setattr(lift_module, "_terminal_normal_form", forbidden)
    monkeypatch.setattr(lift_module, "_canonical_without_bfs", forbidden)
    # Cover direct imports as well as lift-module lookup.  The aliases are intentionally patched only
    # when present so this test remains about the forbidden legacy seam, not an import implementation.
    for name in ("_terminal_normal_form", "_canonical_without_bfs"):
        if hasattr(protostructure_module, name):
            monkeypatch.setattr(protostructure_module, name, forbidden)

    preserved = protostructure_module._canonical_protostructure_asu(source, preserve_chirality=True)
    normalized = protostructure_module._canonical_protostructure_asu(source, preserve_chirality=False)
    assert preserved.spacegroup.it_number == 213
    assert normalized.spacegroup.it_number == 212


@pytest.mark.parametrize(
    "kind",
    ("moments", "assemblies", "molecular"),
)
def test_unsupported_information_is_rejected_before_canonicalization(kind: str) -> None:
    site = WyckoffSite("a", FracVector(()), "Na")
    species = _species("Na")
    kwargs: dict[str, object] = {}
    if kind == "moments":
        site = WyckoffSite("a", FracVector(()), "Na", moment=CollinearSiteMoments((1,)))
    elif kind == "assemblies":
        kwargs["assemblies"] = (Assembly(((0,),), (1,)),)
    elif kind == "molecular":
        kwargs["molecular"] = True
    source = ASUStructure(Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), 221, (site,), species, **kwargs)

    with pytest.raises(ValueError, match=kind[:-1] if kind == "assemblies" else kind):
        protostructure_module._canonical_protostructure_asu(source)


@pytest.mark.parametrize("kind", ("moments", "molecular"))
def test_public_wrapper_rejects_unsupported_information_before_recognition(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = WyckoffSite("a", FracVector(()), "Na")
    kwargs: dict[str, object] = {}
    if kind == "moments":
        site = WyckoffSite("a", FracVector(()), "Na", moment=CollinearSiteMoments((1,)))
    else:
        kwargs["molecular"] = True
    source = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        221,
        (site,),
        _species("Na"),
        **kwargs,
    )

    def reached_recognition(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported data reached P1 preconditioning")

    monkeypatch.setattr(protostructure_module, "_anonymous_p1_frame", reached_recognition)
    with pytest.raises(ValueError, match=kind):
        canonical_asu_protostructure(UnitcellStructureView(source))


def test_species_decorations_survive_exact_canonicalization() -> None:
    decorated = Species(
        "Na",
        ("Na",),
        (1,),
        charges=(F(1),),
        spins=(F(1),),
        labels=("sodium",),
        attached=("H",),
        nattached=(1,),
    )
    source = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        221,
        (WyckoffSite("a", FracVector(()), "Na"),),
        (decorated,),
    )

    result = protostructure_module._canonical_protostructure_asu(source)
    assert result.species == source.species


@pytest.mark.extended
@pytest.mark.parametrize("name", tuple(_SCRAMBLE_BATTERY), ids=tuple(_SCRAMBLE_BATTERY))
def test_scrambled_p1_battery_has_a_single_protostructure_first_result(name: str) -> None:
    pytest.importorskip("spglib")
    reference = _SCRAMBLE_BATTERY[name]
    expected = canonical_asu_protostructure(UnitcellStructureView(reference))
    scrambled = canonical_asu_protostructure(_scrambled_p1(reference, 1))

    assert _key(scrambled) == _key(expected)
