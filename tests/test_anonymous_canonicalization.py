"""Focused contracts for anonymous species geometry canonicalization."""

import datetime
import random
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import httk.core
import pytest
from httk.core import FracVector, SurdVector
from test_lift_p1 import _scrambled_p1

from httk.atomistic import (
    ASUStructure,
    Cell,
    ChemicalComposition,
    Sites,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    canonical_asu_protostructure,
    canonical_asu_protostructure_assignments,
)
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.species.classes import _species_class
from httk.atomistic.symmetry import canonical_classification as classification
from httk.atomistic.symmetry._anonymous import _anonymous_p1_frame
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_protostructure import (
    _apply_action,
    _canonical_protostructure_assignments_asu,
    _derived_precision,
    _mirrored,
    _with_basis,
)
from httk.atomistic.symmetry.subgroups import _standard_input


def _species(name: str, symbol: str, **kwargs: object) -> Species:
    return Species(name, (symbol,), (1,), **kwargs)


def _key(asu: ASUStructure) -> tuple[object, ...]:
    return asu.spacegroup, asu.cell.basis, asu.wyckoff_sites, asu.species, asu.charge


def _anonymous_geometry(asu: ASUStructure) -> tuple[object, ...]:
    """Erase class names while retaining the exact chosen frame and each occupied orbit."""
    classes: dict[str, list[tuple[str, tuple[F, ...]]]] = {}
    for site in asu.wyckoff_sites:
        classes.setdefault(site.species, []).append((site.wyckoff, tuple(site.free_params.to_fractions())))
    return asu.spacegroup, asu.cell.basis, tuple(sorted(tuple(sorted(value)) for value in classes.values()))


def _remap(asu: ASUStructure, names: dict[str, str], species: tuple[Species, ...]) -> ASUStructure:
    return ASUStructure(
        asu.cell,
        asu.spacegroup,
        tuple(WyckoffSite(site.wyckoff, site.free_params, names[site.species]) for site in asu.wyckoff_sites),
        species,
        transform=asu.transform,
        coordinate_precision=asu.coordinate_precision,
        charge=asu.charge,
    )


def _low_symmetry() -> ASUStructure:
    return ASUStructure(
        Cell(((3, 0, 0), (F(1, 3), 4, 0), (F(1, 5), F(2, 7), 5))),
        1,
        (
            WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "rare"),
            WyckoffSite("a", FracVector((F(2, 7), F(3, 11), F(5, 13))), "common"),
            WyckoffSite("a", FracVector((F(4, 7), F(5, 11), F(6, 13))), "common"),
        ),
        (_species("rare", "Na"), _species("common", "Cl")),
    )


def _polar_equal_population() -> ASUStructure:
    return ASUStructure(
        Cell(((4, 0, 0), (0, 5, 0), (0, 0, 6))),
        4,
        (
            WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "left"),
            WyckoffSite("a", FracVector((F(2, 7), F(3, 11), F(5, 13))), "right"),
        ),
        (_species("left", "Si"), _species("right", "O")),
    )


def _nacl() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        225,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")),
        (_species("Na", "Na"), _species("Cl", "Cl")),
    )


def _substitution(source: ASUStructure) -> tuple[ASUStructure, dict[str, str]]:
    names = {species.name: f"changed_{index}" for index, species in enumerate(source.species)}
    changed_species = tuple(
        _species(names[species.name], "K" if index == 0 else "Br") for index, species in enumerate(source.species)
    )
    return _remap(source, names, changed_species), names


def _restore(source: ASUStructure, result: ASUStructure, names: dict[str, str]) -> ASUStructure:
    # the results carry species classes: K and Br stand for the source's first and second species;
    # a source already in classes keeps its species order, any other lists its classes by name
    classes = [_species_class(species) for species in source.species]
    back = {"K": classes[0].name, "Br": classes[1].name}
    order = classes if classes == list(source.species) else sorted(classes, key=lambda species: species.name)
    return _remap(result, back, tuple(order))


@pytest.mark.parametrize("build", (_low_symmetry, _polar_equal_population), ids=("low-symmetry", "polar-equal"))
def test_unique_assignments_are_species_substitution_covariant_and_scramble_invariant(build: object) -> None:
    source = build()  # type: ignore[operator]
    changed, names = _substitution(source)

    family = canonical_asu_protostructure_assignments(UnitcellStructureView(source))
    assert len(family) == 1
    original = canonical_asu_protostructure(UnitcellStructureView(source))
    substituted = canonical_asu_protostructure(UnitcellStructureView(changed))
    assert _key(canonical_asu_protostructure(original)) == _key(original)
    assert _key(_restore(source, substituted, names)) == _key(original)

    scrambled = canonical_asu_protostructure(_scrambled_p1(source, 17))
    scrambled_changed = canonical_asu_protostructure(_scrambled_p1(changed, 17))
    assert _key(scrambled) == _key(original)
    assert _key(_restore(source, scrambled_changed, names)) == _key(original)


def test_na_cl_class_exchange_has_family_covariance_and_scalar_anonymous_invariance() -> None:
    source = _nacl()
    changed, names = _substitution(source)

    family = canonical_asu_protostructure_assignments(UnitcellStructureView(source))
    substituted_family = canonical_asu_protostructure_assignments(UnitcellStructureView(changed))
    restored = {_key(_restore(source, value, names)) for value in substituted_family}

    assert len(family) > 1
    assert len(family) == len({_key(value) for value in family})
    assert {_key(value) for value in family} == restored
    assert len({_anonymous_geometry(value) for value in family}) == 1
    assert _anonymous_geometry(canonical_asu_protostructure(UnitcellStructureView(source))) == _anonymous_geometry(
        canonical_asu_protostructure(UnitcellStructureView(changed))
    )
    scrambled_family = canonical_asu_protostructure_assignments(_scrambled_p1(source, 23))
    scrambled_changed = canonical_asu_protostructure_assignments(_scrambled_p1(changed, 23))
    assert {_key(value) for value in scrambled_family} == {_key(value) for value in family}
    assert {_key(_restore(source, value, names)) for value in scrambled_changed} == {_key(value) for value in family}


def test_exact_assignment_family_is_idempotent_and_keeps_all_ties() -> None:
    family = _canonical_protostructure_assignments_asu(_nacl())
    assert len(family) > 1
    expected = {_key(value) for value in family}
    for value in family:
        assert {_key(item) for item in _canonical_protostructure_assignments_asu(value)} == expected


def test_anonymous_frame_rejects_coincident_species_classes() -> None:
    duplicate = UnitcellStructure(
        Cell(((3, 0, 0), (0, 4, 0), (0, 0, 5))),
        ((F(1, 7), F(2, 11), F(3, 13)), (F(1, 7), F(2, 11), F(3, 13))),
        (_species("first", "Na"), _species("second", "Cl")),
        ("first", "second"),
    )

    with pytest.raises(ValueError):
        _anonymous_p1_frame(duplicate)


def test_exact_result_preserves_decorations_unused_species_charge_and_metadata() -> None:
    active = _species(
        "active",
        "Na",
        mass=(22.99,),
        original_name="Na_source",
        attached=("H",),
        nattached=(1,),
        concentration_precision=(F(1, 100),),
        charges=(F(1),),
        spins=(F(1, 2),),
        labels=("sodium",),
    )
    unused = _species("unused", "Cl", labels=("unused-chlorine",))
    timestamp = datetime.datetime(2026, 9, 24, 12, 0, tzinfo=datetime.UTC)
    source = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5)), precision=F(1, 100)),
        221,
        (WyckoffSite("a", FracVector(()), "active"),),
        (active, unused),
        coordinate_precision=F(1, 1000),
        chemical_composition=ChemicalComposition({"Na": 1}, mode="full"),
        chemical_formula_descriptive="Na",
        chemical_formula_hill="Na",
        optimization_type="experimental",
        immutable_id="source-immutable-id",
        last_modified=timestamp,
        charge=F(3),
    )

    result = _canonical_protostructure_assignments_asu(source)[0]
    assert result.species == source.species
    assert result.charge == source.charge
    assert result.coordinate_precision == source.coordinate_precision
    assert result.chemical_composition == source.chemical_composition
    assert result.chemical_formula_descriptive == source.chemical_formula_descriptive
    assert result.chemical_formula_hill == source.chemical_formula_hill
    assert result.optimization_type == source.optimization_type
    assert result.immutable_id == source.immutable_id
    assert result.last_modified == source.last_modified


def test_precision_is_conservative_after_shear() -> None:
    source = _low_symmetry()
    precise = ASUStructure(
        Cell(source.cell.basis, precision=F(1, 100)),
        source.spacegroup,
        source.wyckoff_sites,
        source.species,
        coordinate_precision=F(1, 1000),
    )
    shear = FracVector(((1, 1, 0), (0, 1, 0), (0, 0, 1)))
    inverse = shear.inv()
    sheared = ASUStructure(
        Cell(SurdVector(shear) * precise.cell.basis, precision=F(1, 50)),
        1,
        tuple(
            WyckoffSite("a", (site.free_params * inverse).normalize(), site.species) for site in precise.wyckoff_sites
        ),
        precise.species,
        coordinate_precision=F(1, 500),
    )

    result = _canonical_protostructure_assignments_asu(sheared)[0]
    assert UnitcellStructureView(result).cartesian_precision() >= UnitcellStructureView(precise).cartesian_precision()


FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_with_precisions(name: str) -> ASUStructure:
    """A fixture in its standard setting, keeping its stated cell and coordinate precisions."""
    try:
        loaded = httk.core.load(str(FIXTURES / name))
    except ValueError:  # an unrecognised declared setting
        loaded = httk.core.load(str(FIXTURES / name), repair=True)
    standard = _standard_input(loaded)
    return ASUStructure(
        standard.cell,
        standard.spacegroup,
        [WyckoffSite(site.wyckoff, site.free_params, site.species) for site in standard.wyckoff_sites],
        standard.species,
        coordinate_precision=standard.coordinate_precision,
    )


@pytest.mark.parametrize("number", (146, 149, 151, 152, 153))
def test_mirror_pair_records_the_source_precision(number: int) -> None:
    # FU4 §6: a hand and its mirror image give one refined value; the recorded precisions were
    # path products (146: 1/50000 and 1/12500) and are now those of the source
    hand = _fixture_with_precisions(f"structreading/{number}.cif")
    values = [
        classification.canonical_protostructure(Protostructure(representative=value), preserve_chirality=False)
        for value in (hand, _mirrored(hand))
    ]
    assert values[0] == values[1]
    representative = values[0].representative
    assert representative is not None
    expected = F(1, 100000000) if number == 149 else F(1, 100000)
    assert hand.cell.precision == hand.coordinate_precision == expected
    assert representative.cell.precision == representative.coordinate_precision == expected


def _rowsum(rows: tuple[tuple[int, ...], ...]) -> int:
    return max(sum(abs(value) for value in row) for row in rows)


def _valid_derivation_least_rowsum(source: ASUStructure, result: ASUStructure) -> int:
    """Return the least row sum of a unimodular ``T`` (entries up to 3) deriving ``result`` from ``source``."""
    gram = source.cell.metric().coefficient(1).to_fractions()
    target = result.cell.metric().coefficient(1).to_fractions()

    def dot(u: tuple[int, ...], v: tuple[int, ...]) -> F:
        return sum((gram[i][j] * u[i] * v[j] for i in range(3) for j in range(3)), start=F(0))

    vectors = [v for v in product(range(-3, 4), repeat=3) if any(v)]
    rows = [[v for v in vectors if dot(v, v) == target[k][k]] for k in range(3)]
    points = [tuple(site.free_params.to_fractions()) for site in source.wyckoff_sites]
    stored = sorted(tuple(value % 1 for value in site.free_params.to_fractions()) for site in result.wyckoff_sites)
    least = None
    for change in product(*rows):
        a, b, c = change
        if (dot(a, b), dot(a, c), dot(b, c)) != (target[0][1], target[0][2], target[1][2]):
            continue
        matrix = FracVector(change)
        if abs(matrix.det()) != 1:
            continue
        inverse = matrix.inv().to_fractions()
        mapped = [tuple(sum(x[i] * inverse[i][j] for i in range(3)) for j in range(3)) for x in points]
        for origin in mapped:
            shift = tuple(stored[0][j] - origin[j] for j in range(3))
            if sorted(tuple((q[j] + shift[j]) % 1 for j in range(3)) for q in mapped) == stored:
                least = _rowsum(change) if least is None else min(least, _rowsum(change))
                break
    assert least is not None
    return least


def _check_least_valid_derivation(trials: int) -> None:
    # the design review's P1 crystals in an fcc metric: the least row sum over all lattice
    # isometries can belong to one that does not derive the stored sites, which would understate
    # the precision; the recorded one is the least over the actual derivations
    rng = random.Random(3)
    for _ in range(trials):
        points = [(F(0), F(0), F(0))] + [tuple(F(rng.randint(1, 30), 31) for _ in range(3)) for _ in range(2)]
        source = ASUStructure(
            Cell(((0, 3, 3), (3, 0, 3), (3, 3, 0)), precision=F(1, 100)),
            1,
            [WyckoffSite("a", point, "Na") for point in points],
            [_species("Na", "Na")],
        )
        result = _canonical_protostructure_assignments_asu(source)[0]
        assert result.cell.precision == F(1, 100) * _valid_derivation_least_rowsum(source, result)


def test_precision_is_epsilon_times_the_least_valid_derivation() -> None:
    _check_least_valid_derivation(4)


@pytest.mark.extended
def test_precision_is_epsilon_times_the_least_valid_derivation_extended() -> None:
    _check_least_valid_derivation(12)


def test_precision_through_a_larger_result_cell() -> None:
    # the primitive rhombohedral cell of structreading/146 is stored in the triple hexagonal R3
    # cell: the derivation has |det T| = 3, so every source site has three images in the result
    view = UnitcellStructureView(httk.core.load(str(FIXTURES / "structreading/146.cif")))
    rhombohedral = FracVector(
        ((F(2, 3), F(1, 3), F(1, 3)), (F(-1, 3), F(1, 3), F(1, 3)), (F(-1, 3), F(-2, 3), F(1, 3)))
    )
    inverse = rhombohedral.inv()
    points = dict.fromkeys(
        (species, tuple((FracVector(row) * inverse).normalize().to_fractions()))
        for species, row in zip(view.species_at_sites, view.sites.reduced_coords.to_fractions(), strict=True)
    )
    basis = SurdVector(rhombohedral) * view.cell.basis
    results = [
        canonical_asu_protostructure(
            UnitcellStructure(
                Cell(hand, precision=F(1, 100000)),
                Sites([point for _species, point in points], precision=F(1, 100000)),
                view.species,
                [species for species, _point in points],
            ),
            preserve_chirality=False,
        )
        for hand in (basis, -basis)
    ]
    assert results[0] == results[1]
    assert results[0].spacegroup.it_number == 146
    assert results[0].cell.precision == F(3, 100000)


def test_precision_without_an_exact_derivation_keeps_the_path_value() -> None:
    # a tolerance-snapped recognition: the stored crystal is no exact re-description of the
    # input, so no derivation exists and the propagated path bound stays
    source = UnitcellStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5)), precision=F(1, 100)),
        Sites([(0, 0, 0), (F(501, 1000), F(1, 2), F(1, 2))], precision=F(1, 1000)),
        [_species("Na", "Na"), _species("Cl", "Cl")],
        ["Na", "Cl"],
    )
    result = canonical_asu_protostructure(source, tolerance=0.05)
    assert result.spacegroup.it_number == 221
    assert _derived_precision(result, source) is result


@pytest.mark.extended
@pytest.mark.parametrize("name", ("cod/7217003.cif", "cod/4131686.cif", "structreading/2.cif"))
def test_p1_fixture_records_its_source_precision(name: str) -> None:
    # stored in the Niggli cell, a signed permutation of the cell as read; a sheared input
    # (row sum 2) needs a derivation of row sum 2 back to it
    source = _fixture_with_precisions(name)
    sheared = _apply_action(source, AffineOperation(((1, 1, 0), (0, 1, 0), (0, 0, 1))))
    for value, factor in ((source, 1), (sheared, 2)):
        representative = classification.canonical_protostructure(Protostructure(representative=value)).representative
        assert representative is not None
        assert value.cell.precision is not None and value.coordinate_precision is not None
        assert representative.cell.precision == factor * value.cell.precision
        assert representative.coordinate_precision == factor * value.coordinate_precision


def test_exact_action_scales_coordinate_and_cell_precision() -> None:
    source = _low_symmetry()
    precise = ASUStructure(
        Cell(source.cell.basis, precision=F(1, 100)),
        1,
        source.wyckoff_sites,
        source.species,
        coordinate_precision=F(1, 1000),
    )
    matrix = FracVector(((1, 1, 0), (0, 1, 0), (0, 0, 1)))
    changed = _apply_action(precise, AffineOperation(matrix, FracVector((0, 0, 0))))
    assert changed.cell.precision == F(1, 50)
    assert changed.coordinate_precision == F(1, 500)
    basis_only = _with_basis(precise, SurdVector(matrix) * precise.cell.basis)
    assert basis_only.cell.precision == F(1, 50)
    assert basis_only.coordinate_precision == precise.coordinate_precision


def test_left_handed_chiral_input_respects_chirality_policy() -> None:
    left_handed = ASUStructure(
        Cell(((-7, 0, 0), (0, 7, 0), (0, 0, 7))),
        213,
        (WyckoffSite("c", FracVector((F(1, 13),)), "Si"),),
        (_species("Si", "Si"),),
    )
    # P4_3 32 written in a left-handed basis is a P4_1 32 crystal: it is read as the crystal
    preserved = _canonical_protostructure_assignments_asu(left_handed, preserve_chirality=True)
    normalized = _canonical_protostructure_assignments_asu(left_handed, preserve_chirality=False)
    assert {value.spacegroup.it_number for value in preserved} == {212}
    assert {value.spacegroup.it_number for value in normalized} == {212}
    assert all(value.cell.basis.det().sign() > 0 for value in (*preserved, *normalized))
    # the right-handed description of the same group keeps 213 by default
    right_handed = ASUStructure(
        Cell(((7, 0, 0), (0, 7, 0), (0, 0, 7))), 213, left_handed.wyckoff_sites, left_handed.species
    )
    assert {value.spacegroup.it_number for value in _canonical_protostructure_assignments_asu(right_handed)} == {213}


@pytest.mark.parametrize(
    "spacegroup,letter,rational",
    (
        pytest.param(1, "a", False, id="p1-irrational"),
        pytest.param(2, "i", False, id="pminus1-irrational"),
        pytest.param(1, "a", True, id="p1-rational-boundary"),
        pytest.param(2, "i", True, id="pminus1-rational-boundary", marks=pytest.mark.extended),
    ),
)
def test_proxy_gram_shear_invariance_and_idempotence(spacegroup: int, letter: str, rational: bool) -> None:
    zero = SurdVector(0)._as_scalar()
    one = SurdVector(1)._as_scalar()
    root_two = SurdVector.sqrt_of(2)
    cell = (
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5)))
        if rational
        else Cell(SurdVector._from_scalar_grid([[one, zero, zero], [root_two, one, zero], [zero, zero, one]], (3, 3)))
    )
    source = ASUStructure(
        cell,
        spacegroup,
        (
            WyckoffSite(letter, FracVector((F(1, 7), F(2, 11), F(3, 13))), "A"),
            WyckoffSite(letter, FracVector((F(2, 7), F(3, 11), F(5, 13))), "B"),
        ),
        (_species("A", "Na"), _species("B", "Cl")),
    )
    shear = FracVector(((1, 1, 0), (0, 1, 0), (0, 0, 1)))
    rebased = ASUStructure(
        Cell(SurdVector(shear) * source.cell.basis),
        spacegroup,
        tuple(
            WyckoffSite(letter, (site.free_params * shear.inv()).normalize(), site.species)
            for site in source.wyckoff_sites
        ),
        source.species,
    )
    first = _canonical_protostructure_assignments_asu(source)
    second = _canonical_protostructure_assignments_asu(rebased)
    assert {_key(value) for value in first} == {_key(value) for value in second}
    expected = {_key(value) for value in first}
    for value in first:
        assert {_key(item) for item in _canonical_protostructure_assignments_asu(value)} == expected


def test_assignment_api_is_exported_from_atomistic_and_symmetry() -> None:
    from httk.atomistic import canonical_asu_protostructure_assignments as atomistic_export
    from httk.atomistic.symmetry import canonical_asu_protostructure_assignments as symmetry_export

    assert atomistic_export is canonical_asu_protostructure_assignments
    assert symmetry_export is canonical_asu_protostructure_assignments
