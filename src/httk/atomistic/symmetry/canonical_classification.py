"""Exact canonicalization for the structural-classification value families.

Input conversion follows the corresponding View contract and can recognize a
structure-like source. Materialized bare values use occupations alone. A refined representative
uses its exact geometric terminal, so projecting it to a bare value can require a
separate bare canonicalization pass.
"""

from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from httk.core import FracVector

from httk.atomistic import data
from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
from httk.atomistic.models.bareprotostructure.like import BareProtostructureLike
from httk.atomistic.models.bareprotostructure.view import BareProtostructureView
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.bareprototype.like import BarePrototypeLike
from httk.atomistic.models.bareprototype.view import BarePrototypeView
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.protonetstructure.like import ProtonetstructureLike
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.protonetstructure.view import ProtonetstructureView
from httk.atomistic.models.protonettype.like import ProtonettypeLike
from httk.atomistic.models.protonettype.protonettype import Protonettype
from httk.atomistic.models.protonettype.view import ProtonettypeView
from httk.atomistic.models.protostructure.like import ProtostructureLike
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.protostructure.view import ProtostructureView
from httk.atomistic.models.prototype.like import PrototypeLike
from httk.atomistic.models.prototype.notation import class_order_key
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.prototype.view import PrototypeView
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structuretype.anonymize import dummy_species
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_protostructure import _canonical_protostructure_asu
from httk.atomistic.symmetry.lift import _discrete_normalizer_translations, _resetting_preserves_group
from httk.atomistic.symmetry.limits import _budget_scope, _checkpoint
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup

if TYPE_CHECKING:
    from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
    from httk.atomistic.models.prototype.occupation import PrototypeOccupation
    from httk.atomistic.symmetry.protonet import _Realisation

__all__ = [
    "canonical_bare_protostructure",
    "canonical_bare_prototype",
    "canonical_protonetstructure",
    "canonical_protonettype",
    "canonical_protostructure",
    "canonical_prototype",
]


def _actions(spacegroup: Spacegroup) -> tuple[AffineOperation, ...]:
    """Return the finite letter-changing same-group action domain."""
    identity = AffineOperation.identity()
    operations = [identity]
    if spacegroup.it_number != 1:
        record = data.affine_normalizer_coset_record(spacegroup.hall_entry)
        operations.extend(AffineOperation.from_record(value) for value in record.get("affine_normalizer_cosets", ()))
    operations = [
        operation if not any(translation) else AffineOperation(FracVector.eye((3, 3)), translation) * operation
        for operation in operations
        for translation in _discrete_normalizer_translations(spacegroup)
    ]
    inversion = AffineOperation(FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1))), (0, 0, 0))
    if _resetting_preserves_group(spacegroup, inversion.matrix):
        operations.extend(inversion * operation for operation in tuple(operations))
    return tuple(dict.fromkeys(operations))


def _mapped_occupations(value: Any, operation: AffineOperation, target: Spacegroup) -> list[tuple[str, Any]]:
    """Map materialized occupation letters through one proven action."""
    return [
        (
            compile_wyckoff_action(
                value.spacegroup, operation, occupation.wyckoff, target_spacegroup=target
            ).target_letter,
            occupation.species if hasattr(occupation, "species") else occupation.label,
        )
        for occupation in value.occupations
    ]


def _class_counts(spacegroup: Spacegroup, groups: dict[Any, tuple[str, ...]]) -> dict[Any, int]:
    """Return each class's summed conventional multiplicity."""
    return {
        key: sum(spacegroup.wyckoff_position(letter).multiplicity for letter in letters)
        for key, letters in groups.items()
    }


def _assigned_key(occupations: Iterable[tuple[str, Any]], spacegroup: Spacegroup) -> tuple[Any, ...]:
    """Order an assigned bare key without inspecting hidden geometry."""
    classes: dict[str, list[str]] = {}
    for letter, species in occupations:
        _checkpoint()
        classes.setdefault(species.name, []).append(letter)
    groups = {name: tuple(sorted(letters)) for name, letters in classes.items()}
    counts = _class_counts(spacegroup, groups)
    return (
        tuple(sorted(class_order_key(letters, counts[name])[:2] for name, letters in groups.items())),
        tuple(sorted(class_order_key(letters, counts[name], name) for name, letters in groups.items())),
    )


def _prototype_key(occupations: Iterable[tuple[str, str]], spacegroup: Spacegroup) -> tuple[Any, ...]:
    """Order an anonymous bare key without using class labels as a tie-break."""
    classes: dict[str, list[str]] = {}
    for letter, label in occupations:
        _checkpoint()
        classes.setdefault(label, []).append(letter)
    groups = {label: tuple(sorted(letters)) for label, letters in classes.items()}
    counts = _class_counts(spacegroup, groups)
    return tuple(sorted(class_order_key(letters, counts[label])[:2] for label, letters in groups.items()))


def _chirality_target(spacegroup: Spacegroup, preserve_chirality: bool) -> Spacegroup:
    """Return the lower enantiomorphic partner when the policy requests it."""
    partner = spacegroup.record["it_number_enantiomorphic"]
    if preserve_chirality or partner is None or partner >= spacegroup.it_number:
        return spacegroup
    return Spacegroup.standard(partner)


def _canonical_bare(value: Any, *, prototype: bool, preserve_chirality: bool) -> Any:
    """Canonicalize one materialized bare value through finite exact letter actions."""
    target = _chirality_target(value.spacegroup, preserve_chirality)
    chirality = AffineOperation.identity()
    if target != value.spacegroup:
        chirality = AffineOperation(FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1))), (0, 0, 0))
    candidates = []
    for action in _actions(target):
        _checkpoint()
        operation = action * chirality
        mapped = _mapped_occupations(value, operation, target)
        candidate = BarePrototype(target, mapped) if prototype else BareProtostructure(target, mapped)
        key = _prototype_key(mapped, target) if prototype else _assigned_key(mapped, target)
        candidates.append((key, candidate))
    return min(candidates, key=lambda item: item[0])[1]


def canonical_bare_protostructure(
    obj: BareProtostructureLike, *, preserve_chirality: bool = True
) -> BareProtostructure:
    """Return the finite-action canonical assigned bare classification.

    :param obj: A bare-protostructure-like value, converted through its View;
        structure-like inputs may require symmetry recognition.
    :param preserve_chirality: Whether to retain the higher enantiomorphic group.
    :return: A standalone canonical bare protostructure.
    """
    value = BareProtostructureView(obj).unview()
    return _canonical_bare(
        BareProtostructure(value.spacegroup, value.occupations), prototype=False, preserve_chirality=preserve_chirality
    )


def canonical_bare_prototype(obj: BarePrototypeLike, *, preserve_chirality: bool = True) -> BarePrototype:
    """Return the finite-action canonical anonymous bare classification.

    :param obj: A bare-prototype-like value, converted through its View;
        structure-like inputs may require symmetry recognition.
    :param preserve_chirality: Whether to retain the higher enantiomorphic group.
    :return: A standalone canonical bare prototype.
    """
    value = BarePrototypeView(obj).unview()
    return _canonical_bare(
        BarePrototype(value.spacegroup, value.occupations), prototype=True, preserve_chirality=preserve_chirality
    )


def _canonical_protonet[V](
    value: Protonettype | Protonetstructure,
    class_of: Sequence[Hashable],
    build: Callable[[int, "_Realisation", list[int], list[tuple[int, int, int, int, int, int]]], V],
    *,
    assigned: bool,
    preserve_chirality: bool,
) -> V:
    """Search the canonical net of one value and build the result with ``build``.

    ``build`` receives the target group, the realisation, the input sites in new site order and
    the 1-based rows; a translation the notation cannot write is reported as such (D4).

    The anonymous level takes the first realisation (all give the same value); the assigned level
    takes the one with the smallest species-name tuple in label order (D3).
    """
    from httk.atomistic.symmetry.protonet import _canonical_net

    occupations: Sequence[PrototypeOccupation | WyckoffOccupation] = value.occupations
    sites = [(cls, o.wyckoff) for cls, o in zip(class_of, occupations, strict=True)]
    rows = [(i - 1, j - 1, (k - 1, n1, n2, n3)) for i, j, k, n1, n2, n3 in value.net]
    result = _canonical_net(value.spacegroup.it_number, sites, rows, preserve_chirality=preserve_chirality)
    real = (
        min(result.realisations, key=lambda r: tuple(str(c) for c in r.classes)) if assigned else result.realisations[0]
    )
    order = sorted(range(len(sites)), key=real.number.__getitem__)
    net = [(i, j, g[0] + 1, g[1], g[2], g[3]) for (i, j), g in zip(result.slots, result.codes, strict=True)]
    try:
        return build(result.it_number, real, order, net)
    except ValueError as exc:
        if all(-5 <= n <= 4 for g in result.codes for n in g[1:]):
            raise
        raise ValueError(
            "the protonet notation cannot write this canonical net: a translation is outside -5..+4 (D4)"
        ) from exc


def canonical_protonettype(
    obj: ProtonettypeLike, *, preserve_chirality: bool = True, timeout: float | None = 120.0
) -> Protonettype:
    """Return the exact canonical anonymous bare prototype with its canonical bonding net.

    The net is minimised over every equivalent description, site numbering and choice of
    reference atoms; classes that tie in count and letters are ordered by the net.
    In triclinic and monoclinic groups the net string is canonical only relative to the cell
    that httk's standardisation chooses; in P-1 the tabulated equivalent descriptions are not
    closed, so two descriptions can give different strings (D7).

    :param obj: A protonettype-like value, converted through its View.
    :param preserve_chirality: Whether to retain the higher enantiomorphic group.
    :param timeout: Cooperative seconds for the search; ``None`` disables the deadline.
    :return: A standalone canonical protonettype.
    :raises ValueError: If the input is invalid, the canonical net needs a translation outside
        -5..+4 that the notation cannot write, or the net search and the bare canonicalization
        disagree on the bare prototype.
    :raises CanonicalizationLimitError: If the search exceeds its limit; no partial result is returned.
    """
    value = ProtonettypeView(obj).unview()
    labels = [o.label for o in value.occupations]
    position = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    class_of = [position[label] for label in labels]

    def build(it_number: int, real: "_Realisation", order: list[int], net: list[Any]) -> Protonettype:
        labels = [(real.letters[s], anonymous_symbol(real.classes.index(class_of[s]))) for s in order]
        return Protonettype(it_number, labels, net)

    with _budget_scope(timeout):
        result = _canonical_protonet(value, class_of, build, assigned=False, preserve_chirality=preserve_chirality)
        expected = canonical_bare_prototype(value.bare_prototype, preserve_chirality=preserve_chirality)
    if result.bare_prototype != expected:
        raise ValueError(
            f"protonet search reached bare prototype {result.bare_prototype.label} "
            f"but the canonical bare prototype is {expected.label}"
        )
    return result


def canonical_protonetstructure(
    obj: ProtonetstructureLike, *, preserve_chirality: bool = True, timeout: float | None = 120.0
) -> Protonetstructure:
    """Return the exact canonical assigned bare protostructure with its canonical bonding net.

    The net is chosen before the species (D3): it is canonicalized anonymously, and only
    descriptions giving the identical net are told apart by the species-name tuple in class
    order, smallest first. With tied classes the result's bare part can therefore differ from
    :func:`canonical_bare_protostructure`.
    In triclinic and monoclinic groups the net string is canonical only relative to the cell
    that httk's standardisation chooses; in P-1 the tabulated equivalent descriptions are not
    closed, so two descriptions can give different strings (D7).

    :param obj: A protonetstructure-like value, converted through its View.
    :param preserve_chirality: Whether to retain the higher enantiomorphic group.
    :param timeout: Cooperative seconds for the search; ``None`` disables the deadline.
    :return: A standalone canonical protonetstructure.
    :raises ValueError: If the input is invalid, the canonical net needs a translation outside
        -5..+4 that the notation cannot write, or the net search and the bare canonicalization
        disagree on the anonymous bare prototype.
    :raises CanonicalizationLimitError: If the search exceeds its limit; no partial result is returned.
    """
    value = ProtonetstructureView(obj).unview()

    def build(it_number: int, real: "_Realisation", order: list[int], net: list[Any]) -> Protonetstructure:
        return Protonetstructure(it_number, [(real.letters[s], value.occupations[s].species) for s in order], net)

    with _budget_scope(timeout):
        names = [o.species.name for o in value.occupations]
        result = _canonical_protonet(value, names, build, assigned=True, preserve_chirality=preserve_chirality)
        expected = canonical_bare_prototype(value.bare_protostructure, preserve_chirality=preserve_chirality)
    reached = canonical_bare_prototype(result.bare_protostructure, preserve_chirality=preserve_chirality)
    if reached != expected:
        raise ValueError(
            f"protonet search reached bare prototype {reached.label} but the canonical bare prototype is {expected.label}"
        )
    return result


def _exact_representative(representative: FundamentalDomainStructure) -> ASUStructure:
    """Rebuild a representative as an exact ASU after rejecting imprecise retained points."""
    sites = []
    for site in representative.wyckoff_sites:
        _checkpoint()
        if site.representative is not None:
            orbit = representative.spacegroup.wyckoff_position(site.wyckoff).coordinates(site.free_params)
            if site.representative.normalize() not in {point.normalize() for point in orbit}:
                raise ValueError(
                    "canonical classification cannot discard an imprecise retained WyckoffSite representative; "
                    "canonicalize the measured structure explicitly"
                )
        sites.append(WyckoffSite(site.wyckoff, site.free_params, site.species, moment=site.moment))
    return ASUStructure(
        representative.cell,
        representative.spacegroup,
        sites,
        representative.species,
        transform=representative.transform,
        coordinate_precision=representative.coordinate_precision,
        molecular=representative.molecular,
        assemblies=representative.assemblies,
        chemical_composition=representative.chemical_composition,
        chemical_formula_descriptive=representative.chemical_formula_descriptive,
        chemical_formula_hill=representative.chemical_formula_hill,
        optimization_type=representative.optimization_type,
        immutable_id=representative.immutable_id,
        last_modified=representative.last_modified,
        charge=representative.charge,
    )


def _structure_representative(result: ASUStructure) -> FundamentalDomainStructure:
    """Rebuild the supported assigned representative carrier from a canonical ASU."""
    return FundamentalDomainStructure(
        result.cell,
        result.spacegroup,
        result.wyckoff_sites,
        result.species,
        transform=SettingTransform.identity(),
        coordinate_precision=result.coordinate_precision,
        molecular=result.molecular,
        assemblies=result.assemblies,
        chemical_composition=result.chemical_composition,
        chemical_formula_descriptive=result.chemical_formula_descriptive,
        chemical_formula_hill=result.chemical_formula_hill,
        optimization_type=result.optimization_type,
        immutable_id=result.immutable_id,
        last_modified=result.last_modified,
        charge=result.charge,
    )


def _template_representative(result: ASUStructure) -> FundamentalDomainTemplate:
    """Rebuild a canonically relabelled anonymous representative carrier."""
    by_name: dict[str, list[WyckoffSite]] = {species.name: [] for species in result.species}
    for site in result.wyckoff_sites:
        _checkpoint()
        by_name.setdefault(site.species, []).append(site)
    groups = {name: tuple(sorted(site.wyckoff for site in sites)) for name, sites in by_name.items() if sites}
    counts = _class_counts(result.spacegroup, groups)
    names = sorted(
        groups,
        key=lambda name: (
            class_order_key(groups[name], counts[name])[:2],
            tuple(sorted((site.wyckoff, tuple(site.free_params.to_fractions())) for site in by_name[name])),
        ),
    )
    names.extend(name for name, sites in by_name.items() if not sites)
    labels = {name: anonymous_symbol(index) for index, name in enumerate(names)}
    sites = tuple(WyckoffSite(site.wyckoff, site.free_params, labels[site.species]) for site in result.wyckoff_sites)
    species = tuple(dummy_species(anonymous_symbol(index)) for index in range(len(names)))
    return FundamentalDomainTemplate(result.cell, result.spacegroup, sites, species, result.coordinate_precision)


def canonical_protostructure(obj: ProtostructureLike, *, preserve_chirality: bool = True) -> Protostructure:
    """Return the exact canonical assigned refined classification.

    :param obj: A protostructure-like value.
    :param preserve_chirality: Whether to retain the higher enantiomorphic group.
    :return: A standalone canonical protostructure.
    """
    value = ProtostructureView(obj).unview()
    if value.representative is None:
        bare = canonical_bare_protostructure(value, preserve_chirality=preserve_chirality)
        return Protostructure(bare.spacegroup, bare.occupations, discriminator=value.discriminator)
    result = _canonical_protostructure_asu(
        _exact_representative(value.representative), preserve_chirality=preserve_chirality
    )
    return Protostructure(representative=_structure_representative(result), discriminator=value.discriminator)


def canonical_prototype(obj: PrototypeLike, *, preserve_chirality: bool = True) -> Prototype:
    """Return the exact canonical anonymous refined classification.

    :param obj: A prototype-like value.
    :param preserve_chirality: Whether to retain the higher enantiomorphic group.
    :return: A standalone canonical prototype.
    """
    value = PrototypeView(obj).unview()
    if value.representative is None:
        bare = canonical_bare_prototype(value, preserve_chirality=preserve_chirality)
        return Prototype(bare.spacegroup, bare.occupations, discriminator=value.discriminator)
    result = _canonical_protostructure_asu(
        _exact_representative(value.representative._domain), preserve_chirality=preserve_chirality
    )
    return Prototype(representative=_template_representative(result), discriminator=value.discriminator)
