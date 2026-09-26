"""Exact canonicalization for the structural-classification value families.

Input conversion follows the corresponding View contract and can recognize a
structure-like source. Materialized bare values use occupations alone. A refined representative
uses its exact geometric terminal, so projecting it to a bare value can require a
separate bare canonicalization pass.
"""

from collections.abc import Iterable
from typing import Any

from httk.core import FracVector

from httk.atomistic import data
from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
from httk.atomistic.models.bareprotostructure.like import BareProtostructureLike
from httk.atomistic.models.bareprotostructure.view import BareProtostructureView
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.bareprototype.like import BarePrototypeLike
from httk.atomistic.models.bareprototype.view import BarePrototypeView
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.protostructure.like import ProtostructureLike
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.protostructure.view import ProtostructureView
from httk.atomistic.models.prototype.like import PrototypeLike
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.prototype.view import PrototypeView
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structuretype.anonymize import dummy_species
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_protostructure import _canonical_protostructure_asu
from httk.atomistic.symmetry.lift import _discrete_normalizer_translations, _resetting_preserves_group
from httk.atomistic.symmetry.limits import _checkpoint
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__ = [
    "canonical_bare_protostructure",
    "canonical_bare_prototype",
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


def _assigned_key(occupations: Iterable[tuple[str, Any]]) -> tuple[Any, ...]:
    """Order an assigned bare key without inspecting hidden geometry."""
    classes: dict[str, list[str]] = {}
    for letter, species in occupations:
        _checkpoint()
        classes.setdefault(species.name, []).append(letter)
    groups = {name: tuple(sorted(letters)) for name, letters in classes.items()}
    return tuple(sorted(groups.values())), tuple(sorted((letters, name) for name, letters in groups.items()))


def _prototype_key(occupations: Iterable[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    """Order an anonymous bare key without using class labels as a tie-break."""
    classes: dict[str, list[str]] = {}
    for letter, label in occupations:
        _checkpoint()
        classes.setdefault(label, []).append(letter)
    return tuple(sorted(tuple(sorted(letters)) for letters in classes.values()))


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
        key = _prototype_key(mapped) if prototype else _assigned_key(mapped)
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
    occupied = [name for name, sites in by_name.items() if sites]
    names = sorted(
        occupied,
        key=lambda name: (
            tuple(sorted(site.wyckoff for site in by_name[name])),
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
