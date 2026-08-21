"""Validation and canonical labelling helpers for chromastructures."""

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.species.species import Species


def dummy_species(label: str) -> Species:
    """Return the exact dummy species used by chromastructures.

    :param label: The anonymous symbol to use as the species name.
    :return: The canonical dummy species.
    """
    return Species(label, ("X",), (1,), labels=(label,))


def is_dummy_species(species: Species) -> bool:
    """Return whether ``species`` has precisely the sanctioned dummy-species shape.

    :param species: The species to inspect.
    :return: Whether the species is a canonical dummy species.
    """
    return (
        species.name == (species.labels[0] if species.labels is not None and len(species.labels) == 1 else None)
        and species.chemical_symbols == ("X",)
        and species.concentration == (Fraction(1),)
        and species.labels == (species.name,)
        and species.mass is None
        and species.attached is None
        and species.nattached is None
        and species.concentration_precision in (None, (None,))
        and species.charges is None
        and species.spins is None
        and species.original_name is None
    )


def canonical_dummy_assignment(amounts: Sequence[tuple[str, Fraction | int]]) -> dict[str, str]:
    """Map element-like keys to anonymous symbols by descending amount.

    :param amounts: The keys and represented site counts.
    :return: The deterministic key-to-anonymous-symbol assignment.
    """
    ordered = sorted(amounts, key=lambda item: (-item[1], item[0]))
    return {element: anonymous_symbol(index) for index, (element, _) in enumerate(ordered)}


def require_anonymizable(structure: Any) -> None:
    """Reject structure features outside this phase's deliberate scope.

    Only fully occupied, single-real-element site species are anonymized. Future phases may
    add merge or label-preserving modes for decorated species; this mode intentionally refuses
    those cases, as well as assemblies, stated compositions, and site moments.

    :param structure: The structure whose representation is being anonymized.
    :raises ValueError: If the structure contains unsupported composition features.
    """
    names = tuple(structure.species_at_sites)
    by_name = {species.name: species for species in structure.species}
    unused = sorted(set(by_name) - set(names))
    if unused:
        raise ValueError(f"cannot anonymize structure: species {unused[0]!r} is unused")
    used: list[Species] = []
    for name in names:
        try:
            species = by_name[name]
        except KeyError as exc:
            raise ValueError(f"cannot anonymize structure: unknown species {name!r}") from exc
        if not species.is_single_element:
            raise ValueError(f"cannot anonymize structure: species {name!r} is not a single real element")
        used.append(species)

    elements: dict[str, str] = {}
    for species in used:
        element = species.chemical_symbols[0]
        previous = elements.get(element)
        if previous is not None and previous != species.name:
            raise ValueError(
                f"cannot anonymize structure: distinct species {previous!r} and {species.name!r} share element {element!r}"
            )
        elements[element] = species.name
    if getattr(structure, "assemblies", None) is not None:
        raise ValueError("cannot anonymize structure: assemblies are not supported")
    if getattr(structure, "chemical_composition", None) is not None:
        raise ValueError("cannot anonymize structure: chemical_composition is not supported")
    if getattr(structure, "site_moments", None) is not None:
        raise ValueError("cannot anonymize structure: site_moments are not supported")
