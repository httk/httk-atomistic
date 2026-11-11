"""Species classes: the chemistry-only identity of a species in the classification families.

Two species are in the same class when they have the same key: the exact concentration of
each constituent symbol other than ``vacancy`` (the vacancy is the remainder), and the count of
each attached symbol. Name, original name, labels, charges, spins, mass and concentration
precision are not part of the key. The class name is derived from the key alone and parses back
to it::

    name   = head { attach }
    head   = SYM                      one constituent at concentration exactly 1
           | SYM frac { SYM frac }    otherwise; symbols alphabetical, vacancy omitted
    frac   = shortest exact decimal (always with "."), else reduced "p/q"
    attach = SYM count                count >= 1, always written; symbols alphabetical

for example ``Ti``, ``Bi0.5Na0.5``, ``Li0.76``, ``Li1/3``, ``OH2``, ``K0.5Na0.5C1H3``. A species
containing the unknown symbol ``X`` (a dummy template species or an unknown site) is not reduced:
its class is its own name.
"""

import re
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from httk.atomistic.models.species.species import Species

if TYPE_CHECKING:
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure, WyckoffSite

type _ClassKey = tuple[tuple[tuple[str, Fraction], ...], tuple[tuple[str, int], ...]]

_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+|/[1-9][0-9]*)?)?")


def _class_key(species: Species) -> _ClassKey:
    """Return the class key of a species without ``X``.

    :param species: The species to classify.
    :return: The sorted constituent concentrations and attached counts.
    :raises ValueError: If the species has no constituent other than ``vacancy``, or the
        constituents of one symbol sum to more than one.
    """
    amounts: dict[str, Fraction] = {}
    for symbol, value in zip(species.chemical_symbols, species.concentration, strict=True):
        if symbol != "vacancy" and value:
            amounts[symbol] = amounts.get(symbol, Fraction()) + value
    attached: dict[str, int] = {}
    for symbol, count in zip(species.attached or (), species.nattached or (), strict=True):
        if count:
            attached[symbol] = attached.get(symbol, 0) + count
    if not amounts:
        raise ValueError(f"species {species.name!r} has no constituent other than vacancy, so it has no class")
    for symbol, value in amounts.items():
        if value > 1:
            raise ValueError(
                f"species {species.name!r} has {symbol} constituents summing to {value}, more than one site, "
                "so it has no class"
            )
    return tuple(sorted(amounts.items())), tuple(sorted(attached.items()))


def _decimal(value: Fraction) -> str:
    """Render a concentration as its shortest exact decimal, else as reduced ``p/q``; never without ``.`` or ``/``."""
    rest, twos, fives = value.denominator, 0, 0
    while rest % 2 == 0:
        rest, twos = rest // 2, twos + 1
    while rest % 5 == 0:
        rest, fives = rest // 5, fives + 1
    if rest != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives, 1)
    digits = str(value.numerator * 10**places // value.denominator).rjust(places + 1, "0")
    text = (digits[:-places] + "." + digits[-places:]).rstrip("0")
    return text + "0" if text.endswith(".") else text


def _class_name(key: _ClassKey) -> str:
    """Return the class name of a key.

    :param key: The class key.
    :return: The class name.
    """
    amounts, attached = key
    if len(amounts) == 1 and amounts[0][1] == 1:
        head = amounts[0][0]
    else:
        head = "".join(symbol + _decimal(value) for symbol, value in amounts)
    return head + "".join(f"{symbol}{count}" for symbol, count in attached)


def _key_species(key: _ClassKey) -> Species:
    """Return the class species of a key: its name, constituents with the vacancy remainder, attachments."""
    amounts, attached = key
    symbols = [symbol for symbol, _ in amounts]
    values = [value for _, value in amounts]
    remainder = 1 - sum(values, Fraction())
    if remainder > 0:
        symbols.append("vacancy")
        values.append(remainder)
    return Species(
        _class_name(key),
        symbols,
        values,
        attached=[symbol for symbol, _ in attached] or None,
        nattached=[count for _, count in attached] or None,
    )


@lru_cache(maxsize=4096)
def _species_class(species: Species) -> Species:
    """Return the class species of a species: key and name only.

    A species containing ``X`` is returned unchanged; its class is its name.

    :param species: The species to reduce.
    :return: The class species.
    :raises ValueError: If the species has no constituent other than ``vacancy``.
    """
    if "X" in species.chemical_symbols or "X" in (species.attached or ()):
        return species
    return _key_species(_class_key(species))


def _species_from_class_name(name: str) -> Species:
    """Parse a strictly canonical class name into its class species.

    :param name: The class name, as in a label suffix.
    :return: The class species.
    :raises ValueError: If ``name`` is not the canonical name of a class.
    """
    tokens: list[tuple[str, str | None]] = []
    position = 0
    while position < len(name):
        match = _TOKEN.match(name, position)
        if match is None:
            raise ValueError(f"invalid species class name {name!r}")
        tokens.append((match.group(1), match.group(2)))
        position = match.end()
    if not tokens:
        raise ValueError("species class name must be non-empty")
    if tokens[0][1] is None:
        amounts, rest = [(tokens[0][0], Fraction(1))], tokens[1:]
    else:
        amounts, rest = [], tokens
        while rest and rest[0][1] is not None and ("." in rest[0][1] or "/" in rest[0][1]):
            amounts.append((rest[0][0], Fraction(rest[0][1])))
            rest = rest[1:]
    attached = []
    for symbol, count in rest:
        if count is None or not count.isdigit() or int(count) == 0:
            raise ValueError(f"invalid species class name {name!r}")
        attached.append((symbol, int(count)))
    key: _ClassKey = (tuple(amounts), tuple(attached))
    for part in key:
        if [symbol for symbol, _ in part] != sorted({symbol for symbol, _ in part}):
            raise ValueError(f"{name!r} is not a canonical species class name: symbols must be sorted and unique")
    if (
        not amounts
        or any(not 0 < value <= 1 for _, value in amounts)
        or "X" in (symbol for part in key for symbol, _ in part)
    ):
        raise ValueError(f"{name!r} is not a canonical species class name")
    if _class_name(key) != name:
        raise ValueError(f"{name!r} is not a canonical species class name")
    return _key_species(key)


def _class_sites(structure: "FundamentalDomainStructure") -> tuple[tuple["WyckoffSite", ...], tuple[Species, ...]]:
    """Return a domain structure's Wyckoff sites renamed to their species classes, and those classes.

    Only the species the sites use are reduced and returned, sorted by class name, so equal
    domains in other spellings give equal results. Site representatives and moments are kept.

    :param structure: The fundamental-domain structure.
    :return: The renamed sites and the used class species.
    :raises ValueError: If a used species has no constituent other than ``vacancy``.
    """
    from httk.atomistic.models.structure.asu import WyckoffSite

    by_name = {value.name: value for value in structure.species}
    classes = {site.species: _species_class(by_name[site.species]) for site in structure.wyckoff_sites}
    sites = tuple(
        WyckoffSite(
            site.wyckoff,
            site.free_params,
            classes[site.species].name,
            representative=site.representative,
            moment=site.moment,
        )
        for site in structure.wyckoff_sites
    )
    return sites, tuple(sorted(set(classes.values()), key=lambda value: value.name))


def _merges_classes(structure: Any) -> bool:
    """Return whether two species a structure's sites use share one class.

    :param structure: A structure with ``species`` and ``wyckoff_sites`` or ``species_at_sites``.
    :return: Whether the used species form fewer classes than names.
    """
    sites = getattr(structure, "wyckoff_sites", None)
    used = {site.species for site in sites} if sites is not None else set(structure.species_at_sites)
    by_name = {value.name: value for value in structure.species}
    classes = set()
    for name in used:
        try:
            classes.add(_species_class(by_name[name]))
        except (KeyError, ValueError):  # unknown or vacancy-only: reported where the structure is read
            classes.add(name)
    return len(classes) < len(used)


def _class_structure(structure: Any, *, drop_molecular: bool = False) -> Any:
    """Return a structure with every used species replaced by its class, for label recognition.

    The labels classify the structure of species classes, whose symmetry can be higher than
    that of the named species (two charge-ordered or differently labelled sites of one
    chemistry). The result is ``structure`` itself when its used species already are their
    classes; otherwise a unit-cell structure with the source's other fields, so the same
    validations apply to it.

    :param structure: A structure-like source.
    :param drop_molecular: Whether to drop molecular metadata, for the anonymous levels.
    :return: The structure in terms of species classes.
    :raises ValueError: If a used species has no constituent other than ``vacancy``.
    """
    from httk.atomistic.models.structure.unitcell import UnitcellStructure
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

    view = UnitcellStructureView(structure)
    used = set(view.species_at_sites)
    species = [value for value in view.species if value.name in used]
    classes = {value.name: _species_class(value) for value in species}
    molecular = getattr(view, "molecular", False) and not drop_molecular
    if (
        len(species) == len(view.species)
        and all(classes[value.name] == value for value in species)
        and molecular == getattr(view, "molecular", False)
    ):
        return structure
    ordered: list[Species] = sorted(set(classes.values()), key=lambda value: value.name)
    return UnitcellStructure(
        view.cell,
        view.sites,
        tuple(ordered),
        tuple(classes[name].name for name in view.species_at_sites),
        site_moments=view.site_moments,
        molecular=molecular,
        assemblies=getattr(view, "assemblies", None),
        chemical_composition=getattr(view, "chemical_composition", None),
        charge=view.charge,
    )
