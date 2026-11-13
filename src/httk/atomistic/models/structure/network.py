"""The bond network a structure carries: its bonds, the shapes of its sites and who assigned them.

Every structure exposes ``bonds``: a :class:`BondNetwork`, or ``None`` when no network is known.
On a fundamental domain (:class:`~httk.atomistic.FundamentalDomainStructure`,
:class:`~httk.atomistic.ASUStructure`) the pairs and shapes index ``wyckoff_sites`` and each
:class:`SiteBond` stands for its bond orbit; on every other structure they index ``sites`` and each
bond's operation is a pure lattice translation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction

from httk.core import FracVector

from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.shapes import _cell_bond_counts, _check_shape

__all__ = ["BondNetwork", "BondSource", "SiteBond"]


@dataclass(frozen=True)
class SiteBond:
    """Represent one bond of a structure's bond network.

    On a fundamental domain the bond is a bond orbit: the reference atom of
    ``wyckoff_sites[site]`` is bonded to ``operation`` applied to the reference atom of
    ``wyckoff_sites[partner]``. A reference atom is the first-listed coordinate of the site's
    Wyckoff position in the structure's stored setting at its free parameters, not reduced modulo
    one. ``operation`` is an element of the stored setting's space group, lattice translation
    included. The bond stands for its whole orbit under the group, so any member of the orbit
    names the same bond.

    On a unit-cell structure ``site`` and ``partner`` index ``sites``, and the atom at
    ``sites[site]`` (as stored, not wrapped) is bonded to the atom at ``sites[partner]`` moved by
    ``operation``, a pure lattice translation; the bond stands for itself and its lattice
    translates.

    :param site: The 0-based index of the first site.
    :param partner: The 0-based index of the second site.
    :param operation: The exact operation applied to the partner's atom.
    """

    site: int
    partner: int
    operation: AffineOperation

    def __post_init__(self) -> None:
        for name in ("site", "partner"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"SiteBond {name} must be a non-negative integer, got {value!r}")
        if not isinstance(self.operation, AffineOperation):
            raise TypeError("SiteBond operation must be an AffineOperation")


@dataclass(frozen=True)
class BondSource:
    """Name who assigned a bond network or its shapes; every field is optional.

    A source is provenance: it never enters equality of a :class:`BondNetwork` or the content
    identity of a structure. An unknown field is ``None``, never ``""`` or only whitespace.

    :param description: Free text, for example the rule used.
    :param software: An identifier of the software or rule set.
    :param version: Its version.
    :param uri: A URL documenting the assignment.
    """

    description: str | None = None
    software: str | None = None
    version: str | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"BondSource {item.name} must be a str or None, got {type(value).__name__}")
            if value is not None and not value.strip():
                raise ValueError(f"BondSource {item.name} must not be empty or blank; use None for unknown")


def _source(value: object, name: str) -> BondSource | None:
    """Coerce a source given as a :class:`BondSource`, a mapping of its fields or ``None``; all-``None`` is ``None``."""
    if isinstance(value, Mapping):
        if unknown := set(value) - {item.name for item in fields(BondSource)}:
            raise ValueError(f"BondNetwork {name} has unknown keys: {sorted(map(str, unknown))!r}")
        value = BondSource(**value)
    if value is not None and not isinstance(value, BondSource):
        raise TypeError(f"BondNetwork {name} must be a BondSource, a mapping or None, got {type(value).__name__}")
    return None if value == BondSource() else value


@dataclass(frozen=True)
class BondNetwork:
    """Represent the bond network of a structure: its bonds, optional shapes and their sources.

    ``pairs`` and ``shapes`` are content: they take part in equality, hashing and the structure's
    content identity. ``source`` and ``shapes_source`` are provenance and take part in neither. A
    network on its own is not put in a normal form; the structure holding it is (see
    :attr:`~httk.atomistic.FundamentalDomainStructure.bonds` and
    :attr:`~httk.atomistic.UnitcellStructure.bonds`).

    :param pairs: The bonds, any sequence of :class:`SiteBond`; empty means the structure is known
        to have no bonds.
    :param shapes: One shape symbol or ``None`` (not assigned) per site, or ``None`` when no shapes
        are assigned; all ``None`` is the same as ``None``. See :mod:`httk.atomistic.symmetry.shapes`
        for the symbols.
    :param source: Who assigned the bonds, as a :class:`BondSource` or a mapping of its fields; all
        fields ``None`` is the same as ``None``.
    :param shapes_source: Who assigned the shapes, in the form of ``source``, or ``None`` when
        ``source`` assigned them (a ``shapes_source`` equal to ``source`` becomes ``None``); it
        needs ``shapes``.
    :raises TypeError: If a value has the wrong type.
    :raises ValueError: If a source mapping has an unknown key, or ``shapes_source`` is given
        without shapes.
    """

    pairs: tuple[SiteBond, ...]
    shapes: tuple[str | None, ...] | None = None
    source: BondSource | None = field(default=None, compare=False)
    shapes_source: BondSource | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.pairs, str) or not isinstance(self.pairs, Sequence):
            raise TypeError("BondNetwork pairs must be a sequence of SiteBond values")
        pairs = tuple(self.pairs)
        if not all(isinstance(bond, SiteBond) for bond in pairs):
            raise TypeError("BondNetwork pairs must contain SiteBond values")
        shapes = self.shapes
        if shapes is not None:
            if isinstance(shapes, str) or not isinstance(shapes, Sequence):
                raise TypeError("BondNetwork shapes must be a sequence of str or None values, one per site, not a str")
            shapes = tuple(shapes)
            if any(symbol is not None and not isinstance(symbol, str) for symbol in shapes):
                raise TypeError("BondNetwork shapes must contain str or None values")
            if all(symbol is None for symbol in shapes):
                shapes = None
        source = _source(self.source, "source")
        shapes_source = _source(self.shapes_source, "shapes_source")
        if shapes_source is not None and shapes is None:
            raise ValueError("BondNetwork shapes_source needs shapes; without shapes there is nothing it assigned")
        if shapes_source == source:  # the same provenance has one spelling: "assigned by source"
            shapes_source = None
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "shapes_source", shapes_source)


_NETWORK_KEYS = frozenset(item.name for item in fields(BondNetwork))


def _bond_network(value: object) -> BondNetwork | None:
    """Coerce a structure's ``bonds=`` argument: a network, a mapping of its fields, a sequence of bonds or ``None``."""
    if value is None or isinstance(value, BondNetwork):
        return value
    if isinstance(value, Mapping):
        if unknown := set(value) - _NETWORK_KEYS:
            raise ValueError(f"bonds has unknown keys: {sorted(map(str, unknown))!r}")
        if "pairs" not in value:
            raise ValueError("bonds given as a mapping needs the key 'pairs'")
        return BondNetwork(**value)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            f"bonds must be a BondNetwork, a mapping or a sequence of SiteBond values, got {type(value).__name__}"
        )
    return BondNetwork(tuple(value))


def _checked_symbols(shapes: Sequence[str | None], counts: Sequence[int], what: str) -> None:
    """Check one shape symbol or ``None`` per site against the sites' bond counts.

    :param shapes: The symbols.
    :param counts: The bond count of each site.
    :param what: The name of a site in the message (``"Wyckoff site"``, ``"site"``).
    :raises ValueError: If the length is not the number of sites or a symbol is invalid for its site.
    """
    if len(shapes) != len(counts):
        raise ValueError(f"shapes must give one symbol or None per {what}: {len(shapes)} for {len(counts)} sites")
    for index, (symbol, count) in enumerate(zip(shapes, counts, strict=True)):
        if symbol is not None:
            try:
                _check_shape(symbol, count)
            except ValueError as error:
                raise ValueError(f"shapes[{index}]: {error}") from None


_EYE = FracVector.eye((3, 3))


def _translation(bond: SiteBond, periodicity: Sequence[bool]) -> tuple[int, int, int]:
    """Return the lattice translation of a unit-cell bond, refusing any other operation."""
    vector = bond.operation.vector.to_fractions()
    if bond.operation.matrix != _EYE or any(Fraction(x).denominator != 1 for x in vector):
        raise ValueError(f"{bond!r}: a unit-cell bond operation must be a lattice translation")
    n = (int(vector[0]), int(vector[1]), int(vector[2]))
    if any(k and not periodic for k, periodic in zip(n, periodicity, strict=True)):
        raise ValueError(f"{bond!r}: a unit-cell bond cannot translate along a non-periodic axis")
    return n


def _cell_pairs(pairs: Sequence[SiteBond], nsites: int, periodicity: Sequence[bool]) -> tuple[SiteBond, ...]:
    """Validate unit-cell bonds and put them in their normal form.

    Each bond is oriented ``site <= partner`` (``(i, j, n)`` with ``i > j`` becomes ``(j, i, -n)``;
    a bond of a site to itself takes the lexicographically smaller of ``n`` and ``-n``), repeats
    collapse, and the bonds are sorted by ``(site, partner, n.n, n)``: the normal form of the P1
    fundamental domain whose sites are these sites.

    :param pairs: The bonds.
    :param nsites: The number of sites.
    :param periodicity: The cell's periodic directions; a translation is zero along the others.
    :return: The normalized bonds.
    :raises ValueError: If a bond refers to a site out of range, its operation is not a lattice
        translation (zero along non-periodic directions), or it bonds an atom to itself.
    """
    keys = set()
    for bond in pairs:
        if not (bond.site < nsites and bond.partner < nsites):
            raise ValueError(f"{bond!r} refers to a site outside 0..{nsites - 1}")
        i, j, n = bond.site, bond.partner, _translation(bond, periodicity)
        minus = (-n[0], -n[1], -n[2])
        if i > j:
            i, j, n = j, i, minus
        elif i == j:
            if not any(n):
                raise ValueError(f"{bond!r} bonds an atom to itself")
            n = min(n, minus)
        keys.add((i, j, sum(k * k for k in n), n))
    return tuple(SiteBond(i, j, AffineOperation(_EYE, n)) for i, j, _length, n in sorted(keys))


def _checked_cell_network(network: BondNetwork, nsites: int, periodicity: Sequence[bool]) -> BondNetwork:
    """Validate a unit-cell structure's network and return it with its pairs in normal form.

    :param network: The network.
    :param nsites: The number of sites.
    :param periodicity: The cell's periodic directions.
    :return: The network with normalized pairs.
    :raises ValueError: As :func:`_cell_pairs`, or if the shapes do not fit the sites' bond counts.
    """
    pairs = _cell_pairs(network.pairs, nsites, periodicity)
    if network.shapes is not None:
        _checked_symbols(network.shapes, _cell_bond_counts(nsites, pairs), "site")
    return network if pairs == network.pairs else replace(network, pairs=pairs)
