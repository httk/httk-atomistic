"""The bond network a structure carries: its bonds, the shapes of its sites, torsions and who assigned them.

Every structure exposes ``bonds``: a :class:`BondNetwork`, or ``None`` when no network is known.
On a fundamental domain (:class:`~httk.atomistic.FundamentalDomainStructure`,
:class:`~httk.atomistic.ASUStructure`) the pairs, shapes and torsions index ``wyckoff_sites``, and
each :class:`SiteBond` and :class:`SiteTorsion` stands for its orbit; on every other structure they
index ``sites`` and each operation is a pure lattice translation.
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction

from httk.core import FracVector

from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.shapes import _cell_bond_counts, _check_shape

__all__ = ["BondNetwork", "BondSource", "SiteBond", "SiteTorsion"]


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


# the symbol a mirror image takes; the other symbols are their own mirror image
_MIRRORED_SYMBOLS = {"+sc": "-sc", "-sc": "+sc", "+ac": "-ac", "-ac": "+ac"}
_TORSION_SYMBOLS = frozenset({"c", "g", "t", "sp", "sc", "ac", "ap", "none", *_MIRRORED_SYMBOLS})


@dataclass(frozen=True)
class SiteTorsion:
    """Represent one torsion statement of a structure's bond network: the torsion of four atoms A-B-C-D.

    Atom ``k`` (A, B, C, D for ``k`` = 0 to 3) is read as the partner of a :class:`SiteBond`. On a
    fundamental domain it is ``operations[k]`` applied to the reference atom of
    ``wyckoff_sites[sites[k]]`` (see :class:`SiteBond`), and the statement stands for its whole
    orbit under the group; its images under improper operations (determinant -1) have the opposite
    sign. On a unit-cell structure it is the atom at ``sites[sites[k]]`` (as stored, not wrapped)
    moved by the lattice translation ``operations[k]``, and the statement stands for itself and its
    lattice translates. The structure checks that consecutive atoms, A and C, and B and D are
    different atoms; B-C need not be a bond of the network.

    The angle is signed by the IUPAC (Klyne-Prelog) convention: viewed along B to C, it is positive
    when the bond A-B turns clockwise, by less than 180 degrees, to eclipse the bond C-D. It
    describes the Cartesian handedness of the four atoms as named; reversing them (D-C-B-A) keeps
    it. The range is [-180, 180], with -180 read as 180 and -0 as 0.

    The symbols: ``c`` (cis), ``g`` (gauche) and ``t`` (trans) are the protonet torsion letters;
    ``sp`` (synperiplanar), ``sc`` (synclinal), ``ac`` (anticlinal) and ``ap`` (antiperiplanar)
    are the Klyne-Prelog ranges (0 to 30, 30 to 90, 90 to 150 and 150 to 180 degrees in magnitude),
    and ``+sc``, ``-sc``, ``+ac``, ``-ac`` the signed ranges; ``none`` states that the bond B-C
    deliberately has no torsion class. Only the sign of a signed symbol is checked against the
    angle (0 and 180 agree with both signs), not its range.

    :param sites: The four 0-based site indices of A, B, C and D.
    :param operations: The four exact operations, as described above.
    :param symbol: The torsion symbol, or ``None``.
    :param angle: The signed torsion angle in degrees, or ``None``.
    :param uncertainty: The standard uncertainty of the angle in degrees (positive), or ``None``.
    :raises TypeError: If a value has the wrong type.
    :raises ValueError: If there are not four sites and operations, a site index is negative, the
        symbol is unknown, the angle is not finite or outside [-180, 180], an uncertainty is given
        without an angle or is not positive, neither symbol nor angle is given, or a signed symbol
        and the angle disagree in sign.
    """

    sites: tuple[int, int, int, int]
    operations: tuple[AffineOperation, AffineOperation, AffineOperation, AffineOperation]
    symbol: str | None = None
    angle: float | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        for name in ("sites", "operations"):
            value = getattr(self, name)
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise TypeError(f"SiteTorsion {name} must be a sequence of four values")
            if len(value) != 4:
                raise ValueError(f"SiteTorsion {name} must give four values, got {len(value)}")
            object.__setattr__(self, name, tuple(value))
        if any(not isinstance(site, int) or isinstance(site, bool) or site < 0 for site in self.sites):
            raise ValueError(f"SiteTorsion sites must be non-negative integers, got {self.sites!r}")
        if not all(isinstance(operation, AffineOperation) for operation in self.operations):
            raise TypeError("SiteTorsion operations must be AffineOperation values")
        if self.symbol is not None and self.symbol not in _TORSION_SYMBOLS:
            raise ValueError(f"SiteTorsion symbol must be one of {sorted(_TORSION_SYMBOLS)}, got {self.symbol!r}")
        angle, uncertainty = (_degrees(self, name) for name in ("angle", "uncertainty"))
        if angle is not None:
            if not -180 <= angle <= 180:
                raise ValueError(f"SiteTorsion angle must lie in [-180, 180], got {angle!r}")
            angle = 180.0 if angle == -180 else angle + 0.0  # -180 is 180, and -0.0 is 0.0
        if uncertainty is not None and (angle is None or uncertainty <= 0):
            raise ValueError("SiteTorsion uncertainty must be positive and needs an angle")
        if self.symbol is None and angle is None:
            raise ValueError("SiteTorsion needs a symbol, an angle or both")
        sign = 0 if angle is None or angle in (0, 180) else math.copysign(1, angle)  # 0 and 180 agree with both
        if self.symbol in _MIRRORED_SYMBOLS and sign and (sign > 0) != str(self.symbol).startswith("+"):
            raise ValueError(f"SiteTorsion symbol {self.symbol!r} and angle {angle!r} disagree in sign")
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "uncertainty", uncertainty)


def _four[T](values: Iterable[T]) -> tuple[T, T, T, T]:
    """Return the four values of a torsion statement's atoms as a 4-tuple, raising ``ValueError`` for another count."""
    a, b, c, d = values
    return a, b, c, d


def _degrees(torsion: SiteTorsion, name: str) -> float | None:
    """Return a torsion's ``angle`` or ``uncertainty`` as a finite float, or ``None``."""
    value = getattr(torsion, name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"SiteTorsion {name} must be a float or None, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"SiteTorsion {name} must be finite, got {value!r}")
    return float(value)


def _mirrored_torsion(torsion: SiteTorsion, sign: int = -1) -> SiteTorsion:
    """Return a torsion statement as its mirror image (negated angle and signed symbol) when ``sign`` is negative."""
    if sign > 0:
        return torsion
    angle = None if torsion.angle is None else -torsion.angle
    symbol = None if torsion.symbol is None else _MIRRORED_SYMBOLS.get(torsion.symbol, torsion.symbol)
    return replace(torsion, symbol=symbol, angle=angle)


def _unsigned_torsion(torsion: SiteTorsion) -> SiteTorsion:
    """Return a torsion statement that its own mirror image must equal: ``|angle|`` and an unsigned symbol."""
    angle = None if torsion.angle is None else abs(torsion.angle)
    symbol = None if torsion.symbol is None else torsion.symbol.lstrip("+-")
    return replace(torsion, symbol=symbol, angle=angle)


class _TorsionConflict(ValueError):
    """Raise when two torsion statements of one orbit differ (after their signs are brought together)."""


# (sites, one standard-setting element key per atom): the order of a normalized torsion statement
type _RowKey = tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]


def _merged_torsions(rows: Sequence[tuple[_RowKey, SiteTorsion]]) -> tuple[SiteTorsion, ...] | None:
    """Collapse normalized torsion statements by their key, sorted by it; ``None`` when there are none.

    :param rows: Each normalized statement with its orbit key.
    :return: One statement per key, in key order.
    :raises _TorsionConflict: If two statements with one key differ in symbol, angle or uncertainty.
    """
    merged: dict[_RowKey, SiteTorsion] = {}
    for key, row in rows:
        if merged.setdefault(key, row) != row:
            raise _TorsionConflict(f"conflicting torsion statements {merged[key]!r} and {row!r} of one orbit")
    return tuple(merged[key] for key in sorted(merged)) or None


def _agreeing_torsions(
    statements: Sequence[SiteTorsion | None],
    normalized: Callable[[Sequence[SiteTorsion]], tuple[SiteTorsion, ...] | None],
) -> tuple[tuple[SiteTorsion, ...] | None, int]:
    """Normalize read torsion statements one by one, dropping those that are not valid or conflict.

    A file reader uses this to keep what it can of a torsion list without failing the load. A
    statement is dropped when it is ``None`` (the reader could not build it) or ``normalized``
    raises ``ValueError`` for it, and so are all statements whose normal forms coincide when
    they differ in data.

    :param statements: The statements as read, ``None`` for a row that could not be read.
    :param normalized: The normal form of a list of statements (one statement here).
    :return: The kept statements in their normal form (``None`` when none is left), and the
        number of dropped ones.
    """
    found: dict[tuple[object, ...], list[SiteTorsion]] = {}
    for statement in statements:
        if statement is None:
            continue
        try:
            result = normalized([statement])
        except ValueError:
            continue
        assert result is not None  # one statement gives one
        (normal,) = result
        found.setdefault((normal.sites, normal.operations), []).append(normal)
    agreeing = [rows for rows in found.values() if len(set(rows)) == 1]
    return tuple(rows[0] for rows in agreeing) or None, len(statements) - sum(map(len, agreeing))


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
    """Represent the bond network of a structure: its bonds, optional shapes and torsions, and their sources.

    ``pairs``, ``shapes`` and ``torsions`` are content: they take part in equality, hashing and the
    structure's content identity. ``source``, ``shapes_source`` and ``torsions_source`` are
    provenance and take part in neither. A network on its own is not put in a normal form; the
    structure holding it is (see
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
    :param torsions: Torsion statements, any sequence of :class:`SiteTorsion`, or ``None`` when
        none are stated; empty is the same as ``None``.
    :param torsions_source: Who assigned the torsions, in the form and with the rules of
        ``shapes_source``; it needs ``torsions``.
    :raises TypeError: If a value has the wrong type.
    :raises ValueError: If a source mapping has an unknown key, or ``shapes_source`` or
        ``torsions_source`` is given without what it assigned.
    """

    pairs: tuple[SiteBond, ...]
    shapes: tuple[str | None, ...] | None = None
    source: BondSource | None = field(default=None, compare=False)
    shapes_source: BondSource | None = field(default=None, compare=False)
    torsions: tuple[SiteTorsion, ...] | None = None
    torsions_source: BondSource | None = field(default=None, compare=False)

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
        torsions = self.torsions
        if torsions is not None:
            if isinstance(torsions, str) or not isinstance(torsions, Sequence):
                raise TypeError("BondNetwork torsions must be a sequence of SiteTorsion values")
            torsions = tuple(torsions) or None
            if torsions is not None and not all(isinstance(row, SiteTorsion) for row in torsions):
                raise TypeError("BondNetwork torsions must contain SiteTorsion values")
        torsions_source = _source(self.torsions_source, "torsions_source")
        if torsions_source is not None and torsions is None:
            raise ValueError(
                "BondNetwork torsions_source needs torsions; without torsions there is nothing it assigned"
            )
        if torsions_source == source:
            torsions_source = None
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "shapes_source", shapes_source)
        object.__setattr__(self, "torsions", torsions)
        object.__setattr__(self, "torsions_source", torsions_source)

    def __repr__(self) -> str:
        # the torsion fields are shown only when set, so a network without torsions reads as before them
        shown = (
            item.name for item in fields(self) if item.name not in _TORSION_KEYS or getattr(self, item.name) is not None
        )
        return f"BondNetwork({', '.join(f'{name}={getattr(self, name)!r}' for name in shown)})"


_TORSION_KEYS = frozenset({"torsions", "torsions_source"})


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
    return _shift(bond, bond.operation, periodicity)


def _shift(
    owner: SiteBond | SiteTorsion, operation: AffineOperation, periodicity: Sequence[bool]
) -> tuple[int, int, int]:
    """Return the lattice translation of an operation of a unit-cell bond or torsion, refusing any other operation."""
    what = "bond" if isinstance(owner, SiteBond) else "torsion"
    vector = operation.vector.to_fractions()
    if operation.matrix != _EYE or any(Fraction(x).denominator != 1 for x in vector):
        raise ValueError(f"{owner!r}: a unit-cell {what} operation must be a lattice translation")
    n = (int(vector[0]), int(vector[1]), int(vector[2]))
    if any(k and not periodic for k, periodic in zip(n, periodicity, strict=True)):
        raise ValueError(f"{owner!r}: a unit-cell {what} cannot translate along a non-periodic axis")
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
    :return: The network with normalized pairs and torsions.
    :raises ValueError: As :func:`_cell_pairs` and :func:`_cell_torsions`, or if the shapes do not
        fit the sites' bond counts.
    """
    pairs = _cell_pairs(network.pairs, nsites, periodicity)
    if network.shapes is not None:
        _checked_symbols(network.shapes, _cell_bond_counts(nsites, pairs), "site")
    torsions = None if network.torsions is None else _cell_torsions(network.torsions, nsites, periodicity)
    if (pairs, torsions) == (network.pairs, network.torsions):
        return network
    return replace(network, pairs=pairs, torsions=torsions)


def _cell_torsions(
    torsions: Sequence[SiteTorsion], nsites: int, periodicity: Sequence[bool]
) -> tuple[SiteTorsion, ...] | None:
    """Validate unit-cell torsion statements and put them in their normal form.

    Each statement is translated so that atom B has the translation 0, and so is its reversal
    (D, C, B, A, with the same angle); the one with the smaller key ``(sites, (n.n, 0, n) per
    atom)`` is kept. Repeats collapse and the statements are sorted by that key: the normal form
    of the P1 fundamental domain whose sites are these sites (a lattice translation is proper,
    so no sign changes).

    :param torsions: The statements.
    :param nsites: The number of sites.
    :param periodicity: The cell's periodic directions; a translation is zero along the others.
    :return: The normalized statements, or ``None`` when there are none.
    :raises ValueError: If a statement refers to a site out of range, an operation is not a
        lattice translation (zero along non-periodic directions), it names one atom twice where
        atoms must differ, or two statements of one atom quadruple differ.
    """
    rows: list[tuple[_RowKey, SiteTorsion]] = []
    for row in torsions:
        if not all(site < nsites for site in row.sites):
            raise ValueError(f"{row!r} refers to a site outside 0..{nsites - 1}")
        atoms = [
            (site, _shift(row, operation, periodicity))
            for site, operation in zip(row.sites, row.operations, strict=True)
        ]
        if any(atoms[i] == atoms[j] for i, j in _DISTINCT):
            raise ValueError(f"{row!r} names one atom twice")
        candidates = []
        for order in (atoms, atoms[::-1]):
            origin = order[1][1]
            moved = [tuple(a - b for a, b in zip(n, origin, strict=True)) for _site, n in order]
            key = (tuple(site for site, _n in order), tuple((sum(k * k for k in n), 0, *n) for n in moved))
            candidates.append((key, moved))
        key, moved = min(candidates)
        operations = _four(AffineOperation(_EYE, n) for n in moved)
        rows.append((key, replace(row, sites=_four(key[0]), operations=operations)))
    return _merged_torsions(rows)


# the atom pairs of a torsion statement that must be different atoms: consecutive ones, A and C, B and D
_DISTINCT = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
