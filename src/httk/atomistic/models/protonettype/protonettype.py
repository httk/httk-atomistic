"""The immutable anonymous protonettype value."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.protonettype.backend import ProtonettypeBackend
from httk.atomistic.models.prototype.notation import (
    _known,
    _label_rows,
    _parse_undetermined,
    _undetermined,
    _unknown,
    class_order_key,
)
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup

if TYPE_CHECKING:
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure


def _standard(spacegroup: Spacegroup | int, owner: str) -> Spacegroup:
    """Resolve a space group and require the IT standard setting."""
    value = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
    if not value.is_standard_setting:
        raise ValueError(f"{owner} records Wyckoff data in the IT standard setting")
    return value


def _class_runs(spacegroup: Spacegroup, pairs: Sequence[tuple[str, str]], owner: str) -> list[str]:
    """Check site-ordered ``(wyckoff, class key)`` pairs and return the class keys in order.

    Each class must be one contiguous run with sorted letters, and the runs must follow
    :func:`~httk.atomistic.models.prototype.notation.class_order_key` (tied classes in
    any order, because their order carries the net's numbering).
    """
    if not pairs:
        raise ValueError(f"{owner} occupations must be non-empty")
    runs: list[tuple[str, list[str]]] = []
    for wyckoff, key in pairs:
        try:
            spacegroup.wyckoff_position(wyckoff)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if runs and runs[-1][0] == key:
            runs[-1][1].append(wyckoff)
        else:
            runs.append((key, [wyckoff]))
    keys = [key for key, _ in runs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{owner} occupations of one class must form one contiguous run")
    order = []
    for _, letters in runs:
        if letters != sorted(letters):
            raise ValueError(f"{owner} occupations within a class must be sorted by Wyckoff letter")
        count = sum(spacegroup.wyckoff_position(letter).multiplicity for letter in letters)
        order.append(class_order_key(tuple(letters), count)[:2])
    if order != sorted(order):
        raise ValueError(f"{owner} classes must run largest site count first, then by sorted Wyckoff letters")
    return keys


def _checked_net(
    net: Sequence[Sequence[int]], nsites: int, nops: int, owner: str, *, writable: bool
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Validate net rows against the site and operation counts and return them as tuples.

    Input rows take any lattice translation and either direction; ``writable`` requires the
    label's range -5..+4.
    """
    rows: list[tuple[int, int, int, int, int, int]] = []
    for row in net:
        values = tuple(row)
        if len(values) != 6 or any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise ValueError(f"{owner} net row {row!r} must be six integers (i, j, k, n1, n2, n3)")
        i, j, k, n1, n2, n3 = (int(v) for v in values)
        if not (1 <= i <= nsites and 1 <= j <= nsites):
            raise ValueError(f"{owner} net row {values!r} needs site numbers in 1..{nsites}")
        if not 1 <= k <= nops:
            raise ValueError(f"{owner} net row {values!r} needs an operation number in 1..{nops}")
        if writable and any(not -5 <= n <= 4 for n in (n1, n2, n3)):
            raise ValueError(f"{owner} net row {values!r} has a translation outside -5..+4")
        if i == j and k == 1 and (n1, n2, n3) == (0, 0, 0):
            raise ValueError(f"{owner} net row {values!r} bonds an atom to itself")
        rows.append((i, j, k, n1, n2, n3))
    if len(set(rows)) != len(rows):
        raise ValueError(f"{owner} net has duplicate rows")
    return tuple(rows)


def _net_counts(
    spacegroup: Spacegroup, letters: Sequence[str], net: Sequence[Sequence[int]], owner: str
) -> tuple[int, ...]:
    """Validate net rows and return the bond count of every site (``shapes._bond_counts``).

    :param spacegroup: The standard-setting space group.
    :param letters: The Wyckoff letter per site, in site order.
    :param net: The rows ``(i, j, k, n1, n2, n3)``.
    :param owner: The name in error messages.
    :return: The number of atoms bonded to an atom of each site.
    :raises ValueError: If a letter is unknown or a row is malformed.
    """
    from httk.atomistic.symmetry.protonet import _tables
    from httk.atomistic.symmetry.shapes import _bond_counts

    tables = _tables(spacegroup.it_number)
    known = {position.letter for position in spacegroup.wyckoff}
    if unknown := sorted(set(letters) - known):
        raise ValueError(f"{owner}: space group {spacegroup.it_number} has no Wyckoff letter {unknown[0]!r}")
    rows = _checked_net(net, len(letters), tables.nops, owner, writable=False)
    stabilisers = [tables.letter(letter).H for letter in letters]
    elements = [(i - 1, j - 1, (k - 1, n1, n2, n3)) for i, j, k, n1, n2, n3 in rows]
    for row, (a, b, g) in zip(rows, elements, strict=True):  # the site symmetry may map the bond onto the atom
        if a == b and tables.canon(stabilisers[a], stabilisers[a], g, same_site=True) == (0, 0, 0, 0):
            raise ValueError(f"{owner} net row {row!r} bonds an atom to itself")
    return _bond_counts(tables, stabilisers, elements)


def _checked_shapes(shapes: Sequence[str | None] | None, counts: Sequence[int], owner: str) -> tuple[str | None, ...]:
    """Validate a value's shapes against its bond counts: a symbol exactly on each site with two or more bonds.

    ``None`` stands for no shapes and is accepted only when no site has two or more bonds.
    """
    from httk.atomistic.symmetry.shapes import _check_shape

    if shapes is None:
        if any(count >= 2 for count in counts):
            raise ValueError(f"{owner} shapes are missing: every site with two or more bonds needs a symbol")
        return (None,) * len(counts)
    if isinstance(shapes, str):
        raise TypeError(f"{owner} shapes must be a sequence of str or None values, one per site, not a str")
    checked = tuple(shapes)
    if len(checked) != len(counts):
        raise ValueError(f"{owner} shapes must give one entry per site: {len(checked)} for {len(counts)} sites")
    for site, (symbol, count) in enumerate(zip(checked, counts, strict=True), 1):
        if symbol is None:
            if count >= 2:
                raise ValueError(f"{owner} site {site} has {count} bonds and needs a shape")
        elif not isinstance(symbol, str):
            raise TypeError(f"{owner} shapes must contain str or None values")
        else:
            try:
                _check_shape(symbol, count)
            except ValueError as exc:
                raise ValueError(f"{owner} site {site}: {exc}") from None
    return checked


# why a lettered row is refused, by the reason of ``torsions._net_judge``; rules 2 and 5 are checked before
_NO_LETTER = {
    "periodic piece": "a bond of a periodic piece (a framework, chain or layer, not a molecule)",
    "end group": "an end group (no other neighbour with more than one bond)",
    "ring bond": "a ring bond",
    "tied reference neighbour": "a bond with a tied reference neighbour at an end",
}


def _checked_torsions(
    spacegroup: Spacegroup,
    letters: Sequence[str],
    rows: Sequence[tuple[int, int, int, int, int, int]],
    shapes: Sequence[str | None],
    torsions: Sequence[str | None] | None,
    owner: str,
) -> dict[tuple[int, int], str]:
    """Validate a value's torsion letters, one per net row, and return them by 1-based site pair ``(i, j)``, ``i <= j``.

    A letter is allowed only on a bond the net and shapes can qualify (see
    :mod:`httk.atomistic.symmetry.torsions`): the sorted pair of its two shapes is in the table
    and the letter is ``t`` or the pair's low letter (rule 5); the site pair has one bond orbit,
    every row of it carrying the same letter; and the bond lies in a molecular piece (rule 1),
    each end has another bonded atom with more than one bond (rule 4), the bond is not in a ring
    (rule 3), and each end has a unique reference neighbour (rule 6, a tie by bond count final at
    a planar end), judged by the same code as the structures' letters on atoms placed at each
    Wyckoff letter's generic probe point. Rule 2 (four or fewer bonds) follows from rule 5 once the shapes are
    validated against the bond counts. Full occupancy and collinear ends are not checked here,
    as they need the structure.

    :param spacegroup: The standard-setting space group.
    :param letters: The Wyckoff letter per site, in site order.
    :param rows: The validated net rows ``(i, j, k, n1, n2, n3)``.
    :param shapes: The validated shape per site.
    :param torsions: One letter or ``None`` per row, or ``None`` for no letters.
    :param owner: The name in error messages.
    :return: The letter per site pair that has one.
    :raises ValueError: If the letters do not give one per row or a letter is not allowed on its row.
    :raises TypeError: If ``torsions`` is a string or holds a value that is neither a string nor ``None``.
    """
    if torsions is None:
        return {}
    if isinstance(torsions, str):
        raise TypeError(f"{owner} torsions must be a sequence of str or None values, one per net row, not a str")
    given = tuple(torsions)
    if len(given) != len(rows):
        raise ValueError(f"{owner} torsions must give one entry per net row: {len(given)} for {len(rows)} rows")
    if any(letter is not None and not isinstance(letter, str) for letter in given):
        raise TypeError(f"{owner} torsions must contain str or None values")
    if not any(given):
        return {}
    from httk.atomistic.symmetry.protonet import _tables
    from httk.atomistic.symmetry.torsions import _edge, _net_judge

    tables = _tables(spacegroup.it_number)
    H = [tables.letter(letter).H for letter in letters]
    elements = [(i - 1, j - 1, (k - 1, n1, n2, n3)) for i, j, k, n1, n2, n3 in rows]
    orbits: dict[tuple[int, int], set[Any]] = {}
    for a, b, g in elements:
        if a > b:
            a, b, g = b, a, tables.inv(g)
        orbits.setdefault((a, b), set()).add(tables.canon(H[a], H[b], g, a == b))
    # rules 1, 3, 4 and 6 on the atoms grown from each letter's generic probe point
    judge = _net_judge(tables, H, [tables.letter(letter).ref for letter in letters], elements, shapes)
    found: dict[tuple[int, int], str] = {}
    for (i, j, k, n1, n2, n3), (a, b, g), letter in zip(rows, elements, given, strict=True):
        if letter is None:
            continue
        where = f"{owner} net row {(i, j, k, n1, n2, n3)!r}"
        edge = _edge(shapes[a], shapes[b])
        if edge is None:
            raise ValueError(f"{where}: shapes {shapes[a]} and {shapes[b]} give no torsion letter")
        allowed = ("t", edge[1])
        if letter not in allowed:
            raise ValueError(f"{where}: torsion letter {letter!r} is not one of {allowed} for its shapes")
        pair = (min(i, j), max(i, j))
        if len(orbits[(pair[0] - 1, pair[1] - 1)]) > 1:
            raise ValueError(f"{where}: a site pair with more than one bond orbit has no torsion letter")
        if isinstance(reason := judge(a, b, g), str):
            raise ValueError(f"{where}: {_NO_LETTER.get(reason, reason)} has no torsion letter")
        if found.setdefault(pair, letter) != letter:
            raise ValueError(f"{owner} rows of the site pair {pair} have different torsion letters")
    for (i, j, *_), letter in zip(rows, given, strict=True):
        if letter is None and (min(i, j), max(i, j)) in found:
            raise ValueError(f"{owner} rows of the site pair {(min(i, j), max(i, j))} have different torsion letters")
    return found


def _assign_net(
    value: Any,
    net: Sequence[Sequence[int]],
    shapes: Sequence[str | None] | None,
    torsions: Sequence[str | None] | None,
    owner: str,
    *,
    writable: bool,
) -> None:
    """Validate and store a value's net, shapes and torsion letters, the rows in label order (both levels)."""
    letters = [o.wyckoff for o in value._occupations]
    nops = len(value._spacegroup.symmetry_operations)
    rows = _checked_net(net, len(letters), nops, owner, writable=writable)
    counts = _net_counts(value._spacegroup, letters, rows, owner)
    value._shapes = _checked_shapes(shapes, counts, owner)
    found = _checked_torsions(value._spacegroup, letters, rows, value._shapes, torsions, owner)
    value._net = _label_rows(rows, counts)
    value._torsions = tuple(found.get((min(i, j), max(i, j))) for i, j, *_ in value._net)


class Protonettype(ProtonettypeBackend):
    """Store an anonymous bare prototype together with a symmetric covalent bonding net.

    Site ``n`` is ``occupations[n-1]``. The occupations form one contiguous run per class,
    the runs ordered largest site count first and then by sorted Wyckoff letters, each run
    sorted by letter; the run at position ``c`` carries the label ``anonymous_symbol(c)``.
    Tied classes may come in any order; the canonical net orders them. A net row
    ``(i, j, k, n1, n2, n3)`` bonds the reference atom of site ``i`` to that of site ``j``
    moved by general-position operation ``k`` (identity = 1) and lattice translation
    ``n``, any integer vector. The anonymous bare description is kept (no affine-normaliser
    pass, as for :class:`~httk.atomistic.BarePrototype`); the net is canonicalized over site
    numbering, reference atoms and the equivalent descriptions with the same anonymous bare
    key, keeping orientation, so one net is one value whatever its spelling.
    :func:`~httk.atomistic.canonical_protonettype` also minimises the bare prototype;
    for a canonical bare prototype the two agree.

    Occupations carry no cell: they are read in a right-handed basis of the standard setting,
    so in a Sohncke group they name one hand of the crystal.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their anonymous labels, in site order.
    :param net: The net rows ``(i, j, k, n1, n2, n3)``, in either direction; stored rows are
        written from their listing site.
    :param shapes: The bond shape symbol per site (see :mod:`httk.atomistic.symmetry.shapes`),
        ``None`` for a site with fewer than two bonds; every site with two or more bonds needs a
        known symbol: a named one ending in its bond count, or ``I`` (irregular) on three or more.
        ``None`` for the whole argument only when no site has two or more bonds.
    :param torsions: The torsion letter per row of ``net`` (see
        :mod:`httk.atomistic.symmetry.torsions`), ``None`` for a row without one; ``None`` for the
        whole argument when no row has one. A letter is checked against what the net and shapes
        determine: its two shapes form a pair of the torsion table and it is ``t`` or that pair's
        low letter, at most four bonds at each end, one bond orbit between its two sites, every
        row of which carries the same letter, a bond in a molecule and outside a ring, another
        neighbour with more than one bond at each end, and a unique reference neighbour at each
        end. Whether its sites are fully occupied and whether an end is straight are not checked,
        as they need the structure.
    :raises ValueError: If the input is malformed, a shape is missing, unknown or has the wrong
        bond count, a torsion letter is not allowed on its row, or the canonical net needs a
        translation outside -5..+4, which the notation cannot write (D4).
    :raises TypeError: If ``shapes`` or ``torsions`` is a string or holds a value that is neither a
        string nor ``None``.
    :raises CanonicalizationLimitError: If the net search exceeds its 120 s deadline, or an
        enclosing canonicalization budget.
    """

    kind: ClassVar[str] = "protonettype"
    _undetermined: str | None = None

    def __init__(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]] = (),
        shapes: Sequence[str | None] | None = None,
        torsions: Sequence[str | None] | None = None,
    ) -> None:
        from httk.atomistic.symmetry.canonical_classification import _protonettype_net
        from httk.atomistic.symmetry.limits import _active, _budget_scope

        self._assign(spacegroup, occupations, net, shapes, torsions, writable=False)
        with _budget_scope(120.0 if _active.get() is None else None):  # an enclosing budget governs
            canonical = _protonettype_net(self, preserve_chirality=True, own_bare=True)
        self._spacegroup, self._occupations = canonical.spacegroup, canonical.occupations
        self._net, self._shapes, self._torsions = canonical.net, canonical.shapes, canonical.torsions

    @classmethod
    def _of(
        cls,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]],
        shapes: Sequence[str | None] | None,
        torsions: Sequence[str | None] | None = None,
    ) -> Self:
        """Build from a trusted canonical net, its shapes and torsion letters: validate only, no canonicalization."""
        instance = object.__new__(cls)
        instance._assign(spacegroup, occupations, net, shapes, torsions, writable=True)
        return instance

    def _assign(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]],
        shapes: Sequence[str | None] | None,
        torsions: Sequence[str | None] | None,
        *,
        writable: bool,
    ) -> None:
        self._spacegroup = _standard(spacegroup, "Protonettype")
        self._occupations = tuple(
            value if isinstance(value, PrototypeOccupation) else PrototypeOccupation(*value) for value in occupations
        )
        labels = _class_runs(self._spacegroup, [(o.wyckoff, o.label) for o in self._occupations], "Protonettype")
        if labels != [anonymous_symbol(index) for index in range(len(labels))]:
            raise ValueError("Protonettype class run c must carry the label anonymous_symbol(c)")
        _assign_net(self, net, shapes, torsions, "Protonettype", writable=writable)

    @classmethod
    def from_asu(
        cls,
        asu: "FundamentalDomainStructure",
        bonds: Iterable[tuple[int, int, int, int, int, int]] | None = None,
        *,
        shapes: Sequence[str | None] | None = None,
        on_limit: Literal["undetermined", "raise"] = "undetermined",
    ) -> "Protonettype":
        """Build the anonymous value of an ASU structure and its covalent bonds.

        The anonymous erasure of :meth:`Protonetstructure.from_asu
        <httk.atomistic.Protonetstructure.from_asu>`, which documents the arguments; without
        ``bonds`` the structure's own bond network is used.

        :param asu: The structure.
        :param bonds: The bonds ``(i, j, k, n1, n2, n3)``, or ``None`` for the structure's own network.
        :param shapes: With ``bonds``, optional supplied shapes, one symbol or ``None`` per Wyckoff site.
        :param on_limit: ``"undetermined"`` or ``"raise"`` when the net search runs out of its budget.
        :return: The protonettype value.
        :raises ValueError: As :meth:`Protonetstructure.from_asu
            <httk.atomistic.Protonetstructure.from_asu>`.
        :raises CanonicalizationLimitError: With ``on_limit="raise"``, if the net search runs out of
            its budget.
        """
        from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
        from httk.atomistic.models.protonettype.view import ProtonettypeView

        return ProtonettypeView(Protonetstructure.from_asu(asu, bonds, shapes=shapes, on_limit=on_limit)).unview()

    @classmethod
    def undetermined(cls, label: str) -> "Protonettype":
        """Return the undetermined protonettype of a label in which ``X`` marks what is not determined.

        The canonicalisers return such a value when a part did not finish within its time budget; it
        holds only its label: ``<Formulatype>_X__X``, ``<Formulatype>_<Pearson>_<number>_X__X``, or
        the determined bare label followed by ``__X`` when only the net was not determined. Two
        undetermined values are equal when their labels are; that does not mean their structures are
        the same.

        :param label: The undetermined label.
        :return: The undetermined value, with ``determined`` false.
        :raises ValueError: If ``label`` is not an undetermined protonettype label.
        """
        _parse_undetermined(label, cls.kind)
        return Protonettype._undetermined_of(label)

    @classmethod
    def _undetermined_of(cls, label: str) -> "Protonettype":
        """Build an undetermined value from a trusted label, without parsing it."""
        instance = object.__new__(Protonettype)
        instance._undetermined = label
        return instance

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "spacegroup", undetermined.spacegroup)
        return self._spacegroup

    @property
    def occupations(self) -> tuple[PrototypeOccupation, ...]:
        """Return the anonymous Wyckoff occupations in site order."""
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "occupations", undetermined.occupations)
        return self._occupations

    @property
    def net(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return the net rows ``(i, j, k, n1, n2, n3)`` in label order, each from its listing site."""
        if _undetermined(self) is not None:
            _unknown(self, "net")
        return self._net

    @property
    def shapes(self) -> tuple[str | None, ...]:
        """Return the bond shape symbol per site, ``None`` for a site with fewer than two bonds."""
        if _undetermined(self) is not None:
            _unknown(self, "shapes")
        return self._shapes

    @property
    def torsions(self) -> tuple[str | None, ...]:
        """Return the torsion letter per net row in label order, ``None`` for a row without one."""
        if _undetermined(self) is not None:
            _unknown(self, "torsions")
        return self._torsions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protonettype):
            return NotImplemented
        if self._undetermined is not None or other._undetermined is not None:
            return self._undetermined == other._undetermined
        return (self.spacegroup, self.occupations, self.net, self.shapes, self.torsions) == (
            other.spacegroup,
            other.occupations,
            other.net,
            other.shapes,
            other.torsions,
        )

    def __hash__(self) -> int:
        if self._undetermined is not None:
            return hash(self._undetermined)
        return hash((self.spacegroup, self.occupations, self.net, self.shapes, self.torsions))

    def __repr__(self) -> str:
        if self._undetermined is not None:
            return f"Protonettype.undetermined({self._undetermined!r})"
        pairs = tuple((o.wyckoff, o.label) for o in self.occupations)
        return f"Protonettype({self.spacegroup.it_number}, {pairs!r}, {self.net!r}, {self.shapes!r}, {self.torsions!r})"
