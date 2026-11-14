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


def _assign_net(
    value: Any,
    net: Sequence[Sequence[int]],
    shapes: Sequence[str | None] | None,
    owner: str,
    *,
    writable: bool,
) -> None:
    """Validate and store a value's net and shapes, the rows in label order (shared by both levels)."""
    letters = [o.wyckoff for o in value._occupations]
    nops = len(value._spacegroup.symmetry_operations)
    rows = _checked_net(net, len(letters), nops, owner, writable=writable)
    counts = _net_counts(value._spacegroup, letters, rows, owner)
    value._shapes = _checked_shapes(shapes, counts, owner)
    value._net = _label_rows(rows, counts)


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
    :raises ValueError: If the input is malformed, a shape is missing, unknown or has the wrong
        bond count, or the canonical net needs a translation outside -5..+4, which the notation
        cannot write (D4).
    :raises TypeError: If ``shapes`` is a string or holds a value that is neither a string nor ``None``.
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
    ) -> None:
        from httk.atomistic.symmetry.canonical_classification import _protonettype_net
        from httk.atomistic.symmetry.limits import _active, _budget_scope

        self._assign(spacegroup, occupations, net, shapes, writable=False)
        with _budget_scope(120.0 if _active.get() is None else None):  # an enclosing budget governs
            canonical = _protonettype_net(self, preserve_chirality=True, own_bare=True)
        self._spacegroup, self._occupations = canonical.spacegroup, canonical.occupations
        self._net, self._shapes = canonical.net, canonical.shapes

    @classmethod
    def _of(
        cls,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]],
        shapes: Sequence[str | None] | None,
    ) -> Self:
        """Build from a trusted canonical net and its shapes: validate the structure only, no canonicalization."""
        instance = object.__new__(cls)
        instance._assign(spacegroup, occupations, net, shapes, writable=True)
        return instance

    def _assign(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]],
        shapes: Sequence[str | None] | None,
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
        _assign_net(self, net, shapes, "Protonettype", writable=writable)

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protonettype):
            return NotImplemented
        if self._undetermined is not None or other._undetermined is not None:
            return self._undetermined == other._undetermined
        return (self.spacegroup, self.occupations, self.net, self.shapes) == (
            other.spacegroup,
            other.occupations,
            other.net,
            other.shapes,
        )

    def __hash__(self) -> int:
        if self._undetermined is not None:
            return hash(self._undetermined)
        return hash((self.spacegroup, self.occupations, self.net, self.shapes))

    def __repr__(self) -> str:
        if self._undetermined is not None:
            return f"Protonettype.undetermined({self._undetermined!r})"
        pairs = tuple((o.wyckoff, o.label) for o in self.occupations)
        return f"Protonettype({self.spacegroup.it_number}, {pairs!r}, {self.net!r}, {self.shapes!r})"
