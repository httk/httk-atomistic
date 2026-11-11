"""The immutable anonymous protonettype value."""

from collections.abc import Sequence
from typing import ClassVar, Self

from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.protonettype.backend import ProtonettypeBackend
from httk.atomistic.models.prototype.notation import _net_row_key, class_order_key
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup


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
    """Validate net rows against the site and operation counts and return them in label order.

    Input rows take any lattice translation; ``writable`` requires the label's range -5..+4.
    """
    rows: list[tuple[int, int, int, int, int, int]] = []
    for row in net:
        values = tuple(row)
        if len(values) != 6 or any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise ValueError(f"{owner} net row {row!r} must be six integers (i, j, k, n1, n2, n3)")
        i, j, k, n1, n2, n3 = (int(v) for v in values)
        if not 1 <= i <= j <= nsites:
            raise ValueError(f"{owner} net row {values!r} needs 1 <= i <= j <= {nsites}")
        if not 1 <= k <= nops:
            raise ValueError(f"{owner} net row {values!r} needs an operation number in 1..{nops}")
        if writable and any(not -5 <= n <= 4 for n in (n1, n2, n3)):
            raise ValueError(f"{owner} net row {values!r} has a translation outside -5..+4")
        if i == j and k == 1 and (n1, n2, n3) == (0, 0, 0):
            raise ValueError(f"{owner} net row {values!r} bonds an atom to itself")
        rows.append((i, j, k, n1, n2, n3))
    if len(set(rows)) != len(rows):
        raise ValueError(f"{owner} net has duplicate rows")
    return tuple(sorted(rows, key=_net_row_key))


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

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their anonymous labels, in site order.
    :param net: The net rows ``(i, j, k, n1, n2, n3)``, with ``i <= j``.
    :raises ValueError: If the input is malformed or the canonical net needs a translation
        outside -5..+4, which the notation cannot write (D4).
    :raises CanonicalizationLimitError: If the net search exceeds its 120 s deadline, or an
        enclosing canonicalization budget.
    """

    kind: ClassVar[str] = "protonettype"

    def __init__(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]] = (),
    ) -> None:
        from httk.atomistic.symmetry.canonical_classification import _protonettype_net
        from httk.atomistic.symmetry.limits import _active, _budget_scope

        self._assign(spacegroup, occupations, net, writable=False)
        with _budget_scope(120.0 if _active.get() is None else None):  # an enclosing budget governs
            canonical = _protonettype_net(self, preserve_chirality=True, own_bare=True)
        self._spacegroup, self._occupations, self._net = canonical.spacegroup, canonical.occupations, canonical.net

    @classmethod
    def _of(
        cls,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]],
    ) -> Self:
        """Build from a trusted canonical net: validate the structure only, no canonicalization."""
        instance = object.__new__(cls)
        instance._assign(spacegroup, occupations, net, writable=True)
        return instance

    def _assign(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]],
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
        nops = len(self._spacegroup.symmetry_operations)
        self._net = _checked_net(net, len(self._occupations), nops, "Protonettype", writable=writable)

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def occupations(self) -> tuple[PrototypeOccupation, ...]:
        """Return the anonymous Wyckoff occupations in site order."""
        return self._occupations

    @property
    def net(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return the net rows ``(i, j, k, n1, n2, n3)`` in label order."""
        return self._net

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protonettype):
            return NotImplemented
        return (self.spacegroup, self.occupations, self.net) == (other.spacegroup, other.occupations, other.net)

    def __hash__(self) -> int:
        return hash((self.spacegroup, self.occupations, self.net))

    def __repr__(self) -> str:
        pairs = tuple((o.wyckoff, o.label) for o in self.occupations)
        return f"Protonettype({self.spacegroup.it_number}, {pairs!r}, {self.net!r})"
