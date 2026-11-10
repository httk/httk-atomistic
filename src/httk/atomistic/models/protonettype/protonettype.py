"""The immutable anonymous protonettype value."""

from collections.abc import Sequence
from typing import ClassVar

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
    net: Sequence[Sequence[int]], nsites: int, nops: int, owner: str
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Validate net rows against the site and operation counts and return them in label order."""
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
        if any(not -5 <= n <= 4 for n in (n1, n2, n3)):
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
    Tied classes keep the given order, since it fixes the net's site numbering. A net row
    ``(i, j, k, n1, n2, n3)`` bonds the reference atom of site ``i`` to that of site ``j``
    moved by general-position operation ``k`` (identity = 1) and lattice translation
    ``n``. No symmetry work is done: any faithful value renders *the* label; only the
    canonicaliser produces the canonical one.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their anonymous labels, in site order.
    :param net: The net rows ``(i, j, k, n1, n2, n3)``, with ``i <= j``.
    """

    kind: ClassVar[str] = "protonettype"

    def __init__(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]],
        net: Sequence[Sequence[int]] = (),
    ) -> None:
        self._spacegroup = _standard(spacegroup, "Protonettype")
        self._occupations = tuple(
            value if isinstance(value, PrototypeOccupation) else PrototypeOccupation(*value) for value in occupations
        )
        labels = _class_runs(self._spacegroup, [(o.wyckoff, o.label) for o in self._occupations], "Protonettype")
        if labels != [anonymous_symbol(index) for index in range(len(labels))]:
            raise ValueError("Protonettype class run c must carry the label anonymous_symbol(c)")
        self._net = _checked_net(net, len(self._occupations), len(self._spacegroup.symmetry_operations), "Protonettype")

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
