"""The immutable assigned-species protonetstructure value."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonettype.protonettype import _checked_net, _class_runs, _standard
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.species.classes import _merges_classes, _species_class
from httk.atomistic.models.species.species import Species
from httk.atomistic.symmetry.spacegroup import Spacegroup

if TYPE_CHECKING:
    from httk.atomistic.models.structure.asu import ASUStructure


class Protonetstructure(ProtonetstructureBackend):
    """Store a bare protostructure together with a symmetric covalent bonding net.

    Site ``n`` is ``occupations[n-1]``. Each class is one contiguous run of a single
    species class (species of equal chemistry are one class, see
    :class:`~httk.atomistic.models.protostructure.occupation.WyckoffOccupation`), the runs
    ordered as for :class:`~httk.atomistic.Protonettype`. Net translations may be any integer
    vectors. The net is canonicalized as for :class:`~httk.atomistic.Protonettype`, over the
    equivalent descriptions with the same anonymous bare key; where several give the canonical
    net, the smallest species-name tuple in class order wins (D3), so species can move between
    the Wyckoff letters of those descriptions (``Protonetstructure(225, [("a", Na), ("b", Cl)])``
    gives ``AB_cF8_225_a_b__:Cl-Na``), unlike in :class:`~httk.atomistic.BareProtostructure`.
    :func:`~httk.atomistic.canonical_protonetstructure` also minimises the bare part; for a
    canonical bare part the two agree.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their species, in site order.
    :param net: The net rows ``(i, j, k, n1, n2, n3)``, with ``i <= j``.
    :raises ValueError: If the input is malformed or the canonical net needs a translation
        outside -5..+4, which the notation cannot write (D4).
    :raises CanonicalizationLimitError: If the net search exceeds its 120 s deadline, or an
        enclosing canonicalization budget.
    """

    kind: ClassVar[str] = "protonetstructure"

    def __init__(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[WyckoffOccupation | tuple[str, Species]],
        net: Sequence[Sequence[int]] = (),
    ) -> None:
        from httk.atomistic.symmetry.canonical_classification import _protonetstructure_net
        from httk.atomistic.symmetry.limits import _active, _budget_scope

        self._assign(spacegroup, occupations, net, writable=False)
        with _budget_scope(120.0 if _active.get() is None else None):  # an enclosing budget governs
            canonical = _protonetstructure_net(self, preserve_chirality=True, own_bare=True)
        self._spacegroup, self._occupations, self._net = canonical.spacegroup, canonical.occupations, canonical.net

    @classmethod
    def _of(
        cls,
        spacegroup: Spacegroup | int,
        occupations: Sequence[WyckoffOccupation | tuple[str, Species]],
        net: Sequence[Sequence[int]],
    ) -> Self:
        """Build from a trusted canonical net: validate the structure only, no canonicalization."""
        instance = object.__new__(cls)
        instance._assign(spacegroup, occupations, net, writable=True)
        return instance

    def _assign(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[WyckoffOccupation | tuple[str, Species]],
        net: Sequence[Sequence[int]],
        *,
        writable: bool,
    ) -> None:
        self._spacegroup = _standard(spacegroup, "Protonetstructure")
        self._occupations = tuple(
            value if isinstance(value, WyckoffOccupation) else WyckoffOccupation(value[0], value[1])
            for value in occupations
        )
        _class_runs(self._spacegroup, [(o.wyckoff, o.species.name) for o in self._occupations], "Protonetstructure")
        nops = len(self._spacegroup.symmetry_operations)
        self._net = _checked_net(net, len(self._occupations), nops, "Protonetstructure", writable=writable)

    @classmethod
    def from_asu(cls, asu: "ASUStructure", bonds: Iterable[tuple[int, int, int, int, int, int]]) -> "Protonetstructure":
        """Build a value from a standard-setting ASU structure and its covalent bonds.

        A bond ``(i, j, k, n1, n2, n3)`` joins the reference atom of ``asu.wyckoff_sites[i-1]``
        to general-position operation ``k`` (identity = 1) plus lattice translation ``n``
        applied to the reference atom of ``asu.wyckoff_sites[j-1]``. A reference atom is the
        first-listed Wyckoff coordinate at the site's free parameters, not reduced modulo one; no
        geometry is used. The translation ``n`` is any integer vector, so the free parameters need
        not be reduced into [0, 1). Each bond stands for its whole symmetry orbit, so listing any
        one member of an orbit is enough and repeats collapse. Which pairs are bonded is the
        caller's responsibility: ``~`` means a covalent bond of unspecified order, dative bonds
        counted as covalent. The result is the constructor's: the anonymous bare description is
        kept and the net is canonicalized (see :class:`Protonetstructure`).

        :param asu: The structure, in its space group's standard setting.
        :param bonds: The bonds ``(i, j, k, n1, n2, n3)``.
        :return: The protonetstructure value.
        :raises ValueError: If the setting is not standard, a site retains a representative, a
            bond is malformed, out of range or bonds an atom to itself, the canonical net needs
            a translation outside -5..+4 (D4), or two species share a class whose structure has
            another space group or conventional cell than the declared one.
        """
        from httk.atomistic.models.prototype.notation import class_order_key
        from httk.atomistic.symmetry.protonet import _tables

        spacegroup = asu.spacegroup
        if not spacegroup.is_standard_setting:
            raise ValueError("Protonetstructure.from_asu requires the space group's standard setting")
        sites = asu.wyckoff_sites
        if any(site.representative is not None for site in sites):
            raise ValueError(
                "Protonetstructure.from_asu cannot use a retained WyckoffSite representative; drop it first"
            )
        if _merges_classes(asu):
            from httk.atomistic.models.bareprotostructure.view import BareProtostructureView

            recognized = BareProtostructureView(asu).unview()
            counts = [
                sum(group.wyckoff_position(letter).multiplicity for letter in letters)
                for group, letters in (
                    (recognized.spacegroup, [o.wyckoff for o in recognized.occupations]),
                    (spacegroup, [site.wyckoff for site in sites]),
                )
            ]
            if recognized.spacegroup.it_number != spacegroup.it_number or counts[0] != counts[1]:
                raise ValueError(
                    "Protonetstructure.from_asu: the ASU's species split a species class whose structure has "
                    f"higher symmetry (IT {recognized.spacegroup.it_number}, not {spacegroup.it_number}), so the "
                    "bonds of its sites do not describe the class structure; use "
                    "canonical_asu_protostructure(structure) and bonds on that"
                )
        species = {value.name: value for value in asu.species}
        class_of = {site.species: _species_class(species[site.species]).name for site in sites}
        letters: dict[str, list[str]] = {}
        for site in sites:
            letters.setdefault(class_of[site.species], []).append(site.wyckoff)

        def class_key(name: str) -> tuple[int, tuple[str, ...], str]:
            group = tuple(sorted(letters[name]))
            return class_order_key(group, sum(spacegroup.wyckoff_position(x).multiplicity for x in group), name)

        rank = {name: index for index, name in enumerate(sorted(letters, key=class_key))}
        order = sorted(range(len(sites)), key=lambda s: (rank[class_of[sites[s].species]], sites[s].wyckoff))
        number = {old: new for new, old in enumerate(order)}
        tables = _tables(spacegroup.it_number)
        rows = set()
        for bond in bonds:
            values = tuple(bond)
            if len(values) != 6 or any(not isinstance(v, int) or isinstance(v, bool) for v in values):
                raise ValueError(f"bond {bond!r} must be six integers (i, j, k, n1, n2, n3)")
            i, j, k, n1, n2, n3 = values
            if not (1 <= i <= len(sites) and 1 <= j <= len(sites)):
                raise ValueError(f"bond {values!r} needs site indices in 1..{len(sites)}")
            if not 1 <= k <= tables.nops:
                raise ValueError(f"bond {values!r} needs an operation number in 1..{tables.nops}")
            a, b, g = number[i - 1], number[j - 1], (k - 1, n1, n2, n3)
            if a > b:
                a, b, g = b, a, tables.inv(g)
            H = [tables.letter(sites[order[s]].wyckoff).H for s in (a, b)]
            g = tables.canon(H[0], H[1], g, same_site=a == b)
            if a == b and g == (0, 0, 0, 0):
                raise ValueError(f"bond {values!r} bonds an atom to itself")
            rows.add((a + 1, b + 1, g[0] + 1, g[1], g[2], g[3]))
        return Protonetstructure(
            spacegroup, [(sites[s].wyckoff, species[sites[s].species]) for s in order], sorted(rows)
        )

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        """Return the occupied Wyckoff positions in site order."""
        return self._occupations

    @property
    def net(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return the net rows ``(i, j, k, n1, n2, n3)`` in label order."""
        return self._net

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protonetstructure):
            return NotImplemented
        return (self.spacegroup, self.occupations, self.net) == (other.spacegroup, other.occupations, other.net)

    def __hash__(self) -> int:
        return hash((self.spacegroup, self.occupations, self.net))

    def __repr__(self) -> str:
        pairs = tuple((o.wyckoff, o.species) for o in self.occupations)
        return f"Protonetstructure({self.spacegroup.it_number}, {pairs!r}, {self.net!r})"
