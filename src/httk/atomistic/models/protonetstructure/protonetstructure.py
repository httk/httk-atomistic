"""The immutable assigned-species protonetstructure value."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonettype.protonettype import _checked_net, _class_runs, _standard
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.prototype.notation import _known, _parse_undetermined, _undetermined, _unknown
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

    Occupations carry no cell: they are read in a right-handed basis of the standard setting,
    so in a Sohncke group they name one hand of the crystal.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their species, in site order.
    :param net: The net rows ``(i, j, k, n1, n2, n3)``, with ``i <= j``.
    :raises ValueError: If the input is malformed or the canonical net needs a translation
        outside -5..+4, which the notation cannot write (D4).
    :raises CanonicalizationLimitError: If the net search exceeds its 120 s deadline, or an
        enclosing canonicalization budget.
    """

    kind: ClassVar[str] = "protonetstructure"
    _undetermined: str | None = None

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
    def from_asu(
        cls,
        asu: "ASUStructure",
        bonds: Iterable[tuple[int, int, int, int, int, int]],
        *,
        on_limit: Literal["undetermined", "raise"] = "undetermined",
    ) -> "Protonetstructure":
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
        kept and the net is canonicalized (see :class:`Protonetstructure`), within 120 s or an
        enclosing budget. When the net search runs out, the result is undetermined: the canonical
        bare protostructure label with ``__X`` for the net
        (``<bare protostructure label>__X:<names>``, see :meth:`undetermined`). The bare part is
        then canonicalized after the search; under an enclosing budget it is canonicalized before
        the search, within that budget, and the label is
        ``<Formulatype>_<Pearson>_<number>_X__X:<names>`` if it did not finish.

        :param asu: The structure, in its space group's standard setting.
        :param bonds: The bonds ``(i, j, k, n1, n2, n3)``.
        :param on_limit: ``"undetermined"`` for the undetermined value when the net search runs out
            of its budget, or ``"raise"`` to raise
            :class:`~httk.atomistic.CanonicalizationLimitError` instead.
        :return: The protonetstructure value.
        :raises ValueError: If the setting is not standard, the cell (after the setting transform)
            is left-handed in a Sohncke group, whose letters and bonds would then describe the
            mirror image, a site retains a representative, a bond is malformed, out of range or
            bonds an atom to itself, the canonical net needs a translation outside -5..+4 (D4), or
            two species share a class whose structure has another space group or conventional cell
            than the declared one, or ``on_limit`` is unknown.
        :raises CanonicalizationLimitError: With ``on_limit="raise"``, if the net search runs out of
            its budget.
        """
        from httk.atomistic.models.prototype.notation import class_order_key
        from httk.atomistic.symmetry.limits import CanonicalizationLimitError, _active, _budget_scope, _check_on_limit
        from httk.atomistic.symmetry.protonet import _tables

        _check_on_limit(on_limit)
        spacegroup = asu.spacegroup
        if not spacegroup.is_standard_setting:
            raise ValueError("Protonetstructure.from_asu requires the space group's standard setting")
        from httk.atomistic.symmetry.canonical_protostructure import _left_handed, _sohncke

        if _sohncke(spacegroup) and _left_handed(asu):
            raise ValueError(
                "Protonetstructure.from_asu: the cell is left-handed in a Sohncke group, so the Wyckoff letters "
                "and bonds would describe the mirror image; describe the crystal in a right-handed basis, or use "
                "canonical_asu_protostructure(structure) and bonds on that"
            )
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
        from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
        from httk.atomistic.models.prototype.notation import _with_net_marker
        from httk.atomistic.symmetry.canonical_classification import _canonical_bare, _wyckoff_undetermined

        occupations = [(sites[s].wyckoff, species[sites[s].species]) for s in order]
        description = BareProtostructure(spacegroup, occupations)
        bare = None
        # an undetermined net bins by the canonical bare part (U4); under an enclosing budget that part
        # must come first, or nothing would remain for it once the net search has run out
        enclosed = _active.get() is not None
        try:
            with _budget_scope(None if enclosed else 120.0):
                if enclosed and on_limit == "undetermined":
                    bare = _canonical_bare(description, prototype=False, preserve_chirality=True)
                return Protonetstructure(spacegroup, occupations, sorted(rows))
        except CanonicalizationLimitError:
            if on_limit == "raise":
                raise
        if not enclosed:  # without an enclosing budget the bare part runs unbounded; it is cheap
            bare = _canonical_bare(description, prototype=False, preserve_chirality=True)
        if bare is None:
            text = _wyckoff_undetermined(description, named=True, net=True, preserve_chirality=True)
        else:
            text = _with_net_marker(str(bare.label))
        return Protonetstructure._undetermined_of(text)

    @classmethod
    def undetermined(cls, label: str) -> "Protonetstructure":
        """Return the undetermined protonetstructure of a label in which ``X`` marks what is not determined.

        The canonicalisers return such a value when a part did not finish within its time budget; it
        holds only its label: ``<Formulatype>_X__X:<names>``,
        ``<Formulatype>_<Pearson>_<number>_X__X:<names>``, or the determined bare label followed by
        ``__X:<names>`` when only the net was not determined. Two undetermined values are equal when
        their labels are; that does not mean their structures are the same.

        :param label: The undetermined label.
        :return: The undetermined value, with ``determined`` false.
        :raises ValueError: If ``label`` is not an undetermined protonetstructure label.
        """
        _parse_undetermined(label, cls.kind)
        return Protonetstructure._undetermined_of(label)

    @classmethod
    def _undetermined_of(cls, label: str) -> "Protonetstructure":
        """Build an undetermined value from a trusted label, without parsing it."""
        instance = object.__new__(Protonetstructure)
        instance._undetermined = label
        return instance

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "spacegroup", undetermined.spacegroup)
        return self._spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        """Return the occupied Wyckoff positions in site order."""
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "occupations", undetermined.occupations)
        return self._occupations

    @property
    def net(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return the net rows ``(i, j, k, n1, n2, n3)`` in label order."""
        if _undetermined(self) is not None:
            _unknown(self, "net")
        return self._net

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protonetstructure):
            return NotImplemented
        if self._undetermined is not None or other._undetermined is not None:
            return self._undetermined == other._undetermined
        return (self.spacegroup, self.occupations, self.net) == (other.spacegroup, other.occupations, other.net)

    def __hash__(self) -> int:
        if self._undetermined is not None:
            return hash(self._undetermined)
        return hash((self.spacegroup, self.occupations, self.net))

    def __repr__(self) -> str:
        if self._undetermined is not None:
            return f"Protonetstructure.undetermined({self._undetermined!r})"
        pairs = tuple((o.wyckoff, o.species) for o in self.occupations)
        return f"Protonetstructure({self.spacegroup.it_number}, {pairs!r}, {self.net!r})"
