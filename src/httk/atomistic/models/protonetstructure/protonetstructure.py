"""The immutable assigned-species protonetstructure value."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonettype.protonettype import _assign_net, _class_runs, _standard
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.prototype.notation import _known, _parse_undetermined, _undetermined, _unknown
from httk.atomistic.models.species.classes import _merges_classes, _species_class
from httk.atomistic.models.species.species import Species
from httk.atomistic.symmetry.spacegroup import Spacegroup

if TYPE_CHECKING:
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure


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
    :param net: The net rows ``(i, j, k, n1, n2, n3)``, in either direction; stored rows are
        written from their listing site.
    :param shapes: The bond shape symbol per site, as for :class:`~httk.atomistic.Protonettype`.
    :raises ValueError: If the input is malformed, a shape is missing, unknown or has the wrong
        bond count, or the canonical net needs a translation outside -5..+4, which the notation
        cannot write (D4).
    :raises TypeError: If ``shapes`` is a string or holds a value that is neither a string nor ``None``.
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
        shapes: Sequence[str | None] | None = None,
    ) -> None:
        from httk.atomistic.symmetry.canonical_classification import _protonetstructure_net
        from httk.atomistic.symmetry.limits import _active, _budget_scope

        self._assign(spacegroup, occupations, net, shapes, writable=False)
        with _budget_scope(120.0 if _active.get() is None else None):  # an enclosing budget governs
            canonical = _protonetstructure_net(self, preserve_chirality=True, own_bare=True)
        self._spacegroup, self._occupations = canonical.spacegroup, canonical.occupations
        self._net, self._shapes = canonical.net, canonical.shapes

    @classmethod
    def _of(
        cls,
        spacegroup: Spacegroup | int,
        occupations: Sequence[WyckoffOccupation | tuple[str, Species]],
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
        occupations: Sequence[WyckoffOccupation | tuple[str, Species]],
        net: Sequence[Sequence[int]],
        shapes: Sequence[str | None] | None,
        *,
        writable: bool,
    ) -> None:
        self._spacegroup = _standard(spacegroup, "Protonetstructure")
        self._occupations = tuple(
            value if isinstance(value, WyckoffOccupation) else WyckoffOccupation(value[0], value[1])
            for value in occupations
        )
        _class_runs(self._spacegroup, [(o.wyckoff, o.species.name) for o in self._occupations], "Protonetstructure")
        _assign_net(self, net, shapes, "Protonetstructure", writable=writable)

    @classmethod
    def from_asu(
        cls,
        asu: "FundamentalDomainStructure",
        bonds: Iterable[tuple[int, int, int, int, int, int]] | None = None,
        *,
        shapes: Sequence[str | None] | None = None,
        on_limit: Literal["undetermined", "raise"] = "undetermined",
    ) -> "Protonetstructure":
        """Build a value from an ASU structure and its covalent bonds.

        Without ``bonds`` the structure's own bond network (``bonds``)
        is used, in any setting and either hand: the structure and its network are re-described in a
        right-handed basis of the standard setting first. With ``bonds``, a bond
        ``(i, j, k, n1, n2, n3)`` joins the reference atom of ``asu.wyckoff_sites[i-1]`` to
        general-position operation ``k`` (identity = 1) plus lattice translation ``n`` applied to the
        reference atom of ``asu.wyckoff_sites[j-1]``, and the structure must be in its space group's
        standard setting without retained site representatives. A reference atom is the first-listed
        Wyckoff coordinate at the site's free parameters, not reduced modulo one, so a structure's own
        network is read whatever representatives its sites retain. The translation ``n`` is any
        integer vector, so the free parameters need not be reduced into [0, 1). With ``bonds`` the
        structure's own network and supplied shapes are ignored; ``shapes`` then gives the supplied
        shapes. Every site with two or more bonds gets a shape: its supplied symbol, else the
        baseline detector's (:mod:`httk.atomistic.symmetry.shapes`), which is the only use of the
        geometry. Each bond stands for its whole
        symmetry orbit, so listing any one member of an orbit is enough and repeats collapse. Which
        pairs are bonded is the caller's responsibility: ``~`` means a covalent bond of unspecified
        order, dative bonds counted as covalent. When two species share a class whose structure has
        higher symmetry than declared, that symmetry is recognized and the bonds are re-expressed in
        it, which needs the bond set to be invariant under it. The result is the constructor's: the
        anonymous bare description is kept and the net is canonicalized (see
        :class:`Protonetstructure`), within 120 s or an enclosing budget. When the net search runs
        out, the result is undetermined: the canonical bare protostructure label with ``__X`` for the
        net (``<bare protostructure label>__X:<names>``, see :meth:`undetermined`). The bare part is
        then canonicalized after the search; under an enclosing budget it is canonicalized before the
        search, within that budget, and the label is ``<Formulatype>_<Pearson>_<number>_X__X:<names>``
        if it did not finish.

        :param asu: The structure.
        :param bonds: The bonds ``(i, j, k, n1, n2, n3)``, or ``None`` for the structure's own network.
        :param shapes: With ``bonds``, optional supplied shapes, one symbol or ``None`` (the
            baseline) per entry of ``asu.wyckoff_sites``, validated as for
            :class:`~httk.atomistic.FundamentalDomainStructure`.
        :param on_limit: ``"undetermined"`` for the undetermined value when the net search runs out
            of its budget, or ``"raise"`` to raise
            :class:`~httk.atomistic.CanonicalizationLimitError` instead.
        :return: The protonetstructure value.
        :raises ValueError: If ``bonds`` is omitted and the structure has no bond network, or
            ``shapes`` is given without ``bonds``; if ``bonds`` is given and the setting is not
            standard, a site retains a representative, a bond is malformed, out of range or bonds an
            atom to itself, or a supplied shape is invalid; if the bond set is not invariant under
            the recognized symmetry of the species classes, or sites with different supplied shapes
            merge under it; if a site lies at a special point of its own Wyckoff letter (its bond
            count and shape are undefined, so it is refused even with supplied shapes and even when
            it has no bonds: give it the special letter); if a site without a supplied shape has a
            zero-length bond; if the canonical net
            needs a translation outside -5..+4 (D4); or if ``on_limit`` is unknown.
        :raises CanonicalizationLimitError: With ``on_limit="raise"``, if the net search runs out of
            its budget.
        """
        from httk.atomistic.models.prototype.notation import class_order_key
        from httk.atomistic.models.structure.asu import _affine, _checked_shapes, _op, _standard_elements
        from httk.atomistic.models.structure.network import BondNetwork, SiteBond
        from httk.atomistic.symmetry.canonical_protostructure import _left_handed, _sohncke
        from httk.atomistic.symmetry.limits import CanonicalizationLimitError, _active, _budget_scope, _check_on_limit
        from httk.atomistic.symmetry.protonet import _tables
        from httk.atomistic.symmetry.shapes import _site_shapes, _stabilisers

        _check_on_limit(on_limit)
        for index, site in enumerate(asu.wyckoff_sites, 1):
            try:
                _stabilisers(asu.spacegroup, [site])
            except ValueError:
                raise ValueError(
                    f"Protonetstructure.from_asu: site {index} ({site.species} on Wyckoff letter {site.wyckoff!r}) "
                    "lies at a special point of its letter, so its bond count and shape are undefined, with or "
                    "without supplied shapes; give it the special letter"
                ) from None
        if bonds is None:
            if shapes is not None:
                raise ValueError(
                    "Protonetstructure.from_asu: shapes= needs bonds given as index tuples; a structure's own "
                    "network carries its own shapes"
                )
            if asu.bonds is None:
                raise ValueError("Protonetstructure.from_asu: the structure has no bond network; pass bonds")
            structure = _protonet_asu(asu)
            elements = None
        else:
            spacegroup = asu.spacegroup
            if not spacegroup.is_standard_setting:
                raise ValueError(
                    "Protonetstructure.from_asu requires the space group's standard setting for bonds given "
                    "as index tuples; give the structure its bond network to label it in any setting"
                )
            if any(site.representative is not None for site in asu.wyckoff_sites):
                raise ValueError(
                    "Protonetstructure.from_asu cannot use a retained WyckoffSite representative; drop it first"
                )
            tables = _tables(spacegroup.it_number)
            elements = [_index_bond(tables, asu, bond) for bond in bonds]
            # the bonds as the structure's network, with the supplied shapes (its own are ignored)
            standard = _standard_elements(spacegroup)
            network = tuple(SiteBond(i, j, _affine(standard.operation(e))) for i, j, e in elements)
            checked = _checked_shapes(spacegroup, asu.wyckoff_sites, network, shapes)
            structure = asu._with_bonds(BondNetwork(network, checked))
            if (_sohncke(spacegroup) and _left_handed(asu)) or _merges_classes(asu):
                # re-described in a right-handed cell or the class symmetry: the network and shapes are carried
                structure = _protonet_asu(structure)
                elements = None
        spacegroup = structure.spacegroup
        sites = structure.wyckoff_sites
        tables = _tables(spacegroup.it_number)
        if elements is None:
            if structure.bonds is None:  # a carry dropped it (a non-invariant set or merged shapes)
                raise ValueError(
                    "Protonetstructure.from_asu: the bond network could not be carried to the description "
                    "the label is read from"
                )
            standard = _standard_elements(spacegroup)
            elements = [
                (bond.site, bond.partner, standard.element(_op(bond.operation))) for bond in structure.bonds.pairs
            ]
        species = {value.name: value for value in structure.species}
        class_of = {site.species: _species_class(species[site.species]).name for site in sites}
        letters: dict[str, list[str]] = {}
        for site in sites:
            letters.setdefault(class_of[site.species], []).append(site.wyckoff)

        def class_key(name: str) -> tuple[int, tuple[str, ...], str]:
            group = tuple(sorted(letters[name]))
            return class_order_key(group, sum(spacegroup.wyckoff_position(x).multiplicity for x in group), name)

        rank = {name: index for index, name in enumerate(sorted(letters, key=class_key))}
        order = sorted(range(len(sites)), key=lambda s: (rank[class_of[sites[s].species]], sites[s].wyckoff))
        site_shapes = _site_shapes(structure)
        number = {old: new for new, old in enumerate(order)}
        rows = set()
        for i, j, g in elements:
            a, b = number[i], number[j]
            if a > b:
                a, b, g = b, a, tables.inv(g)
            H = [tables.letter(sites[order[s]].wyckoff).H for s in (a, b)]
            g = tables.canon(H[0], H[1], g, same_site=a == b)
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
                return Protonetstructure(spacegroup, occupations, sorted(rows), [site_shapes[s] for s in order])
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
        if not isinstance(other, Protonetstructure):
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
            return f"Protonetstructure.undetermined({self._undetermined!r})"
        pairs = tuple((o.wyckoff, o.species) for o in self.occupations)
        return f"Protonetstructure({self.spacegroup.it_number}, {pairs!r}, {self.net!r}, {self.shapes!r})"


def _index_bond(
    tables: Any, asu: "FundamentalDomainStructure", bond: Iterable[int]
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Validate one index-tuple bond ``(i, j, k, n1, n2, n3)`` and return it 0-based as ``(i, j, element)``."""
    values = tuple(bond)
    sites = asu.wyckoff_sites
    if len(values) != 6 or any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        raise ValueError(f"bond {bond!r} must be six integers (i, j, k, n1, n2, n3)")
    i, j, k, n1, n2, n3 = values
    if not (1 <= i <= len(sites) and 1 <= j <= len(sites)):
        raise ValueError(f"bond {values!r} needs site indices in 1..{len(sites)}")
    if not 1 <= k <= tables.nops:
        raise ValueError(f"bond {values!r} needs an operation number in 1..{tables.nops}")
    g = (k - 1, n1, n2, n3)
    if i == j:
        H = tables.letter(sites[i - 1].wyckoff).H
        if tables.canon(H, H, g, same_site=True) == (0, 0, 0, 0):
            raise ValueError(f"bond {values!r} bonds an atom to itself")
    return i - 1, j - 1, g


def _protonet_asu(asu: "FundamentalDomainStructure") -> "FundamentalDomainStructure":
    """Re-describe a structure with a bond network for the protonet labels.

    The result is in a right-handed basis of the standard setting, and when two species share a
    class whose structure has higher symmetry than declared (another space group or conventional
    cell), it is the recognized class structure with the bonds matched onto it.

    :param asu: The structure with a bond network.
    :return: The re-described structure with its network and supplied shapes.
    :raises ValueError: If the bond set is not invariant under the recognized class symmetry, or
        sites with different supplied shapes merge under it.
    """
    from httk.atomistic.models.species.classes import _class_structure
    from httk.atomistic.models.structure.bonds import _matched, _ShapeConflict
    from httk.atomistic.symmetry.canonical_protostructure import _left_handed, _right_handed, _sohncke
    from httk.atomistic.symmetry.recognition import recognize_asu
    from httk.atomistic.symmetry.subgroups import _standard_input

    def oriented(structure: Any) -> Any:
        if not structure.spacegroup.is_standard_setting:
            structure = _standard_input(structure)
        return _right_handed(structure) if _sohncke(structure.spacegroup) and _left_handed(structure) else structure

    structure = oriented(asu)
    if not _merges_classes(structure):
        return structure
    recognized = recognize_asu(_class_structure(structure))
    declared = sum(structure.spacegroup.wyckoff_position(site.wyckoff).multiplicity for site in structure.wyckoff_sites)
    found = sum(recognized.spacegroup.wyckoff_position(site.wyckoff).multiplicity for site in recognized.wyckoff_sites)
    if recognized.spacegroup.it_number == structure.spacegroup.it_number and declared == found:
        return structure
    try:
        network = _matched(
            structure, recognized, recognized.transform.operation.inverse() * structure.transform.operation
        )
    except _ShapeConflict:
        raise ValueError(
            "Protonetstructure.from_asu: sites with different supplied shapes merge under the recognized symmetry "
            f"IT {recognized.spacegroup.it_number}"
        ) from None
    if network is None:
        raise ValueError(
            "Protonetstructure.from_asu: two species share a class whose structure has higher symmetry "
            f"(IT {recognized.spacegroup.it_number}, not {structure.spacegroup.it_number}), and the bond set is not "
            f"invariant under the recognized symmetry IT {recognized.spacegroup.it_number}"
        )
    return oriented(recognized._with_bonds(network))
