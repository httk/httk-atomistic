"""The one home of httk's prototype and protostructure label notation.

An httk label encodes the information content of an unsuffixed AFLOW-style prototype
label: a space group, its occupied Wyckoff letters, and the partition of those
occupations into species classes. The grammar is::

    ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*                 # prototype label
    ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*:NAME(-NAME)*    # protostructure label

A ``GROUP`` is the concatenation of one species class's Wyckoff letters, sorted
alphabetically, a letter occupied ``k >= 2`` times prefixed by the integer ``k``
(``2e``); count ``1`` is omitted. A ``NAME`` is a species class name (``Ti``,
``Bi0.5Na0.5``, ``Li0.333``, ``OH2``; see :mod:`httk.atomistic.models.species.classes`), which
determines the class species and never contains ``-``, ``:`` or ``_``. Groups appear in
the order of :func:`class_order_key`: largest summed conventional multiplicity first, then
sorted Wyckoff letters. ``ANON`` is the anonymous formula built from
:func:`~httk.atomistic.models.formula.notation.anonymous_symbol` in group order with
per-group summed conventional multiplicities reduced by their overall gcd, so it equals
the :class:`~httk.atomistic.Formulatype` of the site counts.

A part that the canonicalisers could not determine within their time budget is written ``X``,
which replaces that part and everything after it that depends on it (see
``determined`` of the label values)::

    ANON_X                         ANON_X__X                 # nor the symmetry
    ANON_PEARSON_ITNUMBER_X        ANON_PEARSON_ITNUMBER_X__X  # nor the Wyckoff part
                                   PROTOTYPELABEL__X           # nor the net

each followed by ``:NAME(-NAME)*`` at the assigned levels. Where the Wyckoff part is
undetermined the names are ordered by site count, largest first, then by name; after a
determined bare part they keep its order. ``X`` never fills those positions in a determined
label.

httk labels are deliberately NOT AFLOW labels. AFLOW orders classes by element symbol
(alphabetically) so its unsuffixed prefix still depends on the chemistry; httk orders
classes by site count and then by Wyckoff letters, so the unsuffixed prefix is
element-agnostic and a protostructure label is exactly the prototype label of the erased
template followed by ``:`` and the class species names. Calcite renders as
``A3BC_hR10_167_e_a_b:O-Ca-C`` here and as ``ABC3_hR10_167_b_a_e:C-Ca-O`` in AFLOW
style. :func:`render_aflow_label` renders the AFLOW-style variant for interoperability;
it has no parser here.

"Canonicality" terminology: any faithful render of an object is *the* prototype or
protostructure label; the *canonical* prototype or protostructure label is the one
obtained from a normalizer-canonical object (for example one derived via
``canonical_asu``). The renderer performs no affine-normalizer pass.

ponytail: the label is canonical only up to the fixed Wyckoff-letter naming of the IT
standard setting; two normalizer-equivalent occupation sets can still render to distinct
labels. The upgrade path is to vendor the per-Hall affine-normalizer coset data in
``httk.atomistic.data``, induce the Wyckoff-letter permutations it generates, and emit the
lexicographic-min label over that orbit. Not needed until labels must match across
normalizer-equivalent settings.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, reduce
from math import gcd
from typing import TYPE_CHECKING, Any, NoReturn

from httk.atomistic.models.formula.notation import (
    anonymous_symbol,
    parse_anonymous_formula,
    reduced_coefficients,
    render_anonymous,
)
from httk.atomistic.models.species.classes import _species_from_class_name
from httk.atomistic.models.species.species import Species
from httk.atomistic.symmetry.spacegroup import Spacegroup

if TYPE_CHECKING:
    from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
    from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
    from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
    from httk.atomistic.models.protonettype.protonettype import Protonettype

# The 27th Wyckoff letter used by a handful of high-multiplicity settings (group 47's
# eightfold orbit). It renders as 'A' and parses back from it; positionally a group token
# never collides with the leading anonymous formula.
_SPECIAL_LETTER = "α"
_SPECIAL_RENDER = "A"

_CRYSTAL_SYSTEM_LETTER = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}

_GROUP_TOKEN = re.compile(r"(\d*)([a-zA-Z])")
_PEARSON = re.compile(r"[a-z][A-Z]([1-9][0-9]*)")
# lattice points of the conventional cell per centring type; R is tabulated on hexagonal axes
_LATTICE_POINTS = {"P": 1, "A": 2, "B": 2, "C": 2, "S": 2, "I": 2, "F": 4, "R": 3}
_NET_TARGET = re.compile(r"([0-9]+)(?:@([0-9]+)(?:@([0-9]{3}))?)?")
_NET_HEAD = re.compile(r"([1-9][0-9]*)([A-Z]+[1-9][0-9]*)?")  # site number, then an optional shape
_NET_RESERVED = frozenset("-=#$:/\\")


def pearson_symbol(spacegroup: Spacegroup, nsites_conventional: int) -> str:
    """Return the Pearson symbol for a setting and its conventional-cell site count.

    The crystal-system letter follows the space group's ``crystal_system``; the centring
    letter follows its ``centring_type`` with the base-centred variants
    ``A``, ``B``, ``C``, and ``S`` folded to ``C``; the count is
    ``nsites_conventional``, except a rhombohedral ``R`` setting (tabulated on hexagonal
    axes) divides it by three. Calcite (167 with 30 conventional sites) yields ``hR10``.

    :param spacegroup: The standard-setting space group.
    :param nsites_conventional: The number of sites in the standard conventional cell.
    :return: The Pearson symbol, such as ``"cF8"``.
    :raises ValueError: If a rhombohedral count is not divisible by three.
    """
    system_letter = _CRYSTAL_SYSTEM_LETTER[spacegroup.crystal_system]
    centring = spacegroup.centring_type
    centring_letter = "C" if centring in ("A", "B", "C", "S") else centring
    count = nsites_conventional
    if centring == "R":
        if count % 3 != 0:
            raise ValueError(
                f"rhombohedral setting {spacegroup.setting} has {count} conventional sites, not a multiple of three"
            )
        count //= 3
    return f"{system_letter}{centring_letter}{count}"


def _render_letter(letter: str) -> str:
    """Return the display form of a Wyckoff letter (the special 27th letter as ``A``)."""
    return _SPECIAL_RENDER if letter == _SPECIAL_LETTER else letter


def _render_group(letters: Sequence[str]) -> str:
    """Render one class's Wyckoff letters, sorted, with counts for repeats."""
    ordered = sorted(letters)
    parts: list[str] = []
    index = 0
    while index < len(ordered):
        letter = ordered[index]
        run = 1
        while index + run < len(ordered) and ordered[index + run] == letter:
            run += 1
        parts.append((str(run) if run >= 2 else "") + _render_letter(letter))
        index += run
    return "".join(parts)


def _classes(
    spacegroup: Spacegroup,
    occupations: Sequence[tuple[str, str]],
) -> dict[str, tuple[tuple[str, ...], int]]:
    """Group ``(wyckoff, class-key)`` occupations into ``key -> (letters, conv mult)``.

    :param spacegroup: The standard-setting space group.
    :param occupations: The occupied ``(wyckoff, class-key)`` pairs.
    :return: For each class key its sorted Wyckoff letters and summed conventional
        multiplicity.
    :raises ValueError: If any Wyckoff letter is absent from the setting.
    """
    letters: dict[str, list[str]] = {}
    multiplicity: dict[str, int] = {}
    for wyckoff, key in occupations:
        try:
            position = spacegroup.wyckoff_position(wyckoff)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        letters.setdefault(key, []).append(wyckoff)
        multiplicity[key] = multiplicity.get(key, 0) + position.multiplicity
    return {key: (tuple(sorted(values)), multiplicity[key]) for key, values in letters.items()}


def _assemble(
    spacegroup: Spacegroup,
    ordered_letters: Sequence[tuple[str, ...]],
    ordered_multiplicity: Sequence[int],
) -> str:
    """Build the unsuffixed label from ordered per-class letters and multiplicities."""
    reduced = _reduce_counts(ordered_multiplicity)
    anon = "".join(anonymous_symbol(index) + (str(count) if count != 1 else "") for index, count in enumerate(reduced))
    pearson = pearson_symbol(spacegroup, sum(ordered_multiplicity))
    groups = "_".join(_render_group(letters) for letters in ordered_letters)
    return f"{anon}_{pearson}_{spacegroup.it_number}_{groups}"


def _reduce_counts(counts: Sequence[int]) -> tuple[int, ...]:
    """Divide integer counts by their overall greatest common divisor."""
    common = reduce(gcd, counts)
    return tuple(count // common for count in counts)


def class_order_key(letters: tuple[str, ...], multiplicity: int, tie: str = "") -> tuple[int, tuple[str, ...], str]:
    """Return the sort key of one species class: Formulatype order, then Wyckoff letters.

    The class with the largest summed conventional multiplicity comes first, as in
    :class:`~httk.atomistic.Formulatype`; equal counts are ordered by the sorted Wyckoff
    letters, and ``tie`` (a species class name or input key) breaks what remains. Label
    rendering, anonymous relabelling and the canonicalisers' choice among equivalent
    descriptions all sort by this key.

    :param letters: The class's sorted Wyckoff letters, repetitions included.
    :param multiplicity: The class's summed conventional Wyckoff multiplicity.
    :param tie: The final tie-break, a species class name or input class key.
    :return: The key; smaller keys come first.
    """
    return (-multiplicity, letters, tie)


def canonical_label_map(spacegroup: Spacegroup, class_letters: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Map input class keys to consecutive anonymous class labels in group order.

    Classes are ordered by :func:`class_order_key`: largest summed conventional
    multiplicity first, then their sorted Wyckoff-letter sequence (including
    repetitions); the input key breaks ties deterministically. Two classes with identical
    letter sequences are interchangeable, so either tie order renders the identical label.

    :param spacegroup: The standard-setting space group the letters belong to.
    :param class_letters: For each input class key its sorted Wyckoff letters.
    :return: A mapping from each input class key to its canonical anonymous label.
    """
    counts = {
        key: sum(spacegroup.wyckoff_position(letter).multiplicity for letter in letters)
        for key, letters in class_letters.items()
    }
    ordered = sorted(class_letters, key=lambda key: class_order_key(class_letters[key], counts[key], key))
    return {key: anonymous_symbol(index) for index, key in enumerate(ordered)}


def render_prototype_label(spacegroup: Spacegroup, occupations: Sequence[tuple[str, str]]) -> str:
    """Render the prototype label of a space group and its class-partitioned Wyckoff letters.

    :param spacegroup: The standard-setting space group.
    :param occupations: The occupied ``(wyckoff, class-key)`` pairs; the class key names
        the anonymous species class an occupation belongs to.
    :return: The prototype label text.
    :raises ValueError: If any Wyckoff letter is absent from the setting.
    """
    classes = _classes(spacegroup, occupations)
    ordered_keys = sorted(classes, key=lambda key: class_order_key(classes[key][0], classes[key][1], key))
    return _assemble(
        spacegroup,
        [classes[key][0] for key in ordered_keys],
        [classes[key][1] for key in ordered_keys],
    )


def render_protostructure_label(spacegroup: Spacegroup, occupations: Sequence[tuple[str, str]]) -> str:
    """Render the httk protostructure label of a space group and its named occupations.

    Classes are ordered by :func:`class_order_key`: largest summed conventional
    multiplicity first, then sorted Wyckoff letters, ties broken by species class name. The
    unsuffixed prefix equals the prototype label of the erased template; the suffix lists
    the species class names in group order.

    :param spacegroup: The standard-setting space group.
    :param occupations: The occupied ``(wyckoff, species-name)`` pairs.
    :return: The protostructure label text.
    :raises ValueError: If any Wyckoff letter is absent from the setting.
    """
    classes = _classes(spacegroup, occupations)
    ordered_keys = sorted(classes, key=lambda key: class_order_key(classes[key][0], classes[key][1], key))
    prefix = _assemble(
        spacegroup,
        [classes[key][0] for key in ordered_keys],
        [classes[key][1] for key in ordered_keys],
    )
    return prefix + ":" + "-".join(ordered_keys)


def render_aflow_label(spacegroup: Spacegroup, occupations: Sequence[tuple[str, str]]) -> str:
    """Render the AFLOW-style label of a space group and its named occupations.

    Unlike the httk label, classes are ordered by species name alphabetically and the
    anonymous symbols are reassigned in that order, so the unsuffixed prefix depends on
    the chemistry. Provided for interoperability only; there is no parser for this form.

    :param spacegroup: The standard-setting space group.
    :param occupations: The occupied ``(wyckoff, species-name)`` pairs.
    :return: The AFLOW-style label text.
    :raises ValueError: If any Wyckoff letter is absent from the setting.
    """
    classes = _classes(spacegroup, occupations)
    ordered_keys = sorted(classes)
    prefix = _assemble(
        spacegroup,
        [classes[key][0] for key in ordered_keys],
        [classes[key][1] for key in ordered_keys],
    )
    return prefix + ":" + "-".join(ordered_keys)


def _split_label(text: str) -> tuple[str, list[str], list[str] | None]:
    """Split a label into its main text, ``_``-separated fields, and optional name suffix."""
    if not isinstance(text, str) or not text:
        raise ValueError("label must be a non-empty string")
    main, sep, suffix = text.partition(":")
    names = suffix.split("-") if sep else None
    fields = main.split("_")
    if len(fields) < 4:
        raise ValueError("label must be ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*")
    return main, fields, names


def _parse_it_number(field: str) -> int:
    """Parse and range-check the International Tables number field."""
    if not field.isdigit():
        raise ValueError(f"label International Tables field {field!r} is not a number")
    it_number = int(field)
    if not 1 <= it_number <= 230:
        raise ValueError(f"label International Tables number {it_number} is out of range [1, 230]")
    return it_number


def _parse_group(token: str) -> list[str]:
    """Parse one group token into its Wyckoff-letter multiset."""
    letters: list[str] = []
    position = 0
    while position < len(token):
        match = _GROUP_TOKEN.match(token, position)
        if match is None or match.start() != position:
            raise ValueError(f"invalid group token {token!r}")
        digits, char = match.groups()
        if char.isupper() and char != _SPECIAL_RENDER:
            raise ValueError(f"invalid Wyckoff letter {char!r} in group token {token!r}")
        count = int(digits) if digits else 1
        if digits and count < 2:
            raise ValueError("explicit letter counts must be at least 2; explicit 1 is invalid")
        letters.extend([_SPECIAL_LETTER if char == _SPECIAL_RENDER else char] * count)
        position = match.end()
    return letters


def parse_bare_prototype_label(text: str) -> "BarePrototype":
    """Parse a strictly canonical prototype label into a prototype.

    Every Wyckoff letter must exist in the resolved standard setting, and the Pearson
    symbol, reduced anonymous counts, and group ordering must all match their recomputed
    canonical values; any deviation is rejected. This canonical-string-only stance mirrors
    :func:`~httk.atomistic.models.formula.notation.parse_anonymous_formula`.

    An undetermined label (``X`` in place of a part) gives its undetermined value
    (:meth:`~httk.atomistic.BarePrototype.undetermined`).

    :param text: The prototype label to parse.
    :return: The parsed :class:`~httk.atomistic.models.bareprototype.bareprototype.BarePrototype`.
    :raises ValueError: If ``text`` is not a canonical prototype label.
    """
    from httk.atomistic.models.bareprototype.bareprototype import BarePrototype

    if _marks_undetermined(text):
        return BarePrototype.undetermined(text)
    main, fields, names = _split_label(text)
    if names is not None:
        raise ValueError("a prototype label carries no ':' species suffix")
    it_number = _parse_it_number(fields[2])
    occupations: list[tuple[str, str]] = []
    for index, token in enumerate(fields[3:]):
        for letter in _parse_group(token):
            occupations.append((letter, anonymous_symbol(index)))
    value = BarePrototype(it_number, occupations)
    if render_prototype_label(value.spacegroup, [(o.wyckoff, o.label) for o in value.occupations]) != main:
        raise ValueError(f"{text!r} is not a canonical prototype label")
    return value


def parse_bare_protostructure_label(text: str) -> "BareProtostructure":
    """Parse a strictly canonical httk protostructure label into a protostructure.

    The unsuffixed part is validated as for a prototype label; each ``:`` name must be a
    canonical species class name (see :mod:`httk.atomistic.models.species.classes`) and
    becomes its class species. Non-canonical labels are rejected. An undetermined label gives
    its undetermined value (:meth:`~httk.atomistic.BareProtostructure.undetermined`).

    :param text: The protostructure label to parse.
    :return: The parsed :class:`~httk.atomistic.models.bareprotostructure.bareprotostructure.BareProtostructure`.
    :raises ValueError: If ``text`` is not a canonical protostructure label.
    """
    from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure

    if _marks_undetermined(text):
        return BareProtostructure.undetermined(text)
    _, fields, names = _split_label(text)
    if names is None:
        raise ValueError("a protostructure label needs a ':' species suffix")
    it_number = _parse_it_number(fields[2])
    groups = fields[3:]
    if len(names) != len(groups):
        raise ValueError(f"label has {len(groups)} class(es) but {len(names)} species name(s)")
    occupations: list[tuple[str, object]] = []
    for name, token in zip(names, groups):
        species = _species_from_class_name(name)
        for letter in _parse_group(token):
            occupations.append((letter, species))
    value = BareProtostructure(it_number, occupations)
    if render_protostructure_label(value.spacegroup, [(o.wyckoff, o.species.name) for o in value.occupations]) != text:
        raise ValueError(f"{text!r} is not a canonical protostructure label")
    return value


def try_parse_bare_prototype(text: str) -> "BarePrototype | None":
    """Return the parsed prototype, or ``None`` when *text* is not a canonical one.

    :param text: The label text to test.
    :return: The parsed prototype, or ``None`` for a non-label string.
    """
    if not isinstance(text, str) or ":" in text:
        return None
    try:
        return parse_bare_prototype_label(text)
    except ValueError:
        return None


def try_parse_bare_protostructure(text: str) -> "BareProtostructure | None":
    """Return the parsed protostructure, or ``None`` when *text* is not a canonical one.

    :param text: The label text to test.
    :return: The parsed protostructure, or ``None`` for a non-label string.
    """
    if not isinstance(text, str) or ":" not in text:
        return None
    try:
        return parse_bare_protostructure_label(text)
    except ValueError:
        return None


def _code_key(row: Sequence[int]) -> tuple[int, ...]:
    """Return the order key of a row's code: ``(n1²+n2²+n3², k, n1, n2, n3)``."""
    _, _, k, n1, n2, n3 = row
    return (n1 * n1 + n2 * n2 + n3 * n3, k, n1, n2, n3)


def _listing_positions(counts: Sequence[int], pairs: Iterable[tuple[int, int]], number: Sequence[int]) -> list[int]:
    """Return each site's 1-based position in the listing order of a net segment.

    A site is *top* when none of its bonded neighbours has a larger bond count. The listing
    order is larger bond count first, top sites first among equal counts, then smaller site
    number.

    :param counts: The bond count per 0-based site.
    :param pairs: The bonded 0-based site pairs, in either direction.
    :param number: The site number per 0-based site.
    :return: The listing position per 0-based site.
    """
    top = [True] * len(counts)
    for a, b in pairs:
        top[a] = top[a] and counts[b] <= counts[a]
        top[b] = top[b] and counts[a] <= counts[b]
    order = sorted(range(len(counts)), key=lambda s: (-counts[s], not top[s], number[s]))
    position = [0] * len(counts)
    for p, s in enumerate(order, 1):
        position[s] = p
    return position


def _label_rows(net: Iterable[Sequence[int]], counts: Sequence[int]) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return net rows, each written from its listing site, in label order.

    Label order sorts by the listing positions of the listing site and the target, then by code.

    :param net: The rows ``(i, j, k, n1, n2, n3)``, each listed from site ``i``.
    :param counts: The bond count per site, in site order.
    :return: The rows in label order.
    """
    rows = [(i, j, k, n1, n2, n3) for i, j, k, n1, n2, n3 in net]
    pos = _listing_positions(counts, [(i - 1, j - 1) for i, j, *_ in rows], range(1, len(counts) + 1))
    return tuple(sorted(rows, key=lambda row: (pos[row[0] - 1], pos[row[1] - 1], *_code_key(row))))


def _render_target(row: Sequence[int]) -> str:
    """Render the destination of one row: ``j``, ``j@k`` or ``j@k@klm``."""
    _, j, k, *shift = row
    if any(not -5 <= value <= 4 for value in shift):
        raise ValueError(f"protonet translation {tuple(shift)} is outside the writable range -5..+4")
    if any(shift):
        return f"{j}@{k}@" + "".join(str(value + 5) for value in shift)
    return f"{j}@{k}" if k != 1 else str(j)


def _render_net(net: Sequence[Sequence[int]], shapes: Sequence[str | None]) -> str:
    """Render the net part of a protonet label from its rows and its sites' shapes.

    The listing order comes from the label alone: a site's bond count is its shape's number, else
    1 if it is in a row, else 0. A site gets a group if it lists a row or its shape differs from
    the default of its bond count.
    """
    from httk.atomistic.symmetry.shapes import _bond_count, _written

    in_row = {site for row in net for site in row[:2]}
    counts = [
        _bond_count(symbol) if symbol is not None else int(site in in_row) for site, symbol in enumerate(shapes, 1)
    ]
    targets: dict[int, list[str]] = {}
    for row in _label_rows(net, counts):
        targets.setdefault(row[0], []).append(_render_target(row))
    pos = _listing_positions(counts, [(row[0] - 1, row[1] - 1) for row in net], range(1, len(shapes) + 1))
    groups = []
    for s in sorted(range(len(shapes)), key=pos.__getitem__):
        written = _written(shapes[s])
        if s + 1 in targets or written:
            groups.append(f"{s + 1}{written}" + ("~" + ".".join(targets[s + 1]) if s + 1 in targets else ""))
    return "_".join(groups)


def render_protonet(
    spacegroup: Spacegroup,
    class_letters: Sequence[tuple[str, ...]],
    net: Sequence[Sequence[int]],
    names: Sequence[str] | None = None,
    *,
    shapes: Sequence[str | None],
) -> str:
    """Render a protonettype or protonetstructure label.

    The bare part lists the classes in the given order, which is the value's class order
    and is not re-sorted, because it fixes the site numbering of the net. ``__`` and the
    net follow; an empty net without written shapes leaves a trailing ``__``. With ``names``
    the ``:`` species suffix closes the label.

    :param spacegroup: The standard-setting space group.
    :param class_letters: Each class's sorted Wyckoff letters, in class order.
    :param net: The rows ``(i, j, k, n1, n2, n3)`` of the bonding net, each listed from site
        ``i``, the end that comes first in the listing order.
    :param names: The class species names in class order, or ``None`` for the anonymous
        level.
    :param shapes: The shape symbol per site, ``None`` for a site with fewer than two bonds.
    :return: The label text.
    :raises ValueError: If a letter is absent from the setting or a translation cannot be
        written.
    """
    try:
        multiplicities = [
            sum(spacegroup.wyckoff_position(letter).multiplicity for letter in letters) for letters in class_letters
        ]
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    text = _assemble(spacegroup, class_letters, multiplicities) + "__" + _render_net(net, shapes)
    return text if names is None else text + ":" + "-".join(names)


def _parse_net(text: str) -> tuple[list[tuple[int, int, int, int, int, int]], dict[int, str]]:
    """Parse the net part of a protonet label into rows and written shapes, without canonicality checks."""
    if "__" in text:
        raise ValueError("a protonet label has exactly one '__' net segment")
    if _NET_RESERVED.intersection(text):
        raise ValueError(f"protonet net {text!r} uses a reserved character")
    rows: list[tuple[int, int, int, int, int, int]] = []
    written: dict[int, str] = {}
    opened: set[int] = set()
    for group in text.split("_") if text else ():
        head, tilde, targets = group.partition("~")
        match = _NET_HEAD.fullmatch(head)
        if match is None or not (tilde or match.group(2)):
            raise ValueError(f"invalid protonet group {group!r}: a site number, then a shape, '~' and bonds, or both")
        site = int(match.group(1))
        if site in opened:
            raise ValueError(f"protonet site {site} opens a second group in {text!r}; one group per site")
        opened.add(site)
        if match.group(2):
            written[site] = match.group(2)
        for target in targets.split(".") if tilde else ():
            target_match = _NET_TARGET.fullmatch(target)
            if target_match is None:
                raise ValueError(f"invalid protonet destination {target!r} in group {group!r}")
            j, k, klm = target_match.groups()
            n1, n2, n3 = (int(digit) - 5 for digit in klm) if klm else (0, 0, 0)
            rows.append((site, int(j), int(k) if k else 1, n1, n2, n3))
    return rows, written


def _shapes_with_defaults(
    spacegroup: Spacegroup,
    letters: Sequence[str],
    rows: Sequence[tuple[int, int, int, int, int, int]],
    written: Mapping[int, str],
) -> tuple[str | None, ...]:
    """Return every site's shape from the written symbols, filling the defaults by bond count.

    :param spacegroup: The standard-setting space group.
    :param letters: The Wyckoff letter per site, in site order.
    :param rows: The parsed net rows.
    :param written: The written symbol per 1-based site.
    :return: The shape per site, ``None`` for fewer than two bonds.
    :raises ValueError: If a row is malformed, a written symbol is unknown, does not end in its
        site's bond count or is on a site with fewer than two bonds, or a bond count without a
        default has no written symbol.
    """
    from httk.atomistic.models.protonettype.protonettype import _net_counts
    from httk.atomistic.symmetry.shapes import _DEFAULTS, _check_shape

    counts = _net_counts(spacegroup, letters, rows, "protonet label")
    if any(not 1 <= site <= len(letters) for site in written):
        raise ValueError(f"protonet shapes name a site outside 1..{len(letters)}")
    shapes: list[str | None] = []
    for site, count in enumerate(counts, 1):
        symbol = written.get(site)
        if symbol is not None:
            try:
                _check_shape(symbol, count)
            except ValueError as exc:
                raise ValueError(f"protonet site {site}: {exc}") from None
        elif count >= 2:
            symbol = _DEFAULTS.get(count)
            if symbol is None:
                raise ValueError(f"protonet site {site} has {count} bonds, which have no default shape; write one")
        shapes.append(symbol)
    return tuple(shapes)


def _parse_protonet(
    text: str,
) -> tuple[
    int, list[tuple[str, ...]], list[str] | None, list[tuple[int, int, int, int, int, int]], tuple[str | None, ...]
]:
    """Split a protonet label into IT number, per-class letters, names, net rows and shapes (defaults filled)."""
    if not isinstance(text, str):
        raise ValueError("label must be a string")
    bare, sep, rest = text.partition("__")
    if not sep:
        raise ValueError("a protonet label needs a '__' net segment")
    net_text, colon, suffix = rest.rpartition(":")
    if not colon:
        net_text, suffix = rest, ""
    _, fields, bare_names = _split_label(bare)
    if bare_names is not None:
        raise ValueError("the ':' species suffix of a protonet label follows the net")
    letters = [tuple(_parse_group(token)) for token in fields[3:]]
    rows, written = _parse_net(net_text)
    it_number = _parse_it_number(fields[2])
    shapes = _shapes_with_defaults(
        Spacegroup.standard(it_number), [letter for group in letters for letter in group], rows, written
    )
    return it_number, letters, suffix.split("-") if colon else None, rows, shapes


def _protonettype_parts(
    text: str,
) -> tuple[int, list[tuple[str, str]], list[tuple[int, int, int, int, int, int]], tuple[str | None, ...]]:
    """Return a protonettype label's IT number, site-ordered occupations, net rows and shapes, without a search."""
    it_number, letters, names, net, shapes = _parse_protonet(text)
    if names is not None:
        raise ValueError("a protonettype label carries no ':' species suffix")
    return (
        it_number,
        [(letter, anonymous_symbol(index)) for index, group in enumerate(letters) for letter in group],
        net,
        shapes,
    )


def _protonetstructure_parts(
    text: str,
) -> tuple[int, list[tuple[str, Species]], list[tuple[int, int, int, int, int, int]], tuple[str | None, ...]]:
    """Return a protonetstructure label's IT number, site-ordered occupations, net rows and shapes, without a search."""
    it_number, letters, names, net, shapes = _parse_protonet(text)
    if names is None:
        raise ValueError("a protonetstructure label needs a ':' species suffix")
    if len(names) != len(letters):
        raise ValueError(f"label has {len(letters)} class(es) but {len(names)} species name(s)")
    occupations = [(letter, _species_from_class_name(name)) for name, group in zip(names, letters) for letter in group]
    return it_number, occupations, net, shapes


def parse_protonettype_label(text: str) -> "Protonettype":
    """Parse a strictly canonical protonettype label.

    Class ``c`` of the bare part gets the anonymous label ``anonymous_symbol(c)``; the
    value is constructed (which validates and canonicalizes the net) and the whole label is
    re-rendered and compared with ``text``, which rejects leading zeros, unsorted or repeated
    groups, non-canonical ``@`` spellings and nets that are not the canonical one. The reserved characters ``- = # $ : / \\`` inside the
    net and a second ``__`` are rejected. An undetermined label gives its undetermined value
    (:meth:`~httk.atomistic.Protonettype.undetermined`), without a search.

    :param text: The protonettype label to parse.
    :return: The parsed :class:`~httk.atomistic.Protonettype`.
    :raises ValueError: If ``text`` is not a canonical protonettype label.
    """
    from httk.atomistic.models.protonettype.protonettype import Protonettype

    if _marks_undetermined(text):
        return Protonettype.undetermined(text)
    value = Protonettype(*_protonettype_parts(text))
    if str(value.label) != text:
        raise ValueError(f"{text!r} is not a canonical protonettype label")
    return value


def parse_protonetstructure_label(text: str) -> "Protonetstructure":
    """Parse a strictly canonical protonetstructure label.

    As :func:`parse_protonettype_label`, but the ``:`` species suffix is required; name
    ``c`` assigns class ``c`` and must be a canonical species class name, which becomes its
    class species.

    :param text: The protonetstructure label to parse.
    :return: The parsed :class:`~httk.atomistic.Protonetstructure`.
    :raises ValueError: If ``text`` is not a canonical protonetstructure label.
    """
    from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure

    if _marks_undetermined(text):
        return Protonetstructure.undetermined(text)
    value = Protonetstructure(*_protonetstructure_parts(text))
    if str(value.label) != text:
        raise ValueError(f"{text!r} is not a canonical protonetstructure label")
    return value


def try_parse_protonettype(text: str) -> "Protonettype | None":
    """Return the parsed protonettype, or ``None`` when *text* is not a canonical one.

    :param text: The label text to test.
    :return: The parsed protonettype, or ``None`` for any other string.
    """
    if not isinstance(text, str) or "__" not in text:
        return None
    try:
        return parse_protonettype_label(text)
    except ValueError:  # deliberately not CanonicalizationLimitError: an unchecked label reports the limit
        return None


def try_parse_protonetstructure(text: str) -> "Protonetstructure | None":
    """Return the parsed protonetstructure, or ``None`` when *text* is not a canonical one.

    :param text: The label text to test.
    :return: The parsed protonetstructure, or ``None`` for any other string.
    """
    if not isinstance(text, str) or "__" not in text:
        return None
    try:
        return parse_protonetstructure_label(text)
    except ValueError:  # deliberately not CanonicalizationLimitError: an unchecked label reports the limit
        return None


_NAMED_KINDS = frozenset({"bare_protostructure", "protostructure", "protonetstructure"})
_REFINED_KINDS = frozenset({"prototype", "protostructure"})


@dataclass(frozen=True)
class _Undetermined:
    """What an undetermined label determines; ``None`` where it does not.

    ``occupations`` are those of a determined bare part: sorted as in the bare value for a refined
    value whose geometry was not determined, in site order (class runs in label order) for a
    protonet value whose net was not determined.
    """

    formulatype: str
    spacegroup: Spacegroup | None
    nsites: int | None
    occupations: tuple[Any, ...] | None


def _marks_undetermined(text: object) -> bool:
    """Return whether a label text marks a part as undetermined: its last bare field or its net is ``X``."""
    return isinstance(text, str) and text.partition(":")[0].endswith("_X")


@lru_cache(maxsize=1024)
def _parse_undetermined(text: str, kind: str) -> _Undetermined:
    """Strictly parse the undetermined label of one label family, without any search.

    :param text: The label text.
    :param kind: The family's ``kind``.
    :return: What the label determines.
    :raises ValueError: If ``text`` is not an undetermined label of that family.
    """
    from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
    from httk.atomistic.models.prototype.occupation import PrototypeOccupation

    if not isinstance(text, str) or not text:
        raise ValueError("label must be a non-empty string")
    named, net = kind in _NAMED_KINDS, kind.startswith("protonet")
    main, colon, suffix = text.partition(":")
    if named != bool(colon):
        raise ValueError(f"a {kind} label {'needs' if named else 'carries no'} ':' species suffix")
    if net:
        main, _, rest = main.partition("__")
        if rest != "X":
            raise ValueError(f"{text!r} is not an undetermined {kind} label: its net segment must be '__X'")
    elif "__" in main:
        raise ValueError(f"a {kind} label has no '__' net segment")
    fields = main.split("_")
    if fields[-1] != "X":
        # a determined bare part: the net, or for a refined value the geometry, was not determined
        if not net and kind not in _REFINED_KINDS:
            raise ValueError(f"{text!r} is not an undetermined {kind} label")
        bare = parse_bare_protostructure_label(f"{main}:{suffix}") if named else parse_bare_prototype_label(main)
        if not net:
            occupations: tuple[Any, ...] = bare.occupations
        else:
            groups = [_parse_group(token) for token in fields[3:]]
            if named:
                occupations = tuple(
                    WyckoffOccupation(letter, _species_from_class_name(name))
                    for name, group in zip(suffix.split("-"), groups, strict=True)
                    for letter in group
                )
            else:
                occupations = tuple(
                    PrototypeOccupation(letter, anonymous_symbol(index))
                    for index, group in enumerate(groups)
                    for letter in group
                )
        return _Undetermined(fields[0], bare.spacegroup, bare.nsites_conventional, occupations)
    spacegroup: Spacegroup | None = None
    nsites: int | None = None
    primitive = 0
    if len(fields) == 4:
        it_number = _parse_it_number(fields[2])
        if str(it_number) != fields[2]:
            raise ValueError(f"{text!r}: the space-group number {fields[2]!r} is not written canonically")
        spacegroup = Spacegroup.standard(it_number)
        match = _PEARSON.fullmatch(fields[1])
        if match is None:
            raise ValueError(f"{text!r}: {fields[1]!r} is not a Pearson symbol")
        lattice_points = _LATTICE_POINTS[spacegroup.centring_type]
        nsites = int(match[1]) * (3 if spacegroup.centring_type == "R" else 1)
        if pearson_symbol(spacegroup, nsites) != fields[1]:
            raise ValueError(f"{text!r} has Pearson symbol {fields[1]!r}, not one of group {spacegroup.it_number}")
        if nsites % lattice_points:
            raise ValueError(f"{text!r}: {nsites} conventional sites do not fill the {lattice_points} lattice points")
        primitive = nsites // lattice_points
    elif len(fields) != 2:
        raise ValueError(
            f"{text!r} is not an undetermined {kind} label: expected ANON_X or ANON_PEARSON_ITNUMBER_X before "
            f"{'__X' if net else 'the end'}"
        )
    counts = [count for _, count in parse_anonymous_formula(fields[0])]
    if nsites is not None and primitive % sum(counts):
        raise ValueError(f"{text!r}: the primitive cell's sites are not a multiple of the Formulatype {fields[0]}")
    if named:
        names = suffix.split("-")
        for name in names:
            _species_from_class_name(name)
        if len(names) != len(counts) or len(set(names)) != len(names):
            raise ValueError(f"{text!r} needs one distinct species class name per Formulatype class")
        if [name for _, name in sorted(zip((-count for count in counts), names, strict=True))] != names:
            raise ValueError(f"{text!r}: names must be ordered by site count, largest first, then by name")
    return _Undetermined(fields[0], spacegroup, nsites, None)


def _render_undetermined(
    counts: Mapping[str, int | Fraction], spacegroup: Spacegroup | None, *, named: bool, net: bool
) -> str:
    """Render an undetermined label whose Wyckoff part (and with ``spacegroup=None`` its symmetry) is unknown.

    :param counts: Each class key (its species class name at the assigned levels) and its site count
        in one cell; with a space group, the summed conventional multiplicity.
    :param spacegroup: The standard-setting space group, or ``None`` when the symmetry is unknown.
    :param named: Whether to close the label with the ``:`` species suffix.
    :param net: Whether to add the protonet levels' ``__X``.
    :return: The label text; classes are ordered by count, largest first, then by key.
    """
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    coefficients = reduced_coefficients([Fraction(counts[key]) for key in ordered])
    assert coefficients is not None  # a value has at least one class
    text = render_anonymous(coefficients)
    if spacegroup is not None:
        text += f"_{pearson_symbol(spacegroup, int(sum(counts.values())))}_{spacegroup.it_number}"
    text += "_X__X" if net else "_X"
    return text + (":" + "-".join(ordered) if named else "")


def _with_net_marker(label: str) -> str:
    """Return a bare label, determined or not, with the undetermined net ``__X`` before any ``:`` suffix."""
    main, colon, names = label.partition(":")
    return f"{main}__X{colon}{names}"


def _undetermined(value: Any) -> _Undetermined | None:
    """Return what an undetermined label value's label determines, or ``None`` for a determined value."""
    text = getattr(value, "_undetermined", None)
    return None if text is None else _parse_undetermined(text, value.kind)


def _unknown(value: object, part: str) -> NoReturn:
    """Raise the error for a part that an undetermined value does not know."""
    raise ValueError(
        f"{type(value).__name__} {getattr(value, '_undetermined', None)!r} cannot give its {part}: "
        "the value is undetermined (determined is False)"
    )


def _known[T](value: object, part: str, known: T | None) -> T:
    """Return a part an undetermined value's label determines, or raise if it does not."""
    if known is None:
        _unknown(value, part)
    return known


def _erased_label(source: Any, kind: str) -> str | None:
    """Return an undetermined source's label cut to the broader level ``kind``, or ``None`` if it is determined.

    The label is cut as the levels nest: the names for an anonymous level, ``__X`` for a level
    without a net. What remains is a determined or undetermined label of that level.

    :param source: A label value, view or backend of a narrower level.
    :param kind: The ``kind`` of the broader level.
    :return: The cut label, or ``None`` for a determined source.
    """
    text = getattr(source, "_undetermined", None)
    if text is None:
        return None
    if kind not in _NAMED_KINDS:
        text = text.partition(":")[0]
    if not kind.startswith("protonet"):
        text = text.replace("__X", "", 1)
    return text


def _erased_undetermined_label(source: Any, kind: str) -> str | None:
    """Return the undetermined label of an undetermined source at the broader level ``kind``.

    ``None`` when the source is determined, or when the cut label is determined there (the bare
    part of a value whose net, or a refined value whose geometry, was not determined).
    """
    text = _erased_label(source, kind)
    if text is None or (kind not in _REFINED_KINDS and not _marks_undetermined(text)):
        return None
    return text


def _erased_undetermined(source: Any, kind: str) -> Any | None:
    """Return an undetermined source's value at the broader level ``kind``, or ``None`` if it is determined.

    :param source: A label value, view or backend of a narrower level.
    :param kind: The ``kind`` of the broader level.
    :return: The determined or undetermined value of that level, or ``None``.
    """
    text = _erased_label(source, kind)
    if text is None:
        return None
    if kind == "prototype":
        from httk.atomistic.models.prototype.prototype import Prototype

        return Prototype.undetermined(text)
    parse = {
        "bare_prototype": parse_bare_prototype_label,
        "bare_protostructure": parse_bare_protostructure_label,
        "protonettype": parse_protonettype_label,
    }[kind]
    return parse(text)
