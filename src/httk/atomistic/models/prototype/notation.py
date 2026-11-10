"""The one home of httk's prototype and protostructure label notation.

An httk label encodes the information content of an unsuffixed AFLOW-style prototype
label: a space group, its occupied Wyckoff letters, and the partition of those
occupations into species classes. The grammar is::

    ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*                 # prototype label
    ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*:NAME(-NAME)*    # protostructure label

A ``GROUP`` is the concatenation of one species class's Wyckoff letters, sorted
alphabetically, a letter occupied ``k >= 2`` times prefixed by the integer ``k``
(``2e``); count ``1`` is omitted. Groups appear in the order of :func:`class_order_key`:
largest summed conventional multiplicity first, then sorted Wyckoff letters. ``ANON`` is
the anonymous formula built from
:func:`~httk.atomistic.models.formula.notation.anonymous_symbol` in group order with
per-group summed conventional multiplicities reduced by their overall gcd, so it equals
the :class:`~httk.atomistic.Formulatype` of the site counts.

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
from collections.abc import Mapping, Sequence
from functools import reduce
from math import gcd
from typing import TYPE_CHECKING

from httk.atomistic.elements import SYMBOLS
from httk.atomistic.models.formula.notation import anonymous_symbol
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
_NET_TARGET = re.compile(r"([0-9]+)(?:@([0-9]+)(?:@([0-9]{3}))?)?")
_NET_RESERVED = frozenset("-=#$:/\\")
_ELEMENTS = frozenset(SYMBOLS)


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
    letters, and ``tie`` (a species name or input key) breaks what remains. Label
    rendering, anonymous relabelling and the canonicalisers' choice among equivalent
    descriptions all sort by this key.

    :param letters: The class's sorted Wyckoff letters, repetitions included.
    :param multiplicity: The class's summed conventional Wyckoff multiplicity.
    :param tie: The final tie-break, a species name or input class key.
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
    multiplicity first, then sorted Wyckoff letters, ties broken by species name. The
    unsuffixed prefix equals the prototype label of the erased template; the suffix lists
    the class species names in group order.

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

    :param text: The prototype label to parse.
    :return: The parsed :class:`~httk.atomistic.models.bareprototype.bareprototype.BarePrototype`.
    :raises ValueError: If ``text`` is not a canonical prototype label.
    """
    from httk.atomistic.models.bareprototype.bareprototype import BarePrototype

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
    known element symbol and becomes ``Species(name, (name,), (1,))``. Non-canonical labels
    are rejected.

    :param text: The protostructure label to parse.
    :return: The parsed :class:`~httk.atomistic.models.bareprotostructure.bareprotostructure.BareProtostructure`.
    :raises ValueError: If ``text`` is not a canonical protostructure label.
    """
    from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
    from httk.atomistic.models.species.species import Species

    _, fields, names = _split_label(text)
    if names is None:
        raise ValueError("a protostructure label needs a ':' species suffix")
    it_number = _parse_it_number(fields[2])
    groups = fields[3:]
    if len(names) != len(groups):
        raise ValueError(f"label has {len(groups)} class(es) but {len(names)} species name(s)")
    occupations: list[tuple[str, object]] = []
    for name, token in zip(names, groups):
        if name not in _ELEMENTS:
            raise ValueError(f"protostructure label name {name!r} is not a known element symbol")
        species = Species(name, (name,), (1,))
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


def _net_row_key(row: Sequence[int]) -> tuple[int, ...]:
    """Return the label order key of a row: site pair, then ``(n1²+n2²+n3², k, n1, n2, n3)``."""
    i, j, k, n1, n2, n3 = row
    return (i, j, n1 * n1 + n2 * n2 + n3 * n3, k, n1, n2, n3)


def _render_target(row: Sequence[int]) -> str:
    """Render the destination of one row: ``j``, ``j@k`` or ``j@k@klm``."""
    _, j, k, *shift = row
    if any(not -5 <= value <= 4 for value in shift):
        raise ValueError(f"protonet translation {tuple(shift)} is outside the writable range -5..+4")
    if any(shift):
        return f"{j}@{k}@" + "".join(str(value + 5) for value in shift)
    return f"{j}@{k}" if k != 1 else str(j)


def _render_net(net: Sequence[Sequence[int]]) -> str:
    """Render the net part of a protonet label from its rows."""
    groups: list[tuple[int, list[str]]] = []
    for row in sorted(net, key=_net_row_key):
        if not groups or groups[-1][0] != row[0]:
            groups.append((row[0], []))
        groups[-1][1].append(_render_target(row))
    return "_".join(f"{site}~" + ".".join(targets) for site, targets in groups)


def render_protonet(
    spacegroup: Spacegroup,
    class_letters: Sequence[tuple[str, ...]],
    net: Sequence[Sequence[int]],
    names: Sequence[str] | None = None,
) -> str:
    """Render a protonettype or protonetstructure label.

    The bare part lists the classes in the given order, which is the value's class order
    and is not re-sorted, because it fixes the site numbering of the net. ``__`` and the
    net follow; an empty net leaves a trailing ``__``. With ``names`` the ``:`` species
    suffix closes the label.

    :param spacegroup: The standard-setting space group.
    :param class_letters: Each class's sorted Wyckoff letters, in class order.
    :param net: The rows ``(i, j, k, n1, n2, n3)`` of the bonding net.
    :param names: The class species names in class order, or ``None`` for the anonymous
        level.
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
    text = _assemble(spacegroup, class_letters, multiplicities) + "__" + _render_net(net)
    return text if names is None else text + ":" + "-".join(names)


def _parse_net(text: str) -> list[tuple[int, int, int, int, int, int]]:
    """Parse the net part of a protonet label into rows, without canonicality checks."""
    if "__" in text:
        raise ValueError("a protonet label has exactly one '__' net segment")
    if _NET_RESERVED.intersection(text):
        raise ValueError(f"protonet net {text!r} uses a reserved character")
    rows: list[tuple[int, int, int, int, int, int]] = []
    for group in text.split("_") if text else ():
        site, tilde, targets = group.partition("~")
        if not tilde or not site.isascii() or not site.isdigit():
            raise ValueError(f"invalid protonet group {group!r}")
        for target in targets.split("."):
            match = _NET_TARGET.fullmatch(target)
            if match is None:
                raise ValueError(f"invalid protonet destination {target!r} in group {group!r}")
            j, k, klm = match.groups()
            n1, n2, n3 = (int(digit) - 5 for digit in klm) if klm else (0, 0, 0)
            rows.append((int(site), int(j), int(k) if k else 1, n1, n2, n3))
    return rows


def _parse_protonet(
    text: str,
) -> tuple[int, list[tuple[str, ...]], list[str] | None, list[tuple[int, int, int, int, int, int]]]:
    """Split a protonet label into IT number, per-class letters, names and net rows."""
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
    return _parse_it_number(fields[2]), letters, suffix.split("-") if colon else None, _parse_net(net_text)


def parse_protonettype_label(text: str) -> "Protonettype":
    """Parse a strictly canonical protonettype label.

    Class ``c`` of the bare part gets the anonymous label ``anonymous_symbol(c)``; the
    value is constructed (which validates the net) and the whole label is re-rendered and
    compared with ``text``, which rejects leading zeros, unsorted or repeated groups and
    non-canonical ``@`` spellings. The reserved characters ``- = # $ : / \\`` inside the
    net and a second ``__`` are rejected.

    :param text: The protonettype label to parse.
    :return: The parsed :class:`~httk.atomistic.Protonettype`.
    :raises ValueError: If ``text`` is not a canonical protonettype label.
    """
    from httk.atomistic.models.protonettype.protonettype import Protonettype

    it_number, letters, names, net = _parse_protonet(text)
    if names is not None:
        raise ValueError("a protonettype label carries no ':' species suffix")
    occupations = [(letter, anonymous_symbol(index)) for index, group in enumerate(letters) for letter in group]
    value = Protonettype(it_number, occupations, net)
    if str(value.label) != text:
        raise ValueError(f"{text!r} is not a canonical protonettype label")
    return value


def parse_protonetstructure_label(text: str) -> "Protonetstructure":
    """Parse a strictly canonical protonetstructure label.

    As :func:`parse_protonettype_label`, but the ``:`` species suffix is required; name
    ``c`` assigns class ``c`` and must be a known element symbol, which becomes
    ``Species(name, (name,), (1,))``.

    :param text: The protonetstructure label to parse.
    :return: The parsed :class:`~httk.atomistic.Protonetstructure`.
    :raises ValueError: If ``text`` is not a canonical protonetstructure label.
    """
    from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
    from httk.atomistic.models.species.species import Species

    it_number, letters, names, net = _parse_protonet(text)
    if names is None:
        raise ValueError("a protonetstructure label needs a ':' species suffix")
    if len(names) != len(letters):
        raise ValueError(f"label has {len(letters)} class(es) but {len(names)} species name(s)")
    occupations: list[tuple[str, Species]] = []
    for name, group in zip(names, letters):
        if name not in _ELEMENTS:
            raise ValueError(f"protonetstructure label name {name!r} is not a known element symbol")
        occupations.extend((letter, Species(name, (name,), (1,))) for letter in group)
    value = Protonetstructure(it_number, occupations, net)
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
    except ValueError:
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
    except ValueError:
        return None
