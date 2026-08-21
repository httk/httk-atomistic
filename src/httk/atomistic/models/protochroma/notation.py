"""The one home of httk's protochroma and protostructure label notation.

An httk label encodes the information content of an unsuffixed AFLOW-style prototype
label: a space group, its occupied Wyckoff letters, and the partition of those
occupations into species classes. The grammar is::

    ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*                 # protochroma label
    ANON_PEARSON_ITNUMBER_GROUP(_GROUP)*:NAME(-NAME)*    # protostructure label

A ``GROUP`` is the concatenation of one species class's Wyckoff letters, sorted
alphabetically, a letter occupied ``k >= 2`` times prefixed by the integer ``k``
(``2e``); count ``1`` is omitted. ``ANON`` is the anonymous formula built from
:func:`~httk.atomistic.models.formula.notation.anonymous_symbol` in group order with
per-group summed conventional multiplicities reduced by their overall gcd.

httk labels are deliberately NOT AFLOW labels. AFLOW orders classes by element symbol
(alphabetically) so its unsuffixed prefix still depends on the chemistry; httk orders
classes by their Wyckoff letters, so the unsuffixed prefix is element-agnostic and a
protostructure label is exactly the protochroma label of the erased form followed
by ``:`` and the class species names. :func:`render_aflow_label` renders the AFLOW-style
variant for interoperability; it has no parser here.

"Canonicality" terminology: any faithful render of an object is *the* protochroma or
protostructure label; the *canonical* protochroma or protostructure label is the one
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
    from httk.atomistic.models.protochroma.protochroma import Protochroma
    from httk.atomistic.models.protostructure.protostructure import Protostructure

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


def canonical_label_map(class_letters: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Map input class keys to consecutive anonymous class labels in group order.

    Classes are ordered lexicographically by their sorted Wyckoff-letter sequence
    (including repetitions); the input key breaks ties deterministically. Two classes
    with identical letter sequences are interchangeable, so either tie order renders the
    identical label.

    :param class_letters: For each input class key its sorted Wyckoff letters.
    :return: A mapping from each input class key to its canonical anonymous label.
    """
    ordered = sorted(class_letters, key=lambda key: (class_letters[key], key))
    return {key: anonymous_symbol(index) for index, key in enumerate(ordered)}


def render_protochroma_label(spacegroup: Spacegroup, occupations: Sequence[tuple[str, str]]) -> str:
    """Render the protochroma label of a space group and its class-partitioned Wyckoff letters.

    :param spacegroup: The standard-setting space group.
    :param occupations: The occupied ``(wyckoff, class-key)`` pairs; the class key names
        the anonymous species class an occupation belongs to.
    :return: The protochroma label text.
    :raises ValueError: If any Wyckoff letter is absent from the setting.
    """
    classes = _classes(spacegroup, occupations)
    ordered_keys = sorted(classes, key=lambda key: (classes[key][0], key))
    return _assemble(
        spacegroup,
        [classes[key][0] for key in ordered_keys],
        [classes[key][1] for key in ordered_keys],
    )


def render_protostructure_label(spacegroup: Spacegroup, occupations: Sequence[tuple[str, str]]) -> str:
    """Render the httk protostructure label of a space group and its named occupations.

    Classes are ordered by their sorted Wyckoff letters, ties broken by species name. The
    unsuffixed prefix equals the protochroma label of the erased form; the suffix lists
    the class species names in group order.

    :param spacegroup: The standard-setting space group.
    :param occupations: The occupied ``(wyckoff, species-name)`` pairs.
    :return: The protostructure label text.
    :raises ValueError: If any Wyckoff letter is absent from the setting.
    """
    classes = _classes(spacegroup, occupations)
    ordered_keys = sorted(classes, key=lambda key: (classes[key][0], key))
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


def parse_protochroma_label(text: str) -> "Protochroma":
    """Parse a strictly canonical protochroma label into a protochroma.

    Every Wyckoff letter must exist in the resolved standard setting, and the Pearson
    symbol, reduced anonymous counts, and group ordering must all match their recomputed
    canonical values; any deviation is rejected. This canonical-string-only stance mirrors
    :func:`~httk.atomistic.models.formula.notation.parse_anonymous_formula`.

    :param text: The protochroma label to parse.
    :return: The parsed :class:`~httk.atomistic.models.protochroma.protochroma.Protochroma`.
    :raises ValueError: If ``text`` is not a canonical protochroma label.
    """
    from httk.atomistic.models.protochroma.protochroma import Protochroma

    main, fields, names = _split_label(text)
    if names is not None:
        raise ValueError("a protochroma label carries no ':' species suffix")
    it_number = _parse_it_number(fields[2])
    occupations: list[tuple[str, str]] = []
    for index, token in enumerate(fields[3:]):
        for letter in _parse_group(token):
            occupations.append((letter, anonymous_symbol(index)))
    value = Protochroma(it_number, occupations)
    if render_protochroma_label(value.spacegroup, [(o.wyckoff, o.label) for o in value.occupations]) != main:
        raise ValueError(f"{text!r} is not a canonical protochroma label")
    return value


def parse_protostructure_label(text: str) -> "Protostructure":
    """Parse a strictly canonical httk protostructure label into a protostructure.

    The unsuffixed part is validated as for a protochroma label; each ``:`` name must be a
    known element symbol and becomes ``Species(name, (name,), (1,))``. Non-canonical labels
    are rejected.

    :param text: The protostructure label to parse.
    :return: The parsed :class:`~httk.atomistic.models.protostructure.protostructure.Protostructure`.
    :raises ValueError: If ``text`` is not a canonical protostructure label.
    """
    from httk.atomistic.models.protostructure.protostructure import Protostructure
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
    value = Protostructure(it_number, occupations)
    if render_protostructure_label(value.spacegroup, [(o.wyckoff, o.species.name) for o in value.occupations]) != text:
        raise ValueError(f"{text!r} is not a canonical protostructure label")
    return value


def try_parse_protochroma(text: str) -> "Protochroma | None":
    """Return the parsed protochroma, or ``None`` when *text* is not a canonical one.

    :param text: The label text to test.
    :return: The parsed protochroma, or ``None`` for a non-label string.
    """
    if not isinstance(text, str) or ":" in text:
        return None
    try:
        return parse_protochroma_label(text)
    except ValueError:
        return None


def try_parse_protostructure(text: str) -> "Protostructure | None":
    """Return the parsed protostructure, or ``None`` when *text* is not a canonical one.

    :param text: The label text to test.
    :return: The parsed protostructure, or ``None`` for a non-label string.
    """
    if not isinstance(text, str) or ":" not in text:
        return None
    try:
        return parse_protostructure_label(text)
    except ValueError:
        return None
